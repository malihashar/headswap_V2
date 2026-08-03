#!/usr/bin/env python3
"""A/B: crop_stitch vs full-frame author-parity on multi-person cases.

Hypothesis: tight head crops (~0.59MP) remove body/shoulder proportion cues
that Krea2 Identity Edit authors assume when documenting full-scene ~1–1.5MP
inputs — causing oversized heads. Isolate that from LoRA weight quality and
ref_boost.

Four arms (production crop_stitch path is NEVER modified by this script's
defaults — overrides are per-arm only):

  (a) crop_stitch + r64 LoRA + ref_boost 3.5     — current production
  (b) full_frame  + r64 LoRA + ref_boost 3.5     — full context alone
  (c) full_frame  + full v1.2 LoRA + ref_boost 3.5 — weight quality
  (d) full_frame  + full v1.2 LoRA + ref_boost 4.0 — author-matched baseline

Usage (GPU + ComfyUI + both LoRAs downloaded):
  PYTHONPATH=src python scripts/ab_full_frame_author_parity.py \\
    --out results/_ab_full_frame_author_parity

  # Prep only (scene MP / face boxes, no Krea2):
  PYTHONPATH=src python scripts/ab_full_frame_author_parity.py --prep-only

  python scripts/download_krea2.py --include-optional   # pulls full v1.2 LoRA
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.metrics.full_synth_eval import score_full_synth_case
from headswap.metrics.scoring import identity_cosine
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import (
    FaceBox,
    detect_best_face,
    detect_faces,
    get_face_landmarks5,
    pil_to_rgb_np,
    resize_max_keep_ar,
    select_face_box,
)


LORA_R64 = "krea2_identity_edit_v1_2_r64.safetensors"
LORA_FULL = "krea2_identity_edit_v1_2.safetensors"

DEFAULT_MULTI = (
    ROOT / "results" / "_krea2_full_vs_localized" / "night_group_001" / "00_body.png"
)
DEFAULT_FACE = ROOT / "data" / "custom" / "face.png"
DEFAULT_OUT = ROOT / "results" / "_ab_full_frame_author_parity"

ARMS: list[dict[str, Any]] = [
    {
        "id": "a_crop_r64_rb35",
        "label": "(a) crop_stitch + r64 + rb3.5",
        "multi_person_edit_mode": "crop_stitch",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "full_frame_identity_lora_name": None,
        "full_frame_ref_boost": None,
    },
    {
        "id": "b_ff_r64_rb35",
        "label": "(b) full_frame + r64 + rb3.5",
        "multi_person_edit_mode": "full_frame",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "full_frame_identity_lora_name": LORA_R64,
        "full_frame_ref_boost": 3.5,
    },
    {
        "id": "c_ff_full_rb35",
        "label": "(c) full_frame + full v1.2 + rb3.5",
        "multi_person_edit_mode": "full_frame",
        "identity_lora_name": LORA_FULL,
        "ref_boost": 3.5,
        "full_frame_identity_lora_name": LORA_FULL,
        "full_frame_ref_boost": 3.5,
    },
    {
        "id": "d_ff_full_rb4",
        "label": "(d) full_frame + full v1.2 + rb4.0",
        "multi_person_edit_mode": "full_frame",
        "identity_lora_name": LORA_FULL,
        "ref_boost": 4.0,
        "full_frame_identity_lora_name": LORA_FULL,
        "full_frame_ref_boost": 4.0,
    },
]


def _mp(size: tuple[int, int]) -> float:
    return (size[0] * size[1]) / 1_000_000.0


def _eye_line_deg(im: Image.Image, cache: Path, prefer: FaceBox | None) -> float | None:
    lm, _, _ = get_face_landmarks5(pil_to_rgb_np(im), cache, prefer_box=prefer)
    if lm is None or len(lm) < 2:
        return None
    le, re = lm[0], lm[1]
    return float(math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0]))))


def head_scale_metrics(
    body: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    cache: Path,
) -> dict[str, Any]:
    """Head-to-body scale: face_h / image_h on body vs result (key head-size metric)."""
    body_rgb = body.convert("RGB")
    res = result.convert("RGB")
    if res.size != body_rgb.size:
        res = res.resize(body_rgb.size, Image.Resampling.LANCZOS)

    body_faces = detect_faces(pil_to_rgb_np(body_rgb), cache, allow_prior=False)
    res_faces = detect_faces(pil_to_rgb_np(res), cache, allow_prior=False)

    def _nearest(target: FaceBox, faces: list[FaceBox]) -> FaceBox | None:
        if not faces:
            return None
        tcx = 0.5 * (target.x0 + target.x1)
        tcy = 0.5 * (target.y0 + target.y1)

        def dist(f: FaceBox) -> float:
            return (0.5 * (f.x0 + f.x1) - tcx) ** 2 + (0.5 * (f.y0 + f.y1) - tcy) ** 2

        return min(faces, key=dist)

    b_face = _nearest(selected, body_faces) or selected
    r_face = _nearest(selected, res_faces)

    ih = max(1, body_rgb.size[1])
    body_frac = float(b_face.height) / float(ih)
    result_frac = None if r_face is None else float(r_face.height) / float(ih)
    ratio = None if result_frac is None else float(result_frac) / max(1e-6, body_frac)
    area_ratio = None
    if r_face is not None:
        area_ratio = (r_face.width * r_face.height) / max(
            1.0, float(b_face.width * b_face.height)
        )

    return {
        "body_face_h_px": int(b_face.height),
        "result_face_h_px": None if r_face is None else int(r_face.height),
        "body_face_h_frac": round(body_frac, 4),
        "result_face_h_frac": None if result_frac is None else round(result_frac, 4),
        "head_to_body_scale_ratio": None if ratio is None else round(ratio, 4),
        "head_area_ratio": None if area_ratio is None else round(float(area_ratio), 4),
        "body_eye_line_deg": _eye_line_deg(body_rgb, cache, b_face),
        "result_eye_line_deg": (
            None if r_face is None else _eye_line_deg(res, cache, r_face)
        ),
    }


def _gaze_delta(hs: dict[str, Any]) -> float | None:
    a, b = hs.get("body_eye_line_deg"), hs.get("result_eye_line_deg")
    if a is None or b is None:
        return None
    return round(abs(float(b) - float(a)), 3)


def score_arm(
    *,
    arm_id: str,
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    all_faces: list[FaceBox],
    cache: Path,
    latency_s: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    hs = head_scale_metrics(body, result, selected, cache)
    full = score_full_synth_case(
        case_id=arm_id,
        pipeline=arm_id,
        body=body,
        face=face,
        result=result,
        selected=selected,
        all_faces=all_faces,
        cache_dir=cache,
        latency_s=latency_s,
    )
    id_cos = identity_cosine(face, result)
    diag = meta.get("face_prep_diag") or {}
    return {
        "arm": arm_id,
        "latency_s": round(float(latency_s), 3),
        "edit_mode": meta.get("edit_mode") or diag.get("edit_mode"),
        "scene_size": meta.get("scene_size") or diag.get("scene_size"),
        "scene_megapixels": diag.get("scene_megapixels")
        or (
            None
            if not meta.get("scene_size")
            else round(_mp(tuple(meta["scene_size"])), 4)
        ),
        "checkpoint_lora": (meta.get("loras_loaded") or [None])[0],
        "ref_boost": meta.get("ref_boost"),
        "identity_cosine": None if id_cos is None else round(float(id_cos), 4),
        "expression_landmark_l2": full.expression_landmark_l2,
        "pose_landmark_l2": full.pose_landmark_l2,
        "head_size_ratio_area": full.head_size_ratio,
        "neighbor_identity_mean": full.neighbor_identity_mean,
        "neighbor_count": full.neighbor_count,
        "gaze_eye_line_delta_deg": _gaze_delta(hs),
        "hair_transfer": "visual_inspection_required",
        **hs,
        "mock": bool(meta.get("mock")),
        "pipeline": meta.get("pipeline"),
    }


def _side_by_side(
    images: list[tuple[Image.Image, str]], *, height: int = 420
) -> Image.Image:
    fitted: list[Image.Image] = []
    for im, _ in images:
        im = im.convert("RGB")
        s = height / max(1, im.size[1])
        fitted.append(
            im.resize((max(1, int(im.size[0] * s)), height), Image.Resampling.LANCZOS)
        )
    gap, label_h = 10, 26
    total_w = sum(f.size[0] for f in fitted) + gap * max(0, len(fitted) - 1)
    canvas = Image.new("RGB", (total_w, height + label_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for f, (_, label) in zip(fitted, images):
        draw.text((x + 2, 4), label, fill=(220, 230, 255))
        canvas.paste(f, (x, label_h))
        x += f.size[0] + gap
    return canvas


def _cfg_for_arm(base: dict, arm: dict[str, Any]) -> dict:
    cfg = deepcopy(base)
    cfg["multi_person_edit_mode"] = arm["multi_person_edit_mode"]
    cfg["identity_lora_name"] = arm["identity_lora_name"]
    cfg["ref_boost"] = float(arm["ref_boost"])
    cfg["full_frame_identity_lora_name"] = arm.get("full_frame_identity_lora_name")
    cfg["full_frame_ref_boost"] = arm.get("full_frame_ref_boost")
    # Author-parity full-frame resolution (no-op for crop_stitch).
    cfg.setdefault("full_frame_target_mp", 1.25)
    cfg.setdefault("full_frame_min_mp", 1.0)
    cfg.setdefault("full_frame_max_mp", 1.5)
    cfg.setdefault("full_frame_max_dim", 2048)
    cfg.setdefault("full_frame_allow_upscale", False)
    cfg["save_debug"] = True
    cfg["verbose"] = False
    # Keep production crop_stitch recipe intact.
    cfg["single_person_parity"] = True
    cfg["mask_crop_stitch"] = True
    return cfg


def _assert_not_mock(meta: dict[str, Any], latency_s: float, arm_id: str) -> None:
    mode = str(meta.get("mode") or "")
    pipe = str(meta.get("pipeline") or "")
    if meta.get("mock") or "mock" in mode.lower():
        raise SystemExit(f"[{arm_id}] MOCK DETECTED — refusing to report results")
    if "krea2" not in pipe.lower() and pipe:
        # empty pipeline name shouldn't happen on real path
        pass
    if latency_s < 2.0 and not meta.get("timing_s"):
        raise SystemExit(
            f"[{arm_id}] latency_s={latency_s:.3f} looks like a stub; aborting"
        )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    cases = payload["cases"]
    arms = payload["arms"]
    lines = [
        "# Full-frame author-parity A/B",
        "",
        f"_Generated {payload.get('generated_at', '')}_",
        "",
        "## Hypothesis",
        "",
        "Tight head crops (~0.59MP) remove body/shoulder proportion cues that the",
        "Krea2 Identity Edit authors assume for full-scene ~1–1.5MP inputs,",
        "causing oversized heads. Isolate full-frame context vs LoRA weight vs ref_boost.",
        "",
        "## Arms",
        "",
        "| id | mode | LoRA | ref_boost |",
        "|---|---|---|---:|",
    ]
    for a in arms:
        lines.append(
            f"| `{a['id']}` | {a['multi_person_edit_mode']} | "
            f"`{a['identity_lora_name']}` | {a['ref_boost']} |"
        )

    lines += [
        "",
        "## Key metric",
        "",
        "`head_to_body_scale_ratio` = (result face_h / image_h) / (body face_h / image_h).",
        "1.0 = same relative head size as the original photo. >1 = oversized head.",
        "",
        "Also track `neighbor_identity_mean` (ArcFace of non-selected faces vs original)",
        "for face-drift between people — a known LoRA limitation when neighbors stay in frame.",
        "",
    ]

    for case in cases:
        cid = case["id"]
        lines += [f"## Case `{cid}`", ""]
        if case.get("skipped"):
            lines += [f"_Skipped: {case.get('reason')}_", ""]
            continue
        lines += [
            f"- body: `{case.get('body')}`",
            f"- face: `{case.get('face')}`",
            f"- faces detected: **{case.get('faces_detected')}**",
            f"- selected: `{case.get('selected_box')}`",
            "",
            f"![side_by_side]({cid}/side_by_side.png)",
            "",
            "| arm | scene MP | head_to_body_scale_ratio | identity_cosine | "
            "expr_l2 | gaze_Δ° | neighbor_id | latency_s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm_id, sc in (case.get("scores") or {}).items():
            if sc.get("error"):
                lines.append(f"| `{arm_id}` | ERR | — | — | — | — | — | — |")
                continue
            lines.append(
                "| `{arm}` | {mp} | **{hs}** | {idc} | {ex} | {gz} | {nb} | {lat} |".format(
                    arm=arm_id,
                    mp=sc.get("scene_megapixels"),
                    hs=sc.get("head_to_body_scale_ratio"),
                    idc=sc.get("identity_cosine"),
                    ex=(
                        None
                        if sc.get("expression_landmark_l2") is None
                        else round(float(sc["expression_landmark_l2"]), 4)
                    ),
                    gz=sc.get("gaze_eye_line_delta_deg"),
                    nb=sc.get("neighbor_identity_mean"),
                    lat=sc.get("latency_s"),
                )
            )
        lines += ["", "### Per-arm images", ""]
        for arm_id in (case.get("scores") or {}):
            lines += [
                f"**{arm_id}**",
                "",
                f"- input: `{cid}/{arm_id}/input_scene.png`",
                f"- final: `{cid}/{arm_id}/final_output.png`",
                f"- pre-stitch/edited: `{cid}/{arm_id}/debug_edited_crop.png` "
                f"(crop_stitch) or full-frame raw edit",
                "",
            ]

    j = payload.get("judgment") or {}
    lines += [
        "## Conclusions",
        "",
        "### Does full context alone (b vs a) fix head scale?",
        "",
        j.get("context_alone", "_run GPU arms to fill_"),
        "",
        "### Does full v1.2 weight quality matter (c vs b)?",
        "",
        j.get("full_lora", "_run GPU arms to fill_"),
        "",
        "### Does author-matched ref_boost=4 help (d vs c)?",
        "",
        j.get("ref_boost", "_run GPU arms to fill_"),
        "",
        "### Face-drift tradeoff (neighbors visible in full_frame)",
        "",
        j.get("face_drift_tradeoff", "_run GPU arms to fill_"),
        "",
        "### Recommendation",
        "",
        j.get("recommendation", "_run GPU arms to fill_"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _judge(cases: list[dict[str, Any]]) -> dict[str, str]:
    """Fill conclusion text from measured head-scale / neighbor metrics."""
    # Aggregate mean head_to_body_scale_ratio per arm across cases.
    by_arm: dict[str, list[float]] = {}
    neigh: dict[str, list[float]] = {}
    for case in cases:
        for arm_id, sc in (case.get("scores") or {}).items():
            if sc.get("error"):
                continue
            hs = sc.get("head_to_body_scale_ratio")
            if hs is not None:
                by_arm.setdefault(arm_id, []).append(float(hs))
            nb = sc.get("neighbor_identity_mean")
            if nb is not None:
                neigh.setdefault(arm_id, []).append(float(nb))

    def mean(arm: str) -> float | None:
        xs = by_arm.get(arm) or []
        return None if not xs else float(sum(xs) / len(xs))

    def nmean(arm: str) -> float | None:
        xs = neigh.get(arm) or []
        return None if not xs else float(sum(xs) / len(xs))

    a, b, c, d = (
        mean("a_crop_r64_rb35"),
        mean("b_ff_r64_rb35"),
        mean("c_ff_full_rb35"),
        mean("d_ff_full_rb4"),
    )
    if a is None or b is None:
        return {
            "context_alone": "Incomplete — need GPU results for arms (a) and (b).",
            "full_lora": "Incomplete.",
            "ref_boost": "Incomplete.",
            "face_drift_tradeoff": "Incomplete.",
            "recommendation": "Re-run without --prep-only / --mock on Colab GPU.",
        }

    def closer_to_one(x: float, y: float) -> bool:
        return abs(x - 1.0) + 1e-6 < abs(y - 1.0)

    ctx = (
        f"Mean head_to_body_scale_ratio (a)={a:.3f} vs (b)={b:.3f}. "
        + (
            "Full-frame context alone moves head scale closer to 1.0 — "
            "supporting the proportion-cue hypothesis."
            if closer_to_one(b, a)
            else "Full-frame context alone does NOT improve head scale vs crop_stitch; "
            "proportion cues are not the primary cause (or freeze/post hides the win)."
        )
    )
    full = "Incomplete (need arm c)."
    if c is not None and b is not None:
        full = (
            f"Mean ratio (b)={b:.3f} vs (c)={c:.3f}. "
            + (
                "Full v1.2 weights improve head-scale vs r64."
                if closer_to_one(c, b)
                else "Full v1.2 weights do not clearly beat r64 on head-scale."
            )
        )
    rb = "Incomplete (need arm d)."
    if d is not None and c is not None:
        rb = (
            f"Mean ratio (c)={c:.3f} vs (d)={d:.3f}. "
            + (
                "ref_boost=4 helps head-scale vs 3.5."
                if closer_to_one(d, c)
                else "ref_boost=4 does not clearly help vs 3.5 on head-scale."
            )
        )

    na, nb_ = nmean("a_crop_r64_rb35"), nmean("b_ff_r64_rb35")
    drift = (
        "Neighbor identity not measured (detector/InsightFace missing)."
        if na is None and nb_ is None
        else (
            f"neighbor_identity_mean (a)={na} vs (b)={nb_}. "
            "Higher = neighbors preserved. If (b) drops, full_frame reintroduces "
            "face-drift between people (author-documented limitation). Compare that "
            "delta against the head-scale win to decide the tradeoff."
        )
    )

    # Recommendation heuristic
    best_arm, best_val = "a_crop_r64_rb35", a
    for arm, val in (
        ("b_ff_r64_rb35", b),
        ("c_ff_full_rb35", c),
        ("d_ff_full_rb4", d),
    ):
        if val is not None and closer_to_one(val, best_val):
            best_arm, best_val = arm, val
    if best_arm == "a_crop_r64_rb35":
        rec = (
            "Keep production crop_stitch — full_frame did not win on head-scale. "
            "Look elsewhere for the oversized-head root cause."
        )
    elif best_arm.startswith("b_"):
        rec = (
            "Promote full_frame (r64, rb3.5) for multi-person if neighbor drift is "
            "acceptable; context alone appears sufficient."
        )
    elif best_arm.startswith("c_"):
        rec = (
            "Promote full_frame with full v1.2 LoRA (rb3.5); weight quality matters "
            "beyond context."
        )
    else:
        rec = (
            "Promote full_frame with full v1.2 LoRA and ref_boost=4 (author-matched). "
            "Watch neighbor face-drift on tight groups."
        )

    return {
        "context_alone": ctx,
        "full_lora": full,
        "ref_boost": rb,
        "face_drift_tradeoff": drift,
        "recommendation": rec,
    }


def _default_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    face = DEFAULT_FACE
    if DEFAULT_MULTI.is_file() and face.is_file():
        cases.append(
            {
                "id": "night_group_001",
                "body": str(DEFAULT_MULTI),
                "face": str(face),
                "policy": "rightmost",
            }
        )
    multi_big = ROOT / "results" / "_geometry_lock_smoke" / "multi_body.png"
    if multi_big.is_file() and face.is_file():
        cases.append(
            {
                "id": "geometry_lock_multi",
                "body": str(multi_big),
                "face": str(face),
                "policy": "largest",
            }
        )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, help="JSON list of {id,body,face,policy?}")
    ap.add_argument("--body", type=Path)
    ap.add_argument("--face", type=Path, default=DEFAULT_FACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument(
        "--arms",
        default="a,b,c,d",
        help="Subset of arms to run, e.g. a,b or a,b,c,d",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.config)

    if args.cases:
        raw = json.loads(Path(args.cases).read_text())
        cases_in = list(raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw)
    elif args.body:
        cases_in = [
            {
                "id": Path(args.body).stem,
                "body": str(args.body),
                "face": str(args.face),
                "policy": "largest",
            }
        ]
    else:
        cases_in = _default_cases()
        if not cases_in:
            raise SystemExit("No default multi-person cases found; pass --body/--face")

    want = {x.strip().lower() for x in str(args.arms).split(",") if x.strip()}
    arms = [
        a
        for a in ARMS
        if a["id"][0] in want or a["id"] in want or a["id"].split("_")[0] in want
    ]
    # Also allow full ids like a_crop_r64_rb35
    if not arms:
        arms = [a for a in ARMS if a["id"] in want]
    if not arms:
        arms = list(ARMS)

    summary_cases: list[dict[str, Any]] = []

    for case in cases_in:
        cid = str(case.get("id") or Path(case["body"]).stem)
        case_dir = out / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        body = Image.open(case["body"]).convert("RGB")
        face = Image.open(case["face"]).convert("RGB")
        policy = str(case.get("policy") or base_cfg.get("body_face_policy") or "largest")

        # Detect on a modest canvas for selection metadata (not the Krea2 scene).
        probe = resize_max_keep_ar(body, int(base_cfg.get("max_body_dim", 1024)), div_by=16)
        selected, all_faces = select_face_box(
            pil_to_rgb_np(probe), cache, policy=policy
        )
        row: dict[str, Any] = {
            "id": cid,
            "body": str(case["body"]),
            "face": str(case["face"]),
            "policy": policy,
            "faces_detected": len(all_faces),
            "selected_box": (
                None
                if selected is None
                else [selected.x0, selected.y0, selected.x1, selected.y1]
            ),
            "body_native_mp": round(_mp(body.size), 4),
            "scores": {},
            "metas": {},
        }
        if selected is None or len(all_faces) < 2:
            row["skipped"] = True
            row["reason"] = f"need >=2 faces (got {len(all_faces)})"
            summary_cases.append(row)
            print(f"[skip] {cid}: {row['reason']}")
            continue

        body.save(case_dir / "00_body.png")
        face.save(case_dir / "00_face.png")
        print(
            f"\n=== {cid}: {len(all_faces)} faces, selected="
            f"{row['selected_box']} native_mp={row['body_native_mp']} ==="
        )

        if args.prep_only:
            # Build conditioning inputs only (no sample) for crop vs full_frame.
            reset_shared_krea2_runtime()
            pipe = Krea2IdentityEditPipeline(cfg=dict(base_cfg), cache_dir=cache)
            from headswap.preprocess import crop_face_reference

            face_crop = crop_face_reference(face, cache)
            for arm in arms:
                cfg = _cfg_for_arm(base_cfg, arm)
                pipe.cfg = cfg
                if arm["multi_person_edit_mode"] == "full_frame":
                    from headswap.preprocess import resize_to_megapixels

                    body_ff = resize_to_megapixels(
                        body,
                        target_mp=float(cfg.get("full_frame_target_mp", 1.25)),
                        min_mp=float(cfg.get("full_frame_min_mp", 1.0)),
                        max_mp=float(cfg.get("full_frame_max_mp", 1.5)),
                        max_dim=int(cfg.get("full_frame_max_dim", 2048)),
                        div_by=int(cfg.get("div_by", 16)),
                        allow_upscale=bool(cfg.get("full_frame_allow_upscale", False)),
                    )
                    sel2, faces2 = select_face_box(
                        pil_to_rgb_np(body_ff), cache, policy=policy
                    )
                    built = pipe._build_full_frame_inputs(
                        body_ff,
                        face_crop,
                        div_by=int(cfg.get("div_by", 16)),
                        selected_face=sel2,
                        all_faces=faces2,
                    )
                else:
                    flags = pipe._tight_crop_flags(probe, selected, all_faces)
                    built = pipe._build_scene_person(
                        probe,
                        face_crop,
                        selected,
                        div_by=int(cfg.get("div_by", 16)),
                        use_tight=bool(flags["use_tight"]),
                        top_ext=float(flags["top_ext"]),
                        side_ext=float(flags["side_ext"]),
                        bot_ext=float(flags["bot_ext"]),
                        expand_px=int(flags["expand_px"]),
                        crop_pad=int(flags["crop_pad"]),
                        all_faces=all_faces,
                        isolate_selected=bool(flags.get("isolate_selected")),
                    )
                arm_dir = case_dir / arm["id"]
                arm_dir.mkdir(parents=True, exist_ok=True)
                built["scene"].save(arm_dir / "scene.png")
                built["person"].save(arm_dir / "person.png")
                diag = built.get("diag") or {}
                scene_mp = diag.get("scene_megapixels") or round(
                    _mp(built["scene"].size), 4
                )
                row["scores"][arm["id"]] = {
                    "arm": arm["id"],
                    "prep_only": True,
                    "edit_mode": arm["multi_person_edit_mode"],
                    "scene_size": list(built["scene"].size),
                    "scene_megapixels": scene_mp,
                    "ref_boost": arm["ref_boost"],
                    "identity_lora_name": arm["identity_lora_name"],
                }
                print(
                    f"  prep {arm['id']}: mode={arm['multi_person_edit_mode']} "
                    f"scene={built['scene'].size} mp={scene_mp}"
                )
            summary_cases.append(row)
            continue

        result_imgs: list[tuple[Image.Image, str]] = [(body, "original")]
        for arm in arms:
            arm_id = arm["id"]
            arm_dir = case_dir / arm_id
            arm_dir.mkdir(parents=True, exist_ok=True)
            cfg = _cfg_for_arm(base_cfg, arm)
            print(
                f"  running {arm_id} mode={arm['multi_person_edit_mode']} "
                f"lora={arm['identity_lora_name']} rb={arm['ref_boost']}…"
            )
            try:
                reset_shared_krea2_runtime()
                if args.mock:
                    from headswap.pipelines import create_pipeline

                    pipe = create_pipeline(cfg, force_mock=True)
                else:
                    pipe = Krea2IdentityEditPipeline(cfg=cfg, cache_dir=cache)
                # Force policy for this case
                pipe.cfg["body_face_policy"] = policy
                t0 = time.perf_counter()
                result = pipe.run(body, face, out_dir=arm_dir)
                latency = time.perf_counter() - t0
                meta = dict(result.meta or {})
                meta["mock"] = bool(args.mock)
                if not args.mock:
                    _assert_not_mock(meta, latency, arm_id)
                result.image.save(arm_dir / "result.png")
                result.image.save(arm_dir / "final_output.png")
                body.save(arm_dir / "input_scene.png")
                # Re-select on the canvas the arm actually used for scoring coords.
                body_for_score = body
                if arm["multi_person_edit_mode"] == "full_frame":
                    # Prefer debug body / meta body size — score on original body
                    # resized to result for fair geometry.
                    body_for_score = body
                sel_score, faces_score = select_face_box(
                    pil_to_rgb_np(
                        resize_max_keep_ar(
                            body_for_score,
                            max(result.image.size),
                            div_by=16,
                        )
                    ),
                    cache,
                    policy=policy,
                )
                if sel_score is None:
                    sel_score, faces_score = selected, all_faces
                sc = score_arm(
                    arm_id=arm_id,
                    body=body,
                    face=face,
                    result=result.image,
                    selected=sel_score,
                    all_faces=faces_score if faces_score else all_faces,
                    cache=cache,
                    latency_s=latency,
                    meta=meta,
                )
                row["scores"][arm_id] = sc
                row["metas"][arm_id] = {
                    "edit_mode": meta.get("edit_mode"),
                    "loras_loaded": meta.get("loras_loaded"),
                    "ref_boost": meta.get("ref_boost"),
                    "scene_size": meta.get("scene_size"),
                    "timing_s": meta.get("timing_s"),
                    "steps": meta.get("steps"),
                }
                result_imgs.append((result.image, arm_id))
                print(
                    f"    id={sc.get('identity_cosine')} "
                    f"head_scale={sc.get('head_to_body_scale_ratio')} "
                    f"scene_mp={sc.get('scene_megapixels')} "
                    f"neighbor_id={sc.get('neighbor_identity_mean')} "
                    f"lat={sc.get('latency_s')}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {arm_id}: {exc}")
                row["scores"][arm_id] = {"arm": arm_id, "error": str(exc)}

        if len(result_imgs) > 1:
            _side_by_side(result_imgs).save(case_dir / "side_by_side.png")
        summary_cases.append(row)

    judgment = _judge(summary_cases)
    if args.mock or args.prep_only:
        judgment = {
            "context_alone": "Incomplete — prep/mock only.",
            "full_lora": "Incomplete — prep/mock only.",
            "ref_boost": "Incomplete — prep/mock only.",
            "face_drift_tradeoff": "Incomplete — prep/mock only.",
            "recommendation": "Re-run on Colab GPU without --prep-only/--mock.",
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "full-scene ~1-1.5MP context fixes head scale vs head crop",
        "arms": arms,
        "cases": summary_cases,
        "judgment": judgment,
        "prep_only": bool(args.prep_only),
        "mock": bool(args.mock),
        "config": str(args.config),
        "note_face_drift": (
            "full_frame keeps neighbors in the Krea2 scene; author changelog warns "
            "two-person inputs can drift faces toward each other. Compare "
            "neighbor_identity_mean vs head_to_body_scale_ratio before promoting."
        ),
    }
    (out / "REPORT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out / "REPORT.md", payload)
    print(f"\nWrote {out / 'REPORT.md'}")
    print("Judgment:", json.dumps(judgment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
