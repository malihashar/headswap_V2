#!/usr/bin/env python3
"""Promote gate: full_frame + full v1.2 (c vs d) across the multi-person set.

Compares only:
  (prod) crop_stitch + r64 + rb3.5   — current production (delta baseline)
  (c)    full_frame  + full v1.2 + rb3.5
  (d)    full_frame  + full v1.2 + rb4.0

Arms (a)/(b) from the author-parity A/B are intentionally omitted.

Does NOT flip production yaml defaults. Writes REPORT.md with:
  - per-case metrics table (c | d | Δ vs prod)
  - aggregate recommendation (prefer d if close)
  - latency + VRAM cost tradeoff
  - whether Procrustes post-align is recommended
  - proposed production config diff (pending confirmation)

Usage (GPU + ComfyUI + full LoRA):
  PYTHONPATH=src python scripts/ab_full_frame_promote_cd.py \\
    --cases data/eval/multi_person_cases.json \\
    --out results/_ab_full_frame_promote_cd
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
    get_face_landmarks5,
    pil_to_rgb_np,
    resize_max_keep_ar,
    select_face_box,
)

# Reuse head-scale / scoring helpers from the author-parity script.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ab_full_frame_author_parity",
    ROOT / "scripts" / "ab_full_frame_author_parity.py",
)
assert _spec and _spec.loader
_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parity)

head_scale_metrics = _parity.head_scale_metrics
_gaze_delta = _parity._gaze_delta
_side_by_side = _parity._side_by_side
_assert_not_mock = _parity._assert_not_mock
_mp = _parity._mp
LORA_R64 = _parity.LORA_R64
LORA_FULL = _parity.LORA_FULL

# Head-scale |ratio-1| above this → recommend Procrustes.
HEAD_SCALE_OFF_THRESH = 0.08
# If |c-d| head-scale error delta below this, prefer d (author-documented rb=4).
CLOSE_HEAD_SCALE = 0.03
CLOSE_ID_COS = 0.015


ARMS: list[dict[str, Any]] = [
    {
        "id": "prod_crop_r64_rb35",
        "label": "production crop_stitch + r64 + rb3.5",
        "role": "production",
        "multi_person_edit_mode": "crop_stitch",
        "identity_lora_name": LORA_R64,
        "ref_boost": 3.5,
        "full_frame_identity_lora_name": None,
        "full_frame_ref_boost": None,
        "full_frame_procrustes_align": False,
    },
    {
        "id": "c_ff_full_rb35",
        "label": "(c) full_frame + full v1.2 + rb3.5",
        "role": "candidate",
        "multi_person_edit_mode": "full_frame",
        "identity_lora_name": LORA_FULL,
        "ref_boost": 3.5,
        "full_frame_identity_lora_name": LORA_FULL,
        "full_frame_ref_boost": 3.5,
        "full_frame_procrustes_align": False,
    },
    {
        "id": "d_ff_full_rb4",
        "label": "(d) full_frame + full v1.2 + rb4.0",
        "role": "candidate",
        "multi_person_edit_mode": "full_frame",
        "identity_lora_name": LORA_FULL,
        "ref_boost": 4.0,
        "full_frame_identity_lora_name": LORA_FULL,
        "full_frame_ref_boost": 4.0,
        "full_frame_procrustes_align": False,
    },
]


def _vram_peak_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except Exception:
        return None


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def hair_region_similarity(
    donor: Image.Image,
    result: Image.Image,
    selected: FaceBox,
    cache: Path,
) -> float | None:
    """HSV hist correlation on upper-head (hair) band vs donor face crop.

    Higher is better (OpenCV HISTCMP_CORREL ∈ [-1, 1]).
    """
    res = result.convert("RGB")
    don = donor.convert("RGB")
    # Crop result selected face with hair pad above.
    pad_top = int(0.70 * selected.height)
    pad_side = int(0.22 * selected.width)
    x0 = max(0, selected.x0 - pad_side)
    y0 = max(0, selected.y0 - pad_top)
    x1 = min(res.size[0], selected.x1 + pad_side)
    y1 = min(res.size[1], selected.y1 + int(0.12 * selected.height))
    if x1 <= x0 or y1 <= y0:
        return None
    head = res.crop((x0, y0, x1, y1))
    # Upper 45% ≈ hair band.
    hh = head.size[1]
    hair = head.crop((0, 0, head.size[0], max(1, int(0.45 * hh))))

    # Donor: prefer landmark-guided crop; else full donor resized.
    don_rgb = pil_to_rgb_np(don)
    lm, _, _ = get_face_landmarks5(don_rgb, cache)
    if lm is not None:
        xs = [float(p[0]) for p in lm]
        ys = [float(p[1]) for p in lm]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        fw = max(8.0, max(xs) - min(xs))
        fh = max(8.0, max(ys) - min(ys))
        dx0 = int(max(0, cx - 0.7 * fw))
        dy0 = int(max(0, cy - 1.6 * fh))
        dx1 = int(min(don.size[0], cx + 0.7 * fw))
        dy1 = int(min(don.size[1], cy + 0.2 * fh))
        donor_head = don.crop((dx0, dy0, dx1, dy1))
    else:
        donor_head = don
    dh = donor_head.size[1]
    donor_hair = donor_head.crop((0, 0, donor_head.size[0], max(1, int(0.45 * dh))))

    target_h = 96
    hair = hair.resize(
        (max(1, int(hair.size[0] * target_h / max(1, hair.size[1]))), target_h),
        Image.Resampling.LANCZOS,
    )
    donor_hair = donor_hair.resize(hair.size, Image.Resampling.LANCZOS)

    a = cv2.cvtColor(np.asarray(hair), cv2.COLOR_RGB2HSV)
    b = cv2.cvtColor(np.asarray(donor_hair), cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([a], [0, 1], None, [24, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [24, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    if math.isnan(corr):
        return None
    return round(corr, 4)


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
    vram_peak_mb: float | None,
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
    hair = hair_region_similarity(face, result, selected, cache)
    diag = meta.get("face_prep_diag") or {}
    samp = meta.get("sampling_diagnostics") or {}
    cuda_after = samp.get("cuda_after_sample") or {}
    return {
        "arm": arm_id,
        "latency_s": round(float(latency_s), 3),
        "vram_peak_mb": vram_peak_mb,
        "vram_allocated_after_sample_mb": cuda_after.get("allocated_mb"),
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
        "gaze_eye_line_delta_deg": _gaze_delta(hs),
        "hair_region_similarity": hair,
        "neighbor_identity_mean": full.neighbor_identity_mean,
        "neighbor_count": full.neighbor_count,
        **hs,
        "mock": bool(meta.get("mock")),
        "pipeline": meta.get("pipeline"),
    }


def _cfg_for_arm(base: dict, arm: dict[str, Any]) -> dict:
    cfg = deepcopy(base)
    cfg["multi_person_edit_mode"] = arm["multi_person_edit_mode"]
    cfg["identity_lora_name"] = arm["identity_lora_name"]
    cfg["ref_boost"] = float(arm["ref_boost"])
    cfg["full_frame_identity_lora_name"] = arm.get("full_frame_identity_lora_name")
    cfg["full_frame_ref_boost"] = arm.get("full_frame_ref_boost")
    cfg["full_frame_procrustes_align"] = bool(
        arm.get("full_frame_procrustes_align", False)
    )
    cfg.setdefault("full_frame_target_mp", 1.25)
    cfg.setdefault("full_frame_min_mp", 1.0)
    cfg.setdefault("full_frame_max_mp", 1.5)
    cfg.setdefault("full_frame_max_dim", 2048)
    cfg.setdefault("full_frame_allow_upscale", False)
    cfg["save_debug"] = True
    cfg["verbose"] = False
    cfg["single_person_parity"] = True
    cfg["mask_crop_stitch"] = True
    return cfg


def _mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return None if not vals else float(sum(vals) / len(vals))


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    arms = ["prod_crop_r64_rb35", "c_ff_full_rb35", "d_ff_full_rb4"]
    keys = [
        "head_to_body_scale_ratio",
        "identity_cosine",
        "gaze_eye_line_delta_deg",
        "expression_landmark_l2",
        "hair_region_similarity",
        "latency_s",
        "vram_peak_mb",
        "neighbor_identity_mean",
    ]
    agg: dict[str, Any] = {a: {} for a in arms}
    for a in arms:
        for k in keys:
            vals = []
            for case in cases:
                sc = (case.get("scores") or {}).get(a) or {}
                if sc.get("error"):
                    continue
                vals.append(sc.get(k))
            agg[a][k] = None if _mean(vals) is None else round(_mean(vals), 4)
            agg[a][f"n_{k}"] = sum(1 for v in vals if v is not None)
    return agg


def _recommend(agg: dict[str, Any]) -> dict[str, Any]:
    c = agg.get("c_ff_full_rb35") or {}
    d = agg.get("d_ff_full_rb4") or {}
    prod = agg.get("prod_crop_r64_rb35") or {}

    c_hs, d_hs = c.get("head_to_body_scale_ratio"), d.get("head_to_body_scale_ratio")
    c_id, d_id = c.get("identity_cosine"), d.get("identity_cosine")

    def err(x: float | None) -> float | None:
        return None if x is None else abs(float(x) - 1.0)

    c_err, d_err = err(c_hs), err(d_hs)
    reason_parts: list[str] = []
    winner = "d_ff_full_rb4"  # safer default when incomplete/close
    winner_ref_boost = 4.0

    if c_err is None or d_err is None:
        reason_parts.append(
            "Incomplete head-scale aggregates — defaulting to d (author rb=4)."
        )
    else:
        delta_err = abs(c_err - d_err)
        if delta_err <= CLOSE_HEAD_SCALE:
            # Close on head-scale → prefer d.
            if c_id is not None and d_id is not None and (c_id - d_id) > CLOSE_ID_COS:
                winner = "c_ff_full_rb35"
                winner_ref_boost = 3.5
                reason_parts.append(
                    f"Head-scale close (|Δerr|={delta_err:.3f}≤{CLOSE_HEAD_SCALE}); "
                    f"c wins on identity (+{c_id - d_id:.3f})."
                )
            else:
                winner = "d_ff_full_rb4"
                winner_ref_boost = 4.0
                reason_parts.append(
                    f"Head-scale close (|Δerr|={delta_err:.3f}≤{CLOSE_HEAD_SCALE}); "
                    "prefer d (ref_boost=4, author-documented)."
                )
        elif c_err + 1e-9 < d_err:
            winner = "c_ff_full_rb35"
            winner_ref_boost = 3.5
            reason_parts.append(
                f"c closer to 1.0 head-scale (err {c_err:.3f} vs {d_err:.3f})."
            )
        else:
            winner = "d_ff_full_rb4"
            winner_ref_boost = 4.0
            reason_parts.append(
                f"d closer to 1.0 head-scale (err {d_err:.3f} vs {c_err:.3f})."
            )

    win = agg.get(winner) or {}
    win_hs = win.get("head_to_body_scale_ratio")
    win_err = err(win_hs)
    need_procrustes = bool(win_err is not None and win_err > HEAD_SCALE_OFF_THRESH)

    # Cost vs production
    cost = {
        "prod_latency_s": prod.get("latency_s"),
        "winner_latency_s": win.get("latency_s"),
        "prod_vram_peak_mb": prod.get("vram_peak_mb"),
        "winner_vram_peak_mb": win.get("vram_peak_mb"),
        "latency_delta_s": (
            None
            if prod.get("latency_s") is None or win.get("latency_s") is None
            else round(float(win["latency_s"]) - float(prod["latency_s"]), 3)
        ),
        "vram_delta_mb": (
            None
            if prod.get("vram_peak_mb") is None or win.get("vram_peak_mb") is None
            else round(float(win["vram_peak_mb"]) - float(prod["vram_peak_mb"]), 1)
        ),
        "note": (
            "Full v1.2 LoRA + full-frame megapixel scenes may cost more latency/VRAM "
            "than crop_stitch+r64. Flag explicitly even when quality wins."
        ),
    }

    proposed = {
        "multi_person_edit_mode": "full_frame",
        "full_frame_identity_lora_name": LORA_FULL,
        "full_frame_ref_boost": winner_ref_boost,
        "full_frame_procrustes_align": need_procrustes,
        "rollback": {
            "multi_person_edit_mode": "crop_stitch",
            "identity_lora_name": LORA_R64,
            "ref_boost": 3.5,
            "full_frame_identity_lora_name": None,
            "full_frame_ref_boost": None,
            "full_frame_procrustes_align": False,
        },
        "single_person_unchanged": True,
        "status": "PENDING — do not apply to yaml until this REPORT has real GPU aggregates",
    }

    return {
        "winner_arm": winner,
        "winner_ref_boost": winner_ref_boost,
        "need_procrustes": need_procrustes,
        "head_scale_err": None if win_err is None else round(win_err, 4),
        "head_scale_off_thresh": HEAD_SCALE_OFF_THRESH,
        "reason": " ".join(reason_parts) or "Insufficient data.",
        "cost_vs_production": cost,
        "proposed_production_diff": proposed,
        "aggregates": agg,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rec = payload.get("recommendation") or {}
    agg = payload.get("aggregates") or {}
    lines = [
        "# Full-frame promote A/B (c vs d vs production)",
        "",
        f"_Generated {payload.get('generated_at', '')}_",
        "",
        "## Scope",
        "",
        "Candidates: **c** (`ff_full_rb35`) and **d** (`ff_full_rb4`) only.",
        "Production `crop_stitch+r64+rb3.5` is measured solely for Δ columns + cost.",
        "Arms a/b (r64 full_frame) are dropped — confirmed inferior on first-case visual.",
        "",
        "**Production yaml defaults are NOT changed by this script.**",
        "",
        "## Key metric",
        "",
        "`head_to_body_scale_ratio` = (result face_h / image_h) / (body face_h / image_h).",
        "1.0 = same relative head size as original. >1 = oversized.",
        "",
    ]

    # Aggregate table
    lines += [
        "## Aggregate (mean across cases)",
        "",
        "| metric | prod | c (rb3.5) | d (rb4) | Δc−prod | Δd−prod |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = [
        ("head_to_body_scale_ratio", True),
        ("identity_cosine", True),
        ("gaze_eye_line_delta_deg", False),
        ("expression_landmark_l2", False),
        ("hair_region_similarity", True),
        ("latency_s", False),
        ("vram_peak_mb", False),
        ("neighbor_identity_mean", True),
    ]
    prod_a = agg.get("prod_crop_r64_rb35") or {}
    c_a = agg.get("c_ff_full_rb35") or {}
    d_a = agg.get("d_ff_full_rb4") or {}
    for key, _higher_better in metrics:
        pv, cv, dv = prod_a.get(key), c_a.get(key), d_a.get(key)

        def _d(a, b):
            if a is None or b is None:
                return "—"
            return round(float(a) - float(b), 4)

        lines.append(
            f"| `{key}` | {pv} | {cv} | {dv} | {_d(cv, pv)} | {_d(dv, pv)} |"
        )

    lines += ["", "## Per-case tables", ""]
    for case in payload.get("cases") or []:
        cid = case["id"]
        lines += [f"### `{cid}`", ""]
        if case.get("skipped"):
            lines += [f"_Skipped: {case.get('reason')}_", ""]
            continue
        lines += [
            f"- body: `{case.get('body')}`",
            f"- faces: **{case.get('faces_detected')}**",
            f"- selected: `{case.get('selected_box')}`",
            "",
            f"![side_by_side]({cid}/side_by_side.png)",
            "",
            "| arm | head_scale | id_cos | gaze_Δ° | expr_l2 | hair_sim | "
            "lat_s | vram_mb | Δhs−prod | Δid−prod |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        scores = case.get("scores") or {}
        prod_sc = scores.get("prod_crop_r64_rb35") or {}
        for arm_id in ("prod_crop_r64_rb35", "c_ff_full_rb35", "d_ff_full_rb4"):
            sc = scores.get(arm_id) or {}
            if sc.get("error"):
                lines.append(f"| `{arm_id}` | ERR | — | — | — | — | — | — | — | — |")
                continue
            hs = sc.get("head_to_body_scale_ratio")
            idc = sc.get("identity_cosine")
            phs = prod_sc.get("head_to_body_scale_ratio")
            pid = prod_sc.get("identity_cosine")
            dhs = (
                "—"
                if hs is None or phs is None or arm_id.startswith("prod")
                else round(float(hs) - float(phs), 4)
            )
            did = (
                "—"
                if idc is None or pid is None or arm_id.startswith("prod")
                else round(float(idc) - float(pid), 4)
            )
            ex = sc.get("expression_landmark_l2")
            lines.append(
                "| `{arm}` | **{hs}** | {idc} | {gz} | {ex} | {hair} | "
                "{lat} | {vram} | {dhs} | {did} |".format(
                    arm=arm_id,
                    hs=hs,
                    idc=idc,
                    gz=sc.get("gaze_eye_line_delta_deg"),
                    ex=None if ex is None else round(float(ex), 4),
                    hair=sc.get("hair_region_similarity"),
                    lat=sc.get("latency_s"),
                    vram=sc.get("vram_peak_mb"),
                    dhs=dhs,
                    did=did,
                )
            )
        lines.append("")

    cost = rec.get("cost_vs_production") or {}
    lines += [
        "## Cost tradeoff (winner vs production)",
        "",
        f"- prod latency: **{cost.get('prod_latency_s')}** s",
        f"- winner latency: **{cost.get('winner_latency_s')}** s "
        f"(Δ {cost.get('latency_delta_s')} s)",
        f"- prod VRAM peak: **{cost.get('prod_vram_peak_mb')}** MB",
        f"- winner VRAM peak: **{cost.get('winner_vram_peak_mb')}** MB "
        f"(Δ {cost.get('vram_delta_mb')} MB)",
        "",
        f"> {cost.get('note')}",
        "",
        "## Recommendation",
        "",
        f"- **Winner:** `{rec.get('winner_arm')}` "
        f"(ref_boost={rec.get('winner_ref_boost')})",
        f"- **Reason:** {rec.get('reason')}",
        f"- **Procrustes needed?** "
        f"**{rec.get('need_procrustes')}** "
        f"(winner |hs−1|={rec.get('head_scale_err')}, "
        f"thresh={rec.get('head_scale_off_thresh')})",
        "",
        "## Proposed production config diff (NOT APPLIED)",
        "",
        "```yaml",
        json.dumps(rec.get("proposed_production_diff") or {}, indent=2),
        "```",
        "",
        "Single-person path stays `crop_stitch` + r64. Rollback keeps old multi path.",
        "",
        "## SPP regression",
        "",
        (payload.get("spp_regression") or {}).get(
            "summary", "_Run scripts/ab_full_frame_spp_check.py after winner session._"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases:
        raw = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        default_face = None
        if isinstance(raw, dict):
            default_face = raw.get("default_face")
            cases_in = list(raw.get("cases") or [])
        else:
            cases_in = list(raw)
        out = []
        for c in cases_in:
            body = Path(c["body"])
            if not body.is_absolute():
                body = ROOT / body
            face = Path(c.get("face") or default_face or args.face)
            if not face.is_absolute():
                face = ROOT / face
            if not body.is_file():
                print(f"[warn] missing body {body}; skip")
                continue
            if not face.is_file():
                print(f"[warn] missing face {face}; skip")
                continue
            out.append(
                {
                    "id": str(c.get("id") or body.stem),
                    "body": str(body),
                    "face": str(face),
                    "policy": str(c.get("policy") or "largest"),
                }
            )
        return out
    if args.body:
        return [
            {
                "id": Path(args.body).stem,
                "body": str(args.body),
                "face": str(args.face),
                "policy": "largest",
            }
        ]
    # Default multi set
    default = ROOT / "data" / "eval" / "multi_person_cases.json"
    if default.is_file():
        args.cases = default
        return _resolve_cases(args)
    raise SystemExit("Pass --cases or --body/--face")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--body", type=Path)
    ap.add_argument("--face", type=Path, default=ROOT / "data" / "custom" / "face.png")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "_ab_full_frame_promote_cd")
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument(
        "--extra-case",
        action="append",
        default=[],
        help="Extra id:body:face[:policy] to append (Colab upload)",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.config)

    cases_in = _resolve_cases(args)
    for raw in args.extra_case or []:
        parts = str(raw).split(":")
        if len(parts) < 3:
            raise SystemExit(f"--extra-case needs id:body:face[:policy], got {raw}")
        cases_in.append(
            {
                "id": parts[0],
                "body": parts[1],
                "face": parts[2],
                "policy": parts[3] if len(parts) > 3 else "largest",
            }
        )

    summary_cases: list[dict[str, Any]] = []
    for case in cases_in:
        cid = case["id"]
        case_dir = out / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        body = Image.open(case["body"]).convert("RGB")
        face = Image.open(case["face"]).convert("RGB")
        policy = str(case.get("policy") or "largest")
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
            f"\n=== {cid}: {len(all_faces)} faces selected={row['selected_box']} "
            f"mp={row['body_native_mp']} ==="
        )

        if args.prep_only:
            row["scores"] = {
                a["id"]: {"prep_only": True, "edit_mode": a["multi_person_edit_mode"]}
                for a in ARMS
            }
            summary_cases.append(row)
            continue

        result_imgs: list[tuple[Image.Image, str]] = [(body, "original")]
        for arm in ARMS:
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
                pipe.cfg["body_face_policy"] = policy
                _reset_vram_peak()
                t0 = time.perf_counter()
                result = pipe.run(body, face, out_dir=arm_dir)
                latency = time.perf_counter() - t0
                vram = _vram_peak_mb()
                meta = dict(result.meta or {})
                meta["mock"] = bool(args.mock)
                if not args.mock:
                    _assert_not_mock(meta, latency, arm_id)
                result.image.save(arm_dir / "result.png")
                result.image.save(arm_dir / "final_output.png")
                body.save(arm_dir / "input_scene.png")
                sel_score, faces_score = select_face_box(
                    pil_to_rgb_np(
                        resize_max_keep_ar(body, max(result.image.size), div_by=16)
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
                    vram_peak_mb=vram,
                )
                row["scores"][arm_id] = sc
                row["metas"][arm_id] = {
                    "edit_mode": meta.get("edit_mode"),
                    "loras_loaded": meta.get("loras_loaded"),
                    "ref_boost": meta.get("ref_boost"),
                    "scene_size": meta.get("scene_size"),
                    "timing_s": meta.get("timing_s"),
                    "vram_peak_mb": vram,
                }
                result_imgs.append((result.image, arm_id.replace("_ff_full_", "_")))
                print(
                    f"    id={sc.get('identity_cosine')} "
                    f"hs={sc.get('head_to_body_scale_ratio')} "
                    f"hair={sc.get('hair_region_similarity')} "
                    f"lat={sc.get('latency_s')}s vram={vram}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {arm_id}: {exc}")
                row["scores"][arm_id] = {"arm": arm_id, "error": str(exc)}

        if len(result_imgs) > 1:
            _side_by_side(result_imgs).save(case_dir / "side_by_side.png")
        summary_cases.append(row)

    aggregates = _aggregate(summary_cases)
    recommendation = _recommend(aggregates)
    if args.mock or args.prep_only:
        recommendation["reason"] = "Incomplete — prep/mock only. Re-run on Colab GPU."
        recommendation["proposed_production_diff"]["status"] = (
            "PENDING — mock/prep only; do not apply"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": ARMS,
        "cases": summary_cases,
        "aggregates": aggregates,
        "recommendation": recommendation,
        "prep_only": bool(args.prep_only),
        "mock": bool(args.mock),
        "config": str(args.config),
        "spp_regression": {
            "summary": "See scripts/ab_full_frame_spp_check.py / Colab §5 post-promote."
        },
        "note": (
            "Do not flip multi_person_edit_mode / LoRA / ref_boost in yaml until "
            "this REPORT has real GPU aggregates across the multi set."
        ),
    }
    (out / "REPORT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "WINNER.json").write_text(
        json.dumps(
            {
                "winner_arm": recommendation.get("winner_arm"),
                "winner_ref_boost": recommendation.get("winner_ref_boost"),
                "need_procrustes": recommendation.get("need_procrustes"),
                "proposed_production_diff": recommendation.get(
                    "proposed_production_diff"
                ),
                "cost_vs_production": recommendation.get("cost_vs_production"),
                "reason": recommendation.get("reason"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out / "REPORT.md", payload)
    print(f"\nWrote {out / 'REPORT.md'}")
    print("Winner:", json.dumps(recommendation, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
