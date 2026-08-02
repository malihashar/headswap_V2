#!/usr/bin/env python3
"""Hypothesis test: is the single vs multi quality gap due to native face resolution?

Degrades a high-quality single-person scene so its face has approximately the same
*effective* native pixel height as a failing multi-person example (downsample →
re-upsample, framing unchanged), then runs the existing single-person Krea2
pipeline unchanged on original vs degraded.

Does NOT modify the Krea2 pipeline, prompts, masks, stitching, or add restorers.

Usage (GPU + ComfyUI + Krea2):
  PYTHONPATH=src python scripts/ab_face_resolution_degradation.py \\
    --out results/_ab_face_resolution_degradation

Prep only (measure + degrade + save inputs, skip Krea2):
  PYTHONPATH=src python scripts/ab_face_resolution_degradation.py --prep-only

Mock smoke (no GPU sample):
  PYTHONPATH=src python scripts/ab_face_resolution_degradation.py --mock
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
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
from headswap.metrics.scoring import identity_cosine
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import (
    FaceBox,
    detect_best_face,
    detect_faces,
    get_face_landmarks5,
    pil_to_rgb_np,
    select_face_box,
)


DEFAULT_SINGLE_BODY = ROOT / "data" / "custom" / "body.png"
DEFAULT_FACE = ROOT / "data" / "custom" / "face.png"
DEFAULT_MULTI_BODY = (
    ROOT / "results" / "_krea2_full_vs_localized" / "night_group_001" / "00_body.png"
)
DEFAULT_OUT = ROOT / "results" / "_ab_face_resolution_degradation"


# ---------------------------------------------------------------------------
# Face geometry / degradation
# ---------------------------------------------------------------------------


def _face_info(box: FaceBox | None) -> dict[str, Any]:
    if box is None:
        return {"box": None, "height_px": None, "width_px": None, "conf": None}
    return {
        "box": [int(box.x0), int(box.y0), int(box.x1), int(box.y1)],
        "height_px": int(box.height),
        "width_px": int(box.width),
        "conf": round(float(box.conf), 4),
    }


def measure_selected_face(
    image: Image.Image,
    cache_dir: Path,
    *,
    policy: str = "largest",
    allow_prior: bool = False,
) -> tuple[FaceBox | None, list[FaceBox], dict[str, Any]]:
    rgb = pil_to_rgb_np(image.convert("RGB"))
    selected, all_faces = select_face_box(
        rgb, cache_dir, policy=policy, conf_thresh=0.30
    )
    if selected is None and allow_prior:
        selected = detect_best_face(rgb, cache_dir)
        all_faces = [selected] if selected is not None else []
    info = {
        "image_size": list(image.size),
        "faces_detected": len(all_faces),
        "selected": _face_info(selected),
        "all_faces": [_face_info(f) for f in all_faces],
        "policy": policy,
    }
    return selected, all_faces, info


def degrade_to_face_height(
    scene: Image.Image,
    *,
    current_face_h: float,
    target_face_h: float,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Downsample then re-upsample so effective native face height ≈ target.

    Framing (final canvas size) is unchanged. Returns
    (degraded_full_res, intermediate_low_res, meta).
    """
    scene = scene.convert("RGB")
    w, h = scene.size
    cur = max(1.0, float(current_face_h))
    tgt = max(1.0, float(target_face_h))
    scale = float(tgt / cur)
    # Never upscale the intermediate — only simulate detail loss.
    scale = min(1.0, scale)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    # BOX/AREA ≈ averaging → information loss; BICUBIC upsample → soft blur.
    arr = np.asarray(scene)
    small_arr = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored_arr = cv2.resize(small_arr, (w, h), interpolation=cv2.INTER_CUBIC)
    intermediate = Image.fromarray(small_arr)
    degraded = Image.fromarray(restored_arr)
    meta = {
        "current_face_h_px": round(cur, 2),
        "target_face_h_px": round(tgt, 2),
        "downsample_scale": round(scale, 6),
        "intermediate_size": [small_w, small_h],
        "restored_size": [w, h],
        "effective_native_face_h_px": round(cur * scale, 2),
        "framing_unchanged": True,
    }
    return degraded, intermediate, meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _crop_face(im: Image.Image, box: FaceBox, pad: float = 0.15) -> Image.Image:
    w, h = im.size
    px = int(round(pad * box.width))
    py = int(round(pad * box.height))
    x0 = max(0, box.x0 - px)
    y0 = max(0, box.y0 - py)
    x1 = min(w, box.x1 + px)
    y1 = min(h, box.y1 + py)
    return im.convert("RGB").crop((x0, y0, x1, y1))


def laplacian_sharpness(im: Image.Image, box: FaceBox | None = None) -> float | None:
    """Variance of Laplacian on the face crop (or full image). Higher = sharper."""
    if box is not None:
        im = _crop_face(im, box)
    if im.size[0] < 8 or im.size[1] < 8:
        return None
    gray = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _eye_stats(
    im: Image.Image, cache_dir: Path, prefer: FaceBox | None = None
) -> dict[str, Any]:
    rgb = pil_to_rgb_np(im.convert("RGB"))
    lm, backend, note = get_face_landmarks5(rgb, cache_dir, prefer_box=prefer)
    out: dict[str, Any] = {
        "landmarks_backend": backend,
        "landmarks_note": note,
        "left_eye": None,
        "right_eye": None,
        "iod": None,
        "eye_line_deg": None,
    }
    if lm is None or len(lm) < 2:
        return out
    le, re = lm[0], lm[1]
    out["left_eye"] = [round(float(le[0]), 2), round(float(le[1]), 2)]
    out["right_eye"] = [round(float(re[0]), 2), round(float(re[1]), 2)]
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    out["iod"] = round(float(math.hypot(dx, dy)), 3)
    out["eye_line_deg"] = round(math.degrees(math.atan2(dy, dx)), 3)
    out["landmarks"] = lm.tolist()
    return out


def landmark_eye_error(
    body: Image.Image,
    result: Image.Image,
    cache_dir: Path,
    *,
    body_box: FaceBox | None,
    result_box: FaceBox | None,
) -> dict[str, Any]:
    """Mean eye landmark L2 error (px) and error / body IOD.

    Aligns result eye positions into body coordinates by translating face
    centers, then compares left/right eye locations. Lower is better.
    """
    be = _eye_stats(body, cache_dir, prefer=body_box)
    re = _eye_stats(result, cache_dir, prefer=result_box)
    out: dict[str, Any] = {
        "body_eyes": {k: be[k] for k in ("left_eye", "right_eye", "iod", "eye_line_deg")},
        "result_eyes": {
            k: re[k] for k in ("left_eye", "right_eye", "iod", "eye_line_deg")
        },
        "eye_error_px": None,
        "eye_error_over_iod": None,
        "eye_line_delta_deg": None,
    }
    if (
        be.get("left_eye") is None
        or be.get("right_eye") is None
        or re.get("left_eye") is None
        or re.get("right_eye") is None
        or body_box is None
        or result_box is None
    ):
        return out

    # Translate result eyes into body face-center frame so absolute coords compare.
    bcx = 0.5 * (body_box.x0 + body_box.x1)
    bcy = 0.5 * (body_box.y0 + body_box.y1)
    rcx = 0.5 * (result_box.x0 + result_box.x1)
    rcy = 0.5 * (result_box.y0 + result_box.y1)
    dx, dy = bcx - rcx, bcy - rcy

    def _err(a: list[float], b: list[float]) -> float:
        return float(math.hypot(a[0] - (b[0] + dx), a[1] - (b[1] + dy)))

    e_l = _err(be["left_eye"], re["left_eye"])
    e_r = _err(be["right_eye"], re["right_eye"])
    mean_e = 0.5 * (e_l + e_r)
    out["eye_error_px"] = round(mean_e, 3)
    iod = float(be["iod"] or 0.0)
    if iod > 1.0:
        out["eye_error_over_iod"] = round(mean_e / iod, 4)
    if be.get("eye_line_deg") is not None and re.get("eye_line_deg") is not None:
        out["eye_line_delta_deg"] = round(
            abs(float(re["eye_line_deg"]) - float(be["eye_line_deg"])), 3
        )
    return out


def score_arm(
    *,
    face_ref: Image.Image,
    body: Image.Image,
    result: Image.Image,
    cache_dir: Path,
    latency_s: float,
    arm: str,
) -> dict[str, Any]:
    body_rgb = body.convert("RGB")
    result_rgb = result.convert("RGB")
    if result_rgb.size != body_rgb.size:
        result_rgb = result_rgb.resize(body_rgb.size, Image.Resampling.LANCZOS)

    body_face = detect_best_face(pil_to_rgb_np(body_rgb), cache_dir)
    result_faces = detect_faces(pil_to_rgb_np(result_rgb), cache_dir, allow_prior=False)
    result_face = result_faces[0] if result_faces else None

    id_cos = identity_cosine(face_ref, result_rgb)
    eye = landmark_eye_error(
        body_rgb, result_rgb, cache_dir, body_box=body_face, result_box=result_face
    )
    sharp_body = laplacian_sharpness(body_rgb, body_face)
    sharp_result = laplacian_sharpness(result_rgb, result_face)

    return {
        "arm": arm,
        "latency_s": round(float(latency_s), 3),
        "identity_cosine": None if id_cos is None else round(float(id_cos), 4),
        "landmark_eye_error_px": eye.get("eye_error_px"),
        "landmark_eye_error_over_iod": eye.get("eye_error_over_iod"),
        "eye_line_delta_deg": eye.get("eye_line_delta_deg"),
        "face_detector_confidence_body": None
        if body_face is None
        else round(float(body_face.conf), 4),
        "face_detector_confidence_result": None
        if result_face is None
        else round(float(result_face.conf), 4),
        "face_sharpness_laplacian_body": None
        if sharp_body is None
        else round(float(sharp_body), 2),
        "face_sharpness_laplacian_result": None
        if sharp_result is None
        else round(float(sharp_result), 2),
        "body_face": _face_info(body_face),
        "result_face": _face_info(result_face),
        "eye_detail": eye,
        "result_size": list(result_rgb.size),
    }


# ---------------------------------------------------------------------------
# Pipeline run (unchanged config)
# ---------------------------------------------------------------------------


def run_single_person(
    body: Image.Image,
    face: Image.Image,
    cfg: dict,
    cache_dir: Path,
    out_dir: Path,
    *,
    mock: bool = False,
) -> tuple[Image.Image, dict[str, Any], float]:
    """Run production Krea2 config as-is (single-person body → crop_stitch path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = deepcopy(cfg)
    run_cfg["save_debug"] = True
    run_cfg["verbose"] = False

    if mock:
        from headswap.pipelines import create_pipeline

        pipe = create_pipeline(run_cfg, force_mock=True)
    else:
        reset_shared_krea2_runtime()
        pipe = Krea2IdentityEditPipeline(cfg=run_cfg, cache_dir=cache_dir)

    t0 = time.perf_counter()
    result = pipe.run(body, face, out_dir=out_dir)
    latency = time.perf_counter() - t0
    result.image.save(out_dir / "result.png")

    # Canonical names requested by the experiment (pipeline already wrote debug_*).
    body.save(out_dir / "input_scene.png")
    debug_map: dict[str, str] = {}
    for name in (
        "debug_edited_crop.png",
        "debug_crop.png",
        "debug_scene.png",
        "debug_person.png",
        "debug_mask.png",
        "debug_final.png",
    ):
        p = out_dir / name
        if p.is_file():
            debug_map[name] = str(p)
    edited = out_dir / "debug_edited_crop.png"
    if edited.is_file():
        shutil.copy2(edited, out_dir / "pre_stitch_edited_crop.png")
        debug_map["pre_stitch_edited_crop.png"] = str(
            out_dir / "pre_stitch_edited_crop.png"
        )
    result.image.save(out_dir / "final_output.png")

    meta = dict(result.meta or {})
    meta["debug_files"] = debug_map
    meta["mock"] = bool(mock)
    return result.image, meta, latency


def _side_by_side(
    images: list[tuple[Image.Image, str]], *, height: int = 512
) -> Image.Image:
    fitted: list[Image.Image] = []
    for im, _ in images:
        im = im.convert("RGB")
        s = height / max(1, im.size[1])
        fitted.append(
            im.resize((max(1, int(im.size[0] * s)), height), Image.Resampling.LANCZOS)
        )
    gap = 12
    label_h = 28
    total_w = sum(f.size[0] for f in fitted) + gap * (len(fitted) - 1)
    canvas = Image.new("RGB", (total_w, height + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (f, (_, label)) in zip(fitted, images):
        draw.text((x + 4, 6), label, fill=(200, 220, 255))
        canvas.paste(f, (x, label_h))
        x += f.size[0] + gap
    return canvas


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 4)


def classify_failure_modes(
    original: dict[str, Any] | None, degraded: dict[str, Any] | None
) -> dict[str, Any]:
    """Heuristic failure-mode flags from original → degraded metric deltas."""
    if not original or not degraded:
        return {
            "identity": "unknown",
            "gaze": "unknown",
            "expression": "unknown",
            "hair": "visual_inspection_required",
            "beard": "visual_inspection_required",
            "notes": "Pipeline metrics unavailable.",
        }

    id_o = original.get("identity_cosine")
    id_d = degraded.get("identity_cosine")
    eye_o = original.get("landmark_eye_error_over_iod")
    eye_d = degraded.get("landmark_eye_error_over_iod")
    gaze_o = original.get("eye_line_delta_deg")
    gaze_d = degraded.get("eye_line_delta_deg")
    conf_o = original.get("face_detector_confidence_result")
    conf_d = degraded.get("face_detector_confidence_result")
    sharp_o = original.get("face_sharpness_laplacian_result")
    sharp_d = degraded.get("face_sharpness_laplacian_result")

    identity = "not_observed"
    if id_o is not None and id_d is not None:
        drop = float(id_o) - float(id_d)
        if drop >= 0.08:
            identity = "appeared"
        elif drop >= 0.03:
            identity = "mild"
        else:
            identity = "not_observed"

    gaze = "not_observed"
    # Prefer normalized eye error; fall back to eye-line angle.
    if eye_o is not None and eye_d is not None:
        if float(eye_d) >= float(eye_o) + 0.08:
            gaze = "appeared"
        elif float(eye_d) >= float(eye_o) + 0.03:
            gaze = "mild"
    elif gaze_o is not None and gaze_d is not None:
        if float(gaze_d) >= float(gaze_o) + 6.0:
            gaze = "appeared"
        elif float(gaze_d) >= float(gaze_o) + 3.0:
            gaze = "mild"

    expression = gaze  # eye-landmark drift is the available expression proxy here

    notes: list[str] = []
    if conf_o is not None and conf_d is not None and float(conf_d) + 0.05 < float(conf_o):
        notes.append("result face detector confidence dropped on degraded arm")
    if sharp_o is not None and sharp_d is not None and float(sharp_d) < 0.7 * float(sharp_o):
        notes.append("result face sharpness dropped substantially on degraded arm")

    return {
        "identity": identity,
        "gaze": gaze,
        "expression": expression,
        "hair": "visual_inspection_required",
        "beard": "visual_inspection_required",
        "notes": "; ".join(notes) if notes else "",
        "deltas": {
            "identity_cosine": _delta(id_o, id_d),
            "landmark_eye_error_over_iod": _delta(eye_o, eye_d),
            "eye_line_delta_deg": _delta(gaze_o, gaze_d),
            "face_detector_confidence_result": _delta(conf_o, conf_d),
            "face_sharpness_laplacian_result": _delta(sharp_o, sharp_d),
        },
    }


def judge_hypothesis(
    *,
    prep: dict[str, Any],
    original: dict[str, Any] | None,
    degraded: dict[str, Any] | None,
    modes: dict[str, Any],
    ran_pipeline: bool,
) -> dict[str, Any]:
    """Answer the three REPORT questions from evidence."""
    if not ran_pipeline or not original or not degraded:
        return {
            "reproduced_multi_failure_mode": "inconclusive",
            "failure_modes_appeared": modes,
            "resolution_role": "inconclusive",
            "rationale": (
                "Krea2 was not run (or metrics missing). Prep confirms the single-person "
                f"face is {prep.get('single_face_h_px')}px vs multi "
                f"{prep.get('multi_face_h_px')}px; re-run without --prep-only on GPU."
            ),
        }

    id_o = original.get("identity_cosine")
    id_d = degraded.get("identity_cosine")
    sharp_body_o = original.get("face_sharpness_laplacian_body")
    sharp_body_d = degraded.get("face_sharpness_laplacian_body")
    id_drop = None if id_o is None or id_d is None else float(id_o) - float(id_d)
    sharp_ok = (
        sharp_body_o is not None
        and sharp_body_d is not None
        and float(sharp_body_d) < 0.5 * float(sharp_body_o)
    )

    appeared = [
        k
        for k in ("identity", "gaze", "expression")
        if modes.get(k) in ("appeared", "mild")
    ]

    if id_drop is not None and id_drop >= 0.08 and sharp_ok:
        reproduced = "yes"
        role = "primary_cause"
        rationale = (
            f"Degrading effective face resolution ({prep.get('single_face_h_px')}→"
            f"{prep.get('multi_face_h_px')}px native) dropped identity_cosine by "
            f"{id_drop:.3f} and reduced input face sharpness; failure modes "
            f"{appeared or ['(see visual)']} appeared. This is strong evidence that "
            "native face resolution is a primary driver of the single/multi gap."
        )
    elif id_drop is not None and id_drop >= 0.03 and sharp_ok:
        reproduced = "partially"
        role = "contributing_factor"
        rationale = (
            f"Identity dropped by {id_drop:.3f} after matching multi native face "
            "height, with clear input sharpness loss, but the magnitude alone may "
            "not fully explain multi-person failures. Treat resolution as a "
            "contributing factor."
        )
    elif sharp_ok and not appeared:
        reproduced = "no"
        role = "not_significant"
        rationale = (
            "Input face detail was successfully degraded to the multi-person native "
            "height, but swap metrics stayed close to the original arm. Effective "
            "face resolution alone does not appear to explain the quality gap."
        )
    elif id_drop is not None and id_drop < 0.03:
        reproduced = "no"
        role = "not_significant"
        rationale = (
            f"Identity cosine barely moved (Δ={id_drop:.3f}) despite matching the "
            "multi-person native face height. Resolution is unlikely to be the "
            "primary cause of the single/multi gap."
        )
    else:
        reproduced = "inconclusive"
        role = "contributing_factor"
        rationale = (
            "Metrics are mixed or incomplete after degradation. Resolution may "
            "contribute, but this run does not isolate it as the primary cause."
        )

    return {
        "reproduced_multi_failure_mode": reproduced,
        "failure_modes_appeared": modes,
        "resolution_role": role,
        "rationale": rationale,
        "identity_drop": id_drop,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    prep = payload["prep"]
    scores = payload.get("scores") or {}
    original = scores.get("original")
    degraded = scores.get("degraded")
    judgment = payload["judgment"]
    modes = judgment["failure_modes_appeared"]

    def _fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
        return str(v)

    lines = [
        "# Face-resolution degradation A/B",
        "",
        f"_Generated {payload.get('generated_at', '')}_",
        "",
        "## Hypothesis",
        "",
        "The gap between single-person and multi-person quality is caused by the "
        "effective native resolution of the target face, not crop/canvas size.",
        "",
        "## Setup",
        "",
        f"- Single-person body: `{prep['single_body']}`",
        f"- Identity face: `{prep['face']}`",
        f"- Multi-person reference body: `{prep['multi_body']}`",
        f"- Single face height (native): **{prep['single_face_h_px']} px**",
        f"- Multi face height (native, before crop/resize): **{prep['multi_face_h_px']} px**",
        f"- Downsample scale: **{prep['degrade']['downsample_scale']}**",
        f"- Effective native face height after degrade: **{prep['degrade']['effective_native_face_h_px']} px**",
        f"- Framing unchanged: **{prep['degrade']['framing_unchanged']}**",
        f"- Pipeline modified: **no** (production config, single-person path)",
        "",
        "## Metrics",
        "",
        "| metric | original | degraded | Δ (deg−orig) |",
        "|---|---:|---:|---:|",
    ]
    keys = [
        ("identity_cosine", "ArcFace identity cosine"),
        ("landmark_eye_error_over_iod", "landmark eye error / IOD"),
        ("landmark_eye_error_px", "landmark eye error (px)"),
        ("eye_line_delta_deg", "eye-line delta (deg)"),
        ("face_detector_confidence_result", "face detector confidence (result)"),
        ("face_sharpness_laplacian_body", "face sharpness Laplacian (input)"),
        ("face_sharpness_laplacian_result", "face sharpness Laplacian (result)"),
        ("latency_s", "latency (s)"),
    ]
    for key, label in keys:
        o = None if not original else original.get(key)
        d = None if not degraded else degraded.get(key)
        lines.append(f"| {label} | {_fmt(o)} | {_fmt(d)} | {_fmt(_delta(o, d))} |")

    lines += [
        "",
        "## Answers",
        "",
        "### Did degrading the effective face resolution reproduce the multi-person failure mode?",
        "",
        f"**{judgment['reproduced_multi_failure_mode']}**",
        "",
        judgment["rationale"],
        "",
        "### Which failure modes appeared? (identity, gaze, expression, hair, beard)",
        "",
        f"- identity: **{modes.get('identity')}**",
        f"- gaze: **{modes.get('gaze')}**",
        f"- expression: **{modes.get('expression')}**",
        f"- hair: **{modes.get('hair')}**",
        f"- beard: **{modes.get('beard')}**",
        "",
    ]
    if modes.get("notes"):
        lines += [f"_Notes:_ {modes['notes']}", ""]

    lines += [
        "### Is effective face resolution likely to be a primary cause, a contributing factor, or not significant?",
        "",
        f"**{judgment['resolution_role']}**",
        "",
        "---",
        "",
        "Artifacts: `original/`, `degraded/`, `side_by_side.png`, `REPORT.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="A/B: degrade single-person face resolution to multi native height"
    )
    ap.add_argument("--single-body", type=Path, default=DEFAULT_SINGLE_BODY)
    ap.add_argument("--face", type=Path, default=DEFAULT_FACE)
    ap.add_argument("--multi-body", type=Path, default=DEFAULT_MULTI_BODY)
    ap.add_argument(
        "--multi-face-policy",
        default="rightmost",
        help="Which multi-person face to measure (default: rightmost, matches night_group)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    ap.add_argument(
        "--target-face-h",
        type=float,
        default=None,
        help="Override measured multi face height (px)",
    )
    ap.add_argument(
        "--prep-only",
        action="store_true",
        help="Measure + degrade + save inputs; skip Krea2",
    )
    ap.add_argument("--mock", action="store_true", help="Use mock pipeline (no GPU)")
    args = ap.parse_args()

    for p, label in (
        (args.single_body, "single-body"),
        (args.face, "face"),
        (args.multi_body, "multi-body"),
    ):
        if not Path(p).is_file():
            raise SystemExit(f"Missing {label}: {p}")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)

    single = Image.open(args.single_body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    multi = Image.open(args.multi_body).convert("RGB")

    print("Measuring faces…")
    single_face, _, single_info = measure_selected_face(
        single, cache, policy="largest", allow_prior=True
    )
    multi_face, _, multi_info = measure_selected_face(
        multi, cache, policy=str(args.multi_face_policy), allow_prior=False
    )
    if single_face is None:
        raise SystemExit("No face detected on single-person body")
    if multi_face is None:
        raise SystemExit(
            f"No face detected on multi-person body with policy={args.multi_face_policy}"
        )

    target_h = (
        float(args.target_face_h)
        if args.target_face_h is not None
        else float(multi_face.height)
    )
    degraded, intermediate, degrade_meta = degrade_to_face_height(
        single,
        current_face_h=float(single_face.height),
        target_face_h=target_h,
    )
    deg_face, _, deg_info = measure_selected_face(
        degraded, cache, policy="largest", allow_prior=True
    )

    # Save prep artifacts
    single.save(out / "00_single_input_scene.png")
    multi.save(out / "00_multi_reference_scene.png")
    face.save(out / "00_identity_face.png")
    intermediate.save(out / "01_degraded_intermediate_lowres.png")
    degraded.save(out / "01_degraded_scene.png")

    # Overlay face boxes for sanity
    def _overlay(im: Image.Image, box: FaceBox | None, label: str, path: Path) -> None:
        vis = im.convert("RGB").copy()
        d = ImageDraw.Draw(vis)
        if box is not None:
            d.rectangle([box.x0, box.y0, box.x1, box.y1], outline=(0, 255, 80), width=3)
            d.text((box.x0, max(0, box.y0 - 14)), label, fill=(0, 255, 80))
        vis.save(path)

    _overlay(
        single,
        single_face,
        f"h={single_face.height}px",
        out / "00_single_face_overlay.png",
    )
    _overlay(
        multi,
        multi_face,
        f"h={multi_face.height}px",
        out / "00_multi_face_overlay.png",
    )
    _overlay(
        degraded,
        deg_face,
        f"canvas_h={deg_face.height if deg_face else '?'} eff={degrade_meta['effective_native_face_h_px']}",
        out / "01_degraded_face_overlay.png",
    )

    prep = {
        "single_body": str(args.single_body),
        "face": str(args.face),
        "multi_body": str(args.multi_body),
        "single_face": single_info,
        "multi_face": multi_info,
        "degraded_face_redetect": deg_info,
        "single_face_h_px": int(single_face.height),
        "multi_face_h_px": int(target_h),
        "degrade": degrade_meta,
        "input_sharpness_original": laplacian_sharpness(single, single_face),
        "input_sharpness_degraded": laplacian_sharpness(degraded, deg_face or single_face),
    }
    # Round sharpness for JSON
    for k in ("input_sharpness_original", "input_sharpness_degraded"):
        if prep[k] is not None:
            prep[k] = round(float(prep[k]), 2)

    print(
        f"  single face h={single_face.height}px  multi face h={int(target_h)}px  "
        f"scale={degrade_meta['downsample_scale']}"
    )
    print(
        f"  input Laplacian sharpness: "
        f"orig={prep['input_sharpness_original']}  deg={prep['input_sharpness_degraded']}"
    )

    scores: dict[str, Any] = {}
    metas: dict[str, Any] = {}
    ran_pipeline = False

    if args.prep_only:
        print("Prep-only: skipping Krea2.")
    else:
        cfg = load_config(args.config)
        for arm, body_im in (("original", single), ("degraded", degraded)):
            arm_dir = out / arm
            print(f"\nRunning single-person Krea2 on {arm}…")
            try:
                result_im, meta, latency = run_single_person(
                    body_im, face, cfg, cache, arm_dir, mock=bool(args.mock)
                )
            except Exception as exc:
                print(f"[error] {arm} failed: {exc}")
                scores[arm] = {"arm": arm, "error": str(exc)}
                continue
            # Always keep the degraded scene next to its outputs.
            if arm == "degraded":
                degraded.save(arm_dir / "degraded_scene.png")
            body_im.save(arm_dir / "input_scene.png")
            result_im.save(arm_dir / "final_output.png")
            sc = score_arm(
                face_ref=face,
                body=body_im,
                result=result_im,
                cache_dir=cache,
                latency_s=latency,
                arm=arm,
            )
            scores[arm] = sc
            metas[arm] = {
                k: meta.get(k)
                for k in (
                    "edit_mode",
                    "multi_person_swap_mode",
                    "single_person_parity",
                    "latency_s",
                    "mock",
                    "debug_files",
                )
            }
            print(
                f"  {arm}: id={sc.get('identity_cosine')} "
                f"eye_err/iod={sc.get('landmark_eye_error_over_iod')} "
                f"sharp_in={sc.get('face_sharpness_laplacian_body')} "
                f"conf={sc.get('face_detector_confidence_result')}"
            )
            ran_pipeline = True

        if "original" in scores and "degraded" in scores and "error" not in scores["original"]:
            o_img = Image.open(out / "original" / "final_output.png").convert("RGB")
            d_img = Image.open(out / "degraded" / "final_output.png").convert("RGB")
            _side_by_side(
                [
                    (single, "input original"),
                    (degraded, "input degraded"),
                    (o_img, "result original"),
                    (d_img, "result degraded"),
                ]
            ).save(out / "side_by_side.png")

    modes = classify_failure_modes(scores.get("original"), scores.get("degraded"))
    judgment = judge_hypothesis(
        prep=prep,
        original=scores.get("original") if "error" not in (scores.get("original") or {}) else None,
        degraded=scores.get("degraded") if "error" not in (scores.get("degraded") or {}) else None,
        modes=modes,
        ran_pipeline=ran_pipeline and not args.mock,
    )
    # Mock runs are wiring checks — do not claim causal answers.
    if args.mock and ran_pipeline:
        judgment = {
            "reproduced_multi_failure_mode": "inconclusive",
            "failure_modes_appeared": modes,
            "resolution_role": "inconclusive",
            "rationale": (
                "Mock pipeline only — metrics are not meaningful. Re-run without "
                "--mock on GPU to answer the hypothesis."
            ),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "single/multi quality gap is caused by effective native face resolution"
        ),
        "prep": prep,
        "scores": scores,
        "metas": metas,
        "judgment": judgment,
        "ran_pipeline": ran_pipeline,
        "prep_only": bool(args.prep_only),
        "mock": bool(args.mock),
        "config": str(args.config),
    }
    (out / "REPORT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out / "REPORT.md", payload)
    print(f"\nWrote {out / 'REPORT.md'}")
    print(f"Judgment: reproduced={judgment['reproduced_multi_failure_mode']} "
          f"role={judgment['resolution_role']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
