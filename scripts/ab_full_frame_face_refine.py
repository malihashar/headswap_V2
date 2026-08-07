#!/usr/bin/env python3
"""A/B: full_frame face quality before vs after the crop_stitch-style refine pass.

Runs the SAME full-body input through Krea2IdentityEditPipeline three ways:
  baseline  - full_frame, refine pass OFF (today's production full_frame)
  refined   - full_frame, refine pass ON (full_frame_face_refine=true, this change)
  crop_only - the refine pass's own crop_stitch-quality output on the
              *baseline* full_frame result, useful as a quality "ceiling"
              reference for the face alone (body geometry not evaluated here)

Body-preserve PSNR (outside the head mask) is reported for baseline vs
refined so a resolution-gain report can also confirm proportions/background
were not disturbed by the second pass.

Requires GPU + ComfyUI + Krea2 nodes set up (same as production). This does
NOT modify the pipeline, config schema, masks, or stitching beyond what is
already in this branch.

Usage (GPU):
  PYTHONPATH=src python scripts/ab_full_frame_face_refine.py \\
    --body /path/to/full_body_photo.jpg --face /path/to/donor_face.jpg \\
    --out results/_ab_full_frame_face_refine

Mock smoke (no GPU, wiring check only -- metrics are not meaningful):
  PYTHONPATH=src python scripts/ab_full_frame_face_refine.py --mock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config
from headswap.metrics.scoring import body_preserve_score, identity_cosine
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline, reset_shared_krea2_runtime
from headswap.preprocess import detect_best_face, pil_to_rgb_np

try:
    import cv2
    import numpy as np

    def _laplacian_sharpness(im: Image.Image) -> float | None:
        if im.size[0] < 8 or im.size[1] < 8:
            return None
        gray = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
except Exception:  # pragma: no cover - cv2 always present in this repo's env
    def _laplacian_sharpness(im: Image.Image) -> float | None:
        return None


DEFAULT_BODY = ROOT / "data" / "custom" / "body.png"
DEFAULT_FACE = ROOT / "data" / "custom" / "face.png"
DEFAULT_OUT = ROOT / "results" / "_ab_full_frame_face_refine"


def run_arm(
    body: Image.Image,
    face: Image.Image,
    cfg: dict,
    cache_dir: Path,
    out_dir: Path,
    *,
    mock: bool,
) -> tuple[Image.Image, dict[str, Any], float]:
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
    result.image.save(out_dir / "final_output.png")
    meta = dict(result.meta or {})
    meta["mock"] = bool(mock)
    return result.image, meta, latency


def score_arm(
    *, face_ref: Image.Image, body: Image.Image, result: Image.Image, cache_dir: Path, latency_s: float, arm: str
) -> dict[str, Any]:
    body_rgb = body.convert("RGB")
    result_rgb = result.convert("RGB")
    if result_rgb.size != body_rgb.size:
        result_rgb = result_rgb.resize(body_rgb.size, Image.Resampling.LANCZOS)

    body_face = detect_best_face(pil_to_rgb_np(body_rgb), cache_dir)
    result_face = detect_best_face(pil_to_rgb_np(result_rgb), cache_dir)

    def _crop(im, box, pad=0.15):
        if box is None:
            return im
        w, h = im.size
        px, py = int(pad * box.width), int(pad * box.height)
        return im.crop(
            (
                max(0, box.x0 - px),
                max(0, box.y0 - py),
                min(w, box.x1 + px),
                min(h, box.y1 + py),
            )
        )

    id_cos = identity_cosine(face_ref, result_rgb)
    face_sharp = _laplacian_sharpness(_crop(result_rgb, result_face)) if result_face else None

    return {
        "arm": arm,
        "latency_s": round(float(latency_s), 3),
        "identity_cosine": None if id_cos is None else round(float(id_cos), 4),
        "face_height_px": None if result_face is None else round(float(result_face.height), 1),
        "face_sharpness_laplacian": None if face_sharp is None else round(float(face_sharp), 2),
        "face_detector_confidence": None
        if result_face is None
        else round(float(result_face.conf), 4),
    }


def _side_by_side(images: list[tuple[Image.Image, str]], *, height: int = 640) -> Image.Image:
    fitted = []
    for im, _ in images:
        im = im.convert("RGB")
        s = height / max(1, im.size[1])
        fitted.append(im.resize((max(1, int(im.size[0] * s)), height), Image.Resampling.LANCZOS))
    gap, label_h = 12, 28
    total_w = sum(f.size[0] for f in fitted) + gap * (len(fitted) - 1)
    canvas = Image.new("RGB", (total_w, height + label_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for f, (_, label) in zip(fitted, images):
        draw.text((x + 4, 6), label, fill=(200, 220, 255))
        canvas.paste(f, (x, label_h))
        x += f.size[0] + gap
    return canvas


def write_report(path: Path, payload: dict[str, Any]) -> None:
    scores = payload["scores"]

    def _fmt(v):
        if v is None:
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "# full_frame face-refine A/B",
        "",
        f"_Generated {payload['generated_at']}_",
        "",
        "## Setup",
        "",
        f"- Body: `{payload['body']}`",
        f"- Face: `{payload['face']}`",
        f"- Mock run: **{payload['mock']}**",
        "",
        "## Metrics",
        "",
        "| metric | baseline (refine off) | refined (refine on) |",
        "|---|---:|---:|",
    ]
    b = scores.get("baseline", {})
    r = scores.get("refined", {})
    for key, label in (
        ("identity_cosine", "ArcFace identity cosine"),
        ("face_height_px", "detected face height (px, final image)"),
        ("face_sharpness_laplacian", "face sharpness (Laplacian var)"),
        ("face_detector_confidence", "face detector confidence"),
        ("latency_s", "latency (s)"),
    ):
        lines.append(f"| {label} | {_fmt(b.get(key))} | {_fmt(r.get(key))} |")

    body_preserve = payload.get("body_preserve_psnr")
    lines += [
        "",
        f"- **Body-preserve PSNR (baseline vs refined, outside head mask)**: "
        f"{_fmt(body_preserve)} dB (high = geometry/background unchanged by the refine pass)",
        "",
        "## Full-frame face-refine diagnostics (refined arm)",
        "",
        "```json",
        json.dumps(payload.get("refine_diag") or {}, indent=2),
        "```",
        "",
        "Artifacts: `baseline/`, `refined/`, `side_by_side.png`, `REPORT.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B: full_frame face refine pass")
    ap.add_argument("--body", type=Path, default=DEFAULT_BODY, help="Full-body photo")
    ap.add_argument("--face", type=Path, default=DEFAULT_FACE, help="Donor face photo")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "krea2_identity_edit.yaml")
    ap.add_argument("--mock", action="store_true", help="Use mock pipeline (no GPU; wiring check only)")
    args = ap.parse_args()

    for p, label in ((args.body, "body"), (args.face, "face")):
        if not Path(p).is_file():
            raise SystemExit(f"Missing {label}: {p}")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)

    body = Image.open(args.body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    body.save(out / "00_input_body.png")
    face.save(out / "00_input_face.png")

    base_cfg = load_config(args.config)
    # Force full_frame explicitly rather than relying on body-route detection,
    # so this A/B isolates the refine pass regardless of the input's framing.
    base_cfg["enable_lighting_route"] = False
    base_cfg["enable_body_route"] = False
    base_cfg["multi_person_edit_mode"] = "full_frame"
    base_cfg["mask_crop_stitch"] = True
    base_cfg["enable_multi_face_features"] = False

    scores: dict[str, Any] = {}
    metas: dict[str, Any] = {}
    images: dict[str, Image.Image] = {}

    for arm, refine_on in (("baseline", False), ("refined", True)):
        cfg = deepcopy(base_cfg)
        cfg["full_frame_face_refine"] = refine_on
        print(f"\n→ Running full_frame ({arm}, refine={refine_on})…")
        try:
            result_im, meta, latency = run_arm(body, face, cfg, cache, out / arm, mock=args.mock)
        except Exception as exc:
            print(f"[error] {arm} failed: {exc}")
            scores[arm] = {"arm": arm, "error": str(exc)}
            continue
        images[arm] = result_im
        metas[arm] = meta
        sc = score_arm(face_ref=face, body=body, result=result_im, cache_dir=cache, latency_s=latency, arm=arm)
        scores[arm] = sc
        print(
            f"  {arm}: id={sc.get('identity_cosine')} face_h={sc.get('face_height_px')}px "
            f"sharp={sc.get('face_sharpness_laplacian')} conf={sc.get('face_detector_confidence')} "
            f"latency={sc.get('latency_s')}s"
        )

    body_preserve = None
    if "baseline" in images and "refined" in images:
        _side_by_side(
            [
                (body, "input body"),
                (images["baseline"], "full_frame (refine off)"),
                (images["refined"], "full_frame (refine on)"),
            ]
        ).save(out / "side_by_side.png")
        body_preserve = body_preserve_score(images["baseline"], images["refined"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "body": str(args.body),
        "face": str(args.face),
        "mock": bool(args.mock),
        "config": str(args.config),
        "scores": scores,
        "metas": metas,
        "body_preserve_psnr": None if body_preserve is None else round(float(body_preserve), 2),
        "refine_diag": (metas.get("refined") or {}).get("full_frame_face_refine"),
    }
    (out / "REPORT.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(out / "REPORT.md", payload)
    print(f"\nWrote {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
