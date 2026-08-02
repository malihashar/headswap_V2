#!/usr/bin/env python3
"""Head-scale geometry RCA for Krea2 crop_stitch.

Runs production multi path with head_scale_trace enabled and writes
FIRST_ENLARGEMENT_STAGE under out/head_scale_trace/.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/trace_head_scale.py \\
    --body path/to/group.png --face path/to/id.png \\
    --out results/_head_scale_trace/run1 [--force-mock]

  # Offline from existing debug dumps:
  PYTHONPATH=src .venv/bin/python scripts/trace_head_scale.py \\
    --from-debug-dir results/_krea2_full_vs_localized/night_group_001/localized \\
    --body results/_krea2_full_vs_localized/night_group_001/00_body.png \\
    --out results/_head_scale_trace/night_group_offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from headswap.config import load_config
from headswap.pipelines import create_pipeline
from headswap.preprocess import (
    FaceBox,
    pil_to_rgb_np,
    resize_max_keep_ar,
    select_face_box,
)
from headswap.profiling.head_scale_trace import analyze_debug_dir


def _offline(args: argparse.Namespace) -> int:
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    dbg = Path(args.from_debug_dir)
    body = Image.open(args.body).convert("RGB")
    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    body_full = resize_max_keep_ar(
        body, int(cfg.get("max_body_dim", 1024)), div_by=int(cfg.get("div_by", 16))
    )
    selected, _faces = select_face_box(
        pil_to_rgb_np(body_full),
        cache,
        policy=str(args.face_policy or cfg.get("body_face_policy", "rightmost")),
    )
    if selected is None:
        raise SystemExit("No face on body for offline analysis")

    scene = Image.open(
        dbg / ("debug_crop.png" if (dbg / "debug_crop.png").is_file() else "debug_scene.png")
    ).convert("RGB")
    edited = Image.open(dbg / "debug_edited_crop.png").convert("RGB")
    result_path = dbg / "result.png"
    if not result_path.is_file():
        result_path = dbg / "debug_final.png"
    if not result_path.is_file():
        # MockHeadSwapPipeline may only leave debug_* dumps; synthesize from edited+body.
        for cand in ("debug_composite.png", "output.png", "final.png"):
            if (dbg / cand).is_file():
                result_path = dbg / cand
                break
    if not result_path.is_file():
        raise SystemExit(
            f"No result image in {dbg} (looked for result.png / debug_final.png). "
            "Pass a directory that contains debug_edited_crop.png and a final result."
        )
    result = Image.open(result_path).convert("RGB")
    if result.size != body_full.size:
        result = result.resize(body_full.size, Image.Resampling.LANCZOS)
    mask = None
    if (dbg / "debug_mask.png").is_file():
        mask = Image.open(dbg / "debug_mask.png")

    # Infer crop box from selected face + scene AR if meta missing.
    # Prefer mapping: scene is resized crop; recover native box via mask bbox or
    # pad around selected face matching scene aspect.
    sw, sh = scene.size
    # Approximate native crop as selected-centered window matching scene AR.
    cx = 0.5 * (selected.x0 + selected.x1)
    cy = 0.5 * (selected.y0 + selected.y1)
    # Use face height * 3.2 as crop height proxy (typical head+hair window).
    ch = max(sh, int(selected.height * 3.2))
    cw = max(sw, int(ch * sw / float(sh)))
    x0 = int(cx - cw / 2)
    y0 = int(cy - ch / 2)
    x0 = max(0, min(x0, body_full.size[0] - cw))
    y0 = max(0, min(y0, body_full.size[1] - ch))
    x1 = min(body_full.size[0], x0 + cw)
    y1 = min(body_full.size[1], y0 + ch)
    crop_box = (x0, y0, x1, y1)

    # If mask exists and is full-body sized, use its bbox as crop.
    if mask is not None and mask.size == body_full.size:
        from headswap.preprocess import mask_bbox

        crop_box = mask_bbox(mask, pad=12)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = analyze_debug_dir(
        body=body_full,
        scene_or_crop=scene,
        edited=edited,
        result=result,
        mask=mask if mask is not None and mask.size == body_full.size else None,
        selected=selected,
        crop_box=crop_box,
        out_dir=out / "head_scale_trace",
        cache_dir=cache,
    )
    verdict = report.get("verdict") or {}
    print(json.dumps(verdict, indent=2))
    print(f"FIRST_ENLARGEMENT_STAGE = {verdict.get('FIRST_ENLARGEMENT_STAGE')}")
    print(f"Wrote {out / 'head_scale_trace'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--face", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force-mock", action="store_true")
    ap.add_argument("--from-debug-dir", type=Path, default=None)
    ap.add_argument("--face-policy", type=str, default="rightmost")
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    args = ap.parse_args()

    if args.from_debug_dir:
        return _offline(args)
    if args.face is None:
        raise SystemExit("--face is required unless --from-debug-dir is set")

    cfg = load_config(args.config)
    cfg = dict(cfg)
    cfg["head_scale_trace"] = True
    cfg["save_debug"] = True
    cfg["multi_person_swap_mode"] = "krea2_crop"
    cfg["single_person_parity"] = True
    cfg["body_face_policy"] = args.face_policy
    if args.force_mock:
        cfg["force_mock"] = True

    args.out.mkdir(parents=True, exist_ok=True)
    pipe = create_pipeline(cfg, force_mock=bool(args.force_mock))
    body = Image.open(args.body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    result = pipe.run(body, face, out_dir=args.out)
    meta = dict(result.meta or {})

    # force_mock uses MockHeadSwapPipeline (not Krea2IdentityEditPipeline), so
    # reconstruct the S0–S4 geometry report from the debug dumps it wrote.
    if args.force_mock and (args.out / "debug_edited_crop.png").is_file():
        print("[trace_head_scale] mock path — analyzing debug dumps for geometry RCA")
        if result.image is not None:
            result.image.save(args.out / "result.png")
        args.from_debug_dir = args.out
        return _offline(args)

    hs = (meta.get("face_prep_diag") or {}).get("head_scale_trace") or {}
    verdict = hs.get("verdict")
    report_path = args.out / "head_scale_trace" / "REPORT.md"
    if report_path.is_file():
        print(report_path.read_text(encoding="utf-8"))
    elif verdict:
        print(json.dumps(verdict, indent=2))
    else:
        alt = args.out / "head_scale_trace" / "REPORT.json"
        if alt.is_file():
            print(alt.read_text(encoding="utf-8"))
        else:
            print("WARNING: head_scale_trace report missing — check edit_mode was crop_stitch")
            print("meta.edit_mode=", meta.get("edit_mode"))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
