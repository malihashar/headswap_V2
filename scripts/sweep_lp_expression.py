#!/usr/bin/env python3
"""Sweep LivePortrait settings on a FIXED T4 swap, to tune the mouth.

The swap costs ~21s; a LivePortrait pass costs ~0.4s warm. So the swap runs
ONCE and every expression variant reuses it -- a dozen arms cost seconds,
not minutes. Reuse a previous swap with --swap-png to make iteration
instant.

The artifact being tuned: opening a closed mouth forces LivePortrait's
spade_generator to INVENT the mouth interior (teeth, cavity), which is
synthesis rather than warping and therefore does not improve with
resolution. The controls that bear on it:

  driving_multiplier   scales the whole transfer; lower opens the mouth
                       less, so less interior has to be invented
  flag_normalize_lip   normalises the source lip toward a canonical state
                       before applying the delta, so the transfer starts
                       from a consistent mouth
  flag_lip_retargeting routes lip motion through the dedicated
                       stitching_retargeting_module instead of the raw
                       keypoint delta -- the mechanism built for this

Montage crops to the FACE, because the mouth is a few dozen pixels in a
1024x576 frame and the artifact is invisible in a full-frame strip.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# label : (driving_multiplier, normalize_lip, lip_retargeting)
DEFAULT_ARMS = [
    ("mult1.0",            1.0, False, False),
    ("mult0.8",            0.8, False, False),
    ("mult0.6",            0.6, False, False),
    ("mult0.8+normlip",    0.8, True,  False),
    ("mult0.8+lipretarg",  0.8, False, True),
    ("mult0.6+norm+retarg", 0.6, True, True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "my_pair"))
    ap.add_argument("--out-dir", default=str(REPO / "results" / "lp_sweep"))
    ap.add_argument("--swap-png", default=None,
                    help="reuse an existing T4 output instead of re-running it")
    ap.add_argument("--lp-dir", default="/content/LivePortrait")
    ap.add_argument("--animation-region", default="lip")
    ap.add_argument("--seed", type=int, default=46)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    paths = m.apply_env(m.default_paths(use_drive=False))
    m.ensure_import_path(REPO)
    comfy = str(paths.get("comfyui", "/content/ComfyUI"))
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    os.chdir(REPO)

    from PIL import Image
    from headswap.expression_transfer import run_expression_transfer

    pair = Path(args.pair_dir)
    if not pair.is_absolute():
        pair = (REPO / pair).resolve()
    body_p, face_p = pair / "body.png", pair / "face.png"
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = (REPO / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.swap_png:
        swap_png = Path(args.swap_png)
        if not swap_png.is_absolute():
            swap_png = (REPO / swap_png).resolve()
        if not swap_png.exists():
            print(f"ERROR: {swap_png} not found", file=sys.stderr)
            return 2
        print(f"Reusing existing swap: {swap_png}", flush=True)
    else:
        from headswap.config import load_config
        from headswap.pipelines import create_pipeline
        from headswap.pipelines.krea2 import get_shared_krea2_runtime

        cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
        cfg.update({"seed": int(args.seed), "verbose": False,
                    "pre_edit_donor_expression": False,
                    "simple_full_body_refine_max_face_frac": 0.25})
        pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))
        print("=== T4 swap (once) ===", flush=True)
        t0 = time.perf_counter()
        res = pipe.run(Image.open(body_p).convert("RGB"),
                       Image.open(face_p).convert("RGB"), out_dir=out)
        swap_png = out / "swap_only.png"
        res.image.save(swap_png)
        print(f"swap done in {time.perf_counter() - t0:.1f}s -> {swap_png}", flush=True)

    tiles = [("T4 only (no LP)", Image.open(swap_png).convert("RGB"))]
    print("\n=== LivePortrait arms ===", flush=True)
    for label, mult, norm, retarg in DEFAULT_ARMS:
        t0 = time.perf_counter()
        try:
            lp = run_expression_transfer(
                source_path=swap_png,
                driving_path=body_p,
                out_dir=out / f"arm_{label}",
                animation_region=args.animation_region,
                driving_multiplier=mult,
                normalize_lip=norm,
                lip_retargeting=retarg,
                live_portrait_dir=args.lp_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:22} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        dt = time.perf_counter() - t0
        prim = lp.get("primary")
        if prim and Path(prim).suffix.lower() != ".mp4":
            tiles.append((label, Image.open(prim).convert("RGB")))
            print(f"  {label:22} {dt:5.1f}s  -> {Path(prim).name}", flush=True)
        else:
            print(f"  {label:22} {dt:5.1f}s  NO IMAGE", flush=True)

    _face_montage(tiles, out / "lp_sweep_faces.png", body_p)
    _montage(tiles, out / "lp_sweep_full.png")
    print(f"\nface montage -> {out / 'lp_sweep_faces.png'}")
    print(f"full montage -> {out / 'lp_sweep_full.png'}")
    print("\nJudge the MOUTH INTERIOR: teeth that look invented, a dark hole,")
    print("or a lower lip that reads wrong. Lower multipliers invent less but")
    print("also transfer less of the expression.")
    return 0


def _crop_face(im, cache_dir):
    """Crop generously around the detected face so the mouth is legible."""
    try:
        from headswap.preprocess import detect_best_face, pil_to_rgb_np
        box = detect_best_face(pil_to_rgb_np(im), cache_dir)
        if box is None:
            return im
        w, h = im.size
        fw, fh = box.width, box.height
        x0 = max(0, int(box.x0 - 0.45 * fw))
        x1 = min(w, int(box.x1 + 0.45 * fw))
        y0 = max(0, int(box.y0 - 0.30 * fh))
        y1 = min(h, int(box.y1 + 0.45 * fh))
        return im.crop((x0, y0, x1, y1))
    except Exception:  # noqa: BLE001
        return im


def _face_montage(items, out_path: Path, _body_p) -> None:
    cache = REPO / ".cache" / "headswap_v2"
    _montage([(lb, _crop_face(im, cache)) for lb, im in items], out_path, h=420)


def _montage(items, out_path: Path, pad: int = 8, h: int = 300) -> None:
    from PIL import Image, ImageDraw
    tiles = [(lb, im.convert("RGB").resize(
        (max(1, int(im.width * h / max(1, im.height))), h),
        Image.Resampling.LANCZOS)) for lb, im in items]
    W = sum(t.width for _, t in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (W, h + 26 + pad * 2), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = pad
    for lb, t in tiles:
        canvas.paste(t, (x, 26 + pad))
        d.text((x + 2, 6), str(lb), fill=(235, 235, 235))
        x += t.width + pad
    canvas.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
