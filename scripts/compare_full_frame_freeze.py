#!/usr/bin/env python3
"""Compare raw full-frame Krea2 vs freeze-outside-selected composite.

Usage (Colab / local after a DEBUG run):
  python scripts/compare_full_frame_freeze.py \\
      --body path/to/body.png \\
      --raw  run_dir/debug/debug_full_frame_raw.png \\
      --frozen run_dir/debug/debug_full_frame_frozen.png \\
      --mask run_dir/debug/debug_freeze_mask.png

Reports PSNR outside the freeze mask (neighbors/BG should be ~identical to
body for frozen; raw will be lower if the model rewrote neighbors).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--frozen", required=True, type=Path)
    ap.add_argument("--mask", required=True, type=Path)
    args = ap.parse_args()

    body = np.asarray(Image.open(args.body).convert("RGB"))
    raw = np.asarray(Image.open(args.raw).convert("RGB").resize(body.shape[1::-1], Image.Resampling.LANCZOS))
    frozen = np.asarray(
        Image.open(args.frozen).convert("RGB").resize(body.shape[1::-1], Image.Resampling.LANCZOS)
    )
    mask = np.asarray(
        Image.open(args.mask).convert("L").resize(body.shape[1::-1], Image.Resampling.BILINEAR)
    )
    outside = mask < 128
    if not np.any(outside):
        print("mask covers entire image — nothing to compare outside")
        return 1

    print(f"outside_pixels={int(outside.sum())}")
    print(f"psnr_raw_vs_body_outside={_psnr(raw[outside], body[outside]):.2f}")
    print(f"psnr_frozen_vs_body_outside={_psnr(frozen[outside], body[outside]):.2f}")
    print(f"psnr_raw_vs_frozen_outside={_psnr(raw[outside], frozen[outside]):.2f}")
    inside = ~outside
    if np.any(inside):
        print(f"psnr_raw_vs_frozen_inside={_psnr(raw[inside], frozen[inside]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
