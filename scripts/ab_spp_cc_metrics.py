#!/usr/bin/env python3
"""A/B metrics harness for SPP-CC vs full_frame vs legacy multi forks.

Computes (no GPU required for offline compare of saved runs):
  - conditioning fingerprint fields from meta JSON
  - neighbor PSNR outside selected mask vs body
  - selected-face MSE/PSNR vs body (swap strength)
  - seam-band mean absolute RGB delta

Usage:
  python scripts/ab_spp_cc_metrics.py \\
    --body body.png --result result.png --mask mask.png \\
    --selected-box 100,80,180,180 \\
    --meta run_meta.json

  # Or compare two result folders:
  python scripts/ab_spp_cc_metrics.py --compare-dirs results/a results/b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import FaceBox, seam_annulus_mask  # noqa: E402


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _parse_box(s: str) -> FaceBox:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise SystemExit("--selected-box must be x0,y0,x1,y1")
    return FaceBox(parts[0], parts[1], parts[2], parts[3], 1.0)


def measure(
    body: Image.Image,
    result: Image.Image,
    mask: Image.Image | None,
    selected: FaceBox | None,
) -> dict:
    body_a = np.asarray(body.convert("RGB"))
    res = result.convert("RGB").resize(body.size, Image.Resampling.LANCZOS)
    res_a = np.asarray(res)
    out: dict = {
        "body_size": list(body.size),
        "psnr_full": round(_psnr(body_a, res_a), 3),
    }
    if mask is not None:
        m = np.asarray(
            mask.convert("L").resize(body.size, Image.Resampling.BILINEAR)
        )
        outside = m < 128
        inside = ~outside
        if np.any(outside):
            out["psnr_outside_mask"] = round(
                _psnr(body_a[outside], res_a[outside]), 3
            )
            out["outside_pixels"] = int(outside.sum())
        if np.any(inside):
            out["psnr_inside_mask"] = round(
                _psnr(body_a[inside], res_a[inside]), 3
            )
            out["mse_inside_mask"] = round(
                float(
                    np.mean(
                        (body_a[inside].astype(np.float64) - res_a[inside].astype(np.float64))
                        ** 2
                    )
                ),
                3,
            )
        band = np.asarray(
            seam_annulus_mask(mask.resize(body.size, Image.Resampling.BILINEAR)).convert(
                "L"
            )
        )
        band_m = band > 128
        if np.any(band_m):
            out["seam_band_mean_abs_rgb"] = round(
                float(
                    np.mean(
                        np.abs(
                            body_a[band_m].astype(np.float64)
                            - res_a[band_m].astype(np.float64)
                        )
                    )
                ),
                3,
            )
    if selected is not None:
        x0, y0 = max(0, selected.x0), max(0, selected.y0)
        x1 = min(body.size[0], selected.x1)
        y1 = min(body.size[1], selected.y1)
        if x1 > x0 and y1 > y0:
            pa, pb = body_a[y0:y1, x0:x1], res_a[y0:y1, x0:x1]
            mse = float(np.mean((pa.astype(np.float64) - pb.astype(np.float64)) ** 2))
            out["selected_face_mse"] = round(mse, 3)
            out["selected_face_psnr"] = round(_psnr(pa, pb), 3)
    return out


def conditioning_fingerprint(meta: dict) -> dict:
    """Extract fields that must match single-person under SPP-CC."""
    keys = [
        "edit_mode",
        "multi_person_edit_mode",
        "single_person_parity",
        "head_mask_backend",
        "seam_refine",
        "steps",
        "cfg",
        "seed",
        "ref_boost",
        "ref_boost_a",
        "fit_mode",
        "grounding_px",
        "loras_loaded",
        "lora_strengths",
        "checkpoint",
        "scene_size",
        "person_size",
    ]
    fp = {k: meta.get(k) for k in keys if k in meta}
    diag = meta.get("face_prep_diag") or {}
    for k in (
        "person_prep",
        "identity_scale_match",
        "isolate_selected",
        "use_tight",
        "single_person_parity",
        "head_mask",
    ):
        if k in diag:
            fp[f"diag_{k}"] = diag[k]
    cond = diag.get("krea2_conditioning") or meta.get("krea2_conditioning")
    if isinstance(cond, dict):
        fp["cond_person_nonblack"] = (cond.get("person") or {}).get("nonblack_frac")
        fp["cond_prompt_sha"] = cond.get("prompt_sha1_12")
        fp["cond_ref_boost_mask"] = cond.get("ref_boost_mask")
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--mask", type=Path, default=None)
    ap.add_argument("--selected-box", type=str, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if not args.body or not args.result:
        ap.error("--body and --result are required")

    body = Image.open(args.body)
    result = Image.open(args.result)
    mask = Image.open(args.mask) if args.mask and args.mask.is_file() else None
    selected = _parse_box(args.selected_box) if args.selected_box else None
    report: dict = {"metrics": measure(body, result, mask, selected)}
    if args.meta and args.meta.is_file():
        meta = json.loads(args.meta.read_text())
        report["conditioning"] = conditioning_fingerprint(meta)
        report["spp_cc_ok"] = bool(
            meta.get("single_person_parity", meta.get("face_prep_diag", {}).get("single_person_parity"))
        ) and str(meta.get("multi_person_edit_mode", "")).startswith("crop")
    print(json.dumps(report, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
