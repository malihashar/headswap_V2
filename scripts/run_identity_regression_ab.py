#!/usr/bin/env python3
"""Colab / local A/B harness for the multi-person identity regression.

Exp A — strong ID baseline (default after hybrid restore):
  multi_person_swap_mode: krea2_crop

Exp B — paste-only isolation (align_paste, no refine/relock/color-match):
  multi_person_swap_mode: align_paste
  align_paste_krea2_refine: false
  align_paste_pose_relock: false
  pre_color_match_strength: 0
  align_paste_post_color_match: 0
  save_debug: true

Usage (Colab, after pull):
  python scripts/run_identity_regression_ab.py \\
    --body /path/to/group.png --face /path/to/gosling.png \\
    --exp A --out results/_id_ab_A

  python scripts/run_identity_regression_ab.py \\
    --body /path/to/group.png --face /path/to/gosling.png \\
    --exp B --out results/_id_ab_B

Compare debug_align_composite vs final (Exp B) and crop edited vs final (Exp A).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _cfg_for_exp(exp: str) -> dict:
    base = {
        "save_debug": True,
        "verbose": True,
        "mask_crop_stitch": True,
        "body_face_policy": "rightmost",  # matches Gosling×rightmost person demos
    }
    exp = exp.upper().strip()
    if exp == "A":
        base.update(
            {
                "multi_person_swap_mode": "krea2_crop",
                "multi_person_edit_mode": "crop_stitch",
                "single_person_parity": True,
                "multi_crop_hard_freeze_neighbors": True,
            }
        )
    elif exp == "B":
        base.update(
            {
                "multi_person_swap_mode": "align_paste",
                "align_paste_krea2_refine": False,
                "align_paste_pose_relock": False,
                "pre_color_match_strength": 0.0,
                "align_paste_post_color_match": 0.0,
            }
        )
    elif exp == "C":
        # Refine on, pose relock off — fork model ID vs relock dilution.
        base.update(
            {
                "multi_person_swap_mode": "align_paste",
                "align_paste_krea2_refine": True,
                "align_paste_pose_relock": False,
            }
        )
    else:
        raise SystemExit(f"Unknown exp {exp!r}; use A, B, or C")
    return base


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body", type=Path, required=True)
    p.add_argument("--face", type=Path, required=True)
    p.add_argument("--exp", choices=["A", "B", "C", "a", "b", "c"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "krea2_identity_edit.yaml",
    )
    args = p.parse_args()

    from PIL import Image

    from headswap.pipelines import create_pipeline_from_config

    overrides = _cfg_for_exp(args.exp)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exp_overrides.json").write_text(json.dumps(overrides, indent=2))

    pipe = create_pipeline_from_config(str(args.config))
    pipe.cfg.update(overrides)

    body = Image.open(args.body).convert("RGB")
    face = Image.open(args.face).convert("RGB")
    result = pipe.run(body=body, face=face, out_dir=args.out)
    out_img = result.image if hasattr(result, "image") else result.get("image")
    if out_img is not None:
        out_path = args.out / f"result_exp_{args.exp.upper()}.png"
        out_img.save(out_path)
        print(f"[id_ab] saved {out_path}")
    meta = result.meta if hasattr(result, "meta") else {}
    print(
        f"[id_ab] exp={args.exp.upper()} edit_mode={meta.get('edit_mode')} "
        f"multi_person_swap_mode={overrides.get('multi_person_swap_mode')}"
    )


if __name__ == "__main__":
    main()
