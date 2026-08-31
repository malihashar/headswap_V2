#!/usr/bin/env python3
"""Entrypoint for chain.sweep_guidance() -- see that function's docstring
for the full diagnosis. Short version: every scene-conditioning carrier
(ref_boost_a, denoise/noise_mask, grounding_px) was already swept and
eliminated -- the cap is not being preserved from the input, it is being
generated fresh regardless of what the source latent contains. Nothing on
the GUIDANCE axis (cfg, ref_boost) has been tried yet.

Also includes the bald-donor headwear wording fix and the sticky-flag fix
now, both landed since the carrier sweeps ran, so this arm 0 (cfg=1.8,
ref_boost=5.5) is a fresh baseline with both fixes ALREADY applied, not the
same run that produced the earlier bald-cap screenshot.

    python scripts/sweep_guidance.py --pair-dir data/custom/chain_pair
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "chain_pair"))
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "guidance_sweep"))
    ap.add_argument(
        "--arms", default="1.8:5.5,3.0:5.5,4.5:5.5,3.0:8.0,1.8:8.0",
        help="comma-separated cfg:ref_boost pairs; first arm is the shipped baseline",
    )
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.apply_env(m.default_paths(use_drive=False))
    m.ensure_import_path(REPO)
    os.chdir(REPO)

    from headswap import chain

    pair = Path(args.pair_dir)
    if not pair.is_absolute():
        pair = (REPO / pair).resolve()
    body_p, face_p = pair / "body.png", pair / "face.png"
    for f in (body_p, face_p):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 2

    arms = tuple(
        tuple(float(v) for v in pair_str.split(":"))
        for pair_str in args.arms.split(",") if pair_str.strip()
    )
    result = chain.sweep_guidance(
        body_p, face_p,
        out_dir=args.out_dir,
        arms=arms,
        seed=args.seed,
    )
    for row in result["arms"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
