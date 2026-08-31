#!/usr/bin/env python3
"""Entrypoint for chain.sweep_grounding() -- see that function's docstring
for the full mechanism (Krea2EditGroundedEncode, the VLM-grounding carrier).

Defaults to grounding_px=256 ONLY, because that is a specific re-test, not a
fresh sweep: an earlier run at 256 produced a bald head with no cap, but that
run used the OLD prompt (884 chars, non-bald-aware "hair is there instead"
wording, plus the garment clause that turned out to be live only because of
the sticky-cfg bug fixed alongside the bald-donor wording). Both of those are
gone now -- current code uses the bald-aware 708-char prompt and
protect_garments defaults False. So this is not "confirm 256 still works",
it is "does 256 still work with the CURRENT prompt", which is a different
question with an unknown answer.

    python scripts/sweep_grounding.py --pair-dir data/custom/chain_pair
    python scripts/sweep_grounding.py --grounding-px 768,512,384,256  # full re-sweep
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
    ap.add_argument("--out-dir", default=str(REPO / "results" / "grounding_sweep"))
    ap.add_argument("--grounding-px", default="256",
                     help="comma-separated values; default is JUST 256, the "
                          "specific value being re-tested under current code")
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

    grounding_px = tuple(int(x) for x in args.grounding_px.split(",") if x.strip())
    result = chain.sweep_grounding(
        body_p, face_p,
        out_dir=args.out_dir,
        grounding_px=grounding_px,
        seed=args.seed,
    )
    for row in result["arms"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
