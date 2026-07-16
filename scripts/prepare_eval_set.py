#!/usr/bin/env python3
"""Prepare eval pairs: synthetic (default) or a single custom real photo pair."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.eval.dataset import (  # noqa: E402
    generate_synthetic_eval_set,
    prepare_custom_eval_set,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "n",
        nargs="?",
        type=int,
        default=24,
        help="Synthetic pair count (ignored with --custom). Positional for back-compat.",
    )
    ap.add_argument("--out", type=str, default=None, help="Eval root (default: data/eval).")
    ap.add_argument(
        "--custom",
        nargs="?",
        const=str(ROOT / "data" / "custom"),
        default=None,
        help=(
            "Build a 1-pair eval set from real photos. "
            "Optional custom dir (default: data/custom). Expects body.png + face.png."
        ),
    )
    args = ap.parse_args()
    root = Path(args.out) if args.out else None

    if args.custom is not None:
        path = prepare_custom_eval_set(custom_dir=Path(args.custom), root=root)
    else:
        path = generate_synthetic_eval_set(root, n_pairs=args.n)
    print(path)


if __name__ == "__main__":
    main()
