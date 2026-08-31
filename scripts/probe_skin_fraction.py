#!/usr/bin/env python3
"""Print the visible-skin fraction for images, with no swap.

Calibration only. The first version of this measurement sampled the whole
rectangle below the chin, which is mostly BACKGROUND -- desert sand sits on
skin hue, so a fully robed subject scored 53% "bare skin" while a tennis
player with bare arms against a dark court scored 5.5%. Both answers were
inverted, and that was only discoverable from a real render.

Runs in seconds against real images so the threshold can be set from data
instead of guessed.

    python scripts/probe_skin_fraction.py a.png b.png
    python scripts/probe_skin_fraction.py --tolerances 8,12,18 a.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--tolerances", default="8,12,18",
                    help="a*b* distances to try")
    ap.add_argument("--face-widths", default="2.0,3.0,5.0",
                    help="column widths, in multiples of the face box width")
    args = ap.parse_args()

    from PIL import Image
    from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

    class _P(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {}
            self.cache_dir = REPO / "results" / "_cache"

    pipe = _P()
    tols = [float(x) for x in args.tolerances.split(",") if x.strip()]
    widths = [float(x) for x in args.face_widths.split(",") if x.strip()]

    print(f"\n{'image':<34}{'cols':>6}{'tol':>6}{'skin%':>9}")
    print("-" * 56)
    for name in args.images:
        f = Path(name)
        if not f.exists():
            print(f"{name}: MISSING")
            continue
        im = Image.open(f).convert("RGB")
        for wfw in widths:
            for tol in tols:
                pipe.cfg["visible_skin_column_face_widths"] = wfw
                pipe.cfg["visible_skin_ab_tolerance"] = tol
                frac, diag = pipe._measure_visible_skin(im)
                shown = f"{frac:.1%}" if frac is not None else diag.get("reason")
                print(f"{f.name:<34}{wfw:>6}{tol:>6}{shown:>9}")
    print("\nA covered subject should read LOW and a bare-armed one HIGH.")
    print("If they do not separate at any setting, the approach is wrong,")
    print("not the threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
