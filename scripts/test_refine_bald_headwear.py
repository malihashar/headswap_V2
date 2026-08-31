#!/usr/bin/env python3
"""Does the LOCAL head-only face_refine pass drop a cap where the global
sampling pass (grounding_px 256-768, cfg 1.8-4.5, ref_boost 5.5-8.0, all
tested and all failed) does not?

Why this is a different lever, not another value of the same one. Every
prior test changed how strongly the GLOBAL sampling pass is pulled toward
the prompt/donor. face_refine is a SECOND, separate pass masked to the head
region only -- it cannot touch the garment because the mask does not reach
it, so this costs nothing on the body-preservation side that every other
lever had to trade against.

Why suspect it specifically for the BALD case. The one pattern across every
failed arm: a donor with dense hair (dreadlocks) already gets the cap
replaced correctly by the SAME global pass that fails for a bald donor. A
bald scalp is visually flat/low-detail; a cap is a rigid, high-contrast
object. At denoise=0.85 the sampler tends to keep whatever already has
strong local structure unless something displaces it -- dreadlocks compete
with the cap on equal footing, a bald head does not. A second pass focused
entirely on the head region, at its own denoise, gives the model another
dedicated attempt at exactly the area that is under-competing.

face_refine was SKIPPED on this pair automatically: chain.py's default
skip_refine=True sets simple_full_body_refine_max_face_frac=0.25, and this
pair's face is 31.1% of the frame -- just over. CHECKPOINT-12 measured
refine as visually indistinguishable from skipping it on a similar bust
shot, but that measurement was about general QUALITY, never specifically
about whether it moves headwear. This is the first time it has been tested
for that.

    python scripts/test_refine_bald_headwear.py --pair-dir data/custom/chain_pair
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "chain_pair"))
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "refine_headwear_test"))
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.apply_env(m.default_paths(use_drive=False))
    m.ensure_import_path(REPO)
    os.chdir(REPO)

    from PIL import Image, ImageDraw
    from headswap import chain

    pair = Path(args.pair_dir)
    if not pair.is_absolute():
        pair = (REPO / pair).resolve()
    body_p, face_p = pair / "body.png", pair / "face.png"
    for f in (body_p, face_p):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 2
    body = Image.open(body_p).convert("RGB")
    face = Image.open(face_p).convert("RGB")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles = [("target", body), ("donor", face)]

    for label, skip_refine in (("A_refine_OFF_baseline", True),
                                ("B_refine_ON", False)):
        pipe = chain.load_models(
            seed=args.seed, skip_refine=skip_refine,
            remove_headwear=True, protect_garments=False,
            skip_skin_clause_when_covered=True,
        )
        print(f"\n=== {label} (skip_refine={skip_refine}) ===\n", flush=True)
        t0 = time.perf_counter()
        res = pipe.run(body, face, out_dir=out)
        wall = time.perf_counter() - t0
        path = out / f"{label}.png"
        res.image.save(path)
        tiles.append((label, res.image))
        meta = res.meta or {}
        refine_diag = meta.get("face_refine")
        print(f"  -> {wall:.0f}s  face_refine={refine_diag}  {path.name}",
              flush=True)

    montage = out / "refine_headwear_montage.png"
    _montage(tiles, montage)
    print(f"\nmontage -> {montage}")
    print(
        "Judge: did B (refine ON) drop the cap where A did not? If yes, "
        "check the SEAM around the head -- refine composites its output "
        "back, which is exactly the boundary class simple_full_body_raw_model "
        "exists to avoid elsewhere. A cap-free head with a visible ring "
        "around it is a new problem, not a clean win.",
        flush=True,
    )
    return 0


def _montage(items, out_path: Path, pad: int = 8, h: int = 420) -> None:
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
