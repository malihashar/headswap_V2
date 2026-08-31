#!/usr/bin/env python3
"""Sweep the SWAP's denoise. The structural lever for headwear, mask-free.

Why denoise and not another prompt or a mask. denoise=0.85 seeds the sampler
from the SOURCE latent, which is exactly why clothing, pose, framing and
background survive -- and equally why a cap does. img2img at that setting
exists to preserve large, high-contrast structure, and a cap is precisely
that. Four prompt wordings could not argue it away (CHECKPOINT-16), because
describing what to draw does not remove what is already in the starting
point.

Raising denoise gives the sampler room to drop it. The cost is the same
anchor that protects the garment, so this sweep must be judged on BOTH:

    hat gone?   AND   is the clothing still the clothing?

CHECKPOINT-10 measured the far end: at denoise 1.0 from pure noise a black
robe returned as a bare torso and a lace dress as a top plus skirt. That was
from EMPTY noise, not from the source latent, so the failure here should be
gentler and gradual -- but it is the direction of travel, and the point of
sweeping rather than guessing a value.

Runs the swap only, no LivePortrait, since expression is not what is being
judged.

    python scripts/sweep_swap_denoise.py --denoise 0.85,0.90,0.95
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
    ap.add_argument("--denoise", default="0.85,0.90,0.95")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "denoise_sweep"))
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
    from headswap.config import load_config
    from headswap.pipelines import create_pipeline
    from headswap.pipelines.krea2 import get_shared_krea2_runtime

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
    if not out.is_absolute():
        out = (REPO / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
    cfg.update({"seed": int(args.seed), "verbose": False,
                "pre_edit_donor_expression": False,
                "simple_full_body_remove_headwear": True,
                "skip_skin_clause_when_covered": True,
                "simple_full_body_refine_max_face_frac": 0.25})
    pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))

    values = [float(x) for x in args.denoise.split(",") if x.strip()]
    print(f"\n=== WARMUP (excluded) ===\n", flush=True)
    pipe.cfg["denoise"] = values[0]
    pipe.run(body, face, out_dir=out)

    tiles = [("target", body)]
    for dn in values:
        pipe.cfg["denoise"] = dn
        print(f"\n=== denoise={dn} ===\n", flush=True)
        t0 = time.perf_counter()
        res = pipe.run(body, face, out_dir=out)
        wall = time.perf_counter() - t0
        path = out / f"swap_denoise_{dn}.png"
        res.image.save(path)
        tiles.append((f"dn={dn}", res.image))
        print(f"  -> {wall:.0f}s  {path.name}", flush=True)

    _montage(tiles, out / "denoise_montage.png")
    print(f"\nmontage -> {out / 'denoise_montage.png'}")
    print("\nJudge BOTH: is the hat gone, AND is the clothing still the")
    print("clothing? Raising denoise loosens the same anchor that protects")
    print("the garment, so a hatless arm with a different outfit is not a win.")
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
