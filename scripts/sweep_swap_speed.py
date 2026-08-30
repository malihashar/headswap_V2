#!/usr/bin/env python3
"""Sweep the T4 swap over (steps, cfg) -- wall time AND identity per arm.

Profiled baseline (A100, warm): 50.7s, of which KSampler is 39.8s (78%),
GPU residency churn 5.9s, encode/vae/misc 5.0s. Detection, rembg and
compositing are ~free once warm, so speed here is bought almost entirely by
doing fewer UNet evaluations. Each pass runs steps x 2 evals at cfg > 1
(classifier-free guidance costs a second evaluation per step), and there are
two passes: the main one and face_refine.

There are two ways to halve the evals and they are NOT equivalent:

  steps 8 -> 4 at cfg 1.8   keeps classifier-free guidance. 4 steps is
                            upstream's own edit value and Turbo is distilled
                            for few steps. Untested here; the expected risk
                            is "less refined", not "worse identity".

  cfg 1.8 -> 1.0            CHECKPOINT-10 measured identity 0.561 at cfg 1.0
                            against 0.671 at cfg 1.8, so this carries a
                            KNOWN identity penalty, and identity is already
                            the weak axis. It also silently disables the
                            skin-tone clause, which needs guidance to work.

Hence the default arms below vary steps and hold cfg at 1.8. Timing alone
would rank these two as identical, which is exactly the trap -- so this
reports ArcFace identity beside every wall time.

Usage:
    python scripts/sweep_swap_speed.py --pair-dir data/custom/my_pair
    python scripts/sweep_swap_speed.py --arms "8:1.8,6:1.8,4:1.8,4:1.0"
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
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "my_pair"))
    ap.add_argument("--arms", default="8:1.8,6:1.8,4:1.8",
                    help="comma-separated steps:cfg pairs")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "swap_speed"))
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py")
    colab_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colab_env)
    paths = colab_env.apply_env(colab_env.default_paths(use_drive=False))
    colab_env.ensure_import_path(REPO)
    comfy = str(paths.get("comfyui", "/content/ComfyUI"))
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    os.chdir(REPO)

    from PIL import Image
    from headswap.config import load_config
    from headswap.metrics.scoring import identity_cosine
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
                "pre_edit_donor_expression": False})
    pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))

    arms = []
    for tok in str(args.arms).split(","):
        st, _, cf = tok.strip().partition(":")
        arms.append((int(st), float(cf or 1.8)))

    print("\n=== WARMUP (excluded; loads models + rembg) ===\n", flush=True)
    pipe.cfg["steps"], pipe.cfg["cfg"] = arms[0]
    pipe.run(body, face, out_dir=out)

    results = []
    for steps, cfgv in arms:
        pipe.cfg["steps"], pipe.cfg["cfg"] = steps, cfgv
        evals = steps * (2 if cfgv > 1.0 + 1e-6 else 1)
        print(f"\n=== steps={steps} cfg={cfgv}  (~{evals} UNet evals/pass) ===\n",
              flush=True)
        t0 = time.perf_counter()
        res = pipe.run(body, face, out_dir=out)
        wall = time.perf_counter() - t0
        idc = identity_cosine(face, res.image)
        path = out / f"swap_s{steps}_cfg{cfgv}.png"
        res.image.save(path)
        results.append({"steps": steps, "cfg": cfgv, "evals": evals,
                        "wall": wall, "identity": idc, "image": res.image,
                        "path": path})
        print(f"  -> {wall:.1f}s  identity_vs_donor={idc}  {path.name}", flush=True)

    print("\n" + "=" * 68)
    print(f"{'steps':>6} {'cfg':>5} {'evals':>6} {'wall':>8} {'identity':>10}")
    print("-" * 68)
    for r in results:
        flag = "  <- TARGET" if r["wall"] <= 25 else ""
        print(f"{r['steps']:>6} {r['cfg']:>5} {r['evals']:>6} "
              f"{r['wall']:>7.1f}s {str(r['identity'])[:8]:>10}{flag}")
    print("=" * 68)
    print("Judge BOTH columns. Fewer evals is always faster; the question is")
    print("whether the face survives it. A fast arm with collapsed identity is")
    print("not a win. Residency churn (~6s, free) is still on the table and is")
    print("not reflected here.")

    _montage([(f"s{r['steps']} cfg{r['cfg']} {r['wall']:.0f}s id={str(r['identity'])[:5]}",
               r["image"]) for r in results], out / "speed_montage.png")
    print(f"\nmontage -> {out / 'speed_montage.png'}")
    return 0


def _montage(items, out_path: Path, pad: int = 8) -> None:
    from PIL import Image, ImageDraw
    h = 320
    tiles = [(lb, im.convert("RGB").resize(
        (max(1, int(im.width * h / max(1, im.height))), h),
        Image.Resampling.LANCZOS)) for lb, im in items]
    W = sum(t.width for _, t in tiles) + pad * (len(tiles) + 1)
    canvas = Image.new("RGB", (W, h + 28 + pad * 2), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = pad
    for lb, t in tiles:
        canvas.paste(t, (x, 28 + pad))
        d.text((x + 2, 6), str(lb), fill=(235, 235, 235))
        x += t.width + pad
    canvas.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
