#!/usr/bin/env python3
"""Profile ONE T4 swap and print where the wall time actually goes.

Target context: the swap measures ~140s on an A100 and the ask is ~25s, a
~5.6x cut. Before trading any quality for speed it is worth knowing what
fraction of that is even sampling.

Reason to doubt that it mostly is: the standalone single-image edit runs 4
UNet evaluations in ~21s INCLUDING model load. The swap runs ~32 evaluations
(8 steps x cfg 1.8 = 16 for the main pass, plus the same again for
face_refine) at a smaller resolution -- which does not extrapolate to 140s.
The gap is model load/offload churn, CPU-side InsightFace, rembg and VAE.

That matters commercially: overhead is free to remove (identical pixels),
while cutting steps/cfg/face_refine costs quality and unwinds decisions made
deliberately in CHECKPOINT-10 and CHECKPOINT-12.

Usage:
    python scripts/profile_swap.py --pair-dir data/custom/my_pair
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "my_pair"))
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument(
        "--out-dir", default=str(REPO / "results" / "profile_swap")
    )
    args = ap.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "colab_env", REPO / "scripts" / "colab_env.py"
    )
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
    from headswap.pipelines import create_pipeline
    from headswap.pipelines.krea2 import get_shared_krea2_runtime

    pair = Path(args.pair_dir)
    body_p, face_p = pair / "body.png", pair / "face.png"
    for f in (body_p, face_p):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 2

    cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
    cfg.update(
        {
            "seed": int(args.seed),
            "verbose": True,
            "pre_edit_donor_expression": False,
            "print_timing_breakdown": True,
        }
    )
    pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== WARMUP (loads models; excluded from the measured run) ===\n")
    body = Image.open(body_p).convert("RGB")
    face = Image.open(face_p).convert("RGB")
    pipe.run(body, face, out_dir=out)

    print("\n=== MEASURED RUN (models already resident) ===\n")
    res = pipe.run(body, face, out_dir=out)

    meta = res.meta or {}
    timings = meta.get("timing_s") or {}
    total = meta.get("total_s") or res.latency_s

    print("\n" + "=" * 66)
    print(f"WARM SWAP TOTAL: {total:.1f}s   (target 25s)")
    print("=" * 66)
    if not timings:
        print("no timing_s in meta -- is the build current?")
        return 1

    sampling = sum(v for k, v in timings.items() if "sampling" in k)
    loading = sum(v for k, v in timings.items() if "load" in k)
    vae = sum(v for k, v in timings.items() if "vae" in k)
    print(f"  sampling      {sampling:7.2f}s  {100*sampling/total:5.1f}%  <- costs QUALITY to cut")
    print(f"  model loading {loading:7.2f}s  {100*loading/total:5.1f}%  <- FREE to cut (residency)")
    print(f"  vae           {vae:7.2f}s  {100*vae/total:5.1f}%")
    other = total - sampling - loading - vae
    print(f"  everything else {other:5.2f}s  {100*other/total:5.1f}%  <- detection/rembg/composite")
    print("\nIf sampling is well under half, the 25s target is mostly an")
    print("infrastructure problem and quality does not have to be traded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
