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

    # ABSOLUTE, resolved before the runtime is built. ComfyUI's init chdirs
    # into its own tree, so a relative --pair-dir passes the existence check
    # here and then fails to open a few lines later, under a different cwd.
    pair = Path(args.pair_dir)
    if not pair.is_absolute():
        pair = (REPO / pair).resolve()
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
    if not out.is_absolute():
        out = (REPO / out).resolve()
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

    # These stages NEST: "sampling" wraps _sample_edit, which contains
    # "diffusion_sampling", which contains "ksampler_only". Summing them
    # counts the same seconds up to three times -- the first version of this
    # report did exactly that and printed a NEGATIVE unaccounted line.
    # Wall time is the two top-level passes; the rest are breakdowns inside.
    main = timings.get("sampling", 0.0)                 # main pass, wall
    refine = timings.get("face_refine_sampling", 0.0)   # refine pass, wall
    ks = timings.get("ksampler_only", 0.0)              # summed over passes
    diff = timings.get("diffusion_sampling", 0.0)       # summed over passes
    churn = max(0.0, diff - ks)
    other = max(0.0, total - main - refine)

    def _row(label, secs, note=""):
        print(f"  {label:<24}{secs:7.2f}s  {100 * secs / total:5.1f}%  {note}")

    print("  -- wall time (these two sum to the total) --")
    _row("main pass", main)
    _row("face_refine pass", refine, "<- a whole 2nd pass; CHECKPOINT-12")
    _row("outside both", other)
    print("  -- what is inside them --")
    _row("KSampler (denoise)", ks, "<- QUALITY cost to cut")
    _row("GPU residency churn", churn, "<- FREE (full_load evict/reload)")
    _row("encode/vae/misc", max(0.0, main + refine - ks - churn))

    print()
    print(f"  Target 25s from {total:.0f}s -> remove {total - 25:.0f}s.")
    print(f"  Killing residency churn alone: -{churn:.0f}s -> {total - churn:.0f}s")
    print(f"  Halving UNet evals as well:    -{ks / 2:.0f}s -> "
          f"{total - churn - ks / 2:.0f}s")
    print()
    print("  Two ways to halve the evals, and they are NOT equivalent:")
    print("    steps 8->4 at cfg 1.8  -- keeps classifier-free guidance;")
    print("                              4 steps is upstream's own edit value")
    print("    cfg 1.8->1.0           -- CHECKPOINT-10 measured identity")
    print("                              0.561 at cfg 1.0 vs 0.671 at 1.8,")
    print("                              so this one has a KNOWN identity cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
