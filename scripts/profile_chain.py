#!/usr/bin/env python3
"""Time the FULL chain -- LivePortrait expression transfer + T4 swap -- cold vs warm.

Both engines load their models ONCE, then the chain runs twice. The first
pass pays every one-time cost (weights off disk, CUDA context, rembg/
InsightFace warmup, LivePortrait's five .pth models); the second is what a
served request would actually cost. Reporting only one of those numbers has
already misled this investigation twice: a 140s "swap" was cold and measured
50.7s warm, and a "30s" LivePortrait figure came from cold, no-op runs.

Stages timed separately, because they have completely different fixes:

    LP model load     one-time; matters only for cold start
    LP execute        the 22s "Animating" for a single frame that looks wrong
    T4 pre_dispatch   InsightFace + rembg, CPU-bound
    T4 KSampler       the only part where cutting costs quality
    T4 other          VAE, grounded encode, compositing

Usage:
    python scripts/profile_chain.py --pair-dir data/custom/my_pair
    python scripts/profile_chain.py --no-liveportrait     # T4 only
    python scripts/profile_chain.py --repeats 3           # more warm samples
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _bootstrap():
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
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", default=str(REPO / "data" / "custom" / "my_pair"))
    ap.add_argument("--out-dir", default=str(REPO / "results" / "chain_profile"))
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--repeats", type=int, default=1, help="warm runs after the cold one")
    ap.add_argument("--no-liveportrait", action="store_true")
    ap.add_argument("--lp-dir", default="/content/LivePortrait")
    ap.add_argument("--animation-region", default="lip")
    args = ap.parse_args()

    _bootstrap()
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

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = (REPO / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    use_lp = not args.no_liveportrait
    if use_lp and not (Path(args.lp_dir) / "inference.py").exists():
        print(f"NOTE: LivePortrait not at {args.lp_dir}; running T4 only.")
        use_lp = False

    cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
    cfg.update({"seed": int(args.seed), "verbose": False,
                "pre_edit_donor_expression": False})
    pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))

    rows = []
    n = 1 + max(0, int(args.repeats))
    for i in range(n):
        label = "COLD" if i == 0 else f"WARM{i}"
        print(f"\n{'=' * 66}\n=== {label} ===\n{'=' * 66}", flush=True)
        row = {"label": label}

        donor_for_swap = face_p
        if use_lp:
            from headswap.expression_transfer import run_expression_transfer
            t0 = time.perf_counter()
            lp = run_expression_transfer(
                source_path=face_p,        # identity to KEEP
                driving_path=body_p,       # expression to TAKE
                out_dir=out / f"lp_{label.lower()}",
                animation_region=args.animation_region,
                live_portrait_dir=args.lp_dir,
            )
            row["lp_total"] = time.perf_counter() - t0
            row["lp_load"] = float(lp.get("model_load_s") or 0.0)
            row["lp_exec"] = float(lp.get("latency_s") or 0.0)
            if lp.get("primary"):
                p = Path(lp["primary"])
                if p.suffix.lower() != ".mp4":
                    donor_for_swap = p
        else:
            row["lp_total"] = row["lp_load"] = row["lp_exec"] = 0.0

        donor_im = Image.open(donor_for_swap).convert("RGB")
        t0 = time.perf_counter()
        res = pipe.run(body, donor_im, out_dir=out / f"swap_{label.lower()}")
        row["t4_total"] = time.perf_counter() - t0
        meta = res.meta or {}
        row["t4_pre"] = float(meta.get("pre_dispatch_s") or 0.0)
        tim = meta.get("timing_s") or {}
        row["t4_ks"] = float(tim.get("ksampler_only") or 0.0)
        row["chain"] = row["lp_total"] + row["t4_total"]
        rows.append(row)
        res.image.save(out / f"final_{label.lower()}.png")

    print("\n" + "=" * 78)
    print(f"{'':8}{'LP load':>9}{'LP exec':>9}{'LP tot':>9}"
          f"{'T4 pre':>9}{'T4 KSamp':>10}{'T4 tot':>9}{'CHAIN':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:8}{r['lp_load']:>8.1f}s{r['lp_exec']:>8.1f}s"
              f"{r['lp_total']:>8.1f}s{r['t4_pre']:>8.1f}s{r['t4_ks']:>9.1f}s"
              f"{r['t4_total']:>8.1f}s{r['chain']:>8.1f}s")
    print("=" * 78)

    warm = [r for r in rows if r["label"] != "COLD"]
    if warm:
        best = min(r["chain"] for r in warm)
        print(f"\nWARM CHAIN: {best:.1f}s   (target 30s)")
        w = min(warm, key=lambda r: r["chain"])
        print(f"  LivePortrait  {w['lp_total']:6.1f}s"
              f"   (exec {w['lp_exec']:.1f}s, load {w['lp_load']:.1f}s)")
        print(f"  T4 swap       {w['t4_total']:6.1f}s"
              f"   (pre {w['t4_pre']:.1f}s, KSampler {w['t4_ks']:.1f}s)")
        gap = best - 30.0
        if gap > 0:
            print(f"\n  {gap:.1f}s over. face_refine is ~half of T4's KSampler")
            print("  time and is the only remaining lever that does not touch")
            print("  steps/cfg/denoise/ref_boost/resolution.")
        else:
            print("\n  UNDER TARGET.")
    if rows:
        c = rows[0]
        print(f"\nCOLD start cost: {c['chain'] - (warm[0]['chain'] if warm else 0):.1f}s "
              "extra on the first request (weights, CUDA context, warmups).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
