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
    ap.add_argument(
        "--lp-order", choices=["before", "after"], default="after",
        help="'before': LP edits the donor, then T4 swaps (T4 REGENERATES the "
             "face and normalises the expression away -- measured). 'after': "
             "T4 swaps, then LP drives the final image, so nothing downstream "
             "can override it and LP works at full resolution.")
    ap.add_argument("--driving-multiplier", type=float, default=1.0)
    ap.add_argument(
        "--skip-refine", action="store_true",
        help="skip face_refine on bust shots (the ~20s second sampling pass). "
             "Uses the EXISTING simple_full_body_refine_max_face_frac gate, "
             "so full-body frames still refine.")
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
        # Auto-install rather than silently degrading to T4-only. A recycled
        # Colab runtime wipes /content, and the previous behaviour printed
        # one NOTE and produced a table of zeros for the half of the budget
        # we were trying to measure.
        print(f"LivePortrait missing at {args.lp_dir} - installing ...",
              flush=True)
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/KwaiVGI/LivePortrait",
                        str(args.lp_dir)], check=True)
        subprocess.run(["pip", "install", "-q", "--no-cache-dir", "tyro",
                        "imageio", "imageio-ffmpeg", "rich", "pykalman",
                        "ffmpeg-python"], check=False)
        w = Path(args.lp_dir) / "pretrained_weights"
        if not (w / "liveportrait").exists():
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="KwaiVGI/LivePortrait",
                              local_dir=str(w), ignore_patterns=["*animal*"])
        print("LivePortrait installed.", flush=True)

    if use_lp and not (Path(args.lp_dir) / "inference.py").exists():
        print(f"NOTE: LivePortrait still missing; running T4 only.")
        use_lp = False

    cfg = load_config(REPO / "configs" / "krea2_identity_edit.yaml")
    cfg.update({"seed": int(args.seed), "verbose": False,
                "pre_edit_donor_expression": False})
    if args.skip_refine:
        # 0.25 matches simple_full_body_restore_max_face_frac, which already
        # encodes "is this a bust shot?". The default is 1.01 (unreachable),
        # so refine always runs; CHECKPOINT-12 measured it at 76s vs 53s and
        # "visually indistinguishable" on a 42% bust shot, then rejected the
        # skip at full size when speed was not a requirement. It is now.
        cfg["simple_full_body_refine_max_face_frac"] = 0.25
        print("NOTE: face_refine will SKIP on bust shots "
              "(simple_full_body_refine_max_face_frac=0.25). Compare the "
              "output against a run without --skip-refine before adopting.",
              flush=True)
    pipe = create_pipeline(cfg, runtime=get_shared_krea2_runtime(init_custom_nodes=True))

    rows = []
    n = 1 + max(0, int(args.repeats))
    for i in range(n):
        label = "COLD" if i == 0 else f"WARM{i}"
        print(f"\n{'=' * 66}\n=== {label} ===\n{'=' * 66}", flush=True)
        row = {"label": label}

        donor_for_swap = face_p
        if use_lp and args.lp_order == "before":
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
            row["lp_placement"] = lp.get("placement")
            # Record WHETHER LivePortrait's output actually became the donor.
            # The fallback below is silent: if LP produced nothing, or an
            # .mp4, T4 runs on the ORIGINAL donor and the chain timing looks
            # identical -- a 21.8s "chain" could be a swap-only number with
            # LP contributing nothing but overhead. That must be a reported
            # value, not an assumption.
            row["lp_used"] = False
            if lp.get("primary"):
                _p = Path(lp["primary"])
                if _p.suffix.lower() != ".mp4":
                    donor_for_swap = _p
                    row["lp_used"] = True
                else:
                    row["lp_note"] = f"output was {_p.suffix}, not an image"
            else:
                row["lp_note"] = "LivePortrait produced no output file"
            print(f"[chain] LP output fed to T4: {row['lp_used']} "
                  f"({row.get('lp_note', donor_for_swap.name)})", flush=True)
        else:
            row["lp_total"] = row["lp_load"] = row["lp_exec"] = 0.0
            row["lp_used"] = False
            row["lp_note"] = ("LivePortrait runs AFTER the swap"
                              if use_lp else "LivePortrait disabled")

        donor_im = Image.open(donor_for_swap).convert("RGB")
        t0 = time.perf_counter()
        res = pipe.run(body, donor_im, out_dir=out / f"swap_{label.lower()}")
        row["t4_total"] = time.perf_counter() - t0
        meta = res.meta or {}
        row["t4_pre"] = float(meta.get("pre_dispatch_s") or 0.0)
        tim = meta.get("timing_s") or {}
        row["t4_ks"] = float(tim.get("ksampler_only") or 0.0)
        row["t4_refine"] = float(tim.get("face_refine_sampling") or 0.0)
        row["refine_applied"] = bool((meta.get("face_refine") or {}).get("applied"))
        row["chain"] = row["lp_total"] + row["t4_total"]
        swap_png = out / f"swap_only_{label.lower()}.png"
        res.image.save(swap_png)
        final_img = swap_png

        if use_lp and args.lp_order == "after":
            # POST-step: drive T4's finished frame with the ORIGINAL target
            # photo. Running LP before the swap does not survive -- T4
            # regenerates the head from the donor crop and normalises the
            # expression back to something plausible, measured directly:
            # LivePortrait produced an open mouth and the final frame came
            # back with an ordinary closed smile.
            #
            # Running last also gives LP the full-resolution result to work
            # on instead of a ~205x186 donor thumbnail.
            from headswap.expression_transfer import run_expression_transfer
            t0 = time.perf_counter()
            lp = run_expression_transfer(
                source_path=swap_png,   # identity to KEEP: T4's own output
                driving_path=body_p,    # expression to TAKE: the target
                out_dir=out / f"lp_{label.lower()}",
                animation_region=args.animation_region,
                driving_multiplier=args.driving_multiplier,
                live_portrait_dir=args.lp_dir,
            )
            row["lp_total"] = time.perf_counter() - t0
            row["lp_load"] = float(lp.get("model_load_s") or 0.0)
            row["lp_exec"] = float(lp.get("latency_s") or 0.0)
            row["lp_placement"] = lp.get("placement")
            row["lp_used"] = False
            if lp.get("primary") and Path(lp["primary"]).suffix.lower() != ".mp4":
                final_img = Path(lp["primary"])
                row["lp_used"] = True
            else:
                row["lp_note"] = "LP produced no usable image; final is swap-only"
            row["chain"] = row["lp_total"] + row["t4_total"]
            print(f"[chain] post-step LP applied: {row['lp_used']}", flush=True)

        from PIL import Image as _I
        _I.open(final_img).convert("RGB").save(out / f"final_{label.lower()}.png")
        rows.append(row)

    # Buffer the summary AND write it to a file. The run emits thousands of
    # ComfyUI/path_proof lines, so the table at the end is unreachable by
    # scrolling in a Colab cell -- three attempts to read these numbers came
    # back truncated to the first 20 lines.
    out_lines: list[str] = []

    def emit(line: str = "") -> None:
        out_lines.append(line)
        print(line, flush=True)

    emit("\n" + "=" * 78)
    emit(f"{'':8}{'LP load':>9}{'LP exec':>9}{'LP tot':>9}"
         f"{'T4 pre':>9}{'T4 KSamp':>10}{'T4 tot':>9}{'CHAIN':>9}")
    emit("-" * 78)
    for r in rows:
        emit(f"{r['label']:8}{r['lp_load']:>8.1f}s{r['lp_exec']:>8.1f}s"
             f"{r['lp_total']:>8.1f}s{r['t4_pre']:>8.1f}s{r['t4_ks']:>9.1f}s"
             f"{r['t4_total']:>8.1f}s{r['chain']:>8.1f}s")
    emit("=" * 78)

    warm = [r for r in rows if r["label"] != "COLD"]
    if warm:
        best = min(r["chain"] for r in warm)
        emit(f"\nWARM CHAIN: {best:.1f}s   (target 30s)")
        w = min(warm, key=lambda r: r["chain"])
        emit(f"  LivePortrait  {w['lp_total']:6.1f}s"
             f"   (exec {w['lp_exec']:.1f}s, load {w['lp_load']:.1f}s)")
        emit(f"  T4 swap       {w['t4_total']:6.1f}s"
             f"   (pre {w['t4_pre']:.1f}s, KSampler {w['t4_ks']:.1f}s, "
             f"face_refine {w.get('t4_refine', 0.0):.1f}s "
             f"applied={w.get('refine_applied')})")
        emit(f"  LP OUTPUT USED BY T4: {w.get('lp_used')}"
             + (f"   <-- {w['lp_note']}" if w.get("lp_note") else ""))
        if not w.get("lp_used"):
            emit("      WARNING: T4 ran on the ORIGINAL donor. This CHAIN "
                 "number is swap-only; LivePortrait changed nothing.")
        if w.get("lp_placement"):
            emit(f"  LP placement  {w['lp_placement']}")
        gap = best - 30.0
        if gap > 0:
            emit(f"\n  {gap:.1f}s over target.")
        else:
            emit("\n  UNDER TARGET.")
    if rows:
        c = rows[0]
        emit(f"\nCOLD start cost: "
             f"{c['chain'] - (warm[0]['chain'] if warm else 0):.1f}s extra on "
             "the first request (weights, CUDA context, warmups).")
    summary_path = out / "SUMMARY.txt"
    summary_path.write_text("\n".join(out_lines) + "\n")
    print(f"\n>>> SUMMARY WRITTEN: {summary_path}")
    print(f">>> Read it with:  !cat {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
