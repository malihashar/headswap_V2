#!/usr/bin/env python3
"""Warm in-process Krea2 Identity Edit runner.

Cold start pays bootstrap (~25s) + model load (~7s). This script keeps one
pipeline / shared Comfy runtime alive and runs the same pair twice so the
second pass measures sampling-dominated wall time.

Usage (Kaggle / Colab, after setup_kaggle.sh --krea2):

  python scripts/run_krea2_warm.py --pair-id custom_001
  python scripts/run_krea2_warm.py --pair-id custom_001 --passes 3

Does not change model weights or sampling quality settings.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from headswap.config import load_config, resolve_out_dir
from headswap.eval.dataset import load_pairs
from headswap.pipelines import create_pipeline
from headswap.pipelines.krea2 import get_shared_krea2_runtime


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/krea2_identity_edit.yaml",
        help="YAML config (default: configs/krea2_identity_edit.yaml)",
    )
    ap.add_argument("--pair-id", default="custom_001")
    ap.add_argument("--passes", type=int, default=2, help="In-process runs (default 2)")
    ap.add_argument("--out", default=None, help="Output root (default: results/<name>_warm)")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = ROOT / args.config
    if not cfg_path.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(cfg_path)
    pairs = [p for p in load_pairs() if p["id"] == args.pair_id]
    if not pairs:
        print(f"Pair not found: {args.pair_id}", file=sys.stderr)
        return 1
    pair = pairs[0]

    out_root = (
        Path(args.out)
        if args.out
        else resolve_out_dir(cfg, None).parent / f"{cfg.get('name', 'krea2')}_warm"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-create shared runtime so pass-1 bootstrap is explicit in timings.
    t_boot = time.perf_counter()
    rt = get_shared_krea2_runtime(init_custom_nodes=True)
    print(f"[warm] shared runtime ready in {time.perf_counter() - t_boot:.2f}s")

    pipe = create_pipeline(cfg, runtime=rt)
    body = Image.open(pair["body_path"]).convert("RGB")
    face = Image.open(pair["face_path"]).convert("RGB")

    for i in range(max(1, int(args.passes))):
        pass_dir = out_root / f"pass_{i + 1}" / pair["id"]
        pass_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[warm] === pass {i + 1}/{args.passes} → {pass_dir} ===")
        t0 = time.perf_counter()
        result = pipe.run(body, face, out_dir=pass_dir)
        wall = time.perf_counter() - t0
        out_file = pass_dir / "result.png"
        result.image.save(out_file)
        timing = (result.meta or {}).get("timing_s") or {}
        sample = timing.get("diffusion_sampling")
        boot = timing.get("bootstrap")
        load = timing.get("model_loading")
        print(
            f"[warm] pass={i + 1} wall={wall:.2f}s latency_s={result.latency_s:.2f}s "
            f"bootstrap={boot} model_loading={load} diffusion={sample} "
            f"cache_hit={result.meta.get('model_cache_hit')} → {out_file}"
        )

    print(
        "\n[warm] Done. Compare pass 1 vs pass 2: bootstrap/load should collapse; "
        "diffusion_sampling should stay similar (GPU-bound)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
