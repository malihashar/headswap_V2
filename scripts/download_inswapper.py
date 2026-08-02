#!/usr/bin/env python3
"""Download InsightFace buffalo_l + InSwapper-128 (and optional GFPGAN).

Idempotent. Default cache: results/_cache/inswap (or --cache-dir / INSIGHTFACE_HOME).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "results" / "_cache" / "inswap",
    )
    parser.add_argument(
        "--with-gfpgan",
        action="store_true",
        help="Also download GFPGANv1.4.pth",
    )
    parser.add_argument(
        "--with-codeformer",
        action="store_true",
        help="Also download codeformer.pth",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    from headswap.inswap.detect import InsightFaceDetector
    from headswap.inswap.engines.inswapper import download_inswapper_model
    from headswap.inswap.restore import FaceRestorer

    print("→ buffalo_l (InsightFace detection + ArcFace)…", flush=True)
    det = InsightFaceDetector()
    try:
        det.load(cache, device=args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED detector: {exc}")
        print("Install: pip install 'insightface>=0.7' onnxruntime-gpu  (or onnxruntime)")
        return 1

    print("→ inswapper_128.onnx…", flush=True)
    model = download_inswapper_model(cache)
    print(f"  OK {model} ({model.stat().st_size / 1e6:.1f} MB)")

    if args.with_gfpgan:
        print("→ GFPGAN…", flush=True)
        FaceRestorer("gfpgan").load(cache, device=args.device)
    if args.with_codeformer:
        print("→ CodeFormer weight…", flush=True)
        FaceRestorer("codeformer").load(cache, device=args.device)

    print("OK: InSwapper assets ready")
    print(f"  cache: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
