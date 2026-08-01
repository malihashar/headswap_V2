#!/usr/bin/env python3
"""Download InsightFace buffalo_l weights used by geometry-locked face swap.

Idempotent. Stores under results/_cache/insightface (or INSIGHTFACE_HOME).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    cache = ROOT / "results" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    from headswap import preprocess as prep

    app = prep.ensure_insightface_app(cache)
    if app is None:
        err = prep._INSIGHTFACE_INIT_ERROR or "unknown"
        print(f"FAILED: {err}")
        print("Install: pip install 'insightface>=0.7' 'onnxruntime>=1.16'")
        return 1
    print("OK: InsightFace buffalo_l ready")
    print(f"  cache root: {cache / 'insightface'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
