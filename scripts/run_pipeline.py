#!/usr/bin/env python3
"""Run a single head-swap pipeline from its YAML config.

Day-to-day benchmarking entrypoint: generates result images, prints stage
timings (when the pipeline reports them), writes metrics.json, and exits.

For full multi-pipeline evaluations, use scripts/run_compare.py instead.

Examples:
  python scripts/run_pipeline.py --config configs/qwen_baseline.yaml --limit 1
  python scripts/run_pipeline.py --config configs/klein4b.yaml --mock --limit 1
  python scripts/run_pipeline.py --config configs/qwen_baseline.yaml --pair-id custom_001
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.cli import main_run


if __name__ == "__main__":
    raise SystemExit(main_run())
