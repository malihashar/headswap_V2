#!/usr/bin/env python3
"""Run all pipelines (mock by default) and write comparison winner."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.cli import main_all_mock
from headswap.eval.runner import compare_reports, run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--gpu", action="store_true", help="Run real ComfyUI pipelines (requires GPU + models)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.gpu:
        configs = [
            ROOT / "configs" / "klein4b.yaml",
            ROOT / "configs" / "qwen_baseline.yaml",
            ROOT / "configs" / "qwen_improved.yaml",
        ]
        reports = []
        for cfg in configs:
            run_eval(cfg, out_dir=ROOT / "results" / cfg.stem, force_mock=False, limit=args.limit)
            reports.append(ROOT / "results" / cfg.stem / "metrics.json")
        compare_reports(reports)
    else:
        main_all_mock()


if __name__ == "__main__":
    main()
