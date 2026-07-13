from __future__ import annotations

import argparse
from pathlib import Path

from headswap.config import project_root
from headswap.eval.dataset import generate_synthetic_eval_set
from headswap.eval.runner import compare_reports, run_eval


def main_run():
    ap = argparse.ArgumentParser(description="Run a head-swap pipeline on the eval set")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--mock", action="store_true", help="Force CPU mock pipeline")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_eval(args.config, out_dir=args.out, force_mock=args.mock, limit=args.limit)


def main_evaluate():
    # alias
    main_run()


def main_compare():
    ap = argparse.ArgumentParser(description="Compare pipeline metric reports and pick a winner")
    ap.add_argument(
        "--reports",
        nargs="+",
        default=None,
        help="metrics.json paths (default: results/*/metrics.json)",
    )
    ap.add_argument("--latency-budget", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.reports:
        paths = [Path(p) for p in args.reports]
    else:
        paths = sorted((project_root() / "results").glob("*/metrics.json"))
    if not paths:
        raise SystemExit("No metrics.json reports found. Run pipelines first.")
    compare_reports(paths, latency_budget_s=args.latency_budget, out_path=Path(args.out) if args.out else None)


def main_all_mock():
    """Prepare eval set, run all configs in mock mode, compare."""
    generate_synthetic_eval_set()
    root = project_root()
    configs = [
        root / "configs" / "klein4b.yaml",
        root / "configs" / "qwen_baseline.yaml",
        root / "configs" / "qwen_improved.yaml",
    ]
    reports = []
    for cfg in configs:
        name = cfg.stem
        report = run_eval(cfg, out_dir=root / "results" / name, force_mock=True)
        reports.append(root / "results" / name / "metrics.json")
        assert report["n_pairs"] > 0
    compare_reports(reports)


if __name__ == "__main__":
    main_run()
