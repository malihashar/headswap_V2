from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from headswap.config import load_config, project_root, resolve_out_dir
from headswap.eval.dataset import generate_synthetic_eval_set
from headswap.eval.runner import compare_reports, run_eval


def _print_run_summary(report: dict[str, Any], results_dir: Path) -> None:
    summary = report.get("summary") or {}
    n = int(report.get("n_pairs") or 0)
    n_ok = int(summary.get("n_success") or 0)
    lat_mean = summary.get("latency_mean")
    lat_p50 = summary.get("latency_p50")
    print()
    print("=" * 60)
    print(f"pipeline:  {report.get('pipeline')}")
    print(f"config:    {report.get('config')}")
    print(f"pairs:     {n_ok}/{n} success")
    if lat_mean is not None:
        print(f"latency:   mean={lat_mean:.2f}s  p50={lat_p50:.2f}s")
    print(f"metrics:   {results_dir / 'metrics.json'}")
    print(f"images:    {results_dir / 'images'}")
    for row in report.get("pairs") or []:
        meta = row.get("meta") or {}
        timing = meta.get("timing_s")
        path = row.get("result_path")
        line = (
            f"  - {row.get('pair_id')}: success={row.get('success')} "
            f"lat={float(row.get('latency_s') or 0):.2f}s"
        )
        if path:
            line += f" → {path}"
        print(line)
        if isinstance(timing, dict) and timing:
            parts = [f"{k}={float(v):.2f}s" for k, v in timing.items()]
            print(f"      timing: {', '.join(parts)}")
    print("=" * 60)


def main_run(argv: list[str] | None = None) -> int:
    """Run one pipeline from a YAML config; write images + metrics.json; exit."""
    ap = argparse.ArgumentParser(
        description=(
            "Run a single head-swap pipeline from its config. "
            "Writes result images and metrics.json, then exits. "
            "Use scripts/run_compare.py for full multi-pipeline evals."
        )
    )
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory (default: results/<config name>)",
    )
    ap.add_argument(
        "--mock",
        action="store_true",
        help="Force CPU mock pipeline (no ComfyUI / GPU)",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Run real ComfyUI GPU pipeline (default when --mock is not set)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max eval pairs to run")
    ap.add_argument(
        "--pair-id",
        action="append",
        default=None,
        dest="pair_ids",
        help="Run only this pair id (repeatable)",
    )
    args = ap.parse_args(argv)

    if args.mock and args.gpu:
        ap.error("Use either --mock or --gpu, not both")

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_config(cfg_path)
    results_dir = resolve_out_dir(cfg, args.out)
    force_mock = bool(args.mock)
    mode = "mock" if force_mock else "gpu"
    print(f"Running {cfg.get('name', cfg_path.stem)} [{mode}] → {results_dir}")

    report = run_eval(
        cfg_path,
        out_dir=results_dir,
        force_mock=force_mock,
        limit=args.limit,
        pair_ids=args.pair_ids,
    )
    _print_run_summary(report, results_dir)

    if int(report.get("n_pairs") or 0) == 0:
        print("No eval pairs ran. Prepare data/eval or pass --pair-id.", file=sys.stderr)
        return 1
    if int((report.get("summary") or {}).get("n_success") or 0) == 0:
        print("All pairs failed.", file=sys.stderr)
        return 1
    return 0


def main_evaluate(argv: list[str] | None = None) -> int:
    return main_run(argv)


def main_compare(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare pipeline metric reports and pick a winner")
    ap.add_argument(
        "--reports",
        nargs="+",
        default=None,
        help="metrics.json paths (default: results/*/metrics.json)",
    )
    ap.add_argument("--latency-budget", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.reports:
        paths = [Path(p) for p in args.reports]
    else:
        paths = sorted((project_root() / "results").glob("*/metrics.json"))
    if not paths:
        print("No metrics.json reports found. Run pipelines first.", file=sys.stderr)
        return 1
    compare_reports(
        paths,
        latency_budget_s=args.latency_budget,
        out_path=Path(args.out) if args.out else None,
    )
    return 0


def main_all_mock() -> None:
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


def entry_run() -> None:
    raise SystemExit(main_run())


def entry_compare() -> None:
    raise SystemExit(main_compare())


if __name__ == "__main__":
    raise SystemExit(main_run())
