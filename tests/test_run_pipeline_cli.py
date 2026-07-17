from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.cli import main_run
from headswap.eval.dataset import generate_synthetic_eval_set


def test_run_pipeline_cli_mock_writes_metrics(tmp_path: Path):
    generate_synthetic_eval_set(n_pairs=2)
    out = tmp_path / "qwen_baseline_run"
    code = main_run(
        [
            "--config",
            str(ROOT / "configs" / "qwen_baseline.yaml"),
            "--mock",
            "--limit",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    metrics_path = out / "metrics.json"
    assert metrics_path.is_file()
    report = json.loads(metrics_path.read_text())
    assert report["n_pairs"] == 1
    assert report["force_mock"] is True
    pair = report["pairs"][0]
    assert Path(pair["result_path"]).is_file()
    assert pair["success"] in (True, False)


def test_run_pipeline_cli_missing_config_exits_2():
    code = main_run(["--config", "/no/such/config.yaml", "--mock"])
    assert code == 2
