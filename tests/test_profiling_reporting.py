from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Minimal torch stub so qwen.run can preprocess without a GPU install in CI.
if "torch" not in sys.modules:
    _torch = MagicMock(name="torch")
    _torch.from_numpy.side_effect = lambda arr: arr
    sys.modules["torch"] = _torch

from headswap.config import load_config
from headswap.eval.runner import run_eval
from headswap.pipelines.errors import PipelineRunError
from headswap.pipelines.qwen import QwenBaselinePipeline
from headswap.profiling.gpu_stages import GpuStageProfiler
from headswap.profiling.reporting import emit_profile_report


def _fake_qwen_patches(sample_side_effect):
    fake_rt = object()
    fake_bundle = {
        "model": object(),
        "clip": object(),
        "vae": object(),
        "load_meta": {"checkpoint": "test.safetensors", "loras_loaded": [], "lora_strengths": {}, "fallbacks": []},
    }
    return (
        patch.object(QwenBaselinePipeline, "_ensure_runtime", return_value=fake_rt),
        patch("headswap.pipelines.qwen._load_qwen_stack", return_value=fake_bundle),
        patch("headswap.pipelines.qwen._sample_qwen", side_effect=sample_side_effect),
    )


def test_emit_profile_report_flushes_and_survives_print_error(capsys):
    profiler = GpuStageProfiler()
    with profiler.stage("preprocessing"):
        pass

    real_print = GpuStageProfiler.print_report

    def boom(self, *, total_s, label="qwen_baseline"):
        raise RuntimeError("print broke")

    GpuStageProfiler.print_report = boom
    try:
        emit_profile_report(profiler, total_s=1.0, label="test_pipe", error="partial")
    finally:
        GpuStageProfiler.print_report = real_print

    err = capsys.readouterr()
    assert "pipeline error" in err.err.lower()
    assert "profile print failed" in err.err or "stages" in err.out


def test_qwen_baseline_emits_profile_on_sample_failure(capsys):
    cfg = load_config(ROOT / "configs" / "qwen_baseline.yaml")
    pipe = QwenBaselinePipeline(cfg)
    body = face = Image.new("RGB", (64, 64), color=(128, 64, 32))

    p1, p2, p3 = _fake_qwen_patches(RuntimeError("boom"))
    with p1, p2, p3:
        with pytest.raises(PipelineRunError, match="boom"):
            pipe.run(body, face)

    out = capsys.readouterr().out
    assert "[qwen_baseline profile]" in out
    assert "preprocessing" in out


def test_run_eval_persists_profile_when_pipeline_fails(tmp_path, capsys):
    cfg = load_config(ROOT / "configs" / "qwen_baseline.yaml")
    pipe = QwenBaselinePipeline(cfg)

    data_dir = tmp_path / "data" / "eval"
    data_dir.mkdir(parents=True)
    body_path = data_dir / "body.png"
    face_path = data_dir / "face.png"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(body_path)
    Image.new("RGB", (64, 64), color=(120, 80, 60)).save(face_path)
    manifest = [
        {
            "id": "test_pair",
            "body_path": str(body_path),
            "face_path": str(face_path),
            "difficulty": "easy",
            "tags": ["test"],
        }
    ]
    (data_dir / "pairs.json").write_text(json.dumps(manifest))

    p1, p2, p3 = _fake_qwen_patches(RuntimeError("gpu died"))
    with patch("headswap.eval.runner.load_pairs", return_value=manifest):
        with patch("headswap.eval.runner.create_pipeline", return_value=pipe):
            with p1, p2, p3:
                report = run_eval(
                    ROOT / "configs" / "qwen_baseline.yaml",
                    out_dir=tmp_path / "results",
                    force_mock=False,
                    limit=1,
                )

    assert report["n_pairs"] == 1
    row = report["pairs"][0]
    assert row["success"] is False
    assert "profile" in row["meta"]
    assert row["meta"]["timing_s"]
    assert (tmp_path / "results" / "metrics.json").is_file()
    assert "[qwen_baseline profile]" in capsys.readouterr().out
