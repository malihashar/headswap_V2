from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.gpu_stages import GpuStageProfiler, describe_latent


def test_describe_latent_dict():
    info = describe_latent({"samples": type("T", (), {"shape": (1, 16, 72, 57), "dtype": "float32", "device": "cuda:0"})()})
    assert info["latent_h"] == 72
    assert info["latent_w"] == 57


def test_profiler_stage_and_timings():
    p = GpuStageProfiler()
    with p.stage("preprocessing"):
        pass
    with p.stage("sampling_total"):
        pass
    d = p.timings_dict()
    assert "preprocessing" in d
    assert "sampling_total" in d
    assert p.to_dict()["colab_reference"]["stages_s"]["total_pipeline"] == 27.9
