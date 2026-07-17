from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.vae_bridge_probe import print_vae_probe


def test_print_vae_probe_emits_prefix_before_cuda(capsys):
    print_vae_probe("unit_test_label", bundle=None, extra={"k": 1})
    out = capsys.readouterr().out
    assert "[vae_probe] === unit_test_label ===" in out
    # Label must appear even when CUDA/comfy are unavailable.
    assert out.index("[vae_probe] === unit_test_label ===") < out.index("extra=")


def test_print_vae_probe_survives_cuda_stats_failure(capsys, monkeypatch):
    import headswap.profiling.vae_bridge_probe as probe

    def boom():
        raise RuntimeError("simulated deferred CUDA OOM")

    monkeypatch.setattr(probe, "_cuda_stats", boom)
    print_vae_probe("after_boom")
    out = capsys.readouterr().out
    assert "[vae_probe] === after_boom ===" in out
    assert "cuda_stats RAISED" in out
