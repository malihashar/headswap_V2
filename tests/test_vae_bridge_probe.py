from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.vae_bridge_probe import (
    cuda_snapshot,
    patcher_weight_devices,
    print_bridge_checkpoint,
)


def test_cuda_snapshot_safe_without_gpu():
    snap = cuda_snapshot()
    assert "cuda_available" in snap
    assert "allocated_mb" in snap


def test_patcher_weight_devices_none():
    info = patcher_weight_devices(None)
    assert info["patcher_type"] is None
    assert info["n_cuda_params"] == 0


def test_print_bridge_checkpoint_without_comfy(capsys):
    payload = print_bridge_checkpoint("unit_test", bundle=None)
    assert payload["label"] == "unit_test"
    out = capsys.readouterr().out
    assert "[vae_bridge] CHECKPOINT: unit_test" in out
