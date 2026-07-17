from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.vae_bridge_probe import print_vae_probe


def test_print_vae_probe_emits_prefix(capsys):
    print_vae_probe("unit_test_label", bundle=None)
    out = capsys.readouterr().out
    assert "[vae_probe] === unit_test_label ===" in out
    assert "allocated_mb=" in out
