from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.config import load_config


def test_qwen_baseline_disables_flux_kontext_image_scale():
    cfg = load_config(ROOT / "configs" / "qwen_baseline.yaml")
    assert cfg.get("flux_kontext_image_scale") is False
    assert cfg.get("max_dim") == 576
    assert cfg.get("steps") == 6


def test_qwen_improved_keeps_flux_kontext_image_scale():
    cfg = load_config(ROOT / "configs" / "qwen_improved.yaml")
    assert cfg.get("flux_kontext_image_scale") is True
