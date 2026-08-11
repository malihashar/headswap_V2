"""Production defaults must not re-enable the rect-box / floating-scalp path.

User regression (2026-08-10): single-person full-body desert/sky swaps showed
(1) a rectangular head patch with mismatched sky inside, (2) a floating ghost
scalp at the top of the frame, (3) a pale double-neck V. Root cause was
e42caad's always-on ``crop_stitch_clamp_head_scale`` (local-box shrink paste)
plus the widened ``head_matte_stitch_feather_px``. Skin-tone flags were a red
herring — those ghosts appear with exposed_skin_tone_match off.

This locks the production yaml to the old soft-stitch defaults.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "krea2_identity_edit.yaml"


def test_production_yaml_disables_crop_stitch_head_clamp():
    cfg = yaml.safe_load(CFG.read_text())
    assert cfg.get("crop_stitch_clamp_head_scale") is False
    assert cfg.get("head_matte_stitch_feather_px", 3) <= 3
    assert cfg.get("exposed_skin_tone_match", False) is False
    assert cfg.get("mask_crop_stitch") is True
    assert cfg.get("neck_stub_tone_match") is True
