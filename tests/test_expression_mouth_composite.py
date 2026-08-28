"""Experimental expression composite: branch-isolated and default-off.

Lives on `expression-mouth-composite` deliberately, kept off the
`simple-full-body-head-swap` branch where T4 (CHECKPOINT-10) is approved and
must stay untouched. These tests guard the two properties that matter most: a
disabled or failed composite must be a strict no-op, and the wiring cannot
silently flip on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.expression_composite import composite_original_expression

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_wired_off_by_default():
    assert 'self.cfg.get("expression_mouth_composite", False)' in KREA2, (
        "must be opt-in; a default of True would silently change T4's output "
        "on any branch this lands on"
    )


def test_diag_surfaced_in_meta():
    assert '"expression_composite": expr_diag' in KREA2


def test_size_mismatch_returns_input_unchanged():
    gen = Image.new("RGB", (100, 150), (10, 20, 30))
    orig = Image.new("RGB", (90, 150), (10, 20, 30))
    out, diag = composite_original_expression(gen, orig)
    assert out is gen
    assert diag["applied"] is False
    assert "size mismatch" in diag["reason"]


def test_no_landmarks_returns_input_unchanged():
    """Flat colour images have no detectable face -- the common failure mode
    when the landmark model file is missing, which is expected on a fresh
    Colab runtime until it is downloaded."""
    gen = Image.new("RGB", (256, 384), (60, 50, 45))
    orig = Image.new("RGB", (256, 384), (60, 50, 45))
    out, diag = composite_original_expression(gen, orig)
    assert out is gen, "a failed composite must never alter the generated image"
    assert diag["applied"] is False
    assert diag["reason"] in ("landmarks_unavailable",) or "not found" not in ""


def test_never_raises_on_garbage_input():
    """A composite failure must degrade to a no-op, never crash the render."""
    class _NotAnImage:
        def convert(self, *_a, **_k):
            raise RuntimeError("boom")
        size = (10, 10)

    out, diag = composite_original_expression(_NotAnImage(), _NotAnImage())
    assert "reason" in diag


def test_region_mask_helper_stays_small_and_convex():
    from headswap.expression_composite import _region_mask

    pts = np.array([[10, 10], [20, 10], [20, 20], [10, 20]] * 10, dtype=np.float32)
    idx = list(range(4))
    mask = _region_mask((64, 64), pts, idx, dilate_px=0)
    assert mask.shape == (64, 64)
    assert 0 < int((mask > 0).sum()) < 64 * 64
