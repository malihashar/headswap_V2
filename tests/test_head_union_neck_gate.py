"""The original-head union must be capped at neck height -- and ONLY it.

The geometric head+hair ellipse (_orig_head) is deliberately over-sized so
the previous person's hair is fully covered. That same over-reach drags it
down across the shoulders, pulling ORIGINAL pixels -- the old hairstyle's
loose strands and pigtail tails -- back into the keep mask, which is the
hair ghosting observed on the shoulders.

Commit 356d344 capped the WHOLE keep mask at neck height and was reverted in
a398bd4. Applied there the gate is harmful: by that point the keep mask also
carries the body-skin mask, so a vertical cut at the neck zeroes every limb
and silently undoes skin-tone correction on arms and legs. The gate must
therefore apply to the ellipse alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _gate(mask: np.ndarray, chin_y: int, face_h: int, frac: float = 0.65) -> np.ndarray:
    """Mirror of the production gate, for behavioural assertions."""
    neck_y = int(chin_y + frac * face_h)
    out = mask.copy()
    if 0 < neck_y < mask.shape[0]:
        g = np.zeros_like(mask, dtype=np.float32)
        g[:neck_y] = 1.0
        fade = int(0.20 * face_h)
        y_end = min(mask.shape[0], neck_y + max(0, fade))
        if y_end > neck_y:
            g[neck_y:y_end] = np.linspace(
                1.0, 0.0, y_end - neck_y, dtype=np.float32
            )[:, None]
        out = out * g
    return out


def test_gate_removes_ellipse_reach_over_the_shoulders():
    ellipse = np.ones((400, 200), dtype=np.float32)  # over-reaching ellipse
    gated = _gate(ellipse, chin_y=100, face_h=80)
    # Well below the neck line (100 + 0.65*80 = 152, fade ends at 168).
    assert gated[300].max() == 0.0, "ellipse still covers the chest"
    assert gated[:100].min() == 1.0, "gate ate into the head itself"


def test_gate_leaves_a_ramp_not_a_hard_edge():
    ellipse = np.ones((400, 200), dtype=np.float32)
    gated = _gate(ellipse, chin_y=100, face_h=80)
    col = gated[:, 100]
    band = col[152:168]
    assert band.max() > 0.0 and band.min() < 1.0, "no ramp — hard shoulder edge"
    assert np.all(np.diff(band) <= 1e-6), "ramp is not monotonically decreasing"


def test_gate_is_applied_to_the_ellipse_not_the_skin_keep_mask():
    """Regression guard on the 356d344 failure mode."""
    i_gate = SRC.find("simple_full_body_head_union_neck_frac")
    assert i_gate > 0, "neck gate on the original-head union is missing"
    i_union = SRC.find("_head = np.maximum(_head, _orig_head)")
    assert i_union > i_gate, (
        "the neck gate must run BEFORE the union into the keep mask, so it "
        "only trims the ellipse"
    )
    # The gate must operate on _orig_head, never on the skin-carrying _head.
    seg = SRC[i_gate - 2000 : i_union]
    assert "_orig_head = _orig_head * _gate" in seg, (
        "gate is not applied to _orig_head -- if it multiplies _head instead, "
        "arms and legs are zeroed and skin-tone correction is silently undone"
    )
    assert "_head = _head * _gate" not in seg, (
        "the gate is multiplying the full keep mask (the reverted 356d344 "
        "behaviour); this zeroes body skin below the neck"
    )
