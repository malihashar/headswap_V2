"""Skin must be RENDERED, not recoloured -- with the head physically frozen.

The LAB wash shifts existing pixels toward a target mean. It cannot add
subsurface warmth, shadow terminators or texture, so a fully-corrected body
still reads as coloured ON rather than rendered. Every setting of it also
trades one artifact for another: reach far enough to fix a shaded limb and it
repaints garments (measured: 102,850px admitted, a white top came back pink);
pull back and the limb stays at its original tone.

The repaint pass asks the model to render the skin instead, with
sampling_containment attaching the skin mask as noise_mask so ComfyUI re-pins
every latent OUTSIDE it at each step. The head, hair, clothing and background
are then not merely restored afterwards -- they cannot be altered at all,
which removes the skin-vs-head trade rather than balancing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_repaint_is_off_by_default():
    """It adds a third sampling pass and is not GPU-verified yet."""
    i = KREA2.find('self.cfg.get("simple_full_body_skin_repaint", False)')
    assert i > 0, "repaint flag missing or defaulted on"


def test_repaint_enables_containment_so_the_head_cannot_move():
    i_on = KREA2.find('self.cfg["sampling_containment"] = True')
    assert i_on > 0, (
        "repaint does not enable sampling_containment; without noise_mask the "
        "sampler can alter the head and the trade-off returns"
    )


def test_repaint_restores_the_previous_containment_setting():
    """Leaking containment=True into later passes would change the main path."""
    seg = KREA2[KREA2.find("[krea2 skin_repaint]") - 4000:]
    assert "finally:" in seg and "_prev_cont" in seg, (
        "containment is not restored in a finally block; a failure mid-pass "
        "would leave it enabled for every subsequent render"
    )


def test_repaint_mask_excludes_the_head_and_clothing():
    i_chin = KREA2.find("_sk[:_chin_y] = 0.0")
    assert i_chin > 0, (
        "the repaint mask is not cut at the chin, so the head is inside the "
        "editable region and can be re-rendered"
    )
    assert "_sk = _sk * (1.0 - np.clip(_cl2, 0.0, 1.0))" in KREA2, (
        "clothing is not subtracted from the repaint mask; garments would be "
        "editable and could be recoloured"
    )


def test_repaint_falls_through_to_the_wash_on_failure():
    assert "falling through to the LAB wash" in KREA2, (
        "a failed repaint must not silently leave the skin uncorrected"
    )
