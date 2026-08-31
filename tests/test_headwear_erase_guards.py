"""Masks are allowed now, but they must not be able to damage the image.

Ali lifted the no-mask constraint on condition that mask artifacts are
guarded against -- an earlier erase "worked but the mask added problems".
The failure mode is specific: headwear_mask() can reach outside the head,
and LaMa then inpaints whatever it covered, inventing background or torso.

Three guards, each aimed at a distinct way that goes wrong.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = (ROOT / "src" / "headswap" / "chain.py").read_text()


def test_guard1_mask_is_clamped_to_a_head_box():
    """Bounded above and beside the face, so it cannot reach into scenery."""
    assert "headwear_up_face_heights" in CHAIN
    assert "headwear_side_face_widths" in CHAIN
    assert "_keep[_ty0:_ty1, _tx0:_tx1] = 1" in CHAIN


def test_guard1_hard_floor_at_the_chin():
    """Nothing below the jaw is headwear, and the torso is where a stray
    erase would be most visible."""
    assert "_ty1 = min(_H, int(_fb.y1))" in CHAIN
    assert "chin: hard floor" in CHAIN


def test_guard1_reports_what_it_clipped():
    """A silent clamp would hide a detector that is reaching far too wide."""
    assert "headwear mask clamped to the head box" in CHAIN


def test_guard2_refuses_an_implausible_mask():
    """A runaway mask is worse than a surviving hat: LaMa invents whatever it
    covers, so past a ceiling this is a detection failure, not a big hat."""
    assert "headwear_coverage_max" in CHAIN
    assert "refusing to inpaint" in CHAIN
    assert "headwear erase REFUSED" in CHAIN


def test_guard2_still_rejects_an_empty_mask():
    assert "no headwear detected" in CHAIN


def test_guard3_dumps_the_mask_for_inspection():
    """When an erase looks wrong, the plate alone cannot say whether the mask
    or the inpaint was at fault."""
    assert 'out / "body_headwear_mask.png"' in CHAIN


def test_erase_is_enabled_and_failure_is_non_fatal():
    assert '"erase_headwear": True' in CHAIN
    assert "returning the swap unmodified" in CHAIN
