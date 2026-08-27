"""Two rules collided, and the losing one produced visible artifacts.

1. Hair over clothing. The restore keep-mask means "use the GENERATED pixel"
   (krea2 composites `gen*hm + orig*(1-hm)`). Clothing protection subtracts
   the garment so clothes always come from the ORIGINAL. Where the previous
   person's hair overhangs their own shoulders the segmenter calls that band
   CLOTHES, so the subtraction pulls it out of the keep mask, the original is
   restored -- and the original still has the strands in it. Because the
   subtraction runs after the head union, widening the union cannot fix it.

2. Missed limb. The multiclass segmenter must label each limb as body-skin
   and does not do so reliably (krea2's restore already works around this).
   When it misses a limb the skin gate contributes nothing there, only the
   graded colour score applies, and a shaded leg is left visibly
   half-corrected beside its fully-corrected pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
HARM = (ROOT / "src" / "headswap" / "skin_harmonize.py").read_text()


def test_clothing_protection_is_lifted_on_original_hair():
    assert "_orig_hair_sem" in KREA2, "original hair class is not captured"
    i_lift = KREA2.find("_cloth = _cloth * (1.0 - np.clip(_hair_lift")
    assert i_lift > 0, "clothing protection is not lifted where hair overlapped"
    i_apply = KREA2.find("_hm = _hm * (1.0 - np.clip(_cloth, 0.0, 1.0))")
    assert 0 < i_lift < i_apply, (
        "the lift must happen BEFORE the clothing mask is subtracted from the "
        "keep mask, or it has no effect"
    )


def test_hair_lift_dilation_is_bounded():
    """An unbounded grow reopens collar protection and the donor collar returns."""
    assert "hair_lift_dilate_face_frac" in KREA2, "hair lift dilation is not configurable"
    assert "0.02 * max(out.size[0], out.size[1])" in KREA2, (
        "the hair-lift dilation has no frame-relative cap; on a large face it "
        "would punch a wide hole in clothing protection"
    )


def test_lift_uses_the_hair_class_not_the_geometric_ellipse():
    """The ellipse would punch a hole in collar protection."""
    i_sem = KREA2.find("_orig_hair_sem = None if _orig_head is None")
    i_ellipse = KREA2.find("_geo_head, _ = build_head_hair_mask(")
    assert 0 < i_sem < i_ellipse, (
        "_orig_hair_sem must be captured BEFORE the geometric ellipse is "
        "unioned into _orig_head, or the lift uses the ellipse and the "
        "donor's collar leaks back in"
    )


def test_skin_gate_has_a_person_minus_clothes_floor():
    assert "person_minus_clothes_mask(result_np, result_pil)" in HARM, (
        "the skin gate has no rembg floor; a limb the class segmenter misses "
        "gets only the colour score and stays half-corrected"
    )


def test_skin_floor_excludes_the_head_so_hair_is_not_recoloured():
    i_pmc = HARM.find("person_minus_clothes_mask(result_np, result_pil)")
    i_head = HARM.find("_pm_head = semantic_head_mask(result_np)", i_pmc)
    i_union = HARM.find("_sem = np.maximum(_sem, _pmc)", i_pmc)
    assert 0 < i_head < i_union, (
        "the head must be subtracted from the rembg floor BEFORE the union, "
        "or overhanging hair is treated as skin and recoloured"
    )


def test_no_neck_gate_shrinks_the_generated_head_coverage():
    """Regression guard: the keep mask means KEEP GENERATED.

    A vertical cap on the original-head union shrinks generated coverage
    around the head, which can only preserve the previous person's hair --
    the opposite of the intent. Measured 60,614 -> 59,824px (1.3%) with no
    visible change.
    """
    assert "simple_full_body_head_union_neck_frac" not in KREA2, (
        "the neck gate is back; it shrinks GENERATED coverage and cannot "
        "remove old hair"
    )
