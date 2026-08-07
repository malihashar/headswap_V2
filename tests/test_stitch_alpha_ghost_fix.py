"""Tests for the stitch alpha-tail cleanup (removes the translucent "ghost"
halo a wide, twice-blurred stitch mask leaves around the head/shoulders).

head_hair_mask_from_face already Gaussian-blurs the mask once at build time;
feathered_soft_composite (used by both crop_stitch's _stitch_edited and
full_frame's freeze compositing) blurs it again for feathering. Two stacked
blurs leave a long, very-low-alpha tail that shows as a faint double-exposure
of the generated crop over the original wherever the two layers aren't
perfectly aligned. clean_alpha_tails snaps that faint tail to 0/255 while
keeping the real transition band soft.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import clean_alpha_tails, feathered_soft_composite


def test_clean_alpha_tails_snaps_faint_values_to_zero():
    mask = Image.new("L", (8, 8), 5)  # below default floor=10
    out = np.asarray(clean_alpha_tails(mask))
    assert (out == 0).all()


def test_clean_alpha_tails_snaps_near_opaque_to_full():
    mask = Image.new("L", (8, 8), 250)  # above default ceil=245
    out = np.asarray(clean_alpha_tails(mask))
    assert (out == 255).all()


def test_clean_alpha_tails_preserves_binary_masks():
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[4:12, 4:12] = 255
    mask = Image.fromarray(arr)
    out = np.asarray(clean_alpha_tails(mask))
    assert np.array_equal(out, arr)


def test_clean_alpha_tails_is_monotonic_in_transition_band():
    ramp = np.linspace(0, 255, 256).astype(np.uint8).reshape(1, 256)
    mask = Image.fromarray(ramp)
    out = np.asarray(clean_alpha_tails(mask))[0]
    assert np.all(np.diff(out.astype(np.int32)) >= 0)
    # Ends fully clamp; middle should not be monotonically flat at 0/255.
    assert out[0] == 0
    assert out[-1] == 255
    assert 0 < out[128] < 255


def test_feathered_soft_composite_removes_faint_ghost_ring():
    """A wide Gaussian-blurred mask leaves a long low-alpha tail where a
    misaligned 'edit' layer bleeds through as a faint ghost. Anywhere the raw
    blurred alpha is below the cleanup floor, the cleaned composite must
    match the base exactly instead of leaking a faint blend."""
    import cv2

    w, h = 200, 200
    base = Image.new("RGB", (w, h), (30, 30, 30))
    edit = Image.new("RGB", (w, h), (220, 20, 20))
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(mask).ellipse([60, 60, 140, 140], fill=255)
    box = (0, 0, w, h)

    # Reproduce the internal blur to find a pixel with a faint (but nonzero)
    # raw tail -- i.e. exactly the "ghost" band this fix targets.
    extra_blur_px = 30
    k = extra_blur_px * 2 + 1
    raw_alpha = cv2.GaussianBlur(np.asarray(mask), (k, k), 0)
    faint = np.argwhere((raw_alpha > 0) & (raw_alpha < 10))
    assert faint.size > 0, "test setup: expected a faint (0<alpha<10) tail band to exist"
    y, x = (int(v) for v in faint[0])

    ghosty = feathered_soft_composite(base, edit, mask, box, extra_blur_px=extra_blur_px, alpha_floor=0, alpha_ceil=255)
    cleaned = feathered_soft_composite(base, edit, mask, box, extra_blur_px=extra_blur_px)

    ghosty_a = np.asarray(ghosty, dtype=np.float64)
    cleaned_a = np.asarray(cleaned, dtype=np.float64)
    base_a = np.asarray(base, dtype=np.float64)

    assert ghosty_a[y, x][0] > base_a[y, x][0], "test setup: uncleaned faint band should show a ghost tint"
    assert cleaned_a[y, x][0] == base_a[y, x][0], "cleaned composite must not leak a ghost tail in the faint band"

    # The mask's actual interior should still be fully edited (unaffected).
    center = (100, 100)
    assert cleaned_a[center][0] > 200
