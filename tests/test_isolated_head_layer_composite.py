"""composite_isolated_head_layer: prepare one donor-content RGBA layer
(color + scale corrected on its own canvas), then composite onto the
untouched original exactly once -- replacing the modify/restore/rewarp/
re-extract cycle (stitch -> harmonize -> restore_background -> clamp) that
produced ghosts, dark rectangles, and identity leaks across earlier fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.preprocess as preprocess_mod
from headswap.preprocess import FaceBox, composite_isolated_head_layer

GEN_COLOR = (220, 180, 150)
TGT_COLOR = (150, 180, 220)


def _bbox_of_color(rgb: np.ndarray, color: tuple[int, int, int]) -> FaceBox | None:
    mask = np.all(np.abs(rgb.astype(int) - np.array(color)) <= 4, axis=-1)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return FaceBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), 0.9)


def _fake_detect_best_face(rgb, cache_dir, conf_thresh=0.30):
    return _bbox_of_color(rgb, GEN_COLOR) or _bbox_of_color(rgb, TGT_COLOR)


def test_background_pixels_are_never_touched(monkeypatch):
    """The core structural guarantee: composite_isolated_head_layer never
    warps, restores, or otherwise modifies body_full -- pixels far from the
    head must be pixel-identical to the original."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), (10, 20, 30))
    ImageDraw.Draw(body_full).ellipse([330, 150, 470, 350], fill=TGT_COLOR)  # height 200

    edited_crop = Image.new("RGB", (300, 400), (200, 200, 200))
    ImageDraw.Draw(edited_crop).ellipse([50, 40, 250, 300], fill=GEN_COLOR)  # height 260 -> oversized

    box = (250, 100, 550, 500)
    hem_box = (100, h - 120, 700, h - 40)
    hem_before = np.asarray(body_full.crop(hem_box))

    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([280, 130, 520, 470], fill=255)

    out, info, final_mask, layer = composite_isolated_head_layer(
        body_full, edited_crop, mask, box, None, ROOT / "results" / "_cache"
    )

    hem_after = np.asarray(out.crop(hem_box))
    assert np.array_equal(hem_before, hem_after), (
        "isolated-layer composite touched pixels far from the head -- "
        "background must never be modified under this architecture"
    )


def test_mask_has_opaque_core_not_uniform_partial_alpha(monkeypatch):
    """Guards the specific 'translucent ghost' failure mode: the interior
    of the mask must hit true 255, not a uniformly-blurred partial value."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), (10, 20, 30))
    ImageDraw.Draw(body_full).ellipse([330, 150, 470, 350], fill=TGT_COLOR)

    edited_crop = Image.new("RGB", (300, 400), (200, 200, 200))
    ImageDraw.Draw(edited_crop).ellipse([80, 60, 220, 260], fill=GEN_COLOR)  # height 200, near target ratio

    box = (250, 100, 550, 500)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([280, 130, 520, 470], fill=255)

    out, info, final_mask, layer = composite_isolated_head_layer(
        body_full, edited_crop, mask, box, None, ROOT / "results" / "_cache",
        feather_px=8,
    )

    m = np.asarray(final_mask)
    # Deep interior (well inside the ellipse) must be fully opaque.
    core_sample = m[250:350, 350:450]
    assert core_sample.min() >= 250, (
        f"mask interior not fully opaque (min={core_sample.min()}) -- "
        "this is the translucent-ghost failure mode"
    )


def test_layer_background_is_transparent(monkeypatch):
    """The isolated layer's alpha must be 0 far from the head -- verifies
    no real photo content leaks into the layer outside the donor region."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), (10, 20, 30))
    ImageDraw.Draw(body_full).ellipse([330, 150, 470, 350], fill=TGT_COLOR)

    edited_crop = Image.new("RGB", (300, 400), (200, 200, 200))
    ImageDraw.Draw(edited_crop).ellipse([80, 60, 220, 260], fill=GEN_COLOR)

    box = (250, 100, 550, 500)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([280, 130, 520, 470], fill=255)

    out, info, final_mask, layer = composite_isolated_head_layer(
        body_full, edited_crop, mask, box, None, ROOT / "results" / "_cache",
        neck_ring_px=20,
    )

    m = np.asarray(final_mask)
    far_corner = m[0:50, 0:50]
    assert far_corner.max() < 5, "layer/mask has content far from the head region"
