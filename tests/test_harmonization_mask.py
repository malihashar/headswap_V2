"""Harmonization mask: gen ⊂ full_harm, ghost-free gen interior lock."""

from __future__ import annotations

import numpy as np
from PIL import Image

from headswap.preprocess import (
    assert_no_gen_interior_leak,
    build_harmonization_mask,
    harmonize_skin_tone,
)


def _synthetic_plate(
    *,
    skin_neck: bool,
    skin_arms: bool,
    size: tuple[int, int] = (400, 600),
) -> tuple[Image.Image, Image.Image]:
    """Body plate + tight generation mask (head oval)."""
    w, h = size
    body = np.zeros((h, w, 3), dtype=np.uint8)
    body[:, :] = (180, 40, 40)
    gen_arr = np.zeros((h, w), dtype=np.uint8)
    gx0, gy0, gx1, gy1 = w // 2 - 40, 80, w // 2 + 40, 200
    gen_arr[gy0:gy1, gx0:gx1] = 255
    gen = Image.fromarray(gen_arr)
    if skin_neck:
        body[gy1 : gy1 + 35, gx0 + 10 : gx1 - 10] = (210, 170, 140)
    if skin_arms:
        body[gy1 + 20 : gy1 + 180, gx0 - 70 : gx0 - 10] = (205, 165, 135)
        body[gy1 + 20 : gy1 + 180, gx1 + 10 : gx1 + 70] = (205, 165, 135)
    return Image.fromarray(body), gen


def test_full_harm_contains_gen_mask():
    body, gen = _synthetic_plate(skin_neck=True, skin_arms=False)
    full_harm, harm_ring, info = build_harmonization_mask(
        body, gen, dilate_px=24, skin_thresh=0.35
    )
    assert info["gen_subset_of_harm"] is True
    gen_arr = np.asarray(gen.convert("L")) > 128
    full_arr = np.asarray(full_harm.convert("L")) > 128
    ring_arr = np.asarray(harm_ring.convert("L")) > 128
    assert np.all(gen_arr <= full_arr)
    assert not np.any(gen_arr & ring_arr)


def test_harm_ring_larger_on_bare_arms_than_collar_only():
    dress_body, dress_gen = _synthetic_plate(skin_neck=True, skin_arms=False)
    runner_body, runner_gen = _synthetic_plate(skin_neck=True, skin_arms=True)
    _, dress_ring, dress_info = build_harmonization_mask(
        dress_body, dress_gen, dilate_px=24, skin_thresh=0.35
    )
    _, runner_ring, runner_info = build_harmonization_mask(
        runner_body, runner_gen, dilate_px=24, skin_thresh=0.35
    )
    dress_area = int(dress_info["harm_ring_area_px"])
    runner_area = int(runner_info["harm_ring_area_px"])
    assert runner_area > dress_area
    assert dress_area > 0


def test_harmonize_zero_diff_inside_gen_core():
    w, h = 200, 300
    body = Image.new("RGB", (w, h), (200, 180, 160))
    gen_arr = np.zeros((h, w), dtype=np.uint8)
    gen_arr[50:120, 70:130] = 255
    gen = Image.fromarray(gen_arr)
    body_arr = np.asarray(body).copy()
    body_arr[130:180, 75:125] = (210, 170, 140)
    body = Image.fromarray(body_arr)
    full_harm, harm_ring, _ = build_harmonization_mask(
        body, gen, dilate_px=20, skin_thresh=0.25
    )
    pre_arr = np.asarray(body).copy()
    pre_arr[50:120, 70:130] = (80, 50, 40)
    pre_harm = Image.fromarray(pre_arr.astype(np.uint8))
    out, info = harmonize_skin_tone(
        pre_harm,
        body,
        gen,
        full_harm,
        harm_ring,
        strength=0.8,
        feather_px=8,
    )
    assert info["harmonization_applied"] is True
    leak = assert_no_gen_interior_leak(pre_harm, out, gen, epsilon=1.0)
    assert leak["gen_interior_ok"] is True
    assert leak["gen_interior_max_diff"] <= 1.0


def test_harmonize_does_not_change_pixels_far_from_harm():
    w, h = 200, 300
    body = Image.new("RGB", (w, h), (200, 180, 160))
    gen_arr = np.zeros((h, w), dtype=np.uint8)
    gen_arr[50:120, 70:130] = 255
    gen = Image.fromarray(gen_arr)
    body_arr = np.asarray(body).copy()
    body_arr[130:160, 80:120] = (210, 170, 140)
    body = Image.fromarray(body_arr)
    full_harm, harm_ring, _ = build_harmonization_mask(body, gen, skin_thresh=0.25)
    pre_arr = np.asarray(body).copy()
    pre_arr[50:120, 70:130] = (80, 50, 40)
    pre_harm = Image.fromarray(pre_arr.astype(np.uint8))
    before_far = pre_harm.getpixel((10, 10))
    out, _ = harmonize_skin_tone(
        pre_harm, body, gen, full_harm, harm_ring, strength=0.8, feather_px=8
    )
    assert out.getpixel((10, 10)) == before_far
