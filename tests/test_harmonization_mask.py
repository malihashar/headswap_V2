"""Harmonization mask adapts to exposed skin (dress vs runner)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from headswap.preprocess import build_harmonization_mask, harmonize_skin_tone


def _synthetic_plate(
    *,
    skin_neck: bool,
    skin_arms: bool,
    size: tuple[int, int] = (400, 600),
) -> tuple[Image.Image, Image.Image]:
    """Body plate + tight generation mask (head oval)."""
    w, h = size
    body = np.zeros((h, w, 3), dtype=np.uint8)
    body[:, :] = (180, 40, 40)  # red dress / jersey
    gen = Image.new("L", (w, h), 0)
    gx0, gy0, gx1, gy1 = w // 2 - 40, 80, w // 2 + 40, 200
    gen_arr = np.zeros((h, w), dtype=np.uint8)
    gen_arr[gy0:gy1, gx0:gx1] = 255
    gen = Image.fromarray(gen_arr)
    # neck skin strip below head
    if skin_neck:
        body[gy1 : gy1 + 35, gx0 + 10 : gx1 - 10] = (210, 170, 140)
    # bare arms lateral
    if skin_arms:
        body[gy1 + 20 : gy1 + 180, gx0 - 70 : gx0 - 10] = (205, 165, 135)
        body[gy1 + 20 : gy1 + 180, gx1 + 10 : gx1 + 70] = (205, 165, 135)
    return Image.fromarray(body), gen


def test_harmonization_mask_larger_on_bare_arms_than_collar_only():
    dress_body, dress_gen = _synthetic_plate(skin_neck=True, skin_arms=False)
    runner_body, runner_gen = _synthetic_plate(skin_neck=True, skin_arms=True)
    _, dress_info = build_harmonization_mask(
        dress_body,
        dress_gen,
        dilate_px=24,
        bot_extend_frac=0.15,
        skin_thresh=0.35,
        include_arms=True,
    )
    _, runner_info = build_harmonization_mask(
        runner_body,
        runner_gen,
        dilate_px=24,
        bot_extend_frac=0.15,
        skin_thresh=0.35,
        include_arms=True,
    )
    dress_area = int(dress_info["harm_area_px"])
    runner_area = int(runner_info["harm_area_px"])
    assert runner_area > dress_area
    assert dress_area > 0


def test_harmonize_skin_tone_is_color_only_geometry_unchanged_outside_harm():
    w, h = 200, 300
    body = Image.new("RGB", (w, h), (200, 180, 160))
    gen = Image.new("L", (w, h), 0)
    gen_arr = np.zeros((h, w), dtype=np.uint8)
    gen_arr[50:120, 70:130] = 255
    gen = Image.fromarray(gen_arr)
    harm = Image.new("L", (w, h), 0)
    harm_arr = np.zeros((h, w), dtype=np.uint8)
    harm_arr[120:160, 80:120] = 255
    harm = Image.fromarray(harm_arr)
    result = body.copy()
    result_arr = np.asarray(result)
    result_arr[50:120, 70:130] = (80, 50, 40)
    result = Image.fromarray(result_arr)
    before_far = result.getpixel((10, 10))
    out, info = harmonize_skin_tone(
        result,
        body,
        gen,
        harm,
        strength=0.8,
        feather_px=8,
    )
    assert info["harmonization_applied"] is True
    assert out.getpixel((10, 10)) == before_far
