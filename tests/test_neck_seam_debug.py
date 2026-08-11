"""Neck seam RCA debug exports."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from headswap.preprocess import FaceBox, dump_neck_seam_debug


def test_neck_seam_debug_crop_dimensions():
    w, h = 400, 600
    img = Image.new("RGB", (w, h), (200, 180, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([140, 80, 260, 220], fill=(120, 90, 70))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([130, 70, 270, 280], fill=255)
    body = img.copy()
    overlay, neck_crop, info = dump_neck_seam_debug(
        img,
        mask,
        body,
        cache_dir=None,
        prefer_box=FaceBox(140, 80, 260, 220, 0.99),
    )
    assert overlay.size == (w, h)
    assert neck_crop.width > 0 and neck_crop.height > 0
    box = info.get("neck_crop_box")
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h
    assert y1 - y0 >= 20
