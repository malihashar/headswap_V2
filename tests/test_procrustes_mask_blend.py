"""Procrustes blend-back uses stitch mask."""

from __future__ import annotations

import numpy as np
from PIL import Image

import headswap.pipelines.krea2 as k2
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline


def test_procrustes_blend_uses_stitch_mask_not_ellipse():
    pipe = Krea2IdentityEditPipeline({"procrustes_mask_blend_feather_px": 4})
    pipe.cache_dir = None
    edited = Image.new("RGB", (100, 100), (200, 0, 0))
    aligned = Image.new("RGB", (100, 100), (0, 200, 0))
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 20:80] = 255
    mask = Image.fromarray(arr)

    calls: list[tuple] = []
    original = k2.feathered_soft_composite

    def fake_composite(base, top, m, box, **kwargs):
        calls.append((base.size, top.size, m.size, box))
        return top

    k2.feathered_soft_composite = fake_composite
    try:
        _out, info = pipe._blend_procrustes_through_mask(edited, aligned, mask)
    finally:
        k2.feathered_soft_composite = original

    assert info.get("mask_blend") is True
    assert info.get("method") == "stitch_mask"
    assert len(calls) == 1
    assert calls[0][2] == (100, 100)
