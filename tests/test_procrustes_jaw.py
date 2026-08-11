"""Jaw-aware Procrustes alignment."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image

from headswap.preprocess import (
    FaceBox,
    _jaw_points_from_landmarks5,
    get_jaw_alignment_points,
    procrustes_align_edited_crop_to_body_box,
)


def test_jaw_points_from_landmarks5_has_chin_below_mouth():
    lm = np.array(
        [
            [50.0, 40.0],
            [90.0, 40.0],
            [70.0, 55.0],
            [60.0, 70.0],
            [80.0, 70.0],
        ],
        dtype=np.float32,
    )
    pts = _jaw_points_from_landmarks5(lm)
    assert pts.shape[0] >= 6
    chin_y = float(pts[3, 1])
    mouth_y = 70.0
    assert chin_y > mouth_y


def test_procrustes_jaw_aligns_chin_to_body():
    w, h = 200, 200
    body = Image.new("RGB", (w, h), (210, 180, 160))
    edited = body.copy()
    box = (40, 30, 160, 170)
    body_box = FaceBox(70, 50, 130, 120, 0.99)

    dst_lm = _jaw_points_from_landmarks5(
        np.array(
            [[80, 60], [120, 60], [100, 75], [90, 90], [110, 90]], dtype=np.float32
        )
    )
    src_lm = dst_lm.copy()
    src_lm[:, 1] += 12.0  # generated chin too low -> double-neck symptom

    call_n = [0]

    def fake_jaw(rgb, cache_dir, prefer_box=None):
        call_n[0] += 1
        if call_n[0] == 1:
            return src_lm.copy(), "test", None
        return dst_lm.copy(), "test", None

    with patch("headswap.preprocess.get_jaw_alignment_points", side_effect=fake_jaw):
        out, info = procrustes_align_edited_crop_to_body_box(
            edited,
            body,
            box,
            cache_dir=None,
            prefer_body_box=body_box,
            alignment="jaw",
            min_inliers=4,
            max_translate_frac=0.5,
        )
    assert info.get("procrustes") is True
    assert info.get("chin_shift_px") is not None
    assert float(info["chin_shift_px"]) >= 0.0


def test_get_jaw_alignment_points_fallback_from_box_prior():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FaceBox(30, 20, 70, 60, 0.9)
    pts, backend, _ = get_jaw_alignment_points(rgb, None, prefer_box=box)
    assert pts is not None
    assert pts.shape[0] >= 6
    assert backend in ("box_prior", "insightface_jaw5", "insightface", "none")
