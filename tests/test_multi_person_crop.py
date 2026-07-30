"""Unit tests for multi-person crop expansion helpers (no GPU)."""
from __future__ import annotations

from headswap.preprocess import (
    FaceBox,
    expand_crop_box_for_face_fill,
    suppress_neighbor_faces_in_mask,
)
from PIL import Image
import numpy as np


def test_expand_crop_reduces_face_fill_and_grows_long_side():
    face = FaceBox(100, 100, 180, 200, conf=0.9)  # 80x100
    # Tiny crop around face → face fill ~1.0
    box = (100, 100, 180, 200)
    new_box, info = expand_crop_box_for_face_fill(
        (640, 640),
        box,
        face,
        target_face_area_frac=0.16,
        min_long_side=448,
        other_faces=[],
        div_by=16,
    )
    assert info["expanded"] == 1.0
    assert info["face_fill_after"] <= info["face_fill_before"]
    assert info["crop_long_after"] >= 448 or new_box == (0, 0, 640, 640)
    assert new_box[2] - new_box[0] >= box[2] - box[0]
    assert new_box[3] - new_box[1] >= box[3] - box[1]


def test_expand_crop_blocks_side_toward_neighbor():
    face = FaceBox(300, 100, 380, 200, conf=0.9)
    neighbor = FaceBox(200, 100, 280, 200, conf=0.8)  # left of selected
    box = (300, 100, 380, 200)
    new_box, _ = expand_crop_box_for_face_fill(
        (640, 640),
        box,
        face,
        target_face_area_frac=0.16,
        min_long_side=448,
        other_faces=[neighbor],
        div_by=16,
    )
    # Should grow more to the right / vertical than left into the neighbor.
    left_grow = box[0] - new_box[0]
    right_grow = new_box[2] - box[2]
    assert right_grow >= left_grow


def test_suppress_neighbor_zeros_other_face():
    mask = Image.new("L", (200, 200), 255)
    selected = FaceBox(20, 20, 60, 70, conf=0.9)
    other = FaceBox(120, 20, 160, 70, conf=0.8)
    out = suppress_neighbor_faces_in_mask(mask, selected, [selected, other])
    arr = np.asarray(out)
    # Center of other face should be suppressed.
    assert arr[45, 140] < 40
    # Selected face region should remain mostly opaque.
    assert arr[45, 40] > 200
