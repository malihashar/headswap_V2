"""Unit tests for multi-person crop expansion helpers (no GPU)."""
from __future__ import annotations

from headswap.preprocess import (
    FaceBox,
    clamp_crop_away_neighbors,
    expand_crop_box_for_face_fill,
    suppress_neighbor_faces_in_mask,
)
from PIL import Image
import numpy as np


def test_clamp_crop_excludes_neighbor_center():
    selected = FaceBox(200, 40, 280, 140, 0.9)
    neighbor = FaceBox(40, 40, 120, 140, 0.9)
    # Wide crop that includes both faces.
    box = (0, 0, 320, 200)
    new_box, info = clamp_crop_away_neighbors(
        (320, 200),
        box,
        selected,
        [selected, neighbor],
        margin_frac=0.1,
    )
    assert info["neighbors_excluded"] >= 1.0
    ncx = 0.5 * (neighbor.x0 + neighbor.x1)
    assert not (new_box[0] <= ncx < new_box[2] and new_box[1] <= 90 < new_box[3])
    # Selected face must remain inside the crop.
    assert new_box[0] <= selected.x0 and new_box[2] >= selected.x1
    # Clamp must stop just past the neighbor (cx + margin), not over-shrink
    # all the way to the protected head boundary: context is precious.
    assert new_box[0] == int(0.5 * (neighbor.x0 + neighbor.x1)) + int(
        0.1 * neighbor.width
    )
    assert new_box[1:] == (0, 320, 200)


def test_clamp_protects_full_head_mask_extents():
    """A vertical clamp must never cut the hair region above the face."""
    selected = FaceBox(200, 300, 280, 400, 0.9)
    # Neighbor above, center inside hair protection band (1.55 x face height).
    neighbor_in_hair = FaceBox(210, 180, 270, 240, 0.9)
    box = (100, 100, 380, 480)
    new_box, info = clamp_crop_away_neighbors(
        (640, 640),
        box,
        selected,
        [selected, neighbor_in_hair],
        margin_frac=0.18,
        protect_top_frac=1.55,
        protect_side_frac=0.60,
        protect_bot_frac=0.40,
    )
    # Cannot exclude without slicing the hair mask — box must stay unchanged.
    assert new_box == box
    assert info["neighbors_excluded"] == 0.0
    assert info["neighbors_unexcludable"] == 1.0


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
