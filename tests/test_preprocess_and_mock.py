from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.eval.dataset import generate_synthetic_eval_set, load_pairs
from headswap.pipelines import create_pipeline_from_config
from headswap.preprocess import (
    FaceBox,
    hard_freeze_neighbor_faces,
    head_hair_mask_from_face,
    soft_composite,
    suppress_neighbor_faces_in_mask,
    evenify,
)


def test_evenify():
    assert evenify(15, 8) == 8
    assert evenify(16, 8) == 16


def test_synthetic_eval_and_mock_klein():
    generate_synthetic_eval_set(n_pairs=4)
    pairs = load_pairs()
    assert len(pairs) >= 4
    pipe = create_pipeline_from_config(ROOT / "configs" / "klein4b.yaml", force_mock=True)
    body = Image.open(pairs[0]["body_path"])
    face = Image.open(pairs[0]["face_path"])
    out = pipe.run(body, face, out_dir=ROOT / "results" / "_test_mock")
    assert out.image.size[0] > 0
    assert out.latency_s >= 0
    assert "debug_mask" in out.debug_paths


def test_soft_composite_preserves_outside_mask():
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    edit = Image.new("RGB", (32, 32), (200, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    # white square in crop region mapped via soft_composite box
    from PIL import ImageDraw

    m = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(m).rectangle([16, 16, 47, 47], fill=255)
    out = soft_composite(base, edit, m, (16, 16, 48, 48))
    arr = np.asarray(out)
    # Corner should stay base color
    assert tuple(arr[0, 0].tolist()) == (10, 20, 30)


def test_hard_freeze_neighbor_faces_restores_pixels():
    original = Image.new("RGB", (100, 80), (10, 20, 30))
    # Paint two "faces"
    from PIL import ImageDraw

    draw = ImageDraw.Draw(original)
    draw.rectangle([10, 10, 35, 40], fill=(255, 0, 0))  # left neighbor
    draw.rectangle([60, 10, 85, 40], fill=(0, 255, 0))  # selected

    # Simulated bad full-frame edit: neighbor corrupted
    edited = original.copy()
    ImageDraw.Draw(edited).rectangle([10, 10, 35, 40], fill=(0, 0, 255))

    selected = FaceBox(60, 10, 85, 40, 0.9)
    neighbor = FaceBox(10, 10, 35, 40, 0.9)
    frozen = hard_freeze_neighbor_faces(
        edited, original, selected, [selected, neighbor], pad_frac=0.0, expand_top_frac=0.0
    )
    arr = np.asarray(frozen)
    # Neighbor center restored to original red
    assert tuple(arr[25, 22].tolist()) == (255, 0, 0)
    # Selected region still edited green
    assert tuple(arr[25, 72].tolist()) == (0, 255, 0)


def test_hard_freeze_does_not_overwrite_selected_on_overlap():
    """Tight group: oversized neighbor paste must not erase the swap."""
    from PIL import ImageDraw

    original = Image.new("RGB", (120, 80), (10, 20, 30))
    ImageDraw.Draw(original).rectangle([20, 15, 55, 50], fill=(255, 0, 0))
    ImageDraw.Draw(original).rectangle([50, 15, 90, 50], fill=(0, 0, 100))  # orig selected

    edited = original.copy()
    ImageDraw.Draw(edited).rectangle([50, 15, 90, 50], fill=(0, 255, 0))  # swapped

    selected = FaceBox(50, 15, 90, 50, 0.9)
    neighbor = FaceBox(20, 15, 55, 50, 0.9)
    # Large pad would cover selected if unprotected
    frozen = hard_freeze_neighbor_faces(
        edited,
        original,
        selected,
        [selected, neighbor],
        pad_frac=0.8,
        expand_top_frac=0.5,
        protect_expand=1.2,
    )
    arr = np.asarray(frozen)
    # Center of selected must stay swapped green, not original blue-ish
    assert tuple(arr[32, 70].tolist()) == (0, 255, 0)


def test_suppress_neighbor_zeros_other_faces():
    mask = Image.new("L", (100, 80), 255)
    selected = FaceBox(60, 10, 85, 40, 0.9)
    neighbor = FaceBox(10, 10, 35, 40, 0.9)
    out = suppress_neighbor_faces_in_mask(mask, selected, [selected, neighbor], shrink=1.0)
    arr = np.asarray(out)
    assert arr[25, 22] == 0
    assert arr[25, 72] == 255
