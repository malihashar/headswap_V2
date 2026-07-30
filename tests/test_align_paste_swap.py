"""Tests for Architecture B: face matte + align-paste swap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.align_paste_swap import measure_align_paste_gates, run_align_paste_swap
from headswap.preprocess import FaceBox, crop_around_face_box, identity_face_only_matte


CACHE = ROOT / "results" / "_cache"


def _fake_group() -> tuple[Image.Image, list[FaceBox]]:
    body = Image.new("RGB", (320, 240), (30, 40, 50))
    d = ImageDraw.Draw(body)
    # Three "faces"
    d.ellipse([20, 40, 70, 100], fill=(200, 160, 140))
    d.ellipse([120, 40, 170, 100], fill=(190, 150, 130))
    d.ellipse([230, 40, 290, 110], fill=(180, 140, 120))
    faces = [
        FaceBox(20, 40, 70, 100, 0.9),
        FaceBox(120, 40, 170, 100, 0.9),
        FaceBox(230, 40, 290, 110, 0.9),
    ]
    return body, faces


def test_identity_face_only_matte_strips_background():
    # Jersey-like donor: face in center, bright clothing below
    donor = Image.new("RGB", (200, 280), (20, 80, 180))  # jersey blue
    ImageDraw.Draw(donor).ellipse([60, 40, 140, 140], fill=(210, 170, 140))
    matte, info = identity_face_only_matte(
        donor, CACHE, top=0.2, bot=0.05, side=0.1, force_ellipse=True
    )
    assert info["identity_face_only"] is True
    arr = np.asarray(matte)
    # Corners of matte canvas should be near white (ellipse matte)
    assert float(arr[2, 2].mean()) > 200


def test_crop_around_face_box_centers_selected():
    body, faces = _fake_group()
    crop, box = crop_around_face_box(body, faces[2], pad_frac=0.5, div_by=8)
    assert crop.size[0] > 0 and crop.size[1] > 0
    assert box[0] <= faces[2].x0 and box[2] >= faces[2].x1


def test_run_align_paste_preserves_outside_mask():
    body, faces = _fake_group()
    # Identity with different face color
    donor = Image.new("RGB", (120, 140), (255, 255, 255))
    ImageDraw.Draw(donor).ellipse([20, 20, 100, 110], fill=(50, 20, 10))
    selected = faces[2]
    out = run_align_paste_swap(
        body,
        donor,
        CACHE,
        selected_face=selected,
        all_faces=faces,
        cfg={
            "align_paste_krea2_refine": False,
            "pre_color_match_strength": 0.0,
            "align_paste_post_color_match": 0.0,
            "div_by": 8,
        },
        refine_fn=None,
    )
    result = out["image"]
    assert result.size == body.size
    assert out["mode"] == "align_paste"
    # Far-left neighbor region should stay close to original (outside mask)
    body_a = np.asarray(body)
    res_a = np.asarray(result)
    left = body_a[50:90, 25:55]
    left_r = res_a[50:90, 25:55]
    mse = float(np.mean((left.astype(np.float64) - left_r.astype(np.float64)) ** 2))
    assert mse < 50.0, f"neighbor MSE too high: {mse}"
    gates = out["gates"]
    assert "neighbor_psnr_outside_mask" in gates or "head_height_ratio" in gates


def test_measure_gates_shape():
    body, faces = _fake_group()
    gates = measure_align_paste_gates(
        body, body, selected=faces[0], cache_dir=CACHE, face_mask=None
    )
    assert isinstance(gates, dict)
