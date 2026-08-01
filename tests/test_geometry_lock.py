"""Geometry-lock production path: landmark prefer_box + seamless paste."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.align_paste_swap import run_align_paste_swap
from headswap.preprocess import FaceBox, paste_aligned_face


CACHE = ROOT / "results" / "_cache"


def test_seamless_paste_blends_without_crash():
    dest = Image.new("RGB", (128, 128), (40, 80, 120))
    ImageDraw.Draw(dest).ellipse([40, 40, 90, 100], fill=(200, 160, 140))
    rgba = np.zeros((128, 128, 4), dtype=np.uint8)
    rgba[45:95, 45:95, :3] = (80, 40, 20)
    rgba[45:95, 45:95, 3] = 220
    out, info = paste_aligned_face(dest, Image.fromarray(rgba, "RGBA"), seamless=True)
    assert out.size == dest.size
    assert info["composite_paste"] is True


def test_geometry_lock_changes_selected_preserves_neighbor():
    body = Image.new("RGB", (320, 200), (20, 20, 20))
    d = ImageDraw.Draw(body)
    d.ellipse([30, 40, 90, 120], fill=(210, 170, 150))  # left neighbor
    d.ellipse([200, 40, 270, 130], fill=(190, 150, 130))  # selected
    faces = [
        FaceBox(30, 40, 90, 120, 0.9),
        FaceBox(200, 40, 270, 130, 0.9),
    ]
    donor = Image.new("RGB", (120, 140), (255, 255, 255))
    ImageDraw.Draw(donor).ellipse([20, 20, 100, 110], fill=(20, 80, 200))  # blue face
    out = run_align_paste_swap(
        body,
        donor,
        CACHE,
        selected_face=faces[1],
        all_faces=faces,
        cfg={
            "align_paste_krea2_refine": False,
            "align_paste_pose_relock": True,
            "align_paste_seamless_clone": True,
            "pre_color_match_strength": 0.3,
            "align_paste_post_color_match": 0.2,
            "div_by": 8,
        },
    )
    result = np.asarray(out["image"])
    body_a = np.asarray(body)
    # Left neighbor center should stay close to original.
    assert float(np.mean(np.abs(result[70:90, 50:70].astype(float) - body_a[70:90, 50:70]))) < 25.0
    assert out["mode"] == "geometry_lock"
    assert out["paste_info"].get("composite_paste") is True
