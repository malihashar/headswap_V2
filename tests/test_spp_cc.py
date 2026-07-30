"""Unit tests for SPP-CC segmentation dispatch and seam refine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import (
    FaceBox,
    narrow_band_seam_refine,
    seam_annulus_mask,
)
from headswap.segmentation import build_head_hair_mask


def test_build_head_hair_mask_ellipse_default():
    body = Image.new("RGB", (200, 240), (40, 40, 40))
    ImageDraw.Draw(body).ellipse([70, 40, 130, 120], fill=(200, 160, 140))
    face = FaceBox(80, 50, 120, 110, 0.9)
    mask, info = build_head_hair_mask(
        body, ROOT / "results" / "_cache", backend="ellipse", face_box=face
    )
    assert mask.size == body.size
    assert info["backend"] == "ellipse"
    arr = np.asarray(mask.convert("L"))
    assert arr[80, 100] > 128


def test_sam2_falls_back_to_ellipse():
    body = Image.new("RGB", (160, 160), (30, 30, 30))
    face = FaceBox(60, 40, 100, 90, 0.9)
    mask, info = build_head_hair_mask(
        body, ROOT / "results" / "_cache", backend="sam2", face_box=face
    )
    assert info["backend"] == "ellipse"
    assert info.get("fallback_reason") or info.get("requested_backend") == "sam2"
    assert mask.size == body.size


def test_seam_annulus_and_refine_changes_band_only():
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    stitched = Image.new("RGB", (64, 64), (200, 100, 50))
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).ellipse([16, 16, 48, 48], fill=255)
    band = seam_annulus_mask(mask, erode_px=4, dilate_px=8)
    band_a = np.asarray(band)
    assert band_a[32, 32] == 0  # core
    assert band_a.max() > 0

    refined = narrow_band_seam_refine(
        original, stitched, mask, erode_px=4, dilate_px=8, strength=1.0, blur_px=0
    )
    r = np.asarray(refined)
    # Far outside mask stays stitched? Actually refine only mixes band;
    # outside band keeps stitched. Corner is outside → stitched orange.
    assert tuple(r[0, 0].tolist()) == (200, 100, 50)
    # Core of mask (eroded) stays stitched
    assert tuple(r[32, 32].tolist()) == (200, 100, 50)


def test_spp_cc_flags_disable_isolate_and_tight():
    from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

    class P(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "single_person_parity": True,
                "force_tight_head_crop": True,
                "mask_top_extend": 1.55,
                "mask_side_extend": 0.60,
                "mask_bot_extend": 0.40,
                "mask_expand_px": 18,
            }
            self.cache_dir = ROOT / "results" / "_cache"

    body = Image.new("RGB", (400, 300), (20, 20, 20))
    faces = [
        FaceBox(20, 40, 80, 120, 0.9),
        FaceBox(200, 40, 260, 120, 0.9),
    ]
    flags = P()._tight_crop_flags(body, faces[0], faces)
    assert flags["single_person_parity"] is True
    assert flags["isolate_selected"] is False
    assert flags["use_tight"] is False
    assert flags["multi_person"] is True
    assert flags["top_ext"] == 1.55
    assert flags["side_ext"] == 0.60
