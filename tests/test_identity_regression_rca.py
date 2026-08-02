"""Isolation tests for the multi-person identity regression (RC1–RC4).

Exp A/B from the RCA plan are encoded as config assertions + a deterministic
stitch-mask proof (RC4). Full Krea2 GPU A/B is documented in
scripts/run_identity_regression_ab.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.align_paste_swap import run_align_paste_swap
from headswap.preprocess import FaceBox, feathered_soft_composite


CACHE = ROOT / "results" / "_cache"
CFG_PATH = ROOT / "configs" / "krea2_identity_edit.yaml"


def test_exp_a_config_default_is_krea2_crop_spp():
    """Exp A / decision-gate YES: production multi = single-person crop path."""
    cfg = yaml.safe_load(CFG_PATH.read_text())
    assert cfg.get("multi_person_swap_mode") == "krea2_crop"
    assert cfg.get("single_person_parity") is True
    assert cfg.get("clamp_crop_away_neighbors") is True
    assert cfg.get("multi_person_edit_mode") == "crop_stitch"


def test_exp_b_paste_only_config_knobs_exist():
    """Exp B knobs: refine/relock/color-match can be disabled for paste isolation."""
    cfg = yaml.safe_load(CFG_PATH.read_text())
    for key in (
        "align_paste_krea2_refine",
        "align_paste_pose_relock",
        "pre_color_match_strength",
        "align_paste_post_color_match",
    ):
        assert key in cfg


def test_rc4_crop_mask_leaks_original_face():
    """
    soft_composite does mask.crop(box). A crop-sized mask + full-image box
    mis-extracts alpha → original body face shows through (RC4).
    """
    w, h = 320, 240
    body = Image.new("RGB", (w, h), (40, 40, 40))
    # Right-side original face = green
    ImageDraw.Draw(body).ellipse([230, 40, 290, 110], fill=(0, 200, 0))
    # Edited crop with red "identity" face in crop-local coords
    box = (200, 20, 320, 140)  # 120x120 crop window
    bw, bh = box[2] - box[0], box[3] - box[1]
    edited_crop = Image.new("RGB", (bw, bh), (40, 40, 40))
    ImageDraw.Draw(edited_crop).ellipse([30, 20, 90, 90], fill=(220, 30, 30))

    # Crop-local opaque mask over the red face (WRONG contract for soft_composite)
    crop_mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(crop_mask).ellipse([30, 20, 90, 90], fill=255)

    # Full-canvas mask (CORRECT contract)
    full_mask = Image.new("L", (w, h), 0)
    full_mask.paste(crop_mask, (box[0], box[1]))

    bad = feathered_soft_composite(body, edited_crop, crop_mask, box, extra_blur_px=0)
    good = feathered_soft_composite(body, edited_crop, full_mask, box, extra_blur_px=0)

    bad_a = np.asarray(bad, dtype=np.float64)
    good_a = np.asarray(good, dtype=np.float64)
    body_a = np.asarray(body, dtype=np.float64)

    # Sample center of right face in full image
    cy, cx = 75, 260
    # Wrong mask: result stays near original green
    assert bad_a[cy, cx, 1] > 100, "crop-mask stitch should leak original green"
    # Correct mask: result should be reddish (identity)
    assert good_a[cy, cx, 0] > 150, "full-mask stitch should keep edited red"
    assert good_a[cy, cx, 1] < 80
    # Neighbor / bg unchanged on good path
    assert float(np.mean(np.abs(good_a[50, 50] - body_a[50, 50]))) < 1.0


def test_align_paste_uses_full_mask_and_exposes_debug_stages():
    """After RC4 fix, align-paste returns full-canvas mask + debug stage images."""
    body = Image.new("RGB", (320, 240), (30, 40, 50))
    d = ImageDraw.Draw(body)
    d.ellipse([20, 40, 70, 100], fill=(200, 160, 140))
    d.ellipse([120, 40, 170, 100], fill=(190, 150, 130))
    d.ellipse([230, 40, 290, 110], fill=(180, 140, 120))
    faces = [
        FaceBox(20, 40, 70, 100, 0.9),
        FaceBox(120, 40, 170, 100, 0.9),
        FaceBox(230, 40, 290, 110, 0.9),
    ]
    donor = Image.new("RGB", (120, 140), (255, 255, 255))
    ImageDraw.Draw(donor).ellipse([20, 20, 100, 110], fill=(50, 20, 10))

    def _fake_refine(composite, id_matte, face_mask_crop):  # noqa: ANN001
        # Simulate Krea2: bright blue face region; return (blended, raw)
        raw = composite.copy()
        ImageDraw.Draw(raw).ellipse([10, 10, 80, 80], fill=(0, 0, 255))
        return composite, raw

    out = run_align_paste_swap(
        body,
        donor,
        CACHE,
        selected_face=faces[2],
        all_faces=faces,
        cfg={
            "align_paste_krea2_refine": True,
            "align_paste_pose_relock": False,
            "pre_color_match_strength": 0.0,
            "align_paste_post_color_match": 0.0,
            "div_by": 8,
        },
        refine_fn=_fake_refine,
    )
    assert out["face_mask"].size == body.size
    assert out["face_mask_crop"].size == out["work_crop"].size
    assert out.get("raw_refined_crop") is not None
    assert out.get("pose_before_relock") is not None
    assert "pose_meta" in out
    assert out["image"].size == body.size


def test_exp_b_paste_only_pipeline_runs():
    """Exp B: refine+relock+color-match off still produces a full-frame result."""
    body = Image.new("RGB", (320, 240), (30, 40, 50))
    ImageDraw.Draw(body).ellipse([230, 40, 290, 110], fill=(180, 140, 120))
    faces = [FaceBox(230, 40, 290, 110, 0.9), FaceBox(20, 40, 70, 100, 0.9)]
    donor = Image.new("RGB", (120, 140), (255, 255, 255))
    ImageDraw.Draw(donor).ellipse([20, 20, 100, 110], fill=(80, 40, 20))
    out = run_align_paste_swap(
        body,
        donor,
        CACHE,
        selected_face=faces[0],
        all_faces=faces,
        cfg={
            "align_paste_krea2_refine": False,
            "align_paste_pose_relock": False,
            "pre_color_match_strength": 0.0,
            "align_paste_post_color_match": 0.0,
            "div_by": 8,
        },
        refine_fn=None,
    )
    assert out["refine_meta"].get("refine_applied") is False
    assert out["pose_meta"].get("pose_relock") is False
    assert out["composite_crop"] is not None
    assert out["image"].size == body.size
