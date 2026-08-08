"""head_matte mask backend: mask follows the real head silhouette.

The `ellipse` backend derives the mask purely from the face box. Measured on a
short-haired subject with a small face in a full-body frame, it leaves ~121px
(0.98x face height) of EMPTY BACKGROUND above the real head.

That is not cosmetic. `crop_with_mask` derives the crop box from the mask, so
the model is handed that background, regenerates it imperfectly (a visible arc
exactly on the mask boundary) and -- asked to fill the space -- invents hair to
occupy it. That is the observed "ghost oval above the head + hair wings at ear
level" production failure.

head_matte intersects the ellipse with a real foreground matte, so the model
can only regenerate actual head pixels. These tests assert the measured
property, not merely that the code runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.segmentation as seg
from headswap.preprocess import FaceBox

H, W = 1024, 768
FH = H * 0.12          # small face, as in a full-body frame
FW = FH * 0.75
FACE = FaceBox(W / 2 - FW / 2, H * 0.22 - FH / 2, W / 2 + FW / 2, H * 0.22 + FH / 2, 0.95)
BODY = Image.new("RGB", (W, H))


def _truth_silhouette() -> np.ndarray:
    """Short-haired head + shoulders; everything else is background."""
    t = np.zeros((H, W), np.uint8)
    cx = int((FACE.x0 + FACE.x1) / 2)
    cy = int((FACE.y0 + FACE.y1) / 2)
    cv2.ellipse(t, (cx, cy - int(FH * 0.15)),
                (int(FW * 0.62), int(FH * 0.70)), 0, 0, 360, 255, -1)
    cv2.rectangle(t, (cx - int(FW * 1.6), int(FACE.y1 + FH * 0.25)), (cx + int(FW * 1.6), H), 255, -1)
    return t


def _headroom(mask: Image.Image, truth: np.ndarray) -> int:
    """Pixels of mask extending ABOVE the real head top -- the empty region
    the model would otherwise fill with invented hair."""
    head_top = np.where(truth.any(axis=1))[0].min()
    ys = np.where((np.asarray(mask.convert("L")) > 16).any(axis=1))[0]
    return int(head_top - ys.min())


def test_head_matte_drastically_reduces_empty_headroom_vs_ellipse():
    truth = _truth_silhouette()
    kw = dict(face_box=FACE, expand_px=18, blur_px=12,
              top_extend=1.55, side_extend=0.60, bot_extend=0.40)

    ell, _ = seg.build_head_hair_mask(BODY, None, backend="ellipse", **kw)
    with mock.patch.object(seg, "_person_matte", lambda img: (truth.copy(), None)):
        matte, info = seg.build_head_hair_mask(BODY, None, backend="head_matte", **kw)

    assert info["backend"] == "head_matte"
    ell_hr, matte_hr = _headroom(ell, truth), _headroom(matte, truth)

    # The ellipse leaves ~a full face-height of empty background above the head.
    assert ell_hr > 0.8 * FH, f"expected the ellipse defect, got {ell_hr}px"
    # head_matte must cut that by at least 5x.
    assert matte_hr * 5 < ell_hr, f"ellipse={ell_hr}px head_matte={matte_hr}px"
    # ...while still leaving SOME room for a larger donor hairstyle.
    assert matte_hr > 0


def test_head_matte_still_covers_the_face_core():
    """Containing the mask must not shrink it off the face itself."""
    truth = _truth_silhouette()
    with mock.patch.object(seg, "_person_matte", lambda img: (truth.copy(), None)):
        m, _ = seg.build_head_hair_mask(
            BODY, None, backend="head_matte", face_box=FACE, expand_px=18,
            blur_px=12, top_extend=1.55, side_extend=0.60, bot_extend=0.40)
    a = np.asarray(m.convert("L"))
    cx, cy = int((FACE.x0 + FACE.x1) / 2), int((FACE.y0 + FACE.y1) / 2)
    assert a[cy, cx] > 200, "face centre must remain fully editable"


def test_hair_margin_frac_controls_growth_room():
    """0.0 pins new hair to the old silhouette; larger values allow growth."""
    truth = _truth_silhouette()
    hrs = []
    for margin in (0.0, 0.20):
        with mock.patch.object(seg, "_person_matte", lambda img: (truth.copy(), None)):
            m, _ = seg._head_matte_mask(
                BODY, None, face_box=FACE, expand_px=18, blur_px=6,
                top_extend=1.55, side_extend=0.60, bot_extend=0.40,
                hair_margin_frac=margin)
        hrs.append(_headroom(m, truth))
    assert hrs[1] > hrs[0], f"margin should add growth room, got {hrs}"


def test_falls_back_to_ellipse_when_no_matting_backend():
    """rembg missing must degrade gracefully, never fail a swap."""
    with mock.patch.object(seg, "_person_matte", lambda img: (None, "rembg_missing:test")):
        m, info = seg.build_head_hair_mask(
            BODY, None, backend="head_matte", face_box=FACE, expand_px=18,
            blur_px=12, top_extend=1.55, side_extend=0.60, bot_extend=0.40)
    assert m is not None
    assert info["backend"] == "ellipse"
    assert "rembg_missing" in info["head_matte_skip"]


def test_production_config_selects_head_matte():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "krea2_identity_edit.yaml").read_text())
    assert cfg["head_mask_backend"] == "head_matte"
    assert 0.0 <= cfg["head_matte_hair_margin_frac"] <= 0.5
