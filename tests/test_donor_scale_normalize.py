"""Donor/target face-scale normalization for the small-face (full-body) case.

Krea2Edit is dual-reference attention transfer with NO inpaint mask -- it
samples the whole crop from an empty latent at denoise=1.0. So the strongest
scale cue it receives is the RELATIVE size of the head in image 1 (scene) vs
image 2 (person/donor).

`resize_contain` fills the donor canvas, putting the donor head at ~55% of
it, while the target head occupies whatever the crop geometry yields:
  portrait  ~58%  (matches -- and only because the mask ellipse gets clipped
                   at the image top, i.e. by accident)
  full-body ~32%  (nothing to clip against, so the full top extension applies)
Handing the model a donor head ~1.7x the target's relative size makes it
render the head at donor scale: oversized head/hair with the original hair
silhouette left uncovered around it.

These tests lock in: (1) the portrait path stays EXACTLY resize_contain (zero
regression on the known-good case), (2) a small-in-frame face gets the donor
rescaled so both fractions match.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.pipelines.krea2 as krea2_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox


def _pipe() -> Krea2IdentityEditPipeline:
    class _P(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "single_person_parity": True,
                "mask_top_extend": 1.55,
                "mask_side_extend": 0.60,
                "mask_bot_extend": 0.40,
                "mask_expand_px": 18,
                "mask_blur_px": 12,
                "crop_long_side": 768,
                "person_match_crop_size": True,
                "head_mask_backend": "ellipse",
                "square_crop": False,
            }
            self.cache_dir = None

    return _P()


# Donor crop whose face occupies 55% of its height, matching what
# crop_face_reference actually produces (~1.8x face height of padding).
_DONOR = Image.new("RGB", (400, 600))
_DONOR_FB = FaceBox(100, 150, 300, 480, 0.95)  # 330/600 = 55%


def _build(face_h_frac: float):
    H, W = 1024, 768
    fh = H * face_h_frac
    fw = fh * 0.75
    fb = FaceBox(W / 2 - fw / 2, H * 0.30 - fh / 2, W / 2 + fw / 2, H * 0.30 + fh / 2, 0.95)
    body = Image.new("RGB", (W, H))
    pipe = _pipe()
    flags = pipe._tight_crop_flags(body, fb, [fb])
    with mock.patch.object(krea2_mod, "detect_best_face", lambda *a, **k: _DONOR_FB):
        built = pipe._build_scene_person(
            body, _DONOR, fb, div_by=16, use_tight=False,
            top_ext=flags["top_ext"], side_ext=flags["side_ext"], bot_ext=flags["bot_ext"],
            expand_px=flags["expand_px"], crop_pad=flags["crop_pad"],
            all_faces=[fb], isolate_selected=False, blur_px=flags["blur_px"],
        )
    box = built["box"]
    target_frac = float(fb.height) / float(box[3] - box[1])
    return built["diag"], target_frac


def test_portrait_donor_prep_is_untouched_resize_contain():
    """The known-good path must stay bit-identical: donor and target already
    agree (~55% vs ~58%), so normalization must short-circuit entirely."""
    diag, _ = _build(0.40)
    assert diag["person_prep"] == "resize_contain"
    assert diag["donor_scale_normalize"]["applied"] is False
    assert diag["donor_scale_normalize"]["reason"] == "already_matched"


def test_small_face_donor_is_rescaled_to_match_target_fraction():
    """Full-body: donor head is ~1.7x the target's relative size. After
    normalization the donor's face must land at the target's fraction."""
    diag, target_frac = _build(0.12)
    info = diag["donor_scale_normalize"]
    assert diag["person_prep"] == "donor_scale_normalized"
    assert info["applied"] is True
    # Donor image scaled so its 55%-of-image face lands at target_frac.
    effective_donor_face_frac = info["donor_face_frac"] * info["donor_img_frac"]
    assert abs(effective_donor_face_frac - target_frac) < 0.01
    # And it genuinely shrank the donor (this is the corrective action).
    assert info["donor_img_frac"] < 0.7


def test_normalization_falls_back_safely_when_donor_face_undetectable():
    """No detected donor face -> plain resize_contain, never a crash."""
    H, W = 1024, 768
    fh = H * 0.12
    fb = FaceBox(W / 2 - fh * 0.375, H * 0.3 - fh / 2, W / 2 + fh * 0.375, H * 0.3 + fh / 2, 0.95)
    body = Image.new("RGB", (W, H))
    pipe = _pipe()
    flags = pipe._tight_crop_flags(body, fb, [fb])
    with mock.patch.object(krea2_mod, "detect_best_face", lambda *a, **k: None):
        built = pipe._build_scene_person(
            body, _DONOR, fb, div_by=16, use_tight=False,
            top_ext=flags["top_ext"], side_ext=flags["side_ext"], bot_ext=flags["bot_ext"],
            expand_px=flags["expand_px"], crop_pad=flags["crop_pad"],
            all_faces=[fb], isolate_selected=False, blur_px=flags["blur_px"],
        )
    assert built["diag"]["person_prep"] == "resize_contain"
    assert built["diag"]["donor_scale_normalize"]["applied"] is False


def test_head_geometry_scale_invariance_leaves_portrait_identical():
    """The absolute-px mask terms (expand/blur/crop_pad/feather) scale DOWN
    only, so portraits/close-ups are untouched and only small-in-frame faces
    (where 30px of absolute growth is 24% of the face, vs 7% for a portrait)
    are corrected."""
    body = Image.new("RGB", (768, 1024))
    pipe = _pipe()
    big = FaceBox(200, 100, 500, 510, 0.95)  # 410px face -> reference
    small = FaceBox(330, 280, 420, 403, 0.95)  # 123px face
    fb_big = pipe._tight_crop_flags(body, big, [big])
    fb_small = pipe._tight_crop_flags(body, small, [small])
    assert fb_big["geom_scale"] == 1.0
    assert fb_big["expand_px"] == 18 and fb_big["blur_px"] == 12
    assert fb_small["geom_scale"] < 1.0
    assert fb_small["expand_px"] < 18 and fb_small["blur_px"] < 12
