"""clamp_edited_head_scale's exposed-border handling.

Three approaches were GPU-tried on 2026-08-09 for the border its shrink/grow
warp uncovers:

  1. cv2.BORDER_REPLICATE -- stretches the warped image's own edge
     row/column outward, producing a warped, vertically-striped duplicate
     of whatever was at that edge (a ghost copy of the subject's feet).
     Rejected outright.
  2. Flat black, relying on a downstream person-matte (restore_background)
     to repair it from the original photo -- sounded cleanest, but
     measured: a solid black rectangle against sandy ground, directly
     below a pair of shoes, is genuinely ambiguous to rembg (46% of gap
     pixels scored "person", 54% "background"), so the matte left most of
     it exactly as-is. Also rejected.
  3. Pasting from ``original_scene`` using the warp's own EXACT geometric
     coverage mask (not a neural guess) -- can echo the person's real,
     unshrunk limbs where the gap overlaps their original position, but
     was the least-bad artifact of the three on GPU inspection. SHIPPED.

This test locks in #1's rejection (no replicate smear) -- the concrete,
regression-prone failure mode -- without over-asserting against #3's
accepted, documented trade-off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.preprocess as preprocess_mod
from headswap.preprocess import FaceBox, clamp_edited_head_scale


def test_shrink_border_is_not_a_replicated_smear(monkeypatch):
    w, h = 300, 300
    original_scene = Image.new("RGB", (w, h), (10, 20, 30))

    edited = Image.new("RGB", (w, h), (200, 100, 50))
    d = ImageDraw.Draw(edited)
    d.ellipse([90, 30, 210, 150], fill=(220, 180, 150))  # oversized generated face

    target_face = FaceBox(110, 110, 190, 190, 0.9)  # height 80
    gen_face = FaceBox(90, 30, 210, 150, 0.9)  # height 120 -> ratio 1.5, oversized
    orig_mean = float(np.asarray(original_scene).mean())

    def _fake(rgb, cache_dir, conf_thresh=0.30):
        return target_face if abs(float(rgb.mean()) - orig_mean) < 5 else gen_face

    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake)

    result, info = clamp_edited_head_scale(
        original_scene, edited, ROOT / "results" / "_cache"
    )
    assert info["clamped"] == 1.0
    assert result.size == (w, h)

    arr = np.asarray(result)
    corner = arr[0:40, 0:40].reshape(-1, 3).astype(int)

    # BORDER_REPLICATE would stretch `edited`'s own flat background color
    # (200, 100, 50) into the corner uniformly -- that must not happen.
    edited_bg = np.array([200, 100, 50])
    exact_replicate = (np.abs(corner - edited_bg[None, :]).max(axis=1) == 0).all()
    assert not exact_replicate, "border looks like an unblended replicate smear"

    # The (documented, accepted) behavior is a patch from original_scene:
    # the corner should match it, not float free of both known sources.
    orig_px = np.asarray(original_scene)[0, 0]
    assert np.abs(corner - orig_px[None, :]).max(axis=1).max() <= 2
