"""_blend_procrustes_face_only must not let warped background/sky leak out.

A real run showed streaky glitches in the sky above the head after
enabling procrustes correction. Root cause: procrustes_align_edited_crop_to_
body_box warps the ENTIRE rectangular head+hair crop (face + hair + whatever
background/sky is baked into the crop's generous top/side padding -- padding
sized for long hair, which is mostly open sky on a short-haired subject).
cv2.warpAffine's border reflection streaks badly across a smooth gradient
sky, and the stitch mask that shows the crop into the final image is that
same generous head+hair mask, so the streaks were visible in the output.

_blend_procrustes_face_only blends the warped and unwarped crops through a
tight face-only mask so only eyes/nose/mouth geometry is corrected --
anything outside that tight region (hair, background) must come from the
unwarped crop untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.pipelines.krea2 as krea2_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox


def _pipe(cfg: dict | None = None) -> Krea2IdentityEditPipeline:
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, cfg):
            self.cfg = dict(cfg or {})
            self.cache_dir = ROOT / "results" / "_cache"

    return _Pipe(cfg)


def test_warped_background_does_not_leak_outside_face(monkeypatch):
    w, h = 300, 400
    face_box = (110, 150, 190, 230)  # center-ish, small relative to frame

    edited = Image.new("RGB", (w, h), (135, 175, 220))  # smooth "sky" blue
    ImageDraw.Draw(edited).ellipse(list(face_box), fill=(200, 160, 130))  # skin

    # Simulate a warp-corrupted crop: same face area, but the background/sky
    # is now a completely different, "streaky" color everywhere -- including
    # the top-left corner, far from the face, exactly where border-reflection
    # streak artifacts showed up in the real bug.
    aligned = Image.new("RGB", (w, h), (255, 0, 0))
    ImageDraw.Draw(aligned).ellipse(list(face_box), fill=(90, 60, 40))  # shifted skin tone

    face = FaceBox(*face_box, 0.95)
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (face, [face])
    )

    pipe = _pipe()
    blended, info = pipe._blend_procrustes_face_only(edited, aligned)

    assert info["face_only_blend"] is True
    blended_a = np.asarray(blended, dtype=np.float64)
    edited_a = np.asarray(edited, dtype=np.float64)

    # Far corner: must match the UNWARPED crop exactly -- no streak leakage.
    corner = (5, 5)
    assert tuple(blended_a[corner].tolist()) == tuple(edited_a[corner].tolist())

    # Face center: should have moved toward the warped/corrected version,
    # not stayed identical to the unwarped original.
    cy, cx = (face_box[1] + face_box[3]) // 2, (face_box[0] + face_box[2]) // 2
    assert blended_a[cy, cx][0] < edited_a[cy, cx][0] - 10  # skin got darker (toward aligned's 90,60,40)


def test_no_face_on_edited_falls_back_to_aligned(monkeypatch):
    edited = Image.new("RGB", (100, 100), (10, 10, 10))
    aligned = Image.new("RGB", (100, 100), (200, 200, 200))
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (None, [])
    )
    pipe = _pipe()
    result, info = pipe._blend_procrustes_face_only(edited, aligned)
    assert result is aligned
    assert info["face_only_blend"] is False
    assert info["face_only_blend_reason"] == "no_face_on_edited_crop"
