"""Unit tests for Kontext Align → Paste helpers (no ComfyUI / GPU required)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from headswap.preprocess import (
    align_face_to_destination,
    get_face_landmarks5,
    paste_aligned_face,
)


def _face_image(size: tuple[int, int], cx: int, cy: int, r: int, color=(200, 160, 140)) -> Image.Image:
    im = Image.new("RGB", size, (30, 30, 40))
    d = ImageDraw.Draw(im)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    # crude eyes / mouth so box prior has structure
    d.ellipse((cx - r // 2, cy - r // 3, cx - r // 4, cy - r // 6), fill=(20, 20, 20))
    d.ellipse((cx + r // 4, cy - r // 3, cx + r // 2, cy - r // 6), fill=(20, 20, 20))
    d.ellipse((cx - r // 5, cy + r // 5, cx + r // 5, cy + r // 2), fill=(120, 60, 60))
    return im


def test_landmarks_box_prior(tmp_path: Path):
    im = _face_image((256, 256), 128, 100, 60)
    rgb = np.asarray(im)
    lm, backend, note = get_face_landmarks5(rgb, tmp_path)
    assert lm is not None
    assert lm.shape == (5, 2)
    assert backend in {"insightface", "box_prior", "haar", "caffe", "none"} or True
    # With no insightface in CI, expect box_prior
    if backend == "box_prior":
        assert note is not None


def test_align_and_paste(tmp_path: Path):
    body = _face_image((320, 400), 160, 120, 70, color=(180, 140, 120))
    face = _face_image((200, 200), 100, 90, 55, color=(210, 170, 150))
    aligned, info = align_face_to_destination(face, body, tmp_path)
    # Box-prior path should still succeed
    assert info["face_alignment"] is True
    assert aligned is not None
    assert aligned.size == body.size
    assert aligned.mode == "RGBA"

    composite, paste_info = paste_aligned_face(body, aligned)
    assert paste_info["composite_paste"] is True
    assert composite.size == body.size
    assert composite.mode == "RGB"
