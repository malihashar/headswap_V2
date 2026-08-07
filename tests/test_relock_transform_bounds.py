"""relock_pose_to_destination must reject implausible transforms.

A real run produced a severely broken result (a large misplaced duplicate of
hair/head content) once head_direction_relock was wired into production --
almost certainly a bad landmark match sending estimateAffine[Partial]2D to a
wild scale/rotation/translation, which then got warped+composited as a large
visible ghost instead of a subtle pose nudge. These tests lock in the
sanity-bound rejection added in response, using monkeypatched landmarks so
they don't depend on real face detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.preprocess as preprocess_mod
from headswap.preprocess import relock_pose_to_destination


def _face_mask(size, box=(60, 60, 140, 140)) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse(list(box), fill=255)
    return m


def _stub_landmarks(monkeypatch, *, src, dst):
    def _fake(rgb, cache_dir, prefer_box=None):
        # Distinguish source (generated) vs destination calls by array identity
        # is unreliable across np.asarray copies, so key off mean pixel value
        # baked into each test's synthetic images instead.
        mean = float(rgb.mean())
        return (src if mean < 128 else dst), "insightface", None

    monkeypatch.setattr(preprocess_mod, "get_face_landmarks5", _fake)


def test_relock_rejects_wildly_scaled_transform(monkeypatch):
    w, h = 200, 200
    generated = Image.new("RGB", (w, h), (10, 10, 10))  # mean < 128 -> "src"
    destination = Image.new("RGB", (w, h), (200, 200, 200))  # mean >= 128 -> "dst"

    # Destination landmarks are a tiny cluster far from source's -- forces a
    # huge scale + large translation in the recovered similarity transform.
    src_lm = np.array([[80, 80], [120, 80], [100, 100], [85, 120], [115, 120]], dtype=np.float32)
    dst_lm = np.array([[10, 10], [12, 10], [11, 12], [10, 13], [12, 13]], dtype=np.float32)
    _stub_landmarks(monkeypatch, src=src_lm, dst=dst_lm)

    mask = _face_mask((w, h))
    out, info = relock_pose_to_destination(
        generated, destination, ROOT / "results" / "_cache", face_mask=mask, use_full_affine=False
    )

    assert info["pose_relock"] is False
    assert "transform_out_of_bounds" in (info.get("pose_relock_reason") or "")
    # Original generated image is returned untouched, not a warped ghost.
    assert np.array_equal(np.asarray(out), np.asarray(generated))


def test_relock_accepts_small_plausible_correction(monkeypatch):
    w, h = 200, 200
    generated = Image.new("RGB", (w, h), (10, 10, 10))
    destination = Image.new("RGB", (w, h), (200, 200, 200))

    # A small, plausible nudge: mild rotation/translation, ~unit scale.
    src_lm = np.array([[80, 80], [120, 80], [100, 100], [85, 120], [115, 120]], dtype=np.float32)
    dst_lm = src_lm + np.array([[3, 2]], dtype=np.float32)
    _stub_landmarks(monkeypatch, src=src_lm, dst=dst_lm)

    mask = _face_mask((w, h))
    out, info = relock_pose_to_destination(
        generated, destination, ROOT / "results" / "_cache", face_mask=mask, use_full_affine=False
    )

    assert info["pose_relock"] is True
    assert info.get("pose_relock_reason") is None
    assert 0.9 < info["scale"] < 1.1
    assert abs(info["rotation_deg"]) < 5.0
