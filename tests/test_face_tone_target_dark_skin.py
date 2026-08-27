"""The transfer TARGET must not read a dark complexion as light.

_face_skin_lab_stats dropped the darkest 35% of warm face pixels before
reading the tone, on the assumption that dark pixels are beard, nostrils and
cast shadow. That holds for a pale donor and is false for a dark-
complexioned one, where the dark pixels ARE the skin -- so the reported tone
was biased brighter.

Because this value is the TARGET of the LAB transfer, the body then aims too
light and stops there. GPU-measured: a dark donor read L=106 here while the
lateral-cheek estimate of the same face read L=81; the body transferred to
106 and stayed permanently lighter than the face. That was misdiagnosed as
mask under-coverage three times, because a body that stops short looks
identical to one that was never fully selected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.skin_harmonize import _cheek_lab_stats, _face_skin_lab_stats


def _face(rgb: tuple[int, int, int], shade: float = 0.55) -> np.ndarray:
    """A face box: flat complexion with a naturally shaded lower third."""
    img = np.zeros((200, 160, 3), dtype=np.uint8)
    img[:, :] = rgb
    img[130:, :] = (np.array(rgb, dtype=np.float32) * shade).astype(np.uint8)
    return img


def _L(rgb_patch: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.median(lab[..., 0]))


def test_dark_complexion_target_is_not_biased_bright():
    dark = (105, 72, 55)
    img = _face(dark)
    stats = _face_skin_lab_stats(img, 0, 0, 160, 200)
    assert stats is not None
    true_L = _L(np.full((8, 8, 3), dark, dtype=np.uint8))
    assert stats[0][0] <= true_L + 10.0, (
        f"target L={stats[0][0]:.0f} reads much brighter than the actual "
        f"complexion L={true_L:.0f}; the body will transfer too light and "
        "stop short of the face"
    )


def test_pale_complexion_still_reads_correctly():
    """The fix must not overcorrect in the other direction."""
    pale = (232, 205, 186)
    img = _face(pale)
    stats = _face_skin_lab_stats(img, 0, 0, 160, 200)
    assert stats is not None
    true_L = _L(np.full((8, 8, 3), pale, dtype=np.uint8))
    assert abs(stats[0][0] - true_L) <= 22.0, (
        f"target L={stats[0][0]:.0f} drifted from the pale complexion "
        f"L={true_L:.0f}"
    )


def test_agrees_with_the_cheek_reading_on_a_dark_face():
    """The two estimators must not disagree by the margin seen on GPU (25 L)."""
    img = _face((105, 72, 55))
    face = _face_skin_lab_stats(img, 0, 0, 160, 200)
    cheek = _cheek_lab_stats(img, 0, 0, 160, 200)
    assert face is not None
    assert face[0][0] - cheek[0][0] <= 12.0, (
        f"face-skin L={face[0][0]:.0f} vs cheek L={cheek[0][0]:.0f} -- the "
        "bright-direction cross-check should have caught this"
    )
