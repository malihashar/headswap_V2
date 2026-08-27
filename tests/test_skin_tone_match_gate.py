"""The LAB wash gate must measure tone, not just which code path ran.

`_wash_default = not _head_restored` gated skin-tone correction purely on
whether the person-skin restore succeeded. That is a boolean about control
flow which never inspects a pixel, so "the model recoloured the body
correctly" and "the model left the body at its original tone" were
indistinguishable -- and the wash, the only mechanism that corrects the
latter, was disabled precisely BECAUSE the restore had worked.

GPU-observed (2026-08-27): a dark-complexioned donor head composited onto a
pale body. The restore reported success (153,949px kept, clothing protected,
no smear) and the wash logged OFF, so nothing in the pipeline ever compared
the face to the body and the output shipped with a visibly mismatched torso
and limbs.

The gate now compares the donor face's cheek L against the visible BODY
skin's L in the same output frame and re-enables the wash on a residual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.skin_harmonize import _cheek_lab_stats, _robust_lab_stats

import cv2


def _lab_L(rgb: tuple[int, int, int]) -> float:
    patch = np.full((8, 8, 3), rgb, dtype=np.uint8)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean, _ = _robust_lab_stats(lab.reshape(-1, 3))
    return float(mean[0])


def test_cheek_sampling_reads_a_dark_face_as_dark():
    """The reference the gate compares against must track real complexion."""
    frame = np.full((400, 300, 3), 240, dtype=np.uint8)  # pale background
    # Face box spanning y=0..200; cheeks are sampled at 0.34-0.54 of height.
    frame[60:120, 40:260] = (90, 60, 45)  # dark cheeks
    mean, _ = _cheek_lab_stats(frame, 40, 0, 260, 200)
    assert mean[0] < 120.0, f"dark cheeks read as L={mean[0]}"


def test_pale_body_against_dark_face_exceeds_default_threshold():
    """The exact production failure: dark face, untouched pale body."""
    face_L = _lab_L((90, 60, 45))
    body_L = _lab_L((225, 200, 185))
    assert abs(face_L - body_L) > 12.0, (
        f"dL={abs(face_L - body_L)} did not exceed the 12.0 default, so the "
        "wash would stay off on a clearly mismatched body"
    )


def test_matching_tone_stays_under_threshold():
    """A body the model DID recolour must not re-trigger the wash."""
    face_L = _lab_L((150, 110, 90))
    body_L = _lab_L((155, 115, 95))
    assert abs(face_L - body_L) <= 12.0, (
        f"dL={abs(face_L - body_L)} would needlessly re-enable the wash on an "
        "already-matching body and risk reintroducing the smear"
    )


def test_gate_source_measures_pixels_not_control_flow():
    """Regression guard on the gate itself."""
    src = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
    assert "simple_full_body_tone_match_max_dl" in src, (
        "the measured tone threshold is gone -- the wash gate has likely "
        "reverted to the control-flow-only `not _head_restored` form"
    )
    assert "_cheek_lab_stats" in src, "face reference sampling missing from the gate"
