"""Tests for the prompt-only head-direction hint.

This replaces pixel-warp head-direction correction (head_direction_relock,
then procrustes) as the primary lever for "the swapped head is looking the
wrong direction": three consecutive pixel-level fixes each corrected the
reported symptom but introduced a new compositing artifact (a whole-image
warp produced a floating duplicate head; a crop-local warp still streaked
warped background into the output even after masking). A wrong or
unhelpful *text* hint can't corrupt pixels, so it's the safer default lever.

estimate_head_direction_label deliberately never asserts a yaw *direction*
(left/right turn) -- only magnitude -- since a 2D nose-offset proxy is too
noisy to confidently commit to a turn direction and telling the model the
wrong direction would be worse than not claiming one. Roll (tilt) IS
asserted directionally, because it's derived from which of the two
(self-sorted-by-x) eye landmarks has the smaller y -- that's unambiguous
and self-consistent by construction, not a guessed anatomical convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.pipelines.krea2 as krea2_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox, estimate_head_direction_label


def _landmarks(eye_a, eye_b, nose, mouth_a=(0, 0), mouth_b=(0, 0)):
    return np.array([eye_a, eye_b, nose, mouth_a, mouth_b], dtype=np.float32)


# ---------------------------------------------------------------------------
# estimate_head_direction_label: pure geometry, no model calls.
# ---------------------------------------------------------------------------


def test_frontal_level_face_has_no_label():
    lm = _landmarks(eye_a=(80, 100), eye_b=(120, 100), nose=(100, 130))
    result = estimate_head_direction_label(lm)
    assert result["label"] == ""
    assert result["roll_deg"] == 0.0
    assert result["yaw_offset"] == 0.0


def test_roll_direction_is_self_consistent_not_guessed():
    # Left-sorted eye (smaller x=80) is higher (smaller y=90, image y grows
    # downward) than the right eye (x=120, y=110) -- the label must say
    # "left" sits higher, since that's exactly what these coordinates show,
    # not an assumed anatomical convention.
    lm = _landmarks(eye_a=(80, 90), eye_b=(120, 110), nose=(100, 130))
    result = estimate_head_direction_label(lm, roll_deg_thresh=1.0)
    assert "left" in result["label"]
    assert "higher" in result["label"]

    # Mirror it: now the right eye is higher -- roll_deg sign must flip too.
    lm2 = _landmarks(eye_a=(80, 110), eye_b=(120, 90), nose=(100, 130))
    result2 = estimate_head_direction_label(lm2, roll_deg_thresh=1.0)
    assert "right" in result2["label"]
    assert "higher" in result2["label"]
    assert (result2["roll_deg"] > 0) != (result["roll_deg"] > 0)


def test_small_roll_below_threshold_is_not_mentioned():
    lm = _landmarks(eye_a=(80, 100), eye_b=(120, 101), nose=(100, 130))
    result = estimate_head_direction_label(lm, roll_deg_thresh=5.0)
    assert result["label"] == ""


def test_yaw_magnitude_scales_with_offset_but_never_asserts_a_direction_word():
    # Nose shifted well off the eye-midpoint -> should mention magnitude,
    # but must never claim "left" or "right" for the TURN itself (only for
    # roll/tilt, which is a separate, unambiguous signal).
    lm = _landmarks(eye_a=(80, 100), eye_b=(120, 100), nose=(140, 130))
    result = estimate_head_direction_label(lm, yaw_thresh=0.1, yaw_strong_thresh=0.3)
    assert "turned off-center" in result["label"]
    assert "left" not in result["label"]
    assert "right" not in result["label"]
    assert "strongly" in result["label"]


def test_moderate_yaw_uses_moderate_wording():
    lm = _landmarks(eye_a=(80, 100), eye_b=(120, 100), nose=(107, 130))
    result = estimate_head_direction_label(lm, yaw_thresh=0.1, yaw_strong_thresh=0.5)
    assert "moderately" in result["label"]


def test_insufficient_landmarks_returns_empty():
    result = estimate_head_direction_label(np.zeros((2, 2)))
    assert result["label"] == ""
    assert result["reason"] == "insufficient_landmarks"


# ---------------------------------------------------------------------------
# Krea2IdentityEditPipeline wiring.
# ---------------------------------------------------------------------------


def _pipe(cfg: dict | None = None) -> Krea2IdentityEditPipeline:
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, cfg):
            self.cfg = dict(cfg or {})
            self.cache_dir = ROOT / "results" / "_cache"

    return _Pipe(cfg)


def test_measure_head_direction_hint_disabled(monkeypatch):
    pipe = _pipe({"head_direction_prompt_hint": False})
    body = Image.new("RGB", (200, 200))
    face = FaceBox(50, 50, 100, 100, 0.9)
    hint, diag = pipe._measure_head_direction_hint(body, face)
    assert hint == ""
    assert diag["reason"] == "disabled"


def test_measure_head_direction_hint_no_face():
    pipe = _pipe({"head_direction_prompt_hint": True})
    body = Image.new("RGB", (200, 200))
    hint, diag = pipe._measure_head_direction_hint(body, None)
    assert hint == ""
    assert diag["reason"] == "no_selected_face"


def test_measure_head_direction_hint_non_insightface_backend_skips(monkeypatch):
    pipe = _pipe({"head_direction_prompt_hint": True})
    body = Image.new("RGB", (200, 200))
    face = FaceBox(50, 50, 100, 100, 0.9)
    monkeypatch.setattr(
        krea2_mod,
        "get_face_landmarks5",
        lambda rgb, cache_dir, prefer_box=None: (np.zeros((5, 2)), "box_prior", "insightface_unavailable"),
    )
    hint, diag = pipe._measure_head_direction_hint(body, face)
    assert hint == ""
    assert diag["reason"] == "insightface_unavailable"


def test_measure_head_direction_hint_success(monkeypatch):
    pipe = _pipe({"head_direction_prompt_hint": True})
    body = Image.new("RGB", (200, 200))
    face = FaceBox(50, 50, 100, 100, 0.9)
    lm = _landmarks(eye_a=(80, 90), eye_b=(120, 110), nose=(100, 130))
    monkeypatch.setattr(
        krea2_mod,
        "get_face_landmarks5",
        lambda rgb, cache_dir, prefer_box=None: (lm, "insightface", None),
    )
    hint, diag = pipe._measure_head_direction_hint(body, face)
    assert hint != ""
    assert diag["applied"] is True
    assert diag["landmarks_backend"] == "insightface"


# ---------------------------------------------------------------------------
# Prompt wiring: the hint must reach the final prompt text in both the
# single_person_parity (default) and non-SPP branches.
# ---------------------------------------------------------------------------


def test_prompt_includes_direction_hint_spp_default():
    pipe = _pipe({"prompt": "base prompt.", "single_person_parity": True})
    out = pipe._prompt_for_edit(
        use_tight=False, multi_person=False, direction_hint="the head is turned"
    )
    assert "the head is turned" in out
    assert "base prompt." in out


def test_prompt_includes_direction_hint_non_spp():
    pipe = _pipe({"prompt": "base prompt.", "single_person_parity": False})
    out = pipe._prompt_for_edit(
        use_tight=False, multi_person=True, direction_hint="tilt detected"
    )
    assert "tilt detected" in out


def test_prompt_direction_hint_can_be_disabled():
    pipe = _pipe(
        {
            "prompt": "base prompt.",
            "single_person_parity": True,
            "head_direction_prompt_hint": False,
        }
    )
    out = pipe._prompt_for_edit(
        use_tight=False, multi_person=False, direction_hint="should not appear"
    )
    assert "should not appear" not in out


def test_prompt_empty_hint_is_a_no_op():
    pipe = _pipe({"prompt": "base prompt.", "single_person_parity": True})
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False, direction_hint="")
    assert out.strip().startswith("base prompt.")
