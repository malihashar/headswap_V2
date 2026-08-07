"""Tests for the post-generation head-direction relock pass (no GPU).

relock_pose_to_destination itself needs real face landmarks (InsightFace),
so it's stubbed here -- these tests cover the wiring: config gating, the
no-face fallback, and that the tight face-only mask (not full head+hair) is
what gets passed through to the relock call.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

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


def test_relock_disabled_is_a_no_op():
    pipe = _pipe({"head_direction_relock": False})
    out = Image.new("RGB", (200, 200), (10, 10, 10))
    dest = Image.new("RGB", (200, 200), (20, 20, 20))
    face = FaceBox(50, 50, 100, 100, 0.9)
    result, diag = pipe._relock_head_direction(out, dest, face)
    assert result is out
    assert diag["pose_relock"] is False
    assert diag["pose_relock_reason"] == "disabled"


def test_relock_no_face_falls_back():
    pipe = _pipe({"head_direction_relock": True})
    out = Image.new("RGB", (200, 200), (10, 10, 10))
    dest = Image.new("RGB", (200, 200), (20, 20, 20))
    result, diag = pipe._relock_head_direction(out, dest, None)
    assert result is out
    assert diag["pose_relock"] is False
    assert diag["pose_relock_reason"] == "no_selected_face"


def test_relock_calls_relock_pose_to_destination_with_safe_defaults(monkeypatch):
    pipe = _pipe({"head_direction_relock": True})
    out = Image.new("RGB", (200, 200), (10, 10, 10))
    dest = Image.new("RGB", (200, 200), (20, 20, 20))
    face = FaceBox(80, 60, 130, 120, 0.9)

    calls = []
    sentinel = Image.new("RGB", (200, 200), (99, 99, 99))

    def _fake_relock(generated, destination, cache_dir, *, face_mask, use_full_affine, core_min_alpha, feather_px, stitch_feather_px):
        calls.append(
            {
                "generated_size": generated.size,
                "destination_size": destination.size,
                "face_mask_size": face_mask.size,
                "use_full_affine": use_full_affine,
            }
        )
        return sentinel, {"pose_relock": True, "rotation_deg": 4.2, "scale": 1.01}

    monkeypatch.setattr(krea2_mod, "relock_pose_to_destination", _fake_relock)

    result, diag = pipe._relock_head_direction(out, dest, face)

    assert result is sentinel
    assert diag["pose_relock"] is True
    assert len(calls) == 1
    # Safe default: similarity transform only, no shear/stretch.
    assert calls[0]["use_full_affine"] is False
    assert calls[0]["face_mask_size"] == out.size


def test_relock_full_affine_opt_in(monkeypatch):
    pipe = _pipe({"head_direction_relock": True, "head_direction_relock_full_affine": True})
    out = Image.new("RGB", (150, 150), (10, 10, 10))
    dest = Image.new("RGB", (150, 150), (20, 20, 20))
    face = FaceBox(40, 40, 90, 90, 0.9)

    calls = []

    def _fake_relock(generated, destination, cache_dir, *, face_mask, use_full_affine, **kwargs):
        calls.append(use_full_affine)
        return generated, {"pose_relock": True}

    monkeypatch.setattr(krea2_mod, "relock_pose_to_destination", _fake_relock)
    pipe._relock_head_direction(out, dest, face)
    assert calls == [True]
