"""Tests for the full_frame refine-before-freeze reorder.

Two independent investigations (docs/full_frame_failure_analysis.md, and a
follow-up architecture review triggered by a persisting ghost/halo after the
head-scale clamp was enabled) converged on the same finding: the refine
pass ran AFTER the single freeze composite, on the already-frozen `out`.
Even after unifying pass-1/pass-2 mask geometry, refine's own _stitch_edited
call still applied an INDEPENDENTLY-parameterized feather (10px vs the
freeze's 30px) and LAB color-match strength (0.35 vs 0.15) in the same
head/hair region -- a second seam. `_maybe_clamp_full_frame_head_scale`
(a whole-image warp) compounded this: it ran before a freeze mask that was
already fixed BEFORE the warp and never re-registered against it.

Fix: reorder so refine runs BEFORE the single freeze composite, on the
raw/clamped (not-yet-frozen) sample, and stop refine from doing its own
independent LAB match / feather radius -- there is now exactly one seam
(the freeze), one feather radius, one LAB match, computed once at the end.

These tests cover the _stitch_edited override wiring (unit-testable without
GPU) and the _refine_full_frame_face call contract; the full _run_after_models
reorder itself is exercised indirectly via test_ab_full_frame_author_parity.py
and the other full_frame test files (no regressions there).
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


def _synthetic_full_frame_output(w=1000, h=1600, face_box=(430, 120, 570, 260)) -> Image.Image:
    im = Image.new("RGB", (w, h), (60, 90, 60))
    d = ImageDraw.Draw(im)
    d.ellipse(list(face_box), fill=(210, 175, 150))
    d.rectangle([w * 0.3, h * 0.3, w * 0.7, h * 0.95], fill=(40, 40, 120))
    return im


def _stub_face_detection(monkeypatch, faces, selected):
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (selected, faces)
    )


def _stub_sample_edit(monkeypatch, edited_image=None):
    def _fake_sample_edit(self, rt, bundle, scene, person, timings, *, prompt, edit_cache_info, seed=None, ref_boost_mask=None):
        img = edited_image if edited_image is not None else scene
        return {
            "edited": img,
            "steps": 8,
            "grounding_px": int(self.cfg.get("grounding_px", 768)),
        }

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_sample_edit", _fake_sample_edit)


# --- _stitch_edited override plumbing (no full_frame-specific setup needed) ---


def test_stitch_edited_feather_override_wins_over_config_default():
    pipe = _pipe({"stitch_feather_px": 10})
    canvas = Image.new("RGB", (100, 100), (10, 10, 10))
    edited = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).ellipse([20, 20, 80, 80], fill=255)
    out = pipe._stitch_edited(
        canvas, edited, mask, (0, 0, 100, 100), None,
        color_ref=canvas, feather_px_override=30,
    )
    # No crash / sane output size is the main contract here; the override's
    # effect on alpha softness is exercised in test_stitch_alpha_ghost_fix.py.
    assert out.size == (100, 100)


def test_stitch_edited_lab_override_zero_skips_color_match():
    pipe = _pipe({"post_color_match_strength": 0.9})
    canvas = Image.new("RGB", (100, 100), (10, 10, 10))
    edited = Image.new("RGB", (100, 100), (200, 200, 200))
    mask = Image.new("L", (100, 100), 255)  # fully opaque -> pure edited color inside
    color_ref = Image.new("RGB", (100, 100), (5, 5, 5))  # very different target color

    with_lab = pipe._stitch_edited(
        canvas, edited, mask, (0, 0, 100, 100), None,
        color_ref=color_ref, post_color_match_strength_override=None,
    )
    without_lab = pipe._stitch_edited(
        canvas, edited, mask, (0, 0, 100, 100), None,
        color_ref=color_ref, post_color_match_strength_override=0.0,
    )
    center = (50, 50)
    with_lab_px = np.asarray(with_lab)[center]
    without_lab_px = np.asarray(without_lab)[center]
    edited_px = np.asarray(edited)[center]
    # LAB-off result should stay much closer to the raw edited pixel than
    # the strongly-LAB-matched (toward a very different color_ref) result.
    assert abs(int(without_lab_px[0]) - int(edited_px[0])) < abs(int(with_lab_px[0]) - int(edited_px[0]))


# --- _refine_full_frame_face now passes the override kwargs ---


def test_refine_full_frame_face_stitch_call_skips_lab_and_uses_freeze_feather(monkeypatch):
    pipe = _pipe(
        {
            "full_frame_face_refine": True,
            "full_frame_face_refine_procrustes": False,
            "full_frame_freeze_feather_px": 30,
        }
    )
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    captured = {}
    real_stitch = Krea2IdentityEditPipeline._stitch_edited

    def _spy_stitch(self, *args, **kwargs):
        captured.update(kwargs)
        return real_stitch(self, *args, **kwargs)

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_stitch_edited", _spy_stitch)

    pipe._refine_full_frame_face(
        out, out, body_full=out, rt=None, bundle={"model": None},
        edit_cache_info={}, div_by=16, timings={}, base_seed=46, prompt="x",
        out_dir=None, stitch_debug={},
    )

    assert captured.get("post_color_match_strength_override") == 0.0
    assert captured.get("feather_px_override") == 30
