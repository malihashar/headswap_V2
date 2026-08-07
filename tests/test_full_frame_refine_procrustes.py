"""Tests for the full_frame refine pass's head-direction procrustes step.

Unlike head_direction_relock (whole-image affine warp, disabled by default
after it produced a severely broken result), this reuses
procrustes_align_edited_crop_to_body_box in content-local (crop-space)
coordinates -- the same mechanism enable_procrustes_correction already uses
for crop_stitch, with built-in inlier/residual/bounds sanity gates. It can
only ever move pixels that started out inside the tight head crop, so it
cannot drag in distant content (hair, body, background) the way a
whole-image pose warp can.
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def test_procrustes_step_runs_and_is_diagnosed(monkeypatch):
    """With no real InsightFace landmarks in a synthetic image, the procrustes
    step should skip gracefully (not crash, not corrupt the result) and
    report why via diag['full_frame_face_refine']['head_direction_procrustes']."""
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": True})
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    refined, diag = pipe._refine_full_frame_face(
        out,
        out,
        body_full=out,
        rt=None,
        bundle={"model": None},
        edit_cache_info={},
        div_by=16,
        timings={},
        base_seed=46,
        prompt="x",
        out_dir=None,
        stitch_debug={},
    )

    assert diag["applied"] is True
    assert "head_direction_procrustes" in diag
    assert refined.size == out.size


def test_procrustes_step_disabled_skips_call(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": False})
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    called = []
    monkeypatch.setattr(
        krea2_mod,
        "procrustes_align_edited_crop_to_body_box",
        lambda *a, **k: called.append(True),
    )

    refined, diag = pipe._refine_full_frame_face(
        out,
        out,
        body_full=out,
        rt=None,
        bundle={"model": None},
        edit_cache_info={},
        div_by=16,
        timings={},
        base_seed=46,
        prompt="x",
        out_dir=None,
        stitch_debug={},
    )

    assert not called
    assert "head_direction_procrustes" not in diag


def test_procrustes_uses_pristine_body_full_not_out(monkeypatch):
    """The alignment target must be the pristine original, not `out` (which
    may itself carry gaze/pose drift from the first full_frame pass)."""
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": True})
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    pristine_body_full = Image.new("RGB", (w, h), (5, 5, 5))
    _stub_sample_edit(monkeypatch)

    seen_destinations = []

    def _fake_procrustes(edited, body, box, cache_dir, **kwargs):
        seen_destinations.append(body)
        return edited, {"procrustes": False, "procrustes_reason": "test_stub"}

    monkeypatch.setattr(krea2_mod, "procrustes_align_edited_crop_to_body_box", _fake_procrustes)

    pipe._refine_full_frame_face(
        out,
        out,
        body_full=pristine_body_full,
        rt=None,
        bundle={"model": None},
        edit_cache_info={},
        div_by=16,
        timings={},
        base_seed=46,
        prompt="x",
        out_dir=None,
        stitch_debug={},
    )

    assert len(seen_destinations) == 1
    assert seen_destinations[0] is pristine_body_full
