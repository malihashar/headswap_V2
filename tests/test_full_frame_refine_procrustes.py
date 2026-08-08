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


# ---------------------------------------------------------------------------
# Mask/face-box unification fix (docs/full_frame_failure_analysis.md §2.1,
# §8). Two independently-detected, independently-shaped masks (pass 1's
# freeze mask vs pass 2's old re-detect + crop_stitch-default mask) was the
# highest-confidence explanation for the persistent ghost/halo artifact.
# These tests lock in: (a) pass 1's face box is reused, not re-detected;
# (b) the mask is built with full_frame_mask_* params, not _tight_crop_flags'
# crop_stitch defaults; (c) color_ref targets pristine body_full, not `out`.
# ---------------------------------------------------------------------------


def test_refine_reuses_pass1_face_box_instead_of_redetecting(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": False})
    w, h = 1000, 1600
    pass1_box = (430, 120, 570, 260)
    pass1_face = FaceBox(*pass1_box, 0.95)
    out = _synthetic_full_frame_output(w, h, pass1_box)
    _stub_sample_edit(monkeypatch)

    # If the function re-detects on `out`, it will call select_face_box and
    # get THIS (deliberately different) box instead of pass1_face.
    redetect_box = (100, 100, 140, 140)
    redetect_face = FaceBox(*redetect_box, 0.5)
    detect_calls = []

    def _tracking_select_face_box(rgb, cache_dir, index=0, policy="largest"):
        detect_calls.append(True)
        return redetect_face, [redetect_face]

    monkeypatch.setattr(krea2_mod, "select_face_box", _tracking_select_face_box)

    built_calls = []
    orig_build_scene_person = Krea2IdentityEditPipeline._build_scene_person

    def _tracking_build_scene_person(self, body_full, face_crop, selected_face, **kwargs):
        built_calls.append(selected_face)
        return orig_build_scene_person(self, body_full, face_crop, selected_face, **kwargs)

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_build_scene_person", _tracking_build_scene_person)

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
        selected_face=pass1_face,
        all_faces=[pass1_face],
    )

    assert diag["reused_pass1_face_box"] is True
    assert not detect_calls, "should not re-detect when selected_face is supplied"
    assert len(built_calls) == 1
    assert built_calls[0] is pass1_face


def test_refine_falls_back_to_redetection_when_no_face_box_supplied(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": False})
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
        # selected_face/all_faces omitted -- must fall back to re-detection.
    )

    assert diag["reused_pass1_face_box"] is False
    assert diag.get("reason") != "no_face_on_full_frame_output"


def test_refine_mask_uses_full_frame_pass1_params_not_crop_stitch_defaults(monkeypatch):
    """The refine mask must be built with full_frame_mask_* (pass 1's shape),
    not _tight_crop_flags'/crop_stitch's mask_* defaults (a different shape:
    side_extend 0.60 vs 0.42, expand_px 18 vs 12, blur_px 12 vs 24)."""
    pipe = _pipe(
        {
            "full_frame_face_refine": True,
            "full_frame_face_refine_procrustes": False,
            # Deliberately distinct values so a mix-up is unambiguous.
            "full_frame_mask_top_extend": 1.5,
            "full_frame_mask_side_extend": 0.42,
            "full_frame_mask_bot_extend": 0.40,
            "full_frame_mask_expand_px": 12,
            "full_frame_mask_blur_px": 24,
            "mask_top_extend": 1.55,
            "mask_side_extend": 0.60,
            "mask_bot_extend": 0.40,
            "mask_expand_px": 18,
            "mask_blur_px": 12,
        }
    )
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    seen_kwargs = {}
    orig_build_scene_person = Krea2IdentityEditPipeline._build_scene_person

    def _capture_build_scene_person(self, body_full, face_crop, selected_face, **kwargs):
        seen_kwargs.update(kwargs)
        return orig_build_scene_person(self, body_full, face_crop, selected_face, **kwargs)

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_build_scene_person", _capture_build_scene_person)

    pipe._refine_full_frame_face(
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
        selected_face=face,
        all_faces=[face],
    )

    assert seen_kwargs["top_ext"] == 1.5
    assert seen_kwargs["side_ext"] == 0.42
    assert seen_kwargs["bot_ext"] == 0.40
    assert seen_kwargs["expand_px"] == 12
    assert seen_kwargs["blur_px"] == 24


def test_refine_color_ref_targets_pristine_body_full_not_out(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True, "full_frame_face_refine_procrustes": False})
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    out = _synthetic_full_frame_output(w, h, face_box)
    pristine_body_full = Image.new("RGB", (w, h), (5, 5, 5))
    _stub_sample_edit(monkeypatch)

    seen_color_refs = []
    orig_stitch_edited = Krea2IdentityEditPipeline._stitch_edited

    def _capture_stitch_edited(self, canvas, edited, mask, box, crop_content_box, *, color_ref, **kwargs):
        seen_color_refs.append(color_ref)
        return orig_stitch_edited(self, canvas, edited, mask, box, crop_content_box, color_ref=color_ref, **kwargs)

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_stitch_edited", _capture_stitch_edited)

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
        selected_face=face,
        all_faces=[face],
    )

    assert len(seen_color_refs) == 1
    assert seen_color_refs[0] is pristine_body_full
