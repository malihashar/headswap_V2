"""Tests for the full_frame -> crop_stitch-quality face refine pass (no GPU).

_sample_edit is stubbed out (it calls the real diffusion model) but the
surrounding wiring -- face re-detection on the full_frame output, crop/mask
geometry via the real _build_scene_person, and the stitch-back via the real
_stitch_edited -- runs for real, so these tests catch wiring/geometry bugs
even without a GPU.
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
    """A drawn 'photo' with an ellipse face -- good enough for the ellipse
    mask fallback and for select_face_box once stubbed."""
    im = Image.new("RGB", (w, h), (60, 90, 60))
    d = ImageDraw.Draw(im)
    d.ellipse(list(face_box), fill=(210, 175, 150))
    d.rectangle([w * 0.3, h * 0.3, w * 0.7, h * 0.95], fill=(40, 40, 120))  # "body"
    return im


def _stub_face_detection(monkeypatch, faces, selected):
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (selected, faces)
    )


def _stub_sample_edit(monkeypatch, *, edited_image=None):
    def _fake_sample_edit(self, rt, bundle, scene, person, timings, *, prompt, edit_cache_info, seed=None, ref_boost_mask=None):
        img = edited_image if edited_image is not None else scene
        return {
            "edited": img,
            "steps": 8,
            "cfg": 1.0,
            "seed": seed,
            "ref_boost": 4.0,
            "ref_boost_a": 1.0,
            "ref_boost_mask_used": ref_boost_mask is not None,
            "grounding_px": int(self.cfg.get("grounding_px", 768)),
            "fit_mode": "fit",
            "sampling_diag": {},
            "negative_mode": "conditioning_zero_out",
            "empty_node": "EmptySD3LatentImage",
            "use_full_load": True,
            "skip_neg_vlm": True,
            "verbose": False,
        }

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_sample_edit", _fake_sample_edit)


def test_refine_disabled_is_a_no_op(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": False})
    out = _synthetic_full_frame_output()
    refined, diag = pipe._refine_full_frame_face(
        out,
        out,
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
    assert refined is out
    assert diag["applied"] is False
    assert diag["reason"] == "disabled"


def test_refine_no_face_on_output_falls_back(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True})
    _stub_face_detection(monkeypatch, [], None)
    out = _synthetic_full_frame_output()
    refined, diag = pipe._refine_full_frame_face(
        out,
        out,
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
    assert refined is out
    assert diag["applied"] is False
    assert diag["reason"] == "no_face_on_full_frame_output"


def test_refine_applies_and_reports_resolution_gain(monkeypatch):
    pipe = _pipe({"full_frame_face_refine": True, "crop_long_side": 768})
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)  # height=140px in a 1600px-tall frame
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    face_crop = out.crop((350, 60, 650, 340))

    refined, diag = pipe._refine_full_frame_face(
        out,
        face_crop,
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
    assert refined.size == out.size
    # The refine crop resolves the face at native crop_stitch resolution
    # (crop_long_side=768), which must be a meaningfully higher effective
    # face resolution than the tiny fraction full_frame rendered it at.
    assert diag["resolution_gain_x"] > 1.5
    assert diag["refine_face_height_px_effective"] > diag["full_frame_face_height_px"]
    assert diag["lora_reloaded"] is False


def test_refine_reload_lora_calls_load_models(monkeypatch):
    pipe = _pipe(
        {
            "full_frame_face_refine": True,
            "full_frame_face_refine_reload_lora": True,
            "multi_person_edit_mode": "full_frame",
        }
    )
    w, h = 1000, 1600
    face_box = (430, 120, 570, 260)
    face = FaceBox(*face_box, 0.95)
    _stub_face_detection(monkeypatch, [face], face)
    out = _synthetic_full_frame_output(w, h, face_box)
    _stub_sample_edit(monkeypatch)

    calls = []

    def _fake_load_models(self, rt, timings):
        calls.append(dict(self.cfg))
        return {"model": "crop_stitch_bundle"}

    monkeypatch.setattr(Krea2IdentityEditPipeline, "_load_models", _fake_load_models)

    refined, diag = pipe._refine_full_frame_face(
        out,
        out,
        rt=None,
        bundle={"model": "full_frame_bundle"},
        edit_cache_info={},
        div_by=16,
        timings={},
        base_seed=46,
        prompt="x",
        out_dir=None,
        stitch_debug={},
    )

    assert len(calls) == 1
    # multi_person_edit_mode was flipped to crop_stitch for the reload call.
    assert calls[0]["multi_person_edit_mode"] == "crop_stitch"
    # ...and restored afterward.
    assert pipe.cfg["multi_person_edit_mode"] == "full_frame"
    assert diag["lora_reloaded"] is True
