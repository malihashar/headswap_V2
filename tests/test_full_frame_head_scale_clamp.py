"""Tests for the full_frame head-scale/position clamp gate fix.

clamp_edited_head_scale (preprocess.py) already existed: it detects the
target's real face (on pristine body_full) and the generated face (on the
raw full_frame sample), and if the generated face is too tall relative to
the target's, shrinks the whole sample about the generated face's own
center and translates it so that center lands on the target's real face
center -- fixing scale AND position (both axes) in one bounded, similarity
(no rotation/shear) transform anchored to face detection.

It was wired in but gated `if multi_person and do_clamp`. full_frame was
originally multi-person-only, so that made sense; _resolve_body_route later
added a single-person full-body route to full_frame without updating this
gate, so the clamp could never fire for that (now primary) trigger case
regardless of config. These tests cover the extracted
_maybe_clamp_full_frame_head_scale wiring (disabled / no-op-when-not-needed
/ clamps-when-oversized), not clamp_edited_head_scale's own geometry (that's
a plain preprocess.py function, exercised directly here too for the "actual
correction happens" case).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.preprocess as preprocess_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox


def _pipe(cfg: dict | None = None) -> Krea2IdentityEditPipeline:
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, cfg):
            self.cfg = dict(cfg or {})
            self.cache_dir = ROOT / "results" / "_cache"

    return _Pipe(cfg)


def _stub_detect_best_face(monkeypatch, boxes_by_mean):
    """Route detect_best_face by the input image's mean pixel value, so
    body_full (one flat color) and out (a different flat color) each get
    their own stubbed FaceBox without needing real face pixels."""

    def _fake(rgb, cache_dir, conf_thresh=0.30):
        mean = float(rgb.mean())
        for target_mean, box in boxes_by_mean:
            if abs(mean - target_mean) < 1.0:
                return box
        return None

    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake)


def test_clamp_disabled_by_default_flag_is_a_no_op():
    pipe = _pipe({"full_frame_clamp_head_scale": False})
    body_full = Image.new("RGB", (200, 200), (10, 10, 10))
    out = Image.new("RGB", (200, 200), (200, 200, 200))
    result, info = pipe._maybe_clamp_full_frame_head_scale(body_full, out)
    assert result is out
    assert info is None


def test_clamp_enabled_but_not_oversized_leaves_image_unchanged(monkeypatch):
    pipe = _pipe({"full_frame_clamp_head_scale": True})
    body_full = Image.new("RGB", (200, 200), (10, 10, 10))
    out = Image.new("RGB", (200, 200), (200, 200, 200))
    target_face = FaceBox(80, 80, 120, 120, 0.9)  # height 40
    gen_face = FaceBox(80, 80, 120, 122, 0.9)  # height 42, ratio 1.05 < 1.08 threshold
    _stub_detect_best_face(monkeypatch, [(10.0, target_face), (200.0, gen_face)])

    result, info = pipe._maybe_clamp_full_frame_head_scale(body_full, out)
    assert info["clamped"] == 0.0
    assert np.array_equal(np.asarray(result), np.asarray(out))


def test_clamp_shrinks_and_recenters_oversized_generated_head(monkeypatch):
    pipe = _pipe({"full_frame_clamp_head_scale": True})
    w, h = 300, 300
    body_full = Image.new("RGB", (w, h), (10, 10, 10))

    # Generated head is both LARGER and OFF-CENTER relative to the target's
    # real face -- the clamp should fix both in one pass.
    out = Image.new("RGB", (w, h), (200, 200, 200))
    d = ImageDraw.Draw(out)
    d.ellipse([120, 40, 220, 160], fill=(220, 180, 150))  # generated face, height 120
    out_mean = float(np.asarray(out).mean())

    target_face = FaceBox(80, 80, 160, 160, 0.9)  # target's real face, height 80, center (120,120)
    gen_face = FaceBox(120, 40, 220, 160, 0.9)  # generated face, height 120 -> ratio 1.5, center (170,100)
    _stub_detect_best_face(monkeypatch, [(10.0, target_face), (out_mean, gen_face)])

    result, info = pipe._maybe_clamp_full_frame_head_scale(body_full, out)

    assert info["clamped"] == 1.0
    assert info["ratio_before"] == 1.5
    # Shrink should pull the effective ratio back down toward target_ratio (0.98).
    assert info["ratio_after"] < info["ratio_before"]
    assert not np.array_equal(np.asarray(result), np.asarray(out))


def test_full_frame_clamp_head_scale_defaults_true_in_production_config():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "krea2_identity_edit.yaml").read_text())
    assert cfg["full_frame_clamp_head_scale"] is True
