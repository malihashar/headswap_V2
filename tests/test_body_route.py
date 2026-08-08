"""Tests for full-body detection -> full_frame auto-routing (no GPU).

Covers the five scenarios called out for the body-route feature:
close-up, portrait, full-body, small-face full-body, and multi-person.
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


def _face(w: int, h: int, *, face_w: int, face_h: int, cx_frac: float, cy_frac: float) -> FaceBox:
    cx, cy = w * cx_frac, h * cy_frac
    return FaceBox(cx - face_w / 2, cy - face_h / 2, cx + face_w / 2, cy + face_h / 2, 0.95)


# ---------------------------------------------------------------------------
# _detect_full_body: pure geometry, no model calls.
# ---------------------------------------------------------------------------


def test_detect_full_body_close_up_is_not_full_body():
    pipe = _pipe()
    w, h = 800, 800
    body = Image.new("RGB", (w, h))
    # Face fills most of the frame (classic close-up crop).
    face = _face(w, h, face_w=560, face_h=640, cx_frac=0.5, cy_frac=0.48)
    result = pipe._detect_full_body(body, face, [face])
    assert result["full_body_detected"] is False
    assert result["face_height_frac"] > 0.5


def test_detect_full_body_portrait_is_not_full_body():
    pipe = _pipe()
    w, h = 900, 1200
    body = Image.new("RGB", (w, h))
    # Head-and-shoulders portrait: face is a moderate fraction of height,
    # not much frame remains below the chin.
    face = _face(w, h, face_w=320, face_h=380, cx_frac=0.5, cy_frac=0.35)
    result = pipe._detect_full_body(body, face, [face])
    assert result["full_body_detected"] is False


def test_detect_full_body_standing_figure_is_full_body():
    pipe = _pipe()
    w, h = 1000, 1600
    body = Image.new("RGB", (w, h))
    # Small face near the top, lots of frame left for torso/legs below.
    face = _face(w, h, face_w=120, face_h=150, cx_frac=0.5, cy_frac=0.12)
    result = pipe._detect_full_body(body, face, [face])
    assert result["full_body_detected"] is True


def test_detect_full_body_small_face_distant_full_body():
    pipe = _pipe()
    w, h = 1200, 1800
    body = Image.new("RGB", (w, h))
    # Distant subject: very small face, most of the frame is body below it.
    face = _face(w, h, face_w=48, face_h=60, cx_frac=0.5, cy_frac=0.08)
    result = pipe._detect_full_body(body, face, [face])
    assert result["full_body_detected"] is True
    assert result["face_height_frac"] < 0.05


def test_detect_full_body_no_face_returns_false():
    pipe = _pipe()
    body = Image.new("RGB", (400, 400))
    result = pipe._detect_full_body(body, None, [])
    assert result["full_body_detected"] is False
    assert result["method"] == "none"


def test_detect_full_body_thresholds_are_configurable():
    pipe = _pipe(
        {
            "body_route_face_height_frac_max": 0.10,
            "body_route_below_face_frac_min": 0.90,
        }
    )
    w, h = 1000, 1600
    body = Image.new("RGB", (w, h))
    face = _face(w, h, face_w=120, face_h=150, cx_frac=0.5, cy_frac=0.12)
    result = pipe._detect_full_body(body, face, [face])
    # Same geometry as the "standing figure" case, but stricter thresholds
    # (below_face_frac_min=0.80) push it back to "not detected".
    assert result["full_body_detected"] is False


# ---------------------------------------------------------------------------
# _resolve_body_route: end-to-end routing decision + cfg mutation.
# ---------------------------------------------------------------------------


def _stub_face_detection(monkeypatch, faces: list[FaceBox], selected: FaceBox | None):
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (selected, faces)
    )


def test_resolve_body_route_single_person_full_body_stays_crop_stitch(monkeypatch):
    # Architecture (see _resolve_body_route docstring): full_frame's whole-
    # scene generation + freeze-composite was the repeated source of
    # ghosting/hand-duplication/scale artifacts. Single-person full-body
    # photos now stay on crop_stitch (proven, ghost-free) with a dedicated
    # scale/position clamp safety net instead of routing to full_frame.
    pipe = _pipe(
        {
            "enable_body_route": True,
            "body_route_use_segmentation": False,
            "max_body_dim": 2000,
        }
    )
    # Dims are multiples of div_by=16 so resize_max_keep_ar is a no-op and the
    # stubbed face coords (computed against w,h) line up with the probe image.
    w, h = 992, 1600
    face = _face(w, h, face_w=120, face_h=150, cx_frac=0.5, cy_frac=0.12)
    _stub_face_detection(monkeypatch, [face], face)
    body = Image.new("RGB", (w, h))

    meta = pipe._resolve_body_route(body)

    assert meta["applied"] is True
    assert meta["full_body_detected"] is True
    assert meta["route"] == "crop_stitch_scale_clamped"
    assert meta["multi_person_edit_mode"] == "crop_stitch"
    assert "multi_person_edit_mode" not in pipe.cfg  # never forced to full_frame
    assert pipe.cfg["crop_stitch_clamp_head_scale"] is True


def test_resolve_body_route_portrait_leaves_crop_stitch_untouched(monkeypatch):
    pipe = _pipe(
        {
            "enable_body_route": True,
            "body_route_use_segmentation": False,
            "max_body_dim": 2000,
        }
    )
    w, h = 896, 1200
    face = _face(w, h, face_w=320, face_h=380, cx_frac=0.5, cy_frac=0.35)
    _stub_face_detection(monkeypatch, [face], face)
    body = Image.new("RGB", (w, h))

    meta = pipe._resolve_body_route(body)

    assert meta["applied"] is False
    assert meta["full_body_detected"] is False
    assert meta["route"] == "crop_stitch_unchanged"
    # Config is untouched -- crop_stitch is whatever it defaulted to.
    assert "multi_person_edit_mode" not in pipe.cfg


def test_resolve_body_route_disabled_is_a_no_op(monkeypatch):
    pipe = _pipe({"enable_body_route": False, "max_body_dim": 2000})
    w, h = 992, 1600
    face = _face(w, h, face_w=120, face_h=150, cx_frac=0.5, cy_frac=0.12)
    _stub_face_detection(monkeypatch, [face], face)
    body = Image.new("RGB", (w, h))

    meta = pipe._resolve_body_route(body)

    assert meta["applied"] is False
    assert meta["reason"] == "disabled"
    assert "multi_person_edit_mode" not in pipe.cfg


def test_resolve_body_route_multi_person_full_body_selected_face(monkeypatch):
    pipe = _pipe(
        {
            "enable_body_route": True,
            "body_route_use_segmentation": False,
            "max_body_dim": 2000,
        }
    )
    w, h = 1600, 1600
    # Two people; the selected (largest) face is small relative to the frame
    # with a full body below it.
    selected = _face(w, h, face_w=110, face_h=140, cx_frac=0.35, cy_frac=0.14)
    other = _face(w, h, face_w=90, face_h=115, cx_frac=0.72, cy_frac=0.16)
    _stub_face_detection(monkeypatch, [selected, other], selected)
    body = Image.new("RGB", (w, h))

    meta = pipe._resolve_body_route(body)

    assert meta["faces_detected"] == 2
    assert meta["full_body_detected"] is True
    assert meta["route"] == "full_frame"


def test_resolve_body_route_no_face_detected(monkeypatch):
    pipe = _pipe({"enable_body_route": True})
    _stub_face_detection(monkeypatch, [], None)
    body = Image.new("RGB", (400, 400))

    meta = pipe._resolve_body_route(body)

    assert meta["applied"] is False
    assert meta["reason"] == "no_face_detected"
