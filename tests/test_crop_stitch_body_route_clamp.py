"""Tests for the crop_stitch body-route scale/position clamp.

Architecture change: single-person full-body photos no longer route to
full_frame (whose whole-scene generation + freeze composite repeatedly
caused ghosting/hand-duplication/scale artifacts across many rounds). They
stay on crop_stitch instead, with this new clamp as the scale/position
safety net -- crop_stitch's own clamp_edited_head_scale call is gated
behind single_person_parity and never fires for a single-subject photo.
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
    def _fake(rgb, cache_dir, conf_thresh=0.30):
        mean = float(rgb.mean())
        for target_mean, box in boxes_by_mean:
            if abs(mean - target_mean) < 1.0:
                return box
        return None

    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake)


def test_disabled_by_default_is_a_no_op():
    pipe = _pipe({})  # crop_stitch_clamp_head_scale not set -> False
    body_full = Image.new("RGB", (200, 200), (10, 10, 10))
    out = Image.new("RGB", (200, 200), (200, 200, 200))
    result, info = pipe._maybe_clamp_crop_stitch_head_scale(body_full, out)
    assert result is out
    assert info is None


def test_enabled_by_body_route_clamps_oversized_stitched_head(monkeypatch):
    pipe = _pipe({"crop_stitch_clamp_head_scale": True})
    w, h = 300, 300
    body_full = Image.new("RGB", (w, h), (10, 10, 10))
    out = Image.new("RGB", (w, h), (200, 200, 200))
    d = ImageDraw.Draw(out)
    d.ellipse([120, 40, 220, 160], fill=(220, 180, 150))
    out_mean = float(np.asarray(out).mean())

    target_face = FaceBox(80, 80, 160, 160, 0.9)  # height 80
    gen_face = FaceBox(120, 40, 220, 160, 0.9)  # height 120 -> ratio 1.5
    _stub_detect_best_face(monkeypatch, [(10.0, target_face), (out_mean, gen_face)])

    result, info = pipe._maybe_clamp_crop_stitch_head_scale(body_full, out)

    assert info["clamped"] == 1.0
    assert info["ratio_before"] == 1.5
    assert info["ratio_after"] < info["ratio_before"]
    assert not np.array_equal(np.asarray(result), np.asarray(out))


def test_resolve_body_route_flag_actually_activates_the_clamp(monkeypatch):
    """End-to-end: _resolve_body_route flags a single-person full-body photo,
    and _maybe_clamp_crop_stitch_head_scale honors that flag."""
    import headswap.pipelines.krea2 as krea2_mod
    from headswap.preprocess import FaceBox as FB

    pipe = _pipe({"enable_body_route": True, "body_route_use_segmentation": False, "max_body_dim": 2000})
    w, h = 992, 1600
    face = FB(*[430, 60, 550, 210], 0.95)
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (face, [face])
    )
    body = Image.new("RGB", (w, h))
    pipe._resolve_body_route(body)

    assert pipe.cfg.get("crop_stitch_clamp_head_scale") is True


def test_resolve_body_route_raises_max_body_dim_for_more_native_face_detail(monkeypatch):
    """crop_long_side=768 (what Krea2 actually samples at) is independent of
    max_body_dim, so raising it for the full-body route costs zero extra
    inference time -- it just gives the crop more native pixels to draw from
    before being downsampled to 768. Should not raise below the configured
    body_route_max_body_dim ceiling, and should never LOWER an already-higher
    user-configured value."""
    import headswap.pipelines.krea2 as krea2_mod
    from headswap.preprocess import FaceBox as FB

    face = FB(*[430, 60, 550, 210], 0.95)
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (face, [face])
    )
    w, h = 992, 1600
    body = Image.new("RGB", (w, h))

    pipe_default = _pipe({"enable_body_route": True, "body_route_use_segmentation": False})
    meta = pipe_default._resolve_body_route(body)
    assert pipe_default.cfg["max_body_dim"] == 2048
    assert meta["max_body_dim_raised"] == [1024, 2048]

    pipe_already_high = _pipe(
        {"enable_body_route": True, "body_route_use_segmentation": False, "max_body_dim": 3000}
    )
    meta2 = pipe_already_high._resolve_body_route(body)
    assert pipe_already_high.cfg["max_body_dim"] == 3000  # never lowered
    assert "max_body_dim_raised" not in meta2


def test_resolve_body_route_scales_mask_feather_with_max_body_dim(monkeypatch):
    """mask_expand_px/mask_blur_px/stitch_feather_px are fixed-pixel values
    applied at body_full's native resolution (never downsampled to
    crop_long_side=768). Raising max_body_dim without scaling them shrinks
    the soft-blend zone's fraction of the crop, risking a harder-edged
    jaw/neck seam -- they must scale by the same ratio."""
    import headswap.pipelines.krea2 as krea2_mod
    from headswap.preprocess import FaceBox as FB

    face = FB(*[430, 60, 550, 210], 0.95)
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (face, [face])
    )
    w, h = 992, 1600
    body = Image.new("RGB", (w, h))

    pipe = _pipe(
        {
            "enable_body_route": True,
            "body_route_use_segmentation": False,
            "mask_expand_px": 18,
            "mask_blur_px": 12,
            "stitch_feather_px": 10,
        }
    )
    meta = pipe._resolve_body_route(body)

    # max_body_dim 1024 -> 2048 is a 2x ratio.
    assert pipe.cfg["mask_expand_px"] == 36
    assert pipe.cfg["mask_blur_px"] == 24
    assert pipe.cfg["stitch_feather_px"] == 20
    assert meta["mask_feather_scaled_by"] == 2.0


def test_scale_match_extends_to_body_route_without_touching_multi_person_gate():
    """identity_scale_match's donor-face scale cue was already gated on for
    multi-person/isolate_selected; this must extend it to the single-person
    full-body route (crop_stitch_clamp_head_scale) without loosening the
    gate for ordinary portrait crop_stitch (flag unset -> resize_contain,
    same as tests/test_spp_strict_architecture.py's existing contract)."""
    from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
    from headswap.preprocess import FaceBox as FB

    class _P(Krea2IdentityEditPipeline):
        def __init__(self, **cfg):
            self.cfg = {
                "single_person_parity": True,
                "mask_top_extend": 1.55,
                "mask_side_extend": 0.60,
                "mask_bot_extend": 0.40,
                "mask_expand_px": 18,
                "mask_blur_px": 4,
                "crop_long_side": 256,
                "person_match_crop_size": True,
                "identity_scale_match": True,
                "square_crop": False,
                "head_mask_backend": "ellipse",
                **cfg,
            }
            self.cache_dir = ROOT / "results" / "_cache"

    body = Image.new("RGB", (300, 400), (25, 25, 25))
    ImageDraw.Draw(body).ellipse([100, 60, 200, 200], fill=(190, 150, 130))
    face = FB(100, 60, 200, 200, 0.9)
    donor = Image.new("RGB", (96, 120), (10, 10, 10))

    ordinary = _P()  # crop_stitch_clamp_head_scale unset -> False
    ordinary_built = ordinary._build_scene_person(
        body, donor, face, div_by=8, use_tight=False,
        top_ext=1.55, side_ext=0.60, bot_ext=0.40, expand_px=18, crop_pad=12,
        all_faces=[face], isolate_selected=False,
    )
    assert ordinary_built["diag"]["identity_scale_match"] is False
    assert ordinary_built["diag"]["person_prep"] == "resize_contain"

    body_routed = _P(crop_stitch_clamp_head_scale=True)
    routed_built = body_routed._build_scene_person(
        body, donor, face, div_by=8, use_tight=False,
        top_ext=1.55, side_ext=0.60, bot_ext=0.40, expand_px=18, crop_pad=12,
        all_faces=[face], isolate_selected=False,
    )
    assert routed_built["diag"]["identity_scale_match"] is True
    assert routed_built["diag"]["person_prep"] == "place_face_at_height_frac"
