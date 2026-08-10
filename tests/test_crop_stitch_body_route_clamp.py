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
    """Content-based fake detector (matches the drawn ellipse's own color),
    not a whole-image-mean stub -- a mean-based stub is fooled by the
    border_fill change (local_out_crop, not local_body_crop) that fixed the
    hairline-leak bug, since it shifts the crop's overall mean.

    Canvas is realistically-proportioned (a small face in a tall full-body
    frame), not the previous tiny 300x300 test image: with the actual
    production box margins (side/top/bot multipliers well over 1x face
    height), a small canvas made the local box cover nearly the entire
    image, which isn't representative of a real render and made this test
    fragile to margin tuning that GPU-renders had already validated."""
    pipe = _pipe({"crop_stitch_clamp_head_scale": True})
    w, h = 1000, 2000
    gen_color = (220, 180, 150)
    tgt_color = (150, 180, 220)
    body_full = Image.new("RGB", (w, h), (10, 10, 10))
    out = Image.new("RGB", (w, h), (200, 200, 200))
    # A realistic oversized ratio (GPU-observed production values run
    # ~1.2-1.3x, never anywhere near 2x).
    cx, cy = w // 2, int(h * 0.18)
    ImageDraw.Draw(body_full).ellipse(
        [cx - 50, cy - 50, cx + 50, cy + 50], fill=tgt_color
    )  # height 100
    ImageDraw.Draw(out).ellipse(
        [cx - 65, cy - 65, cx + 65, cy + 65], fill=gen_color
    )  # height 130 -> ratio 1.3

    def _fake(rgb, cache_dir, conf_thresh=0.30):
        arr = np.asarray(rgb)
        for color in (gen_color, tgt_color):
            mask = np.all(np.abs(arr.astype(int) - np.array(color)) <= 4, axis=-1)
            ys, xs = np.where(mask)
            if ys.size:
                return FaceBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), 0.9)
        return None

    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake)
    import headswap.pipelines.krea2 as krea2_mod
    monkeypatch.setattr(krea2_mod, "detect_best_face", _fake)

    result, info = pipe._maybe_clamp_crop_stitch_head_scale(body_full, out)

    assert info["clamped"] == 1.0
    assert info["ratio_before"] == 1.3
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


def test_resolve_body_route_does_not_touch_max_body_dim(monkeypatch):
    """REVERTED: this route used to also raise max_body_dim 1024->2048 for
    more native face detail. A real GPU render showed massively oversized/
    duplicated hair, traced to two compounding bugs in how that raise scaled
    mask/feather parameters (see the docstring in _resolve_body_route and
    the commit message reverting it). Full-body crop_stitch now uses
    whatever max_body_dim was already configured, completely untouched --
    identical to the portrait path."""
    import headswap.pipelines.krea2 as krea2_mod
    from headswap.preprocess import FaceBox as FB

    face = FB(*[430, 60, 550, 210], 0.95)
    monkeypatch.setattr(
        krea2_mod, "select_face_box", lambda rgb, cache_dir, index=0, policy="largest": (face, [face])
    )
    w, h = 992, 1600
    body = Image.new("RGB", (w, h))

    pipe = _pipe({"enable_body_route": True, "body_route_use_segmentation": False})
    meta = pipe._resolve_body_route(body)
    assert "max_body_dim" not in pipe.cfg
    assert "max_body_dim_raised" not in meta


def test_resolve_body_route_does_not_touch_mask_or_feather_params(monkeypatch):
    """REVERTED: mask_expand_px/mask_blur_px/stitch_feather_px used to be
    scaled proportionally with the (now-reverted) max_body_dim raise. That
    scaling, combined with crop_pad staying fixed, hard-clipped the widened
    stitch-time blur tail at the crop box edge -- a rectangular ghost
    boundary. Full-body crop_stitch now uses these exactly as configured,
    same as an ordinary portrait."""
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

    assert pipe.cfg["mask_expand_px"] == 18
    assert pipe.cfg["mask_blur_px"] == 12
    assert pipe.cfg["stitch_feather_px"] == 10
    assert "mask_feather_scaled_by" not in meta


def test_scale_match_does_not_extend_to_body_route():
    """REVERTED: identity_scale_match's place_face_at_height_frac donor-scale
    cue was briefly extended to the full-body route (crop_stitch_clamp_
    head_scale). A real GPU render showed massively oversized/duplicated
    hair, traced to a latent formula bug: face_h_frac_native is measured
    against the DILATED+BLURRED mask bbox (already smaller than the true
    head silhouette), then identity_hair_height_boost gets applied on top of
    that against face_crop, which crop_face_reference already pads ~1.55x
    for hair -- double-counting the hair padding. multi_person/
    isolate_selected crops don't show this because clamp_crop_away_neighbors
    independently tightens their crop first, masking the same defect --
    not proven safe to reuse without separately fixing that formula.
    Full-body crop_stitch's donor reference now uses plain resize_contain,
    identical to an ordinary portrait, regardless of this flag."""
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

    for flag in (False, True):
        pipe = _P(crop_stitch_clamp_head_scale=flag)
        built = pipe._build_scene_person(
            body, donor, face, div_by=8, use_tight=False,
            top_ext=1.55, side_ext=0.60, bot_ext=0.40, expand_px=18, crop_pad=12,
            all_faces=[face], isolate_selected=False,
        )
        assert built["diag"]["identity_scale_match"] is False
        assert built["diag"]["person_prep"] == "resize_contain"
