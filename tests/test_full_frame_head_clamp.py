"""Full-frame similarity head-scale clamp (post-transform soft composite).

Locks the contract that replaces the local-box path:
  - warps the ENTIRE frame (no pre-sized crop box → no clipped hair)
  - builds the compositing mask AFTER the warp
  - soft-composites over the whole frame
  - head+hair-only mask (no collar/clothing)
  - never composites empty warp-border black pixels
  - dumps mask + transformed frame for debug
  - production flag stays OFF by default
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.pipelines.krea2 as krea2_mod
import headswap.preprocess as preprocess_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox, clamp_edited_head_scale_full_frame

GEN_COLOR = (220, 180, 150)
TGT_COLOR = (150, 180, 220)
BG_COLOR = (180, 210, 240)  # smooth "sky"
HEM_COLOR = (255, 0, 0)
COLLAR_COLOR = (200, 20, 40)  # ornate "donor collar"


def _bbox_of_color(rgb: np.ndarray, color: tuple[int, int, int]) -> FaceBox | None:
    mask = np.all(np.abs(rgb.astype(int) - np.array(color)) <= 4, axis=-1)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return FaceBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), 0.9)


def _fake_detect_best_face(rgb, cache_dir, conf_thresh=0.30):
    return _bbox_of_color(rgb, GEN_COLOR) or _bbox_of_color(rgb, TGT_COLOR)


def _build_images():
    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), BG_COLOR)
    out = Image.new("RGB", (w, h), BG_COLOR)

    # Target face height 140; generated oversized at 220 — same center.
    ImageDraw.Draw(body_full).ellipse([330, 150, 470, 350], fill=TGT_COLOR)
    ImageDraw.Draw(out).ellipse([300, 100, 520, 400], fill=GEN_COLOR)

    # Far-from-head marker (bottom hem). Full-frame warp WILL move this when
    # the clamp fires — that is intentional. Soft composite then restores
    # it from original wherever the head mask is black.
    hem_box = (100, h - 120, 700, h - 40)
    ImageDraw.Draw(body_full).rectangle(list(hem_box), fill=HEM_COLOR)
    ImageDraw.Draw(out).rectangle(list(hem_box), fill=HEM_COLOR)
    return body_full, out, hem_box


def test_production_yaml_keeps_full_frame_clamp_off():
    cfg = yaml.safe_load((ROOT / "configs" / "krea2_identity_edit.yaml").read_text())
    assert cfg.get("crop_stitch_full_frame_head_clamp") is False
    assert cfg.get("crop_stitch_clamp_head_scale") is False
    # Clamp mask must be head+hair-only, not stitch extents.
    assert float(cfg.get("full_frame_head_clamp_mask_bot_extend", 0.40)) <= 0.12
    assert float(cfg.get("full_frame_head_clamp_mask_side_extend", 0.60)) <= 0.40


def test_full_frame_clamp_returns_mask_and_shrinks_head(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    body_full, out, _ = _build_images()
    result, info, mask, transformed = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        max_height_ratio=1.08,
        target_ratio=0.98,
        min_height_ratio=0.92,
        mask_feather_px=24,
    )
    assert info["clamped"] == 1.0
    assert info["shrink"] < 1.0
    assert mask is not None
    assert transformed is not None
    assert mask.size == result.size == out.size == transformed.size
    assert mask.mode == "L"
    arr = np.asarray(mask)
    assert int(arr.min()) == 0
    assert int(arr.max()) >= 250
    mid = int(((arr > 0) & (arr < 255)).sum())
    assert mid > 100, "feathered mask should have soft edge pixels"


def test_full_frame_clamp_restores_far_pixels_via_mask(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    body_full, out, hem_box = _build_images()
    result, info, mask, _ = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_feather_px=16,
    )
    assert info["clamped"] == 1.0
    assert mask is not None

    x0, y0, x1, y1 = hem_box
    mask_hem = np.asarray(mask.crop((x0, y0, x1, y1)))
    assert float(mask_hem.mean()) < 1.0, "head mask leaked into far hem region"

    result_hem = np.asarray(result.crop((x0, y0, x1, y1)))
    body_hem = np.asarray(body_full.crop((x0, y0, x1, y1)))
    assert np.array_equal(result_hem, body_hem)


def test_clamp_mask_excludes_collar_below_chin(monkeypatch):
    """bot_extend must stay tight so donor collar clothing is not composited."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), BG_COLOR)
    out = Image.new("RGB", (w, h), BG_COLOR)
    # Target face y=[200,340]; generated oversized y=[120,400].
    ImageDraw.Draw(body_full).ellipse([330, 200, 470, 340], fill=TGT_COLOR)
    ImageDraw.Draw(out).ellipse([300, 120, 520, 400], fill=GEN_COLOR)
    # Donor collar band well below the chin of the generated face.
    collar_box = (250, 430, 550, 520)
    ImageDraw.Draw(out).rectangle(list(collar_box), fill=COLLAR_COLOR)
    # Original target has plain sky there (no collar).
    ImageDraw.Draw(body_full).rectangle(list(collar_box), fill=BG_COLOR)

    result, info, mask, _ = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_bot_extend=0.08,
        mask_feather_px=8,
    )
    assert info["clamped"] == 1.0
    assert mask is not None
    collar_mask = np.asarray(mask.crop(collar_box))
    assert float(collar_mask.mean()) < 5.0, (
        f"clamp mask leaked into collar region (mean={collar_mask.mean():.1f})"
    )
    result_collar = np.asarray(result.crop(collar_box))
    # Should match original (sky), not red collar.
    assert not np.any(np.all(np.abs(result_collar.astype(int) - np.array(COLLAR_COLOR)) <= 8, axis=-1))


def test_no_black_from_warp_border_in_composite(monkeypatch):
    """Mask ∩ coverage must prevent BORDER_CONSTANT black from compositing."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 600, 900
    body_full = Image.new("RGB", (w, h), BG_COLOR)
    out = Image.new("RGB", (w, h), BG_COLOR)
    # Oversized head near the top so a shrink leaves black border at y≈0.
    ImageDraw.Draw(body_full).ellipse([250, 280, 350, 400], fill=TGT_COLOR)
    ImageDraw.Draw(out).ellipse([200, 20, 400, 320], fill=GEN_COLOR)

    result, info, mask, transformed = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_top_extend=2.0,
        mask_feather_px=16,
    )
    assert info["clamped"] == 1.0
    assert mask is not None and transformed is not None
    # Transformed frame should have some near-black border pixels at top.
    top_strip = np.asarray(transformed.crop((0, 0, w, 30)))
    # Result top strip must not turn solid black where mask was applied over
    # empty warp — should stay near sky/original.
    result_top = np.asarray(result.crop((0, 0, w, 30)))
    black = np.all(result_top <= 8, axis=-1)
    assert int(black.sum()) < 50, (
        f"composite leaked {black.sum()} solid-black pixels at crown"
    )


def test_no_pre_sized_box_clips_content(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 600, 900
    body_full = Image.new("RGB", (w, h), BG_COLOR)
    out = Image.new("RGB", (w, h), BG_COLOR)
    ImageDraw.Draw(body_full).ellipse([250, 300, 350, 420], fill=TGT_COLOR)
    ImageDraw.Draw(out).ellipse([230, 250, 370, 470], fill=GEN_COLOR)
    ImageDraw.Draw(out).rectangle([280, 40, 320, 260], fill=GEN_COLOR)

    result, info, mask, _ = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_top_extend=2.0,
        mask_feather_px=8,
    )
    assert info["clamped"] == 1.0
    assert mask is not None
    upper = np.asarray(result.crop((0, 0, w, h // 2)))
    sky = np.array(BG_COLOR)
    non_sky = np.any(np.abs(upper.astype(int) - sky) > 8, axis=-1).sum()
    assert non_sky > 50


def test_pipeline_flag_off_is_noop(monkeypatch):
    monkeypatch.setattr(krea2_mod, "detect_best_face", _fake_detect_best_face)
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {"crop_stitch_full_frame_head_clamp": False}
            self.cache_dir = ROOT / "results" / "_cache"

    body_full, out, _ = _build_images()
    result, info = _Pipe()._maybe_clamp_crop_stitch_head_scale_full_frame(body_full, out)
    assert info is None
    assert np.array_equal(np.asarray(result), np.asarray(out))


def test_pipeline_flag_on_dumps_mask_and_transformed(monkeypatch, tmp_path):
    monkeypatch.setattr(krea2_mod, "detect_best_face", _fake_detect_best_face)
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "crop_stitch_full_frame_head_clamp": True,
                "clamp_edited_head_scale": True,
                "full_frame_head_clamp_mask_feather_px": 16,
                "full_frame_head_clamp_mask_bot_extend": 0.08,
            }
            self.cache_dir = ROOT / "results" / "_cache"

    body_full, out, _ = _build_images()
    result, info = _Pipe()._maybe_clamp_crop_stitch_head_scale_full_frame(
        body_full, out, out_dir=tmp_path, save_debug=True
    )
    assert info is not None
    assert info["clamped"] == 1.0
    assert not np.array_equal(np.asarray(result), np.asarray(out))
    masks = list(tmp_path.glob("full_frame_head_clamp_mask_*.png"))
    xforms = list(tmp_path.glob("full_frame_head_clamp_transformed_*.png"))
    assert len(masks) == 1
    assert len(xforms) == 1
    assert Image.open(masks[0]).size == out.size
    assert Image.open(xforms[0]).size == out.size
