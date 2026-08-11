"""Full-frame similarity head-scale clamp (post-transform soft composite).

Locks the contract that replaces the local-box path:
  - warps the ENTIRE frame (no pre-sized crop box → no clipped hair)
  - builds the compositing mask AFTER the warp
  - soft-composites over the whole frame
  - returns the feathered mask for debug dump
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


def test_full_frame_clamp_returns_mask_and_shrinks_head(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    body_full, out, _ = _build_images()
    result, info, mask = clamp_edited_head_scale_full_frame(
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
    assert mask.size == result.size == out.size
    assert mask.mode == "L"
    # Mask must be soft (not a hard 0/255 rectangle).
    arr = np.asarray(mask)
    assert int(arr.min()) == 0
    assert int(arr.max()) >= 250  # Gaussian can drop peak slightly below 255
    mid = int(((arr > 0) & (arr < 255)).sum())
    assert mid > 100, "feathered mask should have soft edge pixels"


def test_full_frame_clamp_restores_far_pixels_via_mask(monkeypatch):
    """Outside the head mask, original-frame content must win — so the bottom
    hem (identical in body_full and out) stays pixel-identical after clamp,
    even though the whole-frame warp moved it. This is the soft-sky / far-
    content invariant the local-box path tried (and failed) to guarantee by
    never touching those pixels."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    body_full, out, hem_box = _build_images()
    result, info, mask = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_feather_px=16,
    )
    assert info["clamped"] == 1.0
    assert mask is not None

    x0, y0, x1, y1 = hem_box
    # Hem is far below the head — mask must be ~0 there.
    mask_hem = np.asarray(mask.crop((x0, y0, x1, y1)))
    assert float(mask_hem.mean()) < 1.0, "head mask leaked into far hem region"

    result_hem = np.asarray(result.crop((x0, y0, x1, y1)))
    body_hem = np.asarray(body_full.crop((x0, y0, x1, y1)))
    assert np.array_equal(result_hem, body_hem)


def test_no_pre_sized_box_clips_content(monkeypatch):
    """Hair extending past any local-box size must survive: the warp is applied
    to the full canvas, so a tall generated 'hair' spike above the face stays
    present inside the soft mask rather than being cropped away."""
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    w, h = 600, 900
    body_full = Image.new("RGB", (w, h), BG_COLOR)
    out = Image.new("RGB", (w, h), BG_COLOR)
    # Face mid-frame; a tall hair spike reaching near the top edge.
    ImageDraw.Draw(body_full).ellipse([250, 300, 350, 420], fill=TGT_COLOR)
    ImageDraw.Draw(out).ellipse([230, 250, 370, 470], fill=GEN_COLOR)
    ImageDraw.Draw(out).rectangle([280, 40, 320, 260], fill=GEN_COLOR)

    result, info, mask = clamp_edited_head_scale_full_frame(
        body_full,
        out,
        cache_dir=ROOT / "results" / "_cache",
        mask_top_extend=2.0,
        mask_feather_px=8,
    )
    assert info["clamped"] == 1.0
    assert mask is not None
    # Some of the (now transformed) hair spike must remain non-sky in result
    # inside the upper half of the frame — proving we didn't clip to a box.
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


def test_pipeline_flag_on_fires_and_dumps_mask(monkeypatch, tmp_path):
    monkeypatch.setattr(krea2_mod, "detect_best_face", _fake_detect_best_face)
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "crop_stitch_full_frame_head_clamp": True,
                "clamp_edited_head_scale": True,
                "full_frame_head_clamp_mask_feather_px": 16,
            }
            self.cache_dir = ROOT / "results" / "_cache"

    body_full, out, _ = _build_images()
    result, info = _Pipe()._maybe_clamp_crop_stitch_head_scale_full_frame(
        body_full, out, out_dir=tmp_path, save_debug=True
    )
    assert info is not None
    assert info["clamped"] == 1.0
    assert not np.array_equal(np.asarray(result), np.asarray(out))
    dumps = list(tmp_path.glob("full_frame_head_clamp_mask_*.png"))
    assert len(dumps) == 1
    dumped = Image.open(dumps[0])
    assert dumped.size == out.size
    assert dumped.mode in ("L", "RGB", "RGBA")
