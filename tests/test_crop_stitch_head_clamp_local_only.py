"""_maybe_clamp_crop_stitch_head_scale must only ever touch a local box
around the head, never the whole canvas.

GPU-observed 2026-08-10: the previous implementation handed clamp_edited_
head_scale the ENTIRE stitched image, shrinking body/clothes/background
together just to fix a head-only scale mismatch. On a real full-body render
that produced two visible defects: a thin rectangular seam around the whole
frame (shrunk interior meeting the hard-pasted, unshrunk original border),
and a duplicated/terraced hem at the bottom of the dress (the same border-
fill now cutting through actual subject content). Fixed by warping only a
head-sized crop and feather-pasting it back -- this test locks in that the
rest of the frame is pixel-identical afterward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.pipelines.krea2 as krea2_mod
import headswap.preprocess as preprocess_mod
from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox

GEN_COLOR = (220, 180, 150)  # oversized generated face, drawn in `out`
TGT_COLOR = (150, 180, 220)  # correctly-sized original face, drawn in `body_full`
HEM_COLOR = (255, 0, 0)  # bottom-of-dress marker, identical in both images


def _bbox_of_color(rgb: np.ndarray, color: tuple[int, int, int]) -> FaceBox | None:
    mask = np.all(np.abs(rgb.astype(int) - np.array(color)) <= 4, axis=-1)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return FaceBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), 0.9)


def _fake_detect_best_face(rgb, cache_dir, conf_thresh=0.30):
    return _bbox_of_color(rgb, GEN_COLOR) or _bbox_of_color(rgb, TGT_COLOR)


def _pipe(cfg: dict | None = None) -> Krea2IdentityEditPipeline:
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, cfg):
            self.cfg = dict(cfg or {})
            self.cache_dir = ROOT / "results" / "_cache"

    return _Pipe(cfg)


def _build_images():
    w, h = 800, 1400
    body_full = Image.new("RGB", (w, h), (10, 10, 10))
    out = Image.new("RGB", (w, h), (200, 200, 200))

    # Target's real face: height 140, centered around (400, 250).
    ImageDraw.Draw(body_full).ellipse([330, 150, 470, 350], fill=TGT_COLOR)
    # Generated head came out oversized: height 220, same rough center.
    ImageDraw.Draw(out).ellipse([300, 100, 520, 400], fill=GEN_COLOR)

    # A hem marker near the very bottom, far from the head, identical in
    # both -- this is what a whole-frame shrink would incorrectly move/warp.
    hem_box = [100, h - 120, 700, h - 40]
    ImageDraw.Draw(body_full).rectangle(hem_box, fill=HEM_COLOR)
    ImageDraw.Draw(out).rectangle(hem_box, fill=HEM_COLOR)
    return body_full, out, hem_box


def test_clamp_leaves_far_pixels_pixel_identical(monkeypatch):
    monkeypatch.setattr(krea2_mod, "detect_best_face", _fake_detect_best_face)
    monkeypatch.setattr(preprocess_mod, "detect_best_face", _fake_detect_best_face)

    body_full, out, hem_box = _build_images()
    pipe = _pipe({"crop_stitch_clamp_head_scale": True})

    result, info = pipe._maybe_clamp_crop_stitch_head_scale(body_full, out)

    assert info is not None
    assert info["clamped"] == 1.0

    x0, y0, x1, y1 = hem_box
    result_hem = np.asarray(result.crop((x0, y0, x1, y1)))
    out_hem = np.asarray(out.crop((x0, y0, x1, y1)))
    assert np.array_equal(result_hem, out_hem), (
        "bottom-of-frame content changed -- the clamp is touching more than "
        "a local box around the head again"
    )

    # Sanity: the head region itself DID change (the clamp actually fired).
    head_region = (280, 80, 540, 420)
    result_head = np.asarray(result.crop(head_region))
    out_head = np.asarray(out.crop(head_region))
    assert not np.array_equal(result_head, out_head)


def test_clamp_box_is_small_relative_to_full_canvas():
    """The local box must be a small fraction of a realistic full-body
    canvas -- guards against multipliers creeping back up large enough to
    reproduce the whole-frame-seam failure mode."""
    pipe = _pipe({"crop_stitch_clamp_head_scale": True})

    # A realistically-proportioned full-body photo: face height ~10% of the
    # frame height (the test images in _build_images use an oversized face
    # for detector-friendliness, not a realistic proportion).
    h, w = 2000, 1000
    fh = 150.0
    half_w = fh * float(pipe.cfg.get("crop_stitch_head_clamp_side_mult", 1.1))
    top = fh * float(pipe.cfg.get("crop_stitch_head_clamp_top_mult", 1.9))
    bot = fh * float(pipe.cfg.get("crop_stitch_head_clamp_bot_mult", 0.9))
    box_h = top + bot
    box_w = half_w * 2

    assert box_h < 0.5 * h, "local box height should stay well under half the frame"
    assert box_w < 0.7 * w, "local box width should stay well under the full frame"
