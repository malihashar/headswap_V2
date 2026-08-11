"""crop_stitch_pre_stitch_clamp gates single-person clamp on the edited crop.

Must never call the post-stitch local-box paste (_maybe_clamp_crop_stitch_head_scale).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline


def _pipe(cfg: dict) -> Krea2IdentityEditPipeline:
    class _P(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "single_person_parity": True,
                "clamp_edited_head_scale": True,
                "max_edited_head_height_ratio": 1.08,
                "target_edited_head_height_ratio": 0.98,
                "stitch_feather_px": 4,
                "stitch_mask_dilate_px": 0,
                "post_color_match_strength": 0.0,
                "neck_stub_tone_match": False,
                "collapse_soft_chin_ghost": False,
                "seam_refine": False,
                **cfg,
            }
            self.cache_dir = None

        def _single_person_parity(self) -> bool:
            return True

    return _P()


def test_pre_stitch_clamp_calls_clamp_edited_head_scale_for_single():
    pipe = _pipe({"crop_stitch_pre_stitch_clamp": True})
    canvas = Image.new("RGB", (64, 64), (10, 10, 10))
    edited = Image.new("RGB", (32, 32), (200, 100, 80))
    mask = Image.new("L", (32, 32), 255)
    called = {"n": 0}

    def _fake_clamp(ref, edit, *a, **k):
        called["n"] += 1
        return edit, {"clamped": 0.0, "ratio_before": 1.0, "ratio_after": 1.0, "shrink": 1.0}

    with mock.patch(
        "headswap.pipelines.krea2.clamp_edited_head_scale", side_effect=_fake_clamp
    ):
        pipe._stitch_edited(
            canvas,
            edited,
            mask,
            (16, 16, 48, 48),
            None,
            color_ref=canvas,
            multi_person=False,
            original_scene=edited,
        )
    assert called["n"] == 1


def test_pre_stitch_clamp_off_skips_for_single():
    pipe = _pipe({"crop_stitch_pre_stitch_clamp": False})
    canvas = Image.new("RGB", (64, 64), (10, 10, 10))
    edited = Image.new("RGB", (32, 32), (200, 100, 80))
    mask = Image.new("L", (32, 32), 255)
    called = {"n": 0}

    def _fake_clamp(ref, edit, *a, **k):
        called["n"] += 1
        return edit, {"clamped": 0.0}

    with mock.patch(
        "headswap.pipelines.krea2.clamp_edited_head_scale", side_effect=_fake_clamp
    ):
        pipe._stitch_edited(
            canvas,
            edited,
            mask,
            (16, 16, 48, 48),
            None,
            color_ref=canvas,
            multi_person=False,
            original_scene=edited,
        )
    assert called["n"] == 0
