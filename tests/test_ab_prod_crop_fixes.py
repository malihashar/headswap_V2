"""Tests for prod crop fixes A/B (Procrustes + wide crop) — no GPU."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox, expand_crop_box_wide


CFG_PATH = ROOT / "configs" / "krea2_identity_edit.yaml"


def _load_ab():
    path = ROOT / "scripts" / "ab_prod_crop_fixes.py"
    spec = importlib.util.spec_from_file_location("ab_prod_crop_fixes", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_yaml_defaults_stay_production_safe():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    assert cfg["multi_person_edit_mode"] == "crop_stitch"
    assert cfg.get("crop_margin_mode", "tight") == "tight"
    # Demoted from production default 2026-08-09: GPU-measured on the
    # hat/full-body case, even an in-bounds correction (all sanity gates
    # passing) produced a bright halo hugging the head against the sky --
    # isolated via an edge-energy probe to the face-only ellipse this
    # correction blends through (1.25x elevated Laplacian energy vs the
    # surrounding sky). See the comment above enable_procrustes_correction
    # in the yaml for the full measurement and how the hat-silhouette
    # hypothesis was ruled out first.
    assert cfg.get("enable_procrustes_correction") is True


def test_arms_are_prod_procrustes_wide_only():
    mod = _load_ab()
    ids = [a["id"] for a in mod.ARMS]
    assert ids == ["prod", "prod_procrustes", "prod_wide_crop"]
    assert all(a["multi_person_edit_mode"] == "crop_stitch" for a in mod.ARMS)
    assert mod.ARMS[0]["enable_procrustes_correction"] is False
    assert mod.ARMS[0]["crop_margin_mode"] == "tight"
    assert mod.ARMS[1]["enable_procrustes_correction"] is True
    assert mod.ARMS[1]["crop_margin_mode"] == "tight"
    assert mod.ARMS[2]["crop_margin_mode"] == "wide"
    assert mod.ARMS[2]["enable_procrustes_correction"] is False


def test_cfg_for_arm_isolates_knobs():
    mod = _load_ab()
    base = yaml.safe_load(CFG_PATH.read_text())
    cfg_p = mod._cfg_for_arm(base, mod.ARMS[1])
    assert cfg_p["enable_procrustes_correction"] is True
    assert cfg_p["crop_margin_mode"] == "tight"
    assert cfg_p["multi_person_edit_mode"] == "crop_stitch"
    cfg_w = mod._cfg_for_arm(base, mod.ARMS[2])
    assert cfg_w["crop_margin_mode"] == "wide"
    assert cfg_w["enable_procrustes_correction"] is False


def test_expand_crop_box_wide_larger_than_face():
    face = FaceBox(100, 80, 180, 180, 0.9)
    box = expand_crop_box_wide((800, 600), face)
    assert box[2] - box[0] > face.width
    assert box[3] - box[1] > face.height
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 800 and box[3] <= 600


def test_wide_mode_reports_larger_scene_than_tight():
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, margin: str):
            self.cfg = {
                "crop_margin_mode": margin,
                "crop_long_side": 768,
                "wide_crop_target_mp": 1.0,
                "wide_crop_min_mp": 0.85,
                "wide_crop_max_mp": 1.15,
                "wide_crop_max_dim": 1536,
                "wide_crop_allow_upscale": True,
                "wide_crop_pad_top_frac": 1.2,
                "wide_crop_pad_side_frac": 1.8,
                "wide_crop_pad_bot_frac": 2.5,
                "mask_blur_px": 12,
                "head_mask_backend": "ellipse",
                "person_match_crop_size": True,
                "identity_scale_match": False,
                "clamp_crop_away_neighbors": True,
                "neighbor_crop_margin_frac": 0.18,
                "single_person_parity": True,
            }
            self.cache_dir = ROOT / ".cache" / "headswap_v2"

    body = Image.new("RGB", (1600, 1200), (30, 30, 30))
    d = ImageDraw.Draw(body)
    d.ellipse([700, 200, 900, 450], fill=(200, 160, 140))
    face = FaceBox(700, 200, 900, 450, 0.95)
    donor = Image.new("RGB", (128, 160), (10, 10, 10))
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)

    tight = _Pipe("tight")._build_scene_person(
        body,
        donor,
        face,
        div_by=16,
        use_tight=False,
        top_ext=1.55,
        side_ext=0.60,
        bot_ext=0.40,
        expand_px=18,
        crop_pad=14,
        all_faces=[face],
    )
    wide = _Pipe("wide")._build_scene_person(
        body,
        donor,
        face,
        div_by=16,
        use_tight=False,
        top_ext=1.55,
        side_ext=0.60,
        bot_ext=0.40,
        expand_px=18,
        crop_pad=14,
        all_faces=[face],
    )
    assert tight["diag"]["crop_margin_mode"] == "tight"
    assert wide["diag"]["crop_margin_mode"] == "wide"
    # Wide native crop area should exceed tight head crop.
    t_box = tight["diag"]["crop_box"]
    w_box = wide["diag"]["crop_box"]
    t_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
    w_area = (w_box[2] - w_box[0]) * (w_box[3] - w_box[1])
    assert w_area > t_area
    # Scene should stay cropped (not full body frame).
    assert wide["scene"].size[0] < body.size[0] or wide["scene"].size[1] < body.size[1]
    # Wide scene should land near ~1MP (allow_upscale).
    assert float(wide["diag"]["scene_megapixels"]) >= 0.80
