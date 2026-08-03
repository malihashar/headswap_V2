"""Tests for full-frame megapixel targeting and A/B arm isolation (no GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline
from headswap.preprocess import FaceBox, resize_to_megapixels


CFG_PATH = ROOT / "configs" / "krea2_identity_edit.yaml"


def _load_ab():
    path = ROOT / "scripts" / "ab_full_frame_author_parity.py"
    spec = importlib.util.spec_from_file_location("ab_full_frame_author_parity", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_production_config_stays_crop_stitch():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    assert cfg["multi_person_edit_mode"] == "crop_stitch"
    assert cfg.get("full_frame_target_mp") == 1.25
    assert cfg.get("full_frame_identity_lora_name") in (None, "null")
    assert cfg.get("full_frame_ref_boost") in (None, "null")


def test_resize_to_megapixels_downscales_large_into_band():
    # ~2.36MP → ~1.25MP band, never above max_mp.
    im = Image.new("RGB", (2304, 1024), (30, 30, 30))
    out = resize_to_megapixels(im, target_mp=1.25, min_mp=1.0, max_mp=1.5, max_dim=2048)
    mp = (out.size[0] * out.size[1]) / 1_000_000.0
    assert 1.0 <= mp <= 1.55
    assert max(out.size) <= 2048


def test_resize_to_megapixels_does_not_upscale_tiny_by_default():
    im = Image.new("RGB", (398, 308), (20, 20, 20))
    out = resize_to_megapixels(im, target_mp=1.25, allow_upscale=False)
    assert out.size[0] <= 398 + 16  # evenify slack
    assert out.size[1] <= 308 + 16
    mp = (out.size[0] * out.size[1]) / 1_000_000.0
    assert mp < 0.2


def test_full_frame_inputs_report_scene_megapixels():
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {
                "full_frame_target_mp": 1.25,
                "full_frame_min_mp": 1.0,
                "full_frame_max_mp": 1.5,
                "full_frame_max_dim": 2048,
                "full_frame_allow_upscale": False,
                "full_frame_match_crop_identity": True,
                "identity_scale_match": False,
                "full_frame_freeze_outside": False,
                "full_frame_ref_boost_mask": False,
                "person_match_crop_size": True,
            }
            self.cache_dir = ROOT / "results" / "_cache"

    pipe = _Pipe()
    body = Image.new("RGB", (2304, 1024), (25, 25, 25))
    d = ImageDraw.Draw(body)
    d.ellipse([200, 200, 400, 500], fill=(200, 160, 140))
    d.ellipse([1600, 200, 1800, 500], fill=(190, 150, 130))
    faces = [FaceBox(200, 200, 400, 500, 0.9), FaceBox(1600, 200, 1800, 500, 0.9)]
    donor = Image.new("RGB", (128, 160), (10, 10, 10))
    built = pipe._build_full_frame_inputs(
        body, donor, div_by=16, selected_face=faces[0], all_faces=faces
    )
    mp = built["diag"]["scene_megapixels"]
    assert 1.0 <= float(mp) <= 1.55
    assert built["diag"]["edit_mode"] == "full_frame"
    # Entire frame, not a tight head crop.
    assert built["scene"].size[0] > 800


def test_ab_arms_isolate_variables():
    mod = _load_ab()
    ids = [a["id"] for a in mod.ARMS]
    assert ids == [
        "a_crop_r64_rb35",
        "b_ff_r64_rb35",
        "c_ff_full_rb35",
        "d_ff_full_rb4",
    ]
    assert mod.ARMS[0]["multi_person_edit_mode"] == "crop_stitch"
    assert all(a["multi_person_edit_mode"] == "full_frame" for a in mod.ARMS[1:])
    assert mod.ARMS[0]["identity_lora_name"].endswith("r64.safetensors")
    assert mod.ARMS[2]["identity_lora_name"] == mod.LORA_FULL
    assert mod.ARMS[3]["ref_boost"] == 4.0
    assert mod.ARMS[1]["ref_boost"] == 3.5
