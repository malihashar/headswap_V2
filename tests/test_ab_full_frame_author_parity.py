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
    """The DEFAULT multi-person mode is still crop_stitch (full_frame is
    reached via the body/lighting routes, not as the base default).

    The full_frame_* value assertions this test used to make were a stale
    A/B-era guard: full_frame_identity_lora_name / full_frame_ref_boost have
    been set (not null) since the lighting-route work, and target_mp was
    raised 1.25 -> 1.5 for face quality. Assert the invariant that actually
    matters (the base mode) plus that full_frame stays inside its own
    documented megapixel band.
    """
    cfg = yaml.safe_load(CFG_PATH.read_text())
    assert cfg["multi_person_edit_mode"] == "crop_stitch"
    assert cfg.get("preserve_expression") is True
    assert (
        cfg["full_frame_min_mp"]
        <= cfg["full_frame_target_mp"]
        <= cfg["full_frame_max_mp"]
    )


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
        "e_ff_full_rb5_gp768",
        "f_ff_full_rb4_gp1024",
        "g_ff_full_rb5_gp1024",
        "h_ff_full_rb35_noexpr",
        "i_ff_full_rb4_noexpr",
        "j_ff_full_refboostmask",
    ]
    assert mod.ARMS[0]["multi_person_edit_mode"] == "crop_stitch"
    assert all(a["multi_person_edit_mode"] == "full_frame" for a in mod.ARMS[1:])
    assert mod.ARMS[0]["identity_lora_name"].endswith("r64.safetensors")
    assert mod.ARMS[2]["identity_lora_name"] == mod.LORA_FULL
    assert mod.ARMS[3]["ref_boost"] == 4.0
    assert mod.ARMS[1]["ref_boost"] == 3.5
    # Identity-strength sweep
    e, f, g = mod.ARMS[4], mod.ARMS[5], mod.ARMS[6]
    assert e["ref_boost"] == 5.0 and e["grounding_px"] == 768
    assert f["ref_boost"] == 4.0 and f["grounding_px"] == 1024
    assert g["ref_boost"] == 5.0 and g["grounding_px"] == 1024
    # Expression-freedom variants match c/d knobs with preserve_expression off
    h, i = mod.ARMS[7], mod.ARMS[8]
    assert h["preserve_expression"] is False and h["ref_boost"] == 3.5
    assert i["preserve_expression"] is False and i["ref_boost"] == 4.0
    assert h["identity_lora_name"] == mod.LORA_FULL
    assert i["identity_lora_name"] == mod.LORA_FULL
    # Mask arm: same as d + face-localized ref_boost_mask
    j = mod.ARMS[9]
    assert j["id"] == "j_ff_full_refboostmask"
    assert j["ref_boost"] == 4.0
    assert j["identity_lora_name"] == mod.LORA_FULL
    assert j["full_frame_ref_boost_mask"] is True
    assert j["preserve_expression"] is True
    assert all(
        (not a.get("full_frame_ref_boost_mask")) for a in mod.ARMS if a is not j
    )


def test_cfg_for_arm_sets_grounding_and_expression():
    mod = _load_ab()
    base = yaml.safe_load(CFG_PATH.read_text())
    cfg_e = mod._cfg_for_arm(base, mod.ARMS[4])
    assert cfg_e["grounding_px"] == 768
    assert cfg_e["ref_boost"] == 5.0
    assert cfg_e["preserve_expression"] is True
    assert cfg_e["full_frame_ref_boost_mask"] is False
    cfg_i = mod._cfg_for_arm(base, mod.ARMS[8])
    assert cfg_i["grounding_px"] == 768
    assert cfg_i["ref_boost"] == 4.0
    assert cfg_i["preserve_expression"] is False
    assert cfg_i["multi_person_edit_mode"] == "full_frame"
    cfg_j = mod._cfg_for_arm(base, mod.ARMS[9])
    assert cfg_j["full_frame_ref_boost_mask"] is True
    assert cfg_j["ref_boost"] == 4.0
    assert cfg_j["multi_person_edit_mode"] == "full_frame"
    assert cfg_j["identity_lora_name"] == mod.LORA_FULL


def test_preserve_expression_false_strips_lock_and_allows_donor():
    pipe = Krea2IdentityEditPipeline.__new__(Krea2IdentityEditPipeline)
    pipe.cfg = {
        "preserve_expression": False,
        "prompt": (
            "Replace the head. CRITICAL: copy the facial expression from the first "
            "image exactly — if they are smiling, the result must smile the same way; "
            "if they are not smiling, do not add a smile. Keep mouth shape, "
            "smile/no-smile, eye gaze, and micro-expressions from the first image "
            "only — never from the second image. Keep the body."
        ),
        "single_person_parity": True,
    }
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False)
    assert "copy the facial expression from the first image exactly" not in out.lower()
    assert "Allow the facial expression from the second image" in out
    # Default true leaves lock intact
    pipe.cfg["preserve_expression"] = True
    locked = pipe._prompt_for_edit(use_tight=False, multi_person=False)
    assert "copy the facial expression from the first image exactly" in locked.lower()


def test_side_by_side_wraps_many_arms():
    mod = _load_ab()
    imgs = [
        (Image.new("RGB", (80, 100), (i * 20, 40, 60)), f"arm_{i}") for i in range(10)
    ]
    collage = mod._side_by_side(imgs, height=120, max_per_row=5)
    # Two rows of height 120+label (~146) + gap
    assert collage.size[1] > 200
    assert collage.size[0] > 200
