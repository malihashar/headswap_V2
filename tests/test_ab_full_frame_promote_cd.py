"""Unit tests for full-frame promote gate + Procrustes helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_promote_arms_are_prod_c_d_only():
    mod = _load("ab_full_frame_promote_cd", "scripts/ab_full_frame_promote_cd.py")
    ids = [a["id"] for a in mod.ARMS]
    assert ids == ["prod_crop_r64_rb35", "c_ff_full_rb35", "d_ff_full_rb4"]
    assert all(not a["id"].startswith("b_") for a in mod.ARMS)


def test_recommend_prefers_d_when_close():
    mod = _load("ab_full_frame_promote_cd", "scripts/ab_full_frame_promote_cd.py")
    agg = {
        "prod_crop_r64_rb35": {
            "head_to_body_scale_ratio": 1.25,
            "identity_cosine": 0.70,
            "latency_s": 100.0,
            "vram_peak_mb": 10000.0,
        },
        "c_ff_full_rb35": {
            "head_to_body_scale_ratio": 1.05,
            "identity_cosine": 0.88,
            "latency_s": 140.0,
            "vram_peak_mb": 12000.0,
        },
        "d_ff_full_rb4": {
            "head_to_body_scale_ratio": 1.06,  # within CLOSE_HEAD_SCALE of c
            "identity_cosine": 0.875,
            "latency_s": 142.0,
            "vram_peak_mb": 12100.0,
        },
    }
    rec = mod._recommend(agg)
    assert rec["winner_arm"] == "d_ff_full_rb4"
    assert rec["winner_ref_boost"] == 4.0
    assert rec["need_procrustes"] is False  # |1.06-1| < 0.08
    assert rec["proposed_production_diff"]["status"].startswith("PENDING")


def test_recommend_procrustes_when_head_scale_off():
    mod = _load("ab_full_frame_promote_cd", "scripts/ab_full_frame_promote_cd.py")
    agg = {
        "prod_crop_r64_rb35": {"head_to_body_scale_ratio": 1.40, "identity_cosine": 0.7},
        "c_ff_full_rb35": {"head_to_body_scale_ratio": 1.20, "identity_cosine": 0.85},
        "d_ff_full_rb4": {"head_to_body_scale_ratio": 1.18, "identity_cosine": 0.86},
    }
    rec = mod._recommend(agg)
    assert rec["winner_arm"] == "d_ff_full_rb4"
    assert rec["need_procrustes"] is True
    assert rec["proposed_production_diff"]["full_frame_procrustes_align"] is True


def test_procrustes_align_identity_transform():
    from headswap.preprocess import FaceBox, procrustes_align_generated_to_body

    # Synthetic: same image → near-identity transform (or skip if no face).
    im = Image.new("RGB", (256, 256), (40, 40, 40))
    d = ImageDraw.Draw(im)
    # Rough face oval + eyes (detector may or may not fire; function must not crash).
    d.ellipse((80, 70, 176, 190), fill=(210, 170, 140))
    d.ellipse((100, 110, 118, 122), fill=(20, 20, 20))
    d.ellipse((138, 110, 156, 122), fill=(20, 20, 20))
    cache = ROOT / ".cache" / "headswap_v2"
    cache.mkdir(parents=True, exist_ok=True)
    out, info = procrustes_align_generated_to_body(im, im, cache)
    assert out.size == im.size
    assert "procrustes" in info
    # Either succeeded or skipped cleanly with a reason.
    assert info["procrustes"] is True or info.get("procrustes_reason")


def test_yaml_still_crop_stitch_default():
    from headswap.config import load_config

    cfg = load_config(ROOT / "configs" / "krea2_identity_edit.yaml")
    assert cfg["multi_person_edit_mode"] == "crop_stitch"
    assert cfg.get("full_frame_procrustes_align") is False
    assert "r64" in str(cfg.get("identity_lora_name") or "")
