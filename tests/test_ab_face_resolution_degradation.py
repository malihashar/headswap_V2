"""Unit tests for face-resolution degradation A/B helpers (no GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import FaceBox  # noqa: E402


def _load_script():
    path = ROOT / "scripts" / "ab_face_resolution_degradation.py"
    spec = importlib.util.spec_from_file_location("ab_face_resolution_degradation", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_degrade_to_face_height_preserves_framing_and_scale():
    mod = _load_script()
    scene = Image.new("RGB", (400, 500), (30, 30, 30))
    ImageDraw.Draw(scene).ellipse([150, 100, 250, 260], fill=(200, 160, 140))
    current_h, target_h = 160.0, 55.0
    degraded, intermediate, meta = mod.degrade_to_face_height(
        scene, current_face_h=current_h, target_face_h=target_h
    )
    assert degraded.size == scene.size
    assert meta["framing_unchanged"] is True
    assert abs(meta["downsample_scale"] - (target_h / current_h)) < 1e-6
    assert abs(meta["effective_native_face_h_px"] - target_h) < 0.5
    # Intermediate width ≈ scale * original width
    assert intermediate.size[0] == int(round(400 * target_h / current_h))


def test_degrade_does_not_upscale_when_target_larger():
    mod = _load_script()
    scene = Image.new("RGB", (100, 100), (10, 10, 10))
    degraded, intermediate, meta = mod.degrade_to_face_height(
        scene, current_face_h=40.0, target_face_h=80.0
    )
    assert meta["downsample_scale"] == 1.0
    assert intermediate.size == scene.size
    assert degraded.size == scene.size


def test_laplacian_sharpness_drops_after_degrade():
    mod = _load_script()
    # Sharp synthetic face pattern
    scene = Image.new("RGB", (320, 400), (20, 20, 20))
    d = ImageDraw.Draw(scene)
    d.ellipse([100, 80, 220, 240], fill=(210, 170, 150))
    d.ellipse([130, 130, 150, 150], fill=(20, 20, 20))
    d.ellipse([170, 130, 190, 150], fill=(20, 20, 20))
    box = FaceBox(100, 80, 220, 240, 0.9)
    degraded, _, _ = mod.degrade_to_face_height(
        scene, current_face_h=160.0, target_face_h=40.0
    )
    s0 = mod.laplacian_sharpness(scene, box)
    s1 = mod.laplacian_sharpness(degraded, box)
    assert s0 is not None and s1 is not None
    assert s1 < s0 * 0.6


def test_classify_and_judge_identity_drop():
    mod = _load_script()
    original = {
        "identity_cosine": 0.90,
        "landmark_eye_error_over_iod": 0.02,
        "eye_line_delta_deg": 1.0,
        "face_detector_confidence_result": 0.85,
        "face_sharpness_laplacian_body": 200.0,
        "face_sharpness_laplacian_result": 180.0,
    }
    degraded = {
        "identity_cosine": 0.75,
        "landmark_eye_error_over_iod": 0.15,
        "eye_line_delta_deg": 10.0,
        "face_detector_confidence_result": 0.70,
        "face_sharpness_laplacian_body": 40.0,
        "face_sharpness_laplacian_result": 50.0,
    }
    modes = mod.classify_failure_modes(original, degraded)
    assert modes["identity"] == "appeared"
    assert modes["gaze"] == "appeared"
    judgment = mod.judge_hypothesis(
        prep={"single_face_h_px": 318, "multi_face_h_px": 55},
        original=original,
        degraded=degraded,
        modes=modes,
        ran_pipeline=True,
    )
    assert judgment["reproduced_multi_failure_mode"] == "yes"
    assert judgment["resolution_role"] == "primary_cause"


def test_judge_not_significant_when_identity_stable():
    mod = _load_script()
    original = {
        "identity_cosine": 0.90,
        "landmark_eye_error_over_iod": 0.02,
        "eye_line_delta_deg": 1.0,
        "face_detector_confidence_result": 0.85,
        "face_sharpness_laplacian_body": 200.0,
        "face_sharpness_laplacian_result": 180.0,
    }
    degraded = {
        "identity_cosine": 0.89,
        "landmark_eye_error_over_iod": 0.03,
        "eye_line_delta_deg": 1.5,
        "face_detector_confidence_result": 0.84,
        "face_sharpness_laplacian_body": 40.0,
        "face_sharpness_laplacian_result": 160.0,
    }
    modes = mod.classify_failure_modes(original, degraded)
    judgment = mod.judge_hypothesis(
        prep={"single_face_h_px": 318, "multi_face_h_px": 55},
        original=original,
        degraded=degraded,
        modes=modes,
        ran_pipeline=True,
    )
    assert judgment["reproduced_multi_failure_mode"] == "no"
    assert judgment["resolution_role"] == "not_significant"
