"""face_refine is skipped when the face already fills much of the frame.

The pass exists to recover identity detail when the face is SMALL: it
re-renders a head crop at full resolution and composites it back. On a bust
shot there is nothing to recover, because the main pass already rendered the
face at high resolution.

Measured A/B on a bust shot (face 42.0% of frame), same seed and sampling:
refine ON 76s, refine OFF 53s, outputs visually indistinguishable. ~30% of
wall-clock for no change. Skipping also removes a composite -- and therefore a
boundary -- from exactly the frames where the face is large enough for any
misalignment to be conspicuous.

Full-body frames (face ~8%) sit far below the threshold and keep the refine,
which is where it earns its cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_gate_is_unreachable_by_default():
    """T4 includes refine ON, so the default must never skip it.

    The gate was added on an A/B that read as "visually indistinguishable" in
    a small side-by-side grid. At full size that judgement did not hold -- the
    refined result was preferred and the skip rejected. The 30% saving is real
    but not free, so it is opt-in.
    """
    assert 'self.cfg.get("simple_full_body_refine_max_face_frac", 1.01)' in KREA2


def test_opt_in_value_matches_the_existing_bust_shot_gate():
    """0.25 is the documented opt-in, matching restore's bust-shot threshold."""
    assert 'self.cfg.get("simple_full_body_restore_max_face_frac", 0.25)' in KREA2
    assert "Set simple_full_body_refine_max_face_frac to" in KREA2


def test_detection_failure_returns_none_not_a_plausible_number():
    """Regression: detect_best_face falls back to a CENTRE BOX.

    On a flat grey image it returns height 160/384 = 0.417 -- above the
    bust-shot threshold. A gate built on it would silently skip the refine on
    every frame where detection failed, while looking like a real measurement.
    detect_faces reports real detections only.
    """
    frac = Krea2IdentityEditPipeline._face_frac_of_frame(
        Image.new("RGB", (256, 384), (40, 40, 40)), None, ROOT / "results" / "_cache"
    )
    assert frac is None


def test_helper_never_raises():
    assert (
        Krea2IdentityEditPipeline._face_frac_of_frame(None, None, None) is None
    )


def test_gate_does_not_use_the_fallback_detector():
    assert "detect_faces(pil_to_rgb_np(body_full), cache_dir)" in KREA2, (
        "detect_best_face falls back to a centre box and must not gate this"
    )


def test_gate_only_skips_above_the_threshold():
    """Guard the comparison direction: a small face must still be refined."""
    assert "_rf_frac is not None and _rf_frac > _rf_max" in KREA2
