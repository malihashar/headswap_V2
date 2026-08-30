"""rembg must reuse one session, not rebuild ~1GB of weights per call.

Measured on Colab: _resolve_body_route took 23.6s of a 23.8s pre-dispatch
stage, while lighting_route -- InsightFace on the same image -- took 0.3s.
rembg's remove() builds a NEW session when none is passed, reloading its
~1GB ONNX model every call and ignoring our provider preference, and the
matte is computed more than once per swap.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEG = (ROOT / "src" / "headswap" / "segmentation.py").read_text()


def test_call_sites_go_through_the_cached_helper():
    assert "_rembg_remove_cached(" in SEG
    # No bare remove() left on the matte paths.
    assert "rgba = rembg_remove(body_pil" not in SEG


def test_session_is_module_level_cached():
    assert "_REMBG_SESSION" in SEG
    assert "global _REMBG_SESSION" in SEG


def test_session_uses_the_shared_provider_preference():
    """A cached session on the CPU would fix the reload and still leave the
    matte slow, so it must request CUDA when available."""
    assert "preferred_onnx_providers" in SEG


def test_failure_falls_back_instead_of_breaking_the_swap():
    """rembg missing or unhappy must degrade, never fail a render -- the
    same contract tests/test_head_matte_mask.py already relies on."""
    i = SEG.find("def _rembg_remove_cached")
    body = SEG[i:i + 400]
    assert "if sess is not None:" in body
    assert "return rembg_remove(img)" in body


def test_negative_result_is_not_retried_forever():
    """If the session cannot be built, remember that instead of paying the
    failed construction on every call."""
    assert "_REMBG_SESSION_TRIED" in SEG
