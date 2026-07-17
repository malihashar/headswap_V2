"""Regression: ComfyUI model unload cannot run under torch.inference_mode().

ComfyUI set_attr_param wraps weights in nn.Parameter. Under active
inference_mode, streamed/lowvram weights may be inference tensors and
Parameter() raises RuntimeError: Cannot set version_counter for inference tensor.
ComfyUI only auto-clones those tensors when inference_mode is already disabled.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _real_torch():
    torch = pytest.importorskip("torch")
    # Another test may have installed a MagicMock torch stub in sys.modules.
    if type(torch).__name__ == "MagicMock" or not hasattr(torch, "Tensor"):
        pytest.skip("real torch not available")
    try:
        t = torch.zeros(1)
    except Exception:
        pytest.skip("real torch not available")
    if type(t).__name__ == "MagicMock":
        pytest.skip("real torch not available")
    return torch


def test_parameter_wrap_fails_inside_inference_mode_for_inference_tensor():
    torch = _real_torch()

    with torch.inference_mode():
        w = torch.randn(4, 4)
        assert w.is_inference()
        with pytest.raises(RuntimeError, match="version_counter|inference"):
            torch.nn.Parameter(w, requires_grad=False)


def test_parameter_wrap_ok_under_no_grad():
    torch = _real_torch()

    with torch.no_grad():
        w = torch.randn(4, 4)
        assert not w.is_inference()
        p = torch.nn.Parameter(w.clone(), requires_grad=False)
        assert isinstance(p, torch.nn.Parameter)


def test_comfy_set_attr_param_clone_guard_requires_inference_mode_off():
    """Mirrors ComfyUI comfy.utils.set_attr_param clone condition."""
    torch = _real_torch()

    def set_attr_param_like(value):
        if (not torch.is_inference_mode_enabled()) and value.is_inference():
            value = value.clone()
        return torch.nn.Parameter(value, requires_grad=False)

    with torch.inference_mode():
        inf = torch.randn(2, 2)
        # Still inside inference_mode → ComfyUI does NOT clone → Parameter fails.
        with pytest.raises(RuntimeError, match="version_counter|inference"):
            set_attr_param_like(inf)

    with torch.inference_mode():
        inf = torch.randn(2, 2)
    # Outside inference_mode, clone path allows Parameter wrap.
    p = set_attr_param_like(inf)
    assert isinstance(p, torch.nn.Parameter)


def test_qwen_sample_uses_no_grad_not_inference_mode():
    from headswap.pipelines import qwen as qwen_mod

    src = inspect.getsource(qwen_mod._sample_qwen)
    assert "torch.no_grad()" in src
    assert "torch.inference_mode()" not in src


def test_klein_run_uses_no_grad_not_inference_mode():
    from headswap.pipelines import klein as klein_mod

    src = inspect.getsource(klein_mod.KleinMaskCropPipeline.run)
    assert "torch.no_grad()" in src
    assert "torch.inference_mode()" not in src
