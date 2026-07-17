from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.comfy.full_load import force_sampling_full_load, offload_gpu_models


def _install_fake_comfy(prepare_sampling):
    """Install minimal comfy stubs so force_sampling_full_load can patch them."""
    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.get_torch_device = lambda: "cuda:0"
    # before free, after free, before unload-after, after unload-after
    fake_mm.get_free_memory = MagicMock(side_effect=[5e9, 14e9, 12e6, 13e9])
    fake_mm.free_memory = MagicMock(return_value=[])
    fake_mm.unload_all_models = MagicMock()
    fake_mm.soft_empty_cache = MagicMock()

    fake_sh = types.ModuleType("comfy.sampler_helpers")
    fake_sh.prepare_sampling = prepare_sampling

    fake_comfy = types.ModuleType("comfy")
    saved = {
        k: sys.modules.get(k)
        for k in ("comfy", "comfy.model_management", "comfy.sampler_helpers")
    }
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_management"] = fake_mm
    sys.modules["comfy.sampler_helpers"] = fake_sh
    return fake_mm, fake_sh, saved


def _restore_modules(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_force_sampling_full_load_passes_force_full_load_true():
    calls = []

    def fake_prepare(
        model,
        noise_shape,
        conds,
        model_options=None,
        force_full_load=False,
        force_offload=False,
    ):
        calls.append({"force_full_load": force_full_load, "force_offload": force_offload})
        return ("model", conds, [])

    fake_mm, fake_sh, saved = _install_fake_comfy(fake_prepare)
    try:
        with force_sampling_full_load() as info:
            assert info["enabled"] is True
            assert info["force_full_load"] is True
            assert info["freed_before_sample"] is True
            fake_mm.unload_all_models.assert_called()
            # Caller requests False; patch must override to True.
            fake_sh.prepare_sampling("m", (1, 16, 72, 57), {}, force_full_load=False)

        assert calls and calls[0]["force_full_load"] is True
        # Patch restored on exit
        assert fake_sh.prepare_sampling is fake_prepare
        # UNet must be offloaded after sampling for VAE decode headroom.
        assert info["freed_after_sample"] is True
        assert fake_mm.unload_all_models.call_count == 2
        assert fake_mm.soft_empty_cache.call_count == 2
    finally:
        _restore_modules(saved)


def test_offload_gpu_models_falls_back_to_free_memory():
    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.get_torch_device = lambda: "cuda:0"
    fake_mm.get_free_memory = MagicMock(side_effect=[1e7, 12e9])
    fake_mm.free_memory = MagicMock(return_value=["unet"])
    # No unload_all_models attribute → fallback path.
    fake_mm.soft_empty_cache = MagicMock()

    fake_comfy = types.ModuleType("comfy")
    saved = {
        k: sys.modules.get(k)
        for k in ("comfy", "comfy.model_management", "comfy.sampler_helpers")
    }
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_management"] = fake_mm
    try:
        info = offload_gpu_models(reason="test_fallback")
        assert info["ok"] is True
        assert info["api"] == "free_memory"
        fake_mm.free_memory.assert_called_once_with(1e30, "cuda:0")
        fake_mm.soft_empty_cache.assert_called_once()
    finally:
        _restore_modules(saved)


def test_force_sampling_full_load_without_comfy_yields_safely():
    saved = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "comfy" or k.startswith("comfy.")
    }
    for k in list(saved):
        del sys.modules[k]
    sys.modules["comfy"] = None  # type: ignore[assignment]
    try:
        with force_sampling_full_load() as info:
            assert info["enabled"] is False
            assert info.get("error")
    finally:
        for k in list(sys.modules):
            if k == "comfy" or k.startswith("comfy."):
                del sys.modules[k]
        for k, v in saved.items():
            sys.modules[k] = v
