from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.comfy.full_load import force_sampling_full_load, offload_gpu_models


class _FakeLoadedModel:
    def __init__(self, patcher):
        self._patcher = patcher
        self.unload_calls = 0

    @property
    def model(self):
        return self._patcher

    def is_dead(self):
        return False

    def model_loaded_memory(self):
        return 11_000_000_000

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        self.unload_calls += 1
        self._patcher.detach(unpatch_weights)
        return True


class _FakePatcher:
    def __init__(self):
        self.detach_calls = 0

    def detach(self, unpatch_all=True):
        self.detach_calls += 1
        return self

    def is_clone(self, other):
        return other is self


def _install_fake_comfy(prepare_sampling, loaded=None):
    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.get_torch_device = lambda: "cuda:0"
    fake_mm.get_free_memory = MagicMock(side_effect=[5e9, 14e9, 12e6, 13e9, 13e9, 13e9])
    fake_mm.free_memory = MagicMock(return_value=[])
    fake_mm.unload_all_models = MagicMock()
    fake_mm.unload_model_and_clones = MagicMock()
    fake_mm.soft_empty_cache = MagicMock()
    fake_mm.current_loaded_models = list(loaded or [])

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

    patcher = _FakePatcher()
    loaded = _FakeLoadedModel(patcher)
    fake_mm, fake_sh, saved = _install_fake_comfy(fake_prepare, loaded=[loaded])
    try:
        with force_sampling_full_load(models=(patcher,)) as info:
            assert info["enabled"] is True
            assert info["force_full_load"] is True
            assert info["freed_before_sample"] is True
            fake_sh.prepare_sampling("m", (1, 16, 72, 57), {}, force_full_load=False)

        assert calls and calls[0]["force_full_load"] is True
        assert fake_sh.prepare_sampling is fake_prepare
        assert info["freed_after_sample"] is True
        # Sampling patcher must be detached even if also registry-unloaded.
        assert patcher.detach_calls >= 1
        assert loaded.unload_calls >= 1
        fake_mm.unload_model_and_clones.assert_called()
        fake_mm.unload_all_models.assert_called()
        fake_mm.soft_empty_cache.assert_called()
        # After forced unload, registry should no longer hold the UNet.
        assert loaded not in fake_mm.current_loaded_models
    finally:
        _restore_modules(saved)


def test_offload_detaches_patcher_not_in_current_loaded_models():
    """Regression: unload_all_models no-ops when registry is empty but patcher holds GPU weights."""
    patcher = _FakePatcher()
    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.get_torch_device = lambda: "cuda:0"
    fake_mm.get_free_memory = MagicMock(return_value=14e9)
    fake_mm.free_memory = MagicMock(return_value=[])
    fake_mm.unload_all_models = MagicMock()
    fake_mm.unload_model_and_clones = MagicMock()
    fake_mm.soft_empty_cache = MagicMock()
    fake_mm.current_loaded_models = []  # empty registry — previous bug path

    fake_comfy = types.ModuleType("comfy")
    saved = {
        k: sys.modules.get(k)
        for k in ("comfy", "comfy.model_management", "comfy.sampler_helpers")
    }
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_management"] = fake_mm
    try:
        info = offload_gpu_models(reason="after_sample", patchers=(patcher,))
        assert info["ok"] is True
        assert patcher.detach_calls == 1
        fake_mm.unload_model_and_clones.assert_called_once_with(patcher)
        fake_mm.unload_all_models.assert_called_once()
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
