"""Tests for Krea2 edit-forward static cache installer."""
from __future__ import annotations

import types

from headswap.comfy import krea2_edit_fast as kef


def setup_function():
    kef._INSTALLED = False
    kef._PATCHED_MODULE = None
    kef._STATIC_CACHE["key"] = None
    kef._STATIC_CACHE["payload"] = None


def test_install_without_module_is_safe(monkeypatch):
    monkeypatch.setattr(kef.sys, "modules", {"dummy": types.ModuleType("dummy")})
    info = kef.install_krea2_edit_static_cache()
    assert info.get("installed") is False
    assert info.get("error") == "krea2_edit_forward_not_loaded"


def test_install_patches_forward(monkeypatch):
    mod = types.ModuleType("comfyui_krea2edit")
    mod.__file__ = "/tmp/comfyui-krea2edit/__init__.py"

    def fake_forward(*a, **k):
        return "orig"

    mod.krea2_edit_forward = fake_forward
    mod._fit_src = lambda *a, **k: None
    mod._to_4d = lambda x: x
    mod._imgids = lambda *a, **k: None
    mod._imgids_offset = lambda *a, **k: None
    mod._ref_attn_bias = lambda *a, **k: None
    mod.Krea2EditModelPatch = object

    monkeypatch.setattr(kef.sys, "modules", {"comfyui_krea2edit": mod})
    info = kef.install_krea2_edit_static_cache()
    assert info["installed"] is True
    assert mod.krea2_edit_forward is not fake_forward
    assert getattr(mod, "_headswap_edit_cache") is True

    kef._STATIC_CACHE["key"] = "x"
    kef._STATIC_CACHE["payload"] = {"y": 1}
    kef.clear_krea2_edit_static_cache()
    assert kef._STATIC_CACHE["key"] is None
    assert kef._STATIC_CACHE["payload"] is None

    info2 = kef.install_krea2_edit_static_cache()
    assert info2.get("already") is True


def test_clear_does_not_probe_torch_classes(monkeypatch):
    """Regression: never getattr(__closure__) on random sys.modules entries."""

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError(f"Tried to instantiate class 'krea2_edit_forward.{name}'")

    bad = types.ModuleType("torch_classes_trap")
    bad.krea2_edit_forward = Boom()
    monkeypatch.setattr(kef.sys, "modules", {"torch_classes_trap": bad})
    # Must not raise
    kef.clear_krea2_edit_static_cache()
