"""Tests for Krea2 edit-forward static cache installer."""
from __future__ import annotations

import types

from headswap.comfy import krea2_edit_fast as kef


def test_install_without_module_is_safe(monkeypatch):
    kef._INSTALLED = False
    monkeypatch.setattr(kef.sys, "modules", {"dummy": types.ModuleType("dummy")})
    info = kef.install_krea2_edit_static_cache()
    assert info.get("installed") is False
    assert info.get("error") == "krea2_edit_forward_not_loaded"


def test_install_patches_forward(monkeypatch):
    kef._INSTALLED = False
    mod = types.ModuleType("fake_krea2edit")

    def fake_forward(*a, **k):
        return "orig"

    mod.krea2_edit_forward = fake_forward
    mod._fit_src = lambda *a, **k: None
    mod._to_4d = lambda x: x
    mod._imgids = lambda *a, **k: None
    mod._imgids_offset = lambda *a, **k: None
    mod._ref_attn_bias = lambda *a, **k: None
    mod.Krea2EditModelPatch = object

    monkeypatch.setattr(kef.sys, "modules", {"fake_krea2edit": mod})
    info = kef.install_krea2_edit_static_cache()
    assert info["installed"] is True
    assert mod.krea2_edit_forward is not fake_forward
    assert getattr(mod, "_headswap_edit_cache") is True

    # Second install is idempotent
    info2 = kef.install_krea2_edit_static_cache()
    assert info2.get("already") is True
