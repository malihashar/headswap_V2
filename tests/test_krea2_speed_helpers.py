"""Unit tests for Krea2 software-speed helpers (no GPU required)."""
from __future__ import annotations

import builtins

from headswap.pipelines.krea2 import (
    _silence_krea2edit_step_prints,
    get_shared_krea2_runtime,
    reset_shared_krea2_runtime,
)


def test_silence_filters_krea2edit_prefix(capsys):
    with _silence_krea2edit_step_prints():
        print("[krea2edit] STRIDE1-POS fit: should be hidden", flush=True)
        print("[krea2 sampling] visible", flush=True)
    captured = capsys.readouterr().out
    assert "[krea2edit]" not in captured
    assert "[krea2 sampling] visible" in captured
    # Restored after context
    print("[krea2edit] after restore", flush=True)
    assert "[krea2edit] after restore" in capsys.readouterr().out


def test_silence_restores_builtins_print():
    original = builtins.print
    with _silence_krea2edit_step_prints():
        assert builtins.print is not original
    assert builtins.print is original


def test_shared_runtime_singleton(monkeypatch):
    reset_shared_krea2_runtime()
    calls = {"n": 0}

    class FakeRT:
        def __init__(self, *a, **k):
            calls["n"] += 1

    monkeypatch.setattr(
        "headswap.pipelines.krea2.NodeRuntime", FakeRT
    )
    a = get_shared_krea2_runtime(init_custom_nodes=True)
    b = get_shared_krea2_runtime(init_custom_nodes=True)
    assert a is b
    assert calls["n"] == 1
    reset_shared_krea2_runtime()
