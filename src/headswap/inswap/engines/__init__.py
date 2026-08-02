"""Engine registry — swap models without rewriting the pipeline."""
from __future__ import annotations

from typing import Callable

from headswap.inswap.engines.base import SwapEngine
from headswap.inswap.engines.inswapper import InSwapperEngine

ENGINE_REGISTRY: dict[str, Callable[[], SwapEngine]] = {
    "inswapper": InSwapperEngine,
    # Future drop-ins (register when implemented):
    # "reface": REFaceEngine,
    # "ghost": GhostEngine,
}


def create_engine(name: str = "inswapper") -> SwapEngine:
    key = (name or "inswapper").strip().lower()
    factory = ENGINE_REGISTRY.get(key)
    if factory is None:
        known = ", ".join(sorted(ENGINE_REGISTRY))
        raise KeyError(f"Unknown swap engine '{name}'. Known: {known}")
    return factory()


__all__ = [
    "ENGINE_REGISTRY",
    "InSwapperEngine",
    "SwapEngine",
    "create_engine",
]
