"""Deterministic local face-swap experiment (InsightFace InSwapper).

Completely separate from standalone Krea2 identity-edit. Optional hybrid mode
runs Krea2 only as a head-refinement stage after InSwapper.
"""
from __future__ import annotations

from headswap.inswap.pipeline import InSwapPipeline, InSwapResult
from headswap.inswap.refine_krea2 import HYBRID_REFINE_PROMPT, Krea2HeadRefiner

__all__ = [
    "HYBRID_REFINE_PROMPT",
    "InSwapPipeline",
    "InSwapResult",
    "Krea2HeadRefiner",
]
