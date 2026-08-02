"""Deterministic local face-swap experiment (InsightFace InSwapper).

Completely separate from Krea2 identity-edit. Swap engines are pluggable so
REFace / GHOST / etc. can be dropped in later without rewriting the pipeline.
"""
from __future__ import annotations

from headswap.inswap.pipeline import InSwapPipeline, InSwapResult

__all__ = ["InSwapPipeline", "InSwapResult"]
