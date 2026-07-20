from __future__ import annotations

from typing import Any

from headswap.config import load_config
from headswap.pipelines.base import BasePipeline, MockHeadSwapPipeline
from headswap.pipelines.klein import KleinMaskCropPipeline
from headswap.pipelines.kontext import FluxKontextPipeline
from headswap.pipelines.qwen import QwenBaselinePipeline, QwenImprovedPipeline


PIPELINES: dict[str, type[BasePipeline]] = {
    "klein": KleinMaskCropPipeline,
    "klein4b_mask_crop_stitch": KleinMaskCropPipeline,
    "flux_kontext": FluxKontextPipeline,
    "qwen_baseline": QwenBaselinePipeline,
    "qwen_improved": QwenImprovedPipeline,
    "qwen_improved_mask_crop": QwenImprovedPipeline,
    "mock": MockHeadSwapPipeline,
}


def create_pipeline(cfg: dict[str, Any], runtime=None, force_mock: bool = False) -> BasePipeline:
    if force_mock or cfg.get("force_mock"):
        return MockHeadSwapPipeline(cfg, runtime=runtime)
    key = str(cfg.get("pipeline", cfg.get("name", "mock")))
    cls = PIPELINES.get(key)
    if cls is None:
        raise KeyError(f"Unknown pipeline '{key}'. Known: {sorted(PIPELINES)}")
    return cls(cfg, runtime=runtime)


def create_pipeline_from_config(path: str, force_mock: bool = False) -> BasePipeline:
    return create_pipeline(load_config(path), force_mock=force_mock)
