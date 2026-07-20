from __future__ import annotations

from typing import Any

from headswap.config import load_config
from headswap.pipelines.base import BasePipeline, MockHeadSwapPipeline
from headswap.pipelines.klein import KleinMaskCropPipeline
from headswap.pipelines.kontext import FluxKontextPipeline
from headswap.pipelines.omnigen2 import OmniGen2PipelineRunner
from headswap.pipelines.qwen import QwenBaselinePipeline, QwenImprovedPipeline

# Optional experimental pipelines — missing modules must not break imports.
try:
    from headswap.pipelines.step1x_edit import Step1XEditPipeline
except ImportError:  # pragma: no cover
    Step1XEditPipeline = None  # type: ignore[misc, assignment]


PIPELINES: dict[str, type[BasePipeline]] = {
    "klein": KleinMaskCropPipeline,
    "klein4b_mask_crop_stitch": KleinMaskCropPipeline,
    "flux_kontext": FluxKontextPipeline,
    "qwen_baseline": QwenBaselinePipeline,
    "qwen_improved": QwenImprovedPipeline,
    "qwen_improved_mask_crop": QwenImprovedPipeline,
    "omnigen2": OmniGen2PipelineRunner,
    "mock": MockHeadSwapPipeline,
}
if Step1XEditPipeline is not None:
    PIPELINES["step1x_edit"] = Step1XEditPipeline


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
