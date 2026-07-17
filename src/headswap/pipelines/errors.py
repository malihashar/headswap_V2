from __future__ import annotations

from typing import Any

from PIL import Image

from headswap.pipelines.base import PipelineResult


class PipelineRunError(Exception):
    """Raised after partial profile emission when pipe.run fails mid-flight."""

    def __init__(
        self,
        message: str,
        *,
        meta: dict[str, Any] | None = None,
        latency_s: float = 0.0,
        image: Image.Image | None = None,
        debug_paths: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.meta = meta or {}
        self.latency_s = latency_s
        self.image = image
        self.debug_paths = debug_paths or {}

    def to_partial_result(self) -> PipelineResult | None:
        if self.image is None:
            return None
        return PipelineResult(
            image=self.image,
            latency_s=self.latency_s,
            meta={**self.meta, "run_error": str(self)},
            debug_paths=self.debug_paths,
        )
