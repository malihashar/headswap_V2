"""Pluggable face-swap engine interface.

Implementations must only modify the selected face region of ``target_bgr``.
They must not regenerate the full scene.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetectedFace:
    """Normalized face detection used across engines."""

    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    kps: np.ndarray | None = None  # (5, 2) or None
    det_score: float = 0.0
    embedding: np.ndarray | None = None
    raw: Any = None  # backend-specific object (e.g. insightface Face)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def center_y(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])


@dataclass
class SwapEngineResult:
    image_bgr: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional face crop before paste (engine-native resolution)
    swapped_face_bgr: np.ndarray | None = None
    # Soft mask in full-image coords used for pasting (uint8 L), if any
    paste_mask: np.ndarray | None = None


class SwapEngine(ABC):
    """Abstract local face-swap model."""

    name: str = "base"

    @abstractmethod
    def load(self, cache_dir: Path, *, device: str = "cuda") -> None:
        """Download / load model weights into ``cache_dir``."""

    @abstractmethod
    def swap(
        self,
        target_bgr: np.ndarray,
        target_face: DetectedFace,
        source_face: DetectedFace,
        *,
        paste_back: bool = True,
    ) -> SwapEngineResult:
        """Replace ``target_face`` identity with ``source_face`` identity."""

    def unload(self) -> None:
        """Optional VRAM release."""
        return None
