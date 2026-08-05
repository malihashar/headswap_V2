"""Magic Hour API integrations (optional backends)."""

from headswap.magichour.face_detection import (
    MagicHourFaceDetectionClient,
    MagicHourFaceDetectionError,
    MagicHourFaceDetectionResult,
)

__all__ = [
    "MagicHourFaceDetectionClient",
    "MagicHourFaceDetectionError",
    "MagicHourFaceDetectionResult",
]
