"""Soft blending helpers — keep non-face pixels identical to the original."""
from __future__ import annotations

import cv2
import numpy as np

from headswap.inswap.engines.base import DetectedFace


def face_ellipse_mask(
    shape_hw: tuple[int, int],
    face: DetectedFace,
    *,
    expand: float = 0.15,
    blur_frac: float = 0.12,
) -> np.ndarray:
    """Soft elliptical mask around the face bbox (uint8, full image)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = face.bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx = 0.5 * (x1 - x0) * (1.0 + expand)
    ry = 0.5 * (y1 - y0) * (1.0 + expand * 1.15)
    axes = (max(1, int(round(rx))), max(1, int(round(ry))))
    center = (int(round(cx)), int(round(cy)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    k = max(3, int(round(blur_frac * max(axes))) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def soft_blend(
    original_bgr: np.ndarray,
    swapped_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Alpha-composite ``swapped`` over ``original`` using ``mask_u8``.

    Outside the mask, output equals ``original`` bit-for-bit (within float roundtrip
    of uint8) when strength=1 and mask is zero there.
    """
    if original_bgr.shape != swapped_bgr.shape:
        raise ValueError(
            f"shape mismatch original={original_bgr.shape} swapped={swapped_bgr.shape}"
        )
    a = (mask_u8.astype(np.float32) / 255.0) * float(np.clip(strength, 0.0, 1.0))
    a = a[..., None]
    out = original_bgr.astype(np.float32) * (1.0 - a) + swapped_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def color_match_face(
    swapped_bgr: np.ndarray,
    original_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    strength: float = 0.35,
) -> np.ndarray:
    """Mild mean/std match of swapped face to original lighting inside the mask."""
    if strength <= 0:
        return swapped_bgr
    m = mask_u8 > 32
    if not np.any(m):
        return swapped_bgr
    out = swapped_bgr.astype(np.float32).copy()
    src = original_bgr.astype(np.float32)
    for c in range(3):
        s_pix = out[..., c][m]
        t_pix = src[..., c][m]
        s_mean, s_std = float(s_pix.mean()), float(s_pix.std()) + 1e-6
        t_mean, t_std = float(t_pix.mean()), float(t_pix.std()) + 1e-6
        matched = (out[..., c] - s_mean) * (t_std / s_std) + t_mean
        out[..., c] = out[..., c] * (1.0 - strength) + matched * strength
    return np.clip(out, 0, 255).astype(np.uint8)
