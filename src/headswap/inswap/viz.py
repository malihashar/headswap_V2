"""Visualization helpers for Colab intermediates and comparison."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from headswap.inswap.engines.base import DetectedFace


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_detections(
    image_bgr: np.ndarray,
    faces: list[DetectedFace],
    *,
    selected: DetectedFace | None = None,
) -> np.ndarray:
    """Overlay face boxes; highlight the selected face in green."""
    out = image_bgr.copy()
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = [int(round(v)) for v in f.bbox]
        is_sel = selected is not None and f is selected
        color = (0, 220, 0) if is_sel else (0, 165, 255)
        thick = 3 if is_sel else 2
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        label = f"#{i} {'SEL' if is_sel else ''}".strip()
        cv2.putText(
            out,
            label,
            (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if f.kps is not None:
            for kx, ky in f.kps:
                cv2.circle(out, (int(kx), int(ky)), 2, color, -1)
    return out


def difference_map(
    original_bgr: np.ndarray,
    result_bgr: np.ndarray,
    *,
    amplify: float = 4.0,
) -> np.ndarray:
    """Abs-diff heatmap (BGR) showing where pixels changed."""
    a = original_bgr.astype(np.float32)
    b = result_bgr.astype(np.float32)
    if a.shape != b.shape:
        b_img = cv2.resize(result_bgr, (original_bgr.shape[1], original_bgr.shape[0]))
        b = b_img.astype(np.float32)
    diff = np.mean(np.abs(a - b), axis=2)
    diff = np.clip(diff * float(amplify), 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    return heat


def side_by_side(
    original_bgr: np.ndarray,
    result_bgr: np.ndarray,
    diff_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """Horizontal strip: original | swapped | difference."""
    h = original_bgr.shape[0]
    panels = [original_bgr]
    r = result_bgr
    if r.shape[0] != h:
        r = cv2.resize(r, (int(r.shape[1] * h / r.shape[0]), h))
    panels.append(r)
    if diff_bgr is not None:
        d = diff_bgr
        if d.shape[0] != h:
            d = cv2.resize(d, (int(d.shape[1] * h / d.shape[0]), h))
        panels.append(d)
    # Match heights already; pad widths independently via hstack after resize height.
    return np.hstack(panels)
