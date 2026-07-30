"""Head/hair/neck mask backends for SPP-CC (ellipse | sam2 | birefnet).

Segmentation is optional: missing deps or failures fall back to the portable
ellipse prior so Colab/CPU environments keep working.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from headswap.preprocess import FaceBox, head_hair_mask_from_face, pil_to_rgb_np


def _ellipse_mask(
    body_pil: Image.Image,
    cache_dir,
    *,
    face_box: FaceBox | None,
    expand_px: int,
    blur_px: int,
    top_extend: float,
    side_extend: float,
    bot_extend: float,
) -> tuple[Image.Image, dict[str, Any]]:
    mask = head_hair_mask_from_face(
        body_pil,
        cache_dir,
        expand_px=expand_px,
        blur_px=blur_px,
        top_extend=top_extend,
        side_extend=side_extend,
        bot_extend=bot_extend,
        face_box=face_box,
    )
    return mask, {"backend": "ellipse", "fallback_reason": None}


def _try_sam2_mask(
    body_pil: Image.Image,
    face_box: FaceBox | None,
    *,
    blur_px: int,
) -> tuple[Image.Image | None, str | None]:
    """Box-prompted SAM2 / segment-anything. Returns (mask, skip_reason)."""
    if face_box is None:
        return None, "sam2_no_face_box"
    try:
        # Prefer SAM2 if installed; else classic segment_anything.
        predictor = None
        backend_name = "sam2"
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            # Caller must have set checkpoint via env or default HF id; we only
            # use an already-constructible predictor API if a global exists.
            # Without a configured checkpoint this path is skipped.
            _ = SAM2ImagePredictor
            return None, "sam2_predictor_not_configured"
        except Exception:
            backend_name = "segment_anything"
            try:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore
            except Exception as exc:
                return None, f"sam_import_failed:{exc}"

            # Without a local checkpoint we cannot load weights portably.
            return None, f"{backend_name}_checkpoint_not_configured"
    except Exception as exc:
        return None, f"sam2_failed:{exc}"


def _try_birefnet_mask(
    body_pil: Image.Image,
    face_box: FaceBox | None,
    *,
    blur_px: int,
) -> tuple[Image.Image | None, str | None]:
    """
    BiRefNet / rembg person matting cropped to the selected face region.

    Prefer rembg (widely available). Restrict the matte to an expanded face box
    so other people in the frame are not included in the stitch mask.
    """
    rgb = pil_to_rgb_np(body_pil)
    h, w = rgb.shape[:2]
    try:
        from rembg import remove as rembg_remove  # type: ignore
    except Exception as exc:
        # Optional transformers BiRefNet path
        try:
            import torch
            from transformers import AutoModelForImageSegmentation  # type: ignore
        except Exception as exc2:
            return None, f"birefnet_deps_missing:rembg={exc};transformers={exc2}"
        return None, "birefnet_transformers_not_auto_loaded"

    try:
        rgba = rembg_remove(body_pil.convert("RGB"))
        if not isinstance(rgba, Image.Image):
            rgba = Image.fromarray(np.asarray(rgba))
        alpha = np.asarray(rgba.convert("RGBA"))[:, :, 3]
    except Exception as exc:
        return None, f"rembg_failed:{exc}"

    if face_box is not None:
        fw, fh = max(1, face_box.width), max(1, face_box.height)
        x0 = max(0, int(face_box.x0 - 0.55 * fw))
        x1 = min(w, int(face_box.x1 + 0.55 * fw))
        y0 = max(0, int(face_box.y0 - 1.40 * fh))
        y1 = min(h, int(face_box.y1 + 0.55 * fh))
        gate = np.zeros((h, w), dtype=np.uint8)
        gate[y0:y1, x0:x1] = 255
        alpha = np.minimum(alpha, gate)

    if float((alpha > 16).mean()) < 0.002:
        return None, "birefnet_empty_matte"

    if blur_px > 0:
        k = blur_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return Image.fromarray(alpha), None


def build_head_hair_mask(
    body_pil: Image.Image,
    cache_dir,
    *,
    backend: str = "ellipse",
    face_box: FaceBox | None = None,
    expand_px: int = 18,
    blur_px: int = 12,
    top_extend: float = 1.55,
    side_extend: float = 0.60,
    bot_extend: float = 0.40,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Dispatch head+hair(+neck) mask by backend.

    Always returns a usable L mask. ``info`` records which backend produced it.
    """
    name = str(backend or "ellipse").strip().lower()
    info: dict[str, Any] = {"requested_backend": name}

    if name in ("sam2", "sam"):
        mask, reason = _try_sam2_mask(body_pil, face_box, blur_px=blur_px)
        if mask is not None:
            info.update({"backend": "sam2", "fallback_reason": None})
            return mask, info
        info["sam_skip"] = reason
        name = "ellipse"  # fall through

    if name in ("birefnet", "rembg", "matting"):
        mask, reason = _try_birefnet_mask(body_pil, face_box, blur_px=blur_px)
        if mask is not None:
            info.update({"backend": "birefnet", "fallback_reason": None})
            return mask, info
        info["birefnet_skip"] = reason
        name = "ellipse"

    mask, meta = _ellipse_mask(
        body_pil,
        cache_dir,
        face_box=face_box,
        expand_px=expand_px,
        blur_px=blur_px,
        top_extend=top_extend,
        side_extend=side_extend,
        bot_extend=bot_extend,
    )
    if info.get("requested_backend") not in ("ellipse", ""):
        meta["fallback_reason"] = (
            info.get("sam_skip") or info.get("birefnet_skip") or "unknown"
        )
        meta["requested_backend"] = info["requested_backend"]
    return mask, meta
