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


_MATTE_AVAILABLE: bool | None = None


def matte_backend_available() -> bool:
    """Is a real silhouette matte actually installed? Cached.

    Lets callers pick edge softness by what the mask ACTUALLY is: a real
    matte follows the true silhouette and can take a tight edge, while the
    geometric ellipse is only an approximation and needs generous feather to
    hide the mismatch. Deciding this up front avoids matting twice.
    """
    global _MATTE_AVAILABLE
    if _MATTE_AVAILABLE is None:
        try:
            import rembg  # noqa: F401
            _MATTE_AVAILABLE = True
        except Exception:
            _MATTE_AVAILABLE = False
    return _MATTE_AVAILABLE


def _person_matte(body_pil: Image.Image) -> tuple[np.ndarray | None, str | None]:
    """Raw, UNGATED foreground/person alpha matte (rembg / BiRefNet)."""
    try:
        from rembg import remove as rembg_remove  # type: ignore
    except Exception as exc:
        return None, f"rembg_missing:{exc}"
    try:
        rgba = rembg_remove(body_pil.convert("RGB"))
        if not isinstance(rgba, Image.Image):
            rgba = Image.fromarray(np.asarray(rgba))
        return np.asarray(rgba.convert("RGBA"))[:, :, 3], None
    except Exception as exc:
        return None, f"rembg_failed:{exc}"


def _head_matte_mask(
    body_pil: Image.Image,
    cache_dir,
    *,
    face_box: FaceBox | None,
    expand_px: int,
    blur_px: int,
    top_extend: float,
    side_extend: float,
    bot_extend: float,
    hair_margin_frac: float = 0.08,
) -> tuple[Image.Image | None, str | None]:
    """Head mask that follows the REAL silhouette: ellipse AND person matte.

    The plain ``ellipse`` backend is a geometric prior derived only from the
    face box. Measured on a short-haired subject with a small face in a
    full-body frame, it covers ~6.5x the face-box AREA and reaches ~1.3x
    face-height above the face -- and for short hair essentially all of that
    upper region is BACKGROUND, not hair.

    That is not merely a compositing concern. ``crop_with_mask`` derives the
    crop box from the mask, and the model regenerates everything inside it.
    So the model is handed a large oval of sky, regenerates it (imperfectly,
    producing a visible arc exactly on the mask boundary) and -- having been
    asked to fill that space -- invents hair to occupy it. This is the
    observed "ghost oval above the head + hair wings at ear level" failure.

    Intersecting the ellipse with a foreground/person matte removes the
    background from the mask entirely, so the model can only ever regenerate
    actual head/hair pixels. The matte is dilated by ``hair_margin_frac`` of
    face height first, so a donor hairstyle slightly larger than the target's
    still has room to grow (an un-dilated intersection would pin the new hair
    to the old silhouette exactly -- the failure HID names, arXiv 2503.00861).

    Returns ``(mask, None)`` or ``(None, reason)`` so the caller can fall back
    to the ellipse when no matting backend is installed.
    """
    if face_box is None:
        return None, "head_matte_no_face_box"
    alpha, reason = _person_matte(body_pil)
    if alpha is None:
        return None, reason

    # Region of interest: the existing ellipse prior, UNBLURRED (we blur the
    # intersection at the end -- blurring twice would re-grow a soft tail
    # into the background we are trying to exclude).
    ellipse, _ = _ellipse_mask(
        body_pil,
        cache_dir,
        face_box=face_box,
        expand_px=expand_px,
        blur_px=0,
        top_extend=top_extend,
        side_extend=side_extend,
        bot_extend=bot_extend,
    )
    ell = np.asarray(ellipse.convert("L"))

    fh = max(1.0, float(face_box.height))
    margin = int(round(max(0.0, hair_margin_frac) * fh))
    if margin > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        alpha = cv2.dilate(alpha, k)

    mask = np.minimum(ell, alpha)

    # Rigid headwear safety net: person-segmentation backends are trained on
    # human bodies, not accessories -- a stiff hat/headpiece is frequently
    # NOT marked as foreground at all (measured on this exact case: matte
    # alpha is uniformly 0 in the entire crown band above the ellipse's
    # face-anchored top edge). `min(ell, alpha)` then chops the headwear out
    # regardless of how generous `top_extend` is, since growing the ellipse
    # only ever intersects more zeros. The excluded headwear sits inside
    # this mask's own blur band as real, un-regenerated content, and the
    # composite blends it toward transparent -- a translucent "ghost" of
    # the ORIGINAL headwear, not a compositing glitch (confirmed via a real
    # pixel diff: the composited output exactly matches
    # layer*mask + original*(1-mask) using this exact mask).
    #
    # A blanket "trust the ellipse over the matte" bypass would revive the
    # older, opposite bug this intersection exists to prevent (short hair +
    # bare ellipse regenerating a ghost oval of sky). So only bypass the
    # matte within the crown band directly above the face -- and only where
    # that band looks like a worn object rather than sky: real headwear has
    # local texture (brim edges, fabric folds, embroidery); open sky is
    # smooth. Laplacian variance is a standard cheap texture proxy for this.
    fx0 = max(0, int(face_box.x0))
    fx1 = min(alpha.shape[1], int(face_box.x1))
    crown_bottom = max(0, int(face_box.y0))
    ell_ys, _ = np.where(ell > 16)
    crown_top = int(ell_ys.min()) if ell_ys.size else crown_bottom
    if fx1 > fx0 and crown_top < crown_bottom:
        crown_gray = cv2.cvtColor(
            np.asarray(body_pil.convert("RGB"))[crown_top:crown_bottom, fx0:fx1],
            cv2.COLOR_RGB2GRAY,
        )
        texture = float(cv2.Laplacian(crown_gray, cv2.CV_64F).var()) if crown_gray.size else 0.0
        is_object_like = texture > 40.0
        print(
            f"[head_matte diag] crown=[{crown_top}:{crown_bottom},{fx0}:{fx1}] "
            f"texture_var={texture:.1f} is_object_like={is_object_like} "
            f"alpha_max_in_crown={int(alpha[crown_top:crown_bottom, fx0:fx1].max())}",
            flush=True,
        )
        if is_object_like:
            mask[crown_top:crown_bottom, fx0:fx1] = np.maximum(
                mask[crown_top:crown_bottom, fx0:fx1],
                ell[crown_top:crown_bottom, fx0:fx1],
            )

    # The face core must always be editable even if the matte is imperfect
    # (sunglasses, motion blur, low contrast against the background).
    core = np.zeros_like(mask)
    cx = int(0.5 * (face_box.x0 + face_box.x1))
    cy = int(0.5 * (face_box.y0 + face_box.y1))
    cv2.ellipse(
        core,
        (cx, cy),
        (max(1, int(face_box.width * 0.50)), max(1, int(fh * 0.55))),
        0, 0, 360, 255, -1,
    )
    mask = np.maximum(mask, core)

    if float((mask > 16).mean()) < 0.0005:
        return None, "head_matte_empty"
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (blur_px * 2 + 1, blur_px * 2 + 1), 0)
    return Image.fromarray(mask), None


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
    hair_margin_frac: float = 0.08,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Dispatch head+hair(+neck) mask by backend.

    Always returns a usable L mask. ``info`` records which backend produced it.
    """
    name = str(backend or "ellipse").strip().lower()
    info: dict[str, Any] = {"requested_backend": name}

    if name in ("head_matte", "matte", "silhouette"):
        mask, reason = _head_matte_mask(
            body_pil,
            cache_dir,
            face_box=face_box,
            expand_px=expand_px,
            blur_px=blur_px,
            top_extend=top_extend,
            side_extend=side_extend,
            bot_extend=bot_extend,
            hair_margin_frac=hair_margin_frac,
        )
        if mask is not None:
            info.update({"backend": "head_matte", "fallback_reason": None})
            return mask, info
        info["head_matte_skip"] = reason
        name = "ellipse"  # fall through -- never fail a swap on a missing dep

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
            info.get("head_matte_skip")
            or info.get("sam_skip")
            or info.get("birefnet_skip")
            or "unknown"
        )
        if info.get("head_matte_skip"):
            meta["head_matte_skip"] = info["head_matte_skip"]
        meta["requested_backend"] = info["requested_backend"]
    return mask, meta
