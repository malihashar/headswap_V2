"""Erase headwear and reconstruct what it occluded, before the swap.

Why this exists
---------------
A hat that covers the skull makes a HEAD swap impossible: preserve it and the
donor's hair is hidden, so only the face can change and the result reads as a
face swap. Removing it is not something the edit model can do on its own --
asked to delete a large opaque object it has no way to reconstruct the sky
behind it, and fills the void with a bright blob (observed on every
prompt-only removal attempt).

So removal is done deterministically FIRST, by an inpainting model, and the
swap then runs on a genuinely bare-headed plate. This is the "erase then
regenerate" split GHOST 2.0 uses (arXiv 2502.18417), where LaMa extrapolates
the background before the head is composited.

Measured on the reference case: LaMa removes the crown AND the stiff side
flaps and reconstructs the sky and hairline convincingly; the subsequent swap
then produces the donor's real hairstyle instead of a hidden-hair face swap.

IMPORTANT: erasing does NOT license a bigger edit mask. Every attempt that
paired removal with an enlarged mask blew up into a glowing oval. Keep the
normal (tight matte) mask geometry -- the plate is what changes, not the mask.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from headswap.preprocess import FaceBox, pil_to_rgb_np


def headwear_mask(
    body: Image.Image,
    face: FaceBox,
    person_matte: np.ndarray,
    *,
    brow_frac: float = 0.18,
    chin_pad_frac: float = 0.10,
    dark_sum_max: int = 260,
    dilate_px: int = 11,
    face_keep_side_frac: float = 0.15,
) -> np.ndarray:
    """Pixels belonging to a hat/headwear, as a uint8 0/255 mask.

    Headwear is taken to be foreground (inside ``person_matte``) within the
    head band that is either ABOVE the brow line -- the crown -- or DARK
    fabric beside the face -- the flaps. The face itself is explicitly
    protected so it can never be erased.
    """
    h, w = person_matte.shape[:2]
    rgb = pil_to_rgb_np(body).astype(int)
    lum = rgb.sum(axis=2)
    yy, xx = np.mgrid[0:h, 0:w]
    brow = int(face.y0 + brow_frac * face.height)
    chin = int(face.y1 + chin_pad_frac * face.height)
    person = person_matte > 40
    face_keep = (
        (yy > brow) & (yy < chin)
        & (xx > face.x0 - face_keep_side_frac * face.width)
        & (xx < face.x1 + face_keep_side_frac * face.width)
    )
    hat = person & (yy < chin) & ((yy < brow) | (lum < dark_sum_max)) & (~face_keep)
    out = hat.astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        out = cv2.dilate(out, k)
    return out


def erase_headwear(
    body: Image.Image, mask: np.ndarray
) -> tuple[Image.Image, dict[str, Any]]:
    """Inpaint ``mask`` out of ``body``. Returns ``(plate, info)``.

    Falls back to the untouched image when the inpainting backend is absent,
    so a missing optional dependency degrades to today's behaviour instead of
    failing a swap.
    """
    info: dict[str, Any] = {"applied": False, "mask_px": int((mask > 0).sum())}
    if info["mask_px"] < 16:
        info["reason"] = "empty_headwear_mask"
        return body, info
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except Exception as exc:
        info["reason"] = f"simple_lama_missing:{exc}"
        return body, info
    try:
        plate = SimpleLama()(body.convert("RGB"), Image.fromarray(mask))
    except Exception as exc:
        info["reason"] = f"lama_failed:{exc}"
        return body, info
    if plate.size != body.size:
        plate = plate.resize(body.size, Image.Resampling.LANCZOS)
    info["applied"] = True
    return plate.convert("RGB"), info


def restore_background(
    result: Image.Image,
    plate: Image.Image,
    *,
    dilate_px: int = 5,
    blur_px: int = 9,
) -> tuple[Image.Image, dict[str, Any]]:
    """Force every pixel outside the person back to ``plate``.

    The edit model regenerates the whole crop, so background inside the crop
    box comes back subtly different -- brighter sky, shifted gradient -- and
    the mask boundary then reads as a glowing arc around the head. That is
    the "ghost oval" this project chased for a long time.

    Rather than tuning the blend to hide it, this asserts the invariant
    directly: only the PERSON may differ from the source. The union of the
    result's and the plate's person mattes defines who; everything else is
    copied back verbatim. Measured on the reference case, worst-case altered
    background pixel dropped 81.3 -> 4.3 out of 255.

    Note the matte is taken from the RESULT too, not just the plate, so newly
    generated hair extending beyond the original silhouette is kept.
    """
    from headswap.segmentation import _person_matte

    if result.size != plate.size:
        result = result.resize(plate.size, Image.Resampling.LANCZOS)
    m_res, _ = _person_matte(result)
    m_plate, _ = _person_matte(plate)
    if m_res is None or m_plate is None:
        return result, {"applied": False, "reason": "no_matte_backend"}
    m = np.maximum(m_res, m_plate)
    if dilate_px > 0:
        m = cv2.dilate(
            m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        )
    if blur_px > 0:
        k = blur_px if blur_px % 2 else blur_px + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    a = (m.astype(np.float32) / 255.0)[..., None]
    out = (
        np.asarray(result).astype(np.float32) * a
        + np.asarray(plate).astype(np.float32) * (1.0 - a)
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), {"applied": True}
