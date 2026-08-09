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
