"""Super-resolve the pre-swap plate without hallucinating sky/background detail.

Why this exists
----------------
Real-ESRGAN x4 was added to sharpen a tiny detected face (43px -> 172px) before
the swap. Running it on the WHOLE frame also super-resolves the background --
and on a smooth, low-detail region (a gradient sunset sky) it hallucinates
texture that was never there. GPU-measured on the reference case: a row of
small arc/bird-like shapes appeared in the sky at a fixed position, present
in the plate immediately after Real-ESRGAN and absent from the original
source and from a plain Lanczos upsample of the same region.

The face is the only region SR quality actually matters for -- identity comes
from a tight face crop, and ``restore_background`` already asserts that
everything outside the person must be pixel-identical to the plate, so any
hallucinated background detail ships straight to the final image. So only a
padded crop around the face is super-resolved; the rest of the frame is a
plain Lanczos upsample (no hallucination risk) and the sharpened crop is
pasted back at its scaled location.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from headswap.preprocess import FaceBox

_SR_MODEL: Any = None
_SR_SCALE: int | None = None


def _shim_huggingface_hub() -> None:
    """RealESRGAN imports the removed ``cached_download`` API at MODULE load
    time (``RealESRGAN/__init__.py`` -> ``model.py``), so the shim must run
    BEFORE any ``import RealESRGAN`` -- including the availability probe
    below, not just the model-loading path. Applying it only in
    ``_get_sr_model`` left ``sr_backend_available()`` observing the bare,
    unshimmed import and reporting the backend missing even when installed.
    """
    import huggingface_hub as hh

    if not hasattr(hh, "cached_download"):
        hh.cached_download = hh.hf_hub_download


def sr_backend_available() -> bool:
    try:
        _shim_huggingface_hub()
        import RealESRGAN  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _get_sr_model(scale: int):
    global _SR_MODEL, _SR_SCALE
    if _SR_MODEL is not None and _SR_SCALE == scale:
        return _SR_MODEL
    _shim_huggingface_hub()
    import huggingface_hub as hh
    import torch
    from RealESRGAN import RealESRGAN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=scale)
    weights_name = {2: "RealESRGAN_x2.pth", 4: "RealESRGAN_x4.pth", 8: "RealESRGAN_x8.pth"}[scale]
    model.load_weights(hh.hf_hub_download("ai-forever/Real-ESRGAN", weights_name), download=False)
    _SR_MODEL, _SR_SCALE = model, scale
    return model


def super_resolve_plate(
    plate: Image.Image,
    face: FaceBox,
    *,
    scale: int = 4,
    face_pad_side_frac: float = 1.3,
    face_pad_top_frac: float = 2.0,
    face_pad_bot_frac: float = 1.6,
) -> tuple[Image.Image, dict[str, Any]]:
    """Upsample ``plate`` by ``scale``, sharpening only a padded face crop.

    Falls back to a plain Lanczos upsample (no super-resolution, but also no
    hallucination risk) when the optional Real-ESRGAN dependency is absent.
    """
    w, h = plate.size
    base_hr = plate.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    info: dict[str, Any] = {"applied": False, "scale": scale}

    if not sr_backend_available():
        info["reason"] = "realesrgan_missing"
        return base_hr, info

    pad_w = int(face.width * face_pad_side_frac)
    cx0 = max(0, int(face.x0 - pad_w))
    cx1 = min(w, int(face.x1 + pad_w))
    cy0 = max(0, int(face.y0 - face.height * face_pad_top_frac))
    cy1 = min(h, int(face.y1 + face.height * face_pad_bot_frac))
    if cx1 <= cx0 or cy1 <= cy0:
        info["reason"] = "empty_face_crop"
        return base_hr, info

    try:
        model = _get_sr_model(scale)
        face_crop = plate.crop((cx0, cy0, cx1, cy1))
        face_crop_hr = model.predict(face_crop.convert("RGB"))
    except Exception as exc:
        info["reason"] = f"sr_failed:{exc}"
        return base_hr, info

    box_hr = (cx0 * scale, cy0 * scale, cx1 * scale, cy1 * scale)
    expect = (box_hr[2] - box_hr[0], box_hr[3] - box_hr[1])
    if face_crop_hr.size != expect:
        face_crop_hr = face_crop_hr.resize(expect, Image.Resampling.LANCZOS)

    out = base_hr.copy()
    out.paste(face_crop_hr, (box_hr[0], box_hr[1]))
    info["applied"] = True
    info["face_crop_box_hr"] = box_hr
    return out, info
