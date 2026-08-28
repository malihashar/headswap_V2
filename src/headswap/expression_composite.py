"""Composite the ORIGINAL photo's mouth/eye expression onto a generated head.

Experimental. Lives on branch `expression-mouth-composite` deliberately kept
off the `simple-full-body-head-swap` branch, where T4 (docs/PIPELINE_STATE.md
CHECKPOINT-10) is the approved recipe and must stay untouched.

Why this exists: CHECKPOINT-11/12/13 tested three independent channels for
keeping the BODY photo's expression -- prompt text, the face_refine pass, and
donor conditioning strength (ref_boost) -- and none of them moved the
expression at all. The donor's expression appears to be carried by the
identity LoRA itself as part of "the head," not as a separable signal on any
tested surface. This module is the one approach left that does not touch T4:
composite the expression back in as a POST-PROCESS step, not as a change to
generation.

Known risk, stated up front rather than discovered again: this is exactly the
class of masked composite that produced ghosting and seams throughout this
session (hair strands, donor-collar fringe, halo rings). A mouth lifted from
one face and pasted onto a different face's bone structure is a plausible new
failure mode, not a guaranteed fix. Every guard in this file exists because of
a specific failure observed earlier in this project on a similar composite.
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Mediapipe FaceMesh landmark indices for the mouth AND eyes. Chosen wider
# than "just the lips" -- pasting only the inner mouth against differently
# textured skin is what produces a visible seam; including a soft margin of
# surrounding skin lets the feather do its job across cheek/chin texture
# rather than across a lip boundary.
_MOUTH_IDX = [
    61, 76, 62, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Outer ring, widened toward cheeks/chin for feather margin.
    57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287,
]
_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144, 163, 7, 246, 161, 159, 157, 173]
_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380, 390, 249, 466, 388, 386, 384, 398]

_MODEL_PATHS = (
    "/content/models/face_landmarker.task",
    "/root/.cache/headswap/face_landmarker.task",
    os.path.expanduser("~/.cache/headswap/face_landmarker.task"),
)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _find_model() -> str | None:
    for p in _MODEL_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 100_000:
            return p
    return None


def _landmarks468(rgb_np: np.ndarray) -> np.ndarray | None:
    """Full-face mediapipe landmarks in PIXEL coords, or None.

    None on ANY failure -- missing model, missing package, no face found, or
    an unexpected mediapipe API shape. The caller must treat None as "skip
    entirely, return the input unchanged," never as a partial result.
    """
    model_path = _find_model()
    if model_path is None:
        print(
            "[expr_composite] face_landmarker.task NOT found -- searched "
            f"{list(_MODEL_PATHS)}. Download with:\n"
            f"  curl -fsSL -o {_MODEL_PATHS[0]} '{_MODEL_URL}'",
            flush=True,
        )
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1
        )
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = rgb_np.shape[:2]
        pts = np.array(
            [[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        return pts
    except Exception as exc:  # noqa: BLE001
        print(f"[expr_composite] landmark detection FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None


def _region_mask(shape_hw: tuple[int, int], pts: np.ndarray, idx: list[int], dilate_px: int) -> np.ndarray:
    hull = cv2.convexHull(pts[idx].astype(np.int32))
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


def composite_original_expression(
    generated_pil: Image.Image,
    original_body_pil: Image.Image,
    *,
    feather_px: int = 14,
    color_match: bool = True,
    max_region_frac: float = 0.06,
) -> tuple[Image.Image, dict[str, Any]]:
    """Paste the ORIGINAL photo's mouth+eyes onto the GENERATED head.

    Both images must be the same size (same coordinate frame as the rest of
    run_simple_full_body, since head position/scale is preserved by the
    img2img path -- see CHECKPOINT-10). If they are not, or if landmarks
    cannot be found on EITHER face, this returns generated_pil UNCHANGED. A
    failed expression composite must never be worse than no composite.
    """
    diag: dict[str, Any] = {"applied": False}
    try:
        if generated_pil.size != original_body_pil.size:
            diag["reason"] = (
                f"size mismatch gen={generated_pil.size} "
                f"orig={original_body_pil.size}"
            )
            return generated_pil, diag

        gen_rgb = np.asarray(generated_pil.convert("RGB"), dtype=np.uint8)
        orig_rgb = np.asarray(original_body_pil.convert("RGB"), dtype=np.uint8)
        h, w = gen_rgb.shape[:2]

        gen_pts = _landmarks468(gen_rgb)
        orig_pts = _landmarks468(orig_rgb)
        if gen_pts is None or orig_pts is None:
            diag["reason"] = "landmarks_unavailable"
            return generated_pil, diag

        region_idx = _MOUTH_IDX + _LEFT_EYE_IDX + _RIGHT_EYE_IDX

        # Sanity guard: the expression region must be a SMALL fraction of the
        # frame. On a bad detection (wrong face, degenerate landmarks) the
        # hull can balloon to cover most of the image -- pasting "expression"
        # at that scale would silently overwrite the whole head. This is the
        # same plausibility guard pattern already used elsewhere in this repo
        # for exactly this failure class (a mask that is technically valid
        # but obviously too large to be what it claims to be).
        orig_hull_mask = _region_mask((h, w), orig_pts, region_idx, dilate_px=0)
        orig_frac = float((orig_hull_mask > 0).mean())
        if orig_frac > max_region_frac:
            diag["reason"] = (
                f"region too large ({orig_frac:.3f} > {max_region_frac}); "
                "landmarks likely wrong, refusing to composite"
            )
            return generated_pil, diag

        # Align the ORIGINAL expression region onto the GENERATED face via a
        # similarity transform (rotation + uniform scale + translation) fit
        # from eye-corner + mouth-corner correspondences. This is deliberately
        # NOT a full warp of every landmark: a similarity transform cannot
        # distort the pasted region's own shape, only place and scale it,
        # which keeps the source expression's geometry intact rather than
        # bending it to match a different face's proportions.
        anchor_idx = [33, 263, 61, 291]  # left eye outer, right eye outer, mouth corners
        src = orig_pts[anchor_idx].astype(np.float32)
        dst = gen_pts[anchor_idx].astype(np.float32)
        M, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            diag["reason"] = "alignment_failed"
            return generated_pil, diag

        warped_orig = cv2.warpAffine(
            orig_rgb, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        # Mask built in the GENERATED frame from the WARPED original hull, not
        # from gen_pts -- the pasted content must be masked to where the
        # SOURCE pixels actually landed, or a scale mismatch pastes expression
        # pixels next to a mask boundary that doesn't correspond to them.
        warped_hull_pts = cv2.transform(
            orig_pts[region_idx].reshape(-1, 1, 2), M
        ).reshape(-1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(warped_hull_pts.astype(np.int32)), 255)

        mask_f = mask.astype(np.float32) / 255.0
        if feather_px > 0:
            k = max(3, feather_px * 2 + 1)
            mask_f = cv2.GaussianBlur(mask_f, (k, k), 0)

        source = warped_orig.astype(np.float32)
        if color_match:
            # Match the pasted region's mean LAB to the GENERATED face's own
            # skin tone in the same region, preserving the source's own
            # deviation-from-mean (texture, shading) -- same Reinhard-style
            # split already used and documented in skin_harmonize.py, for the
            # same reason: a pure paste carries the wrong skin tone and reads
            # as an obvious patch, but a full recolour destroys the expression
            # texture that is the entire point of this composite.
            sel = mask_f > 0.5
            if sel.sum() > 50:
                src_lab = cv2.cvtColor(warped_orig, cv2.COLOR_RGB2LAB).astype(np.float32)
                dst_lab = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                src_mean = src_lab[sel].mean(axis=0)
                dst_mean = dst_lab[sel].mean(axis=0)
                matched_lab = src_lab.copy()
                for c in range(3):
                    matched_lab[..., c] = matched_lab[..., c] - src_mean[c] + dst_mean[c]
                source = cv2.cvtColor(
                    np.clip(matched_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
                ).astype(np.float32)

        mask_3 = mask_f[..., None]
        composited = gen_rgb.astype(np.float32) * (1.0 - mask_3) + source * mask_3
        out = Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8))

        diag = {
            "applied": True,
            "region_px": int((mask_f > 0.5).sum()),
            "region_frac": round(float((mask_f > 0.5).mean()), 4),
            "color_matched": color_match,
            "feather_px": feather_px,
        }
        return out, diag
    except Exception as exc:  # noqa: BLE001 -- must never break a render
        diag["reason"] = f"{type(exc).__name__}: {exc}"
        return generated_pil, diag
