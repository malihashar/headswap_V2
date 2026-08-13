"""
Post-composite body-skin harmonization.

Shifts visible body skin (arms, shoulders, décolletage) to match the
already-composited donor head tone so the swap reads as one person.

Safety guarantees
-----------------
- Runs after ALL generation and compositing — zero calls to Krea2 or any
  generative model.
- The skin mask is built from body-region landmarks BELOW the head box;
  head pixels are structurally excluded and a byte-identity assertion
  confirms this before returning.
- Original pixels are blended (not replaced), so feathering at sleeve
  cuffs / collar lines is inherent.

Gate: ``extend_skin_harmonization: true`` in config (default true).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Person matte (thin wrapper so segmentation is an optional dep)
# ---------------------------------------------------------------------------

def _get_person_matte(body_pil: Image.Image) -> np.ndarray | None:
    """Return uint8 foreground alpha (0-255) or None if rembg unavailable."""
    try:
        from headswap.segmentation import _person_matte  # noqa: PLC0415
        alpha, _ = _person_matte(body_pil)
        return alpha
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Limb-region mask
# ---------------------------------------------------------------------------

def _limb_mask_mediapipe(rgb_np: np.ndarray, H: int, W: int) -> np.ndarray | None:
    """Thickened shoulder→elbow→wrist and hip→knee→ankle lines via MediaPipe."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        PAIRS = [
            (11, 13), (13, 15),   # L shoulder→elbow→wrist
            (12, 14), (14, 16),   # R shoulder→elbow→wrist
            (23, 25), (25, 27),   # L hip→knee→ankle
            (24, 26), (26, 28),   # R hip→knee→ankle
            (11, 12),             # shoulder bar
            (23, 24),             # hip bar
        ]
        thickness = max(28, W // 8)
        mask = np.zeros((H, W), dtype=np.uint8)
        with mp.solutions.pose.Pose(
            static_image_mode=True, min_detection_confidence=0.4
        ) as pose:
            res = pose.process(rgb_np)
        if not (res and res.pose_landmarks):
            return None
        lm = res.pose_landmarks.landmark
        for i, j in PAIRS:
            x1, y1 = int(lm[i].x * W), int(lm[i].y * H)
            x2, y2 = int(lm[j].x * W), int(lm[j].y * H)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        return mask
    except Exception:  # noqa: BLE001
        return None


def _limb_mask_geometric(
    person_matte: np.ndarray,
    head_bottom: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Fallback: body-matte below the head box."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if head_bottom < H:
        mask[head_bottom:] = np.where(person_matte[head_bottom:] > 16, 255, 0)
    return mask


# ---------------------------------------------------------------------------
# HSV skin heuristic
# ---------------------------------------------------------------------------

def _hsv_skin_mask(rgb_np: np.ndarray) -> np.ndarray:
    """
    Loose HSV heuristic covering common complexions.
    Constrained downstream by limb+matte intersection so can afford to be
    inclusive rather than precise.
    """
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_ok = (h <= 25) | (h >= 155)   # red-yellow + wraparound
    sat_ok = (s >= 28) & (s <= 220)   # not white, not neon
    val_ok = v >= 55                   # not near-black shadow
    return (hue_ok & sat_ok & val_ok).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# LAB cheek sampling + Reinhard transfer
# ---------------------------------------------------------------------------

def _cheek_lab_stats(
    result_np: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(mean_lab, std_lab) from the central cheek band of the donor head."""
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    py0 = y0 + int(fh * 0.40)
    py1 = y0 + int(fh * 0.72)
    px0 = x0 + int(fw * 0.18)
    px1 = x0 + int(fw * 0.82)
    patch = result_np[py0:py1, px0:px1]
    if patch.size == 0:
        patch = result_np[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-6


def _reinhard_transfer(
    region_rgb: np.ndarray,
    src_mean: np.ndarray,
    src_std: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
) -> np.ndarray:
    """Per-channel LAB statistical transfer (Reinhard et al.)."""
    lab = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        lab[:, :, ch] = (
            (lab[:, :, ch] - src_mean[ch]) * (tgt_std[ch] / src_std[ch])
            + tgt_mean[ch]
        )
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extend_skin_harmonization(
    result_pil: Image.Image,
    body_full_pil: Image.Image,
    head_x0: int,
    head_y0: int,
    head_x1: int,
    head_y1: int,
    *,
    body_visible_thresh: float = 0.09,
    feather_px: int = 20,
    transfer_strength: float = 0.80,
) -> tuple[Image.Image, dict]:
    """
    Shift body-skin tone to match the donor head already composited into
    ``result_pil``.

    Parameters
    ----------
    result_pil      : fully composited output (head already swapped)
    body_full_pil   : original un-edited body image (for person matte)
    head_{x0,y0,x1,y1} : head bounding box in result coordinates
    body_visible_thresh : fraction of person pixels below head required to proceed
    feather_px      : Gaussian blur radius on the skin mask
    transfer_strength : blend weight (1.0 = full transfer)

    Returns
    -------
    (result, info_dict)
    """
    info: dict = {"applied": False}
    W, H = result_pil.size
    result_np = np.asarray(result_pil.convert("RGB"), dtype=np.uint8).copy()
    original_np = result_np.copy()

    head_bottom = max(0, head_y1)
    head_excl = min(H, head_bottom + 10)   # safety margin below head

    # ------------------------------------------------------------------
    # 1. Body-visibility routing
    # ------------------------------------------------------------------
    person_matte = _get_person_matte(body_full_pil)
    if person_matte is None:
        info["skip_reason"] = "no_person_matte_backend"
        print("[skin_harm] skipped — no rembg/matte backend", flush=True)
        return result_pil, info

    if person_matte.shape[:2] != (H, W):
        person_matte = cv2.resize(
            person_matte, (W, H), interpolation=cv2.INTER_LINEAR
        )

    below = person_matte[head_bottom:]
    body_frac = float((below > 16).mean()) if below.size > 0 else 0.0
    info["body_frac"] = round(body_frac, 4)
    print(
        f"[skin_harm] body_frac_below_head={body_frac:.3f} "
        f"thresh={body_visible_thresh}",
        flush=True,
    )
    if body_frac < body_visible_thresh:
        info["skip_reason"] = "body_not_visible"
        print("[skin_harm] skipped — insufficient body pixels", flush=True)
        return result_pil, info

    # ------------------------------------------------------------------
    # 2. Limb-region mask  ∩  HSV skin heuristic  ∩  person matte
    # ------------------------------------------------------------------
    mp_mask = _limb_mask_mediapipe(result_np, H, W)
    if mp_mask is not None:
        limb_mask = mp_mask
        info["limb_backend"] = "mediapipe"
    else:
        limb_mask = _limb_mask_geometric(person_matte, head_bottom, H, W)
        info["limb_backend"] = "geometric"
    print(f"[skin_harm] limb_backend={info['limb_backend']}", flush=True)

    skin_hsv = _hsv_skin_mask(result_np)
    skin_mask = np.minimum(limb_mask, skin_hsv)
    skin_mask = np.minimum(skin_mask, person_matte)

    # Hard-exclude head region
    skin_mask[:head_excl] = 0

    skin_px = int((skin_mask > 0).sum())
    info["skin_px"] = skin_px
    print(f"[skin_harm] skin_px={skin_px}", flush=True)
    if skin_px < 300:
        info["skip_reason"] = "too_few_skin_px"
        print("[skin_harm] skipped — too few skin pixels", flush=True)
        return result_pil, info

    # ------------------------------------------------------------------
    # 3. Sample cheek tone from donor head; compute body-skin stats
    # ------------------------------------------------------------------
    tgt_mean, tgt_std = _cheek_lab_stats(
        result_np, head_x0, head_y0, head_x1, head_y1
    )

    ys, xs = np.where(skin_mask > 0)
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    region_rgb = result_np[by0:by1, bx0:bx1].copy()

    region_lab = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = region_lab.reshape(-1, 3)
    # Only compute stats over pixels actually in the skin mask
    region_mask_flat = skin_mask[by0:by1, bx0:bx1].ravel() > 0
    src_mean = flat[region_mask_flat].mean(axis=0)
    src_std = flat[region_mask_flat].std(axis=0) + 1e-6
    info["src_lab_mean"] = [round(float(x), 2) for x in src_mean]
    info["tgt_lab_mean"] = [round(float(x), 2) for x in tgt_mean]

    # ------------------------------------------------------------------
    # 4. Reinhard transfer + feathered blend
    # ------------------------------------------------------------------
    matched_rgb = _reinhard_transfer(region_rgb, src_mean, src_std, tgt_mean, tgt_std)

    k = feather_px * 2 + 1
    region_skin = skin_mask[by0:by1, bx0:bx1].astype(np.float32)
    soft = cv2.GaussianBlur(region_skin, (k, k), 0) / 255.0
    soft = np.clip(soft * transfer_strength, 0.0, 1.0)[..., None]

    result_np[by0:by1, bx0:bx1] = np.clip(
        matched_rgb * soft + result_np[by0:by1, bx0:bx1] * (1.0 - soft),
        0, 255,
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Safety: head region must be byte-identical
    # ------------------------------------------------------------------
    hx0 = max(0, head_x0)
    hy0 = max(0, head_y0)
    hx1 = min(W, head_x1)
    head_before = original_np[hy0:head_excl, hx0:hx1]
    head_after = result_np[hy0:head_excl, hx0:hx1]
    head_identical = bool(np.array_equal(head_before, head_after))
    info["head_identical"] = head_identical
    if head_identical:
        print("[skin_harm] head region byte-identical ✓", flush=True)
    else:
        diff = np.abs(head_before.astype(int) - head_after.astype(int))
        info["head_max_diff"] = int(diff.max())
        print(
            f"[skin_harm] WARNING head region modified max_diff={diff.max()}",
            flush=True,
        )

    info["applied"] = True
    print(
        f"[skin_harm] done — strength={transfer_strength} feather={feather_px}px",
        flush=True,
    )
    return Image.fromarray(result_np), info
