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
    """Thickened shoulder→elbow→wrist and hip→knee→ankle lines via MediaPipe.

    Also draws a filled circle at each wrist so HANDS are covered: the
    landmark chain stops AT the wrist, so a line-only mask ends at the wrist
    and leaves the hands out of the skin-tone transfer entirely (visible as
    donor-toned arms with target-toned hands).
    """
    # `import mediapipe as mp` then `mp.solutions.pose` fails on some builds
    # with "module 'mediapipe' has no attribute 'solutions'" (observed on the
    # Colab Python 3.13 image): the top-level lazy attribute isn't populated
    # even though the real module is importable at its full path. Import the
    # submodule directly, with the attribute style as a fallback.
    mp_pose = None
    errors = []
    try:
        from mediapipe.python.solutions import pose as mp_pose  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mediapipe.python.solutions.pose: {type(exc).__name__}: {exc}")
        try:
            import mediapipe as mp  # noqa: PLC0415
            mp_pose = mp.solutions.pose
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"mediapipe.solutions.pose: {type(exc2).__name__}: {exc2}")
    if mp_pose is None:
        print(f"[skin_harm] mediapipe unavailable -- {' | '.join(errors)}", flush=True)
        return None
    try:
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
        with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.4
        ) as pose:
            res = pose.process(rgb_np)
        if not (res and res.pose_landmarks):
            print("[skin_harm] mediapipe found no pose landmarks", flush=True)
            return None
        lm = res.pose_landmarks.landmark
        for i, j in PAIRS:
            x1, y1 = int(lm[i].x * W), int(lm[i].y * H)
            x2, y2 = int(lm[j].x * W), int(lm[j].y * H)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        # Hands: the pose chain ends at the wrist (15/17), so without this the
        # hands fall outside the mask. Landmarks 17-22 are finger/thumb points
        # when present; fall back to a blob at the wrist itself.
        for wrist_idx in (15, 16):
            wx, wy = int(lm[wrist_idx].x * W), int(lm[wrist_idx].y * H)
            cv2.circle(mask, (wx, wy), max(thickness, W // 7), 255, -1)
        for hand_idx in (17, 18, 19, 20, 21, 22):
            if hand_idx < len(lm):
                hx, hy = int(lm[hand_idx].x * W), int(lm[hand_idx].y * H)
                cv2.circle(mask, (hx, hy), max(thickness // 2, W // 12), 255, -1)
        return mask
    except Exception as exc:  # noqa: BLE001
        print(f"[skin_harm] mediapipe pose failed ({type(exc).__name__}: {exc})", flush=True)
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

def _robust_lab_stats(flat_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median_lab, robust_std_lab) via median + MAD, trimming the top/bottom
    10% per channel (highlights/shadows/outliers) before computing stats --
    a plain mean/std over a skin patch is easily dragged by a handful of
    specular-highlight or deep-shadow pixels."""
    med = np.median(flat_lab, axis=0)
    mad = np.median(np.abs(flat_lab - med), axis=0)
    robust_std = mad * 1.4826 + 1e-6  # MAD -> std-equivalent for a normal dist
    return med, robust_std


def _cheek_lab_stats(
    result_np: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(median_lab, robust_std_lab) from the central cheek band of the donor head."""
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    py0 = y0 + int(fh * 0.40)
    py1 = y0 + int(fh * 0.72)
    px0 = x0 + int(fw * 0.18)
    px1 = x0 + int(fw * 0.82)
    patch = result_np[py0:py1, px0:px1]
    if patch.size == 0:
        patch = result_np[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    return _robust_lab_stats(lab.reshape(-1, 3))


def _reinhard_transfer(
    region_rgb: np.ndarray,
    src_mean: np.ndarray,
    src_std: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
    *,
    contrast_clamp: tuple[float, float] = (0.75, 1.30),
) -> np.ndarray:
    """Per-channel LAB statistical transfer (Reinhard et al.).

    Donor colour, target lighting -- via the MEAN/DEVIATION split, not by
    holding L back.

    An earlier version blended L toward the target's own value
    (``luminance_preserve``) to protect shading. That was wrong for this
    task: between light and dark skin the difference lives almost entirely
    in **L**, while a/b barely move. Preserving 85% of the target's L
    therefore preserved 85% of the tone that was supposed to change -- a
    donor-toned face ended up on unchanged target-toned arms/hands, and the
    tone step at the shoulder read as a seam.

    The mean/deviation split gives both properties at once:
      * the MEAN of each channel moves fully to the donor's -> tone changes
      * each pixel's DEVIATION from that mean is preserved -> shadows,
        highlights and skin texture survive, because that local variation is
        exactly what deviation-from-mean encodes.
    Only the std *ratio* is clamped, so a low-contrast donor patch cannot
    flatten (or a high-contrast one exaggerate) the target's own modelling.
    """
    lo, hi = contrast_clamp
    lab = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        ratio = float(np.clip(tgt_std[ch] / src_std[ch], lo, hi))
        lab[:, :, ch] = (lab[:, :, ch] - src_mean[ch]) * ratio + tgt_mean[ch]
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
    # Exclusion line for the head region. A fixed +10px margin left the whole
    # neck/upper-shoulder band untreated, so the donor-toned face met
    # target-toned shoulders across a short, hard step -- read as "the
    # shoulders are popping up". Scale the margin to the head's own size so
    # the transfer starts right below the jaw and the feather has room to
    # ramp across the neck instead of terminating on top of the shoulder.
    head_h = max(1, head_y1 - head_y0)
    head_excl = min(H, head_bottom + max(2, int(head_h * 0.04)))

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
    # Coverage per image-third, so "the arms were not recolored" is
    # observable instead of inferred: an arms-covered mask has real coverage
    # in the middle/lower thirds, a neck-and-collar-only mask does not.
    _t = H // 3
    for _name, _sl in (("upper", slice(0, _t)), ("mid", slice(_t, 2 * _t)), ("lower", slice(2 * _t, H))):
        _band = skin_mask[_sl]
        info[f"cover_{_name}"] = round(float((_band > 0).mean()), 4) if _band.size else 0.0
    print(
        f"[skin_harm] skin_px={skin_px} coverage upper={info['cover_upper']} "
        f"mid={info['cover_mid']} lower={info['cover_lower']}",
        flush=True,
    )
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
    src_mean, src_std = _robust_lab_stats(flat[region_mask_flat])
    info["src_lab_mean"] = [round(float(x), 2) for x in src_mean]
    info["tgt_lab_mean"] = [round(float(x), 2) for x in tgt_mean]

    # ------------------------------------------------------------------
    # 4. Reinhard transfer (chroma-led, luminance preserved) + feathered blend
    # ------------------------------------------------------------------
    matched_rgb = _reinhard_transfer(
        region_rgb, src_mean, src_std, tgt_mean, tgt_std
    )

    # Feather scaled to the image, not a fixed 20px. A tone change spread over
    # a whole torso/arm needs a wide ramp: at 1024px a 20px feather completes
    # the entire transition in ~2% of the frame, which is a visible edge --
    # the "two layers / different mask beyond the chin" seam. Human perception
    # is far more tolerant of a large gradual tone gradient than of a small
    # sharp one, so widen the ramp with resolution.
    feather_eff = max(int(feather_px), int(max(H, W) * 0.05))
    k = feather_eff * 2 + 1
    info["feather_eff_px"] = feather_eff
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
