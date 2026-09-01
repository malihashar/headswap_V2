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

import os

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

_SEM_MODEL_PATHS = (
    "/content/models/selfie_multiclass_256x256.tflite",
    "/root/.cache/headswap/selfie_multiclass_256x256.tflite",
)
# MediaPipe multiclass selfie segmentation labels.
_SEM_BODY_SKIN, _SEM_FACE_SKIN = 2, 3

# Skin-likeness score above which a pixel overrides a CLOTHES class veto.
#
# DISABLED BY DEFAULT (2.0 is unreachable; _skin_likeness maxes at 1.0).
#
# At 0.75 this rescued 102,850px on a single frame and what it rescued was
# GARMENT: a white top came back pink and a patterned skirt came back brown.
# That is the exact regression this file already documents -- chroma distance
# alone scored a blue jersey at 1.00 -- and it reappeared because a garment
# photographed under the same light as its wearer sits close to that wearer's
# face in chroma. The reference cannot separate them, so no threshold on this
# score is safe: raising it far enough to exclude clothing also excludes the
# shaded skin the rescue existed to reach.
#
# Kept as a knob rather than deleted because the underlying observation still
# holds -- a bare limb rendered in a garment-like tone IS vetoed by the
# clothes class. That has to be fixed with a signal that distinguishes cloth
# from skin (texture//geometry), not with a chroma threshold.
_SKIN_RESCUE_FROM_CLOTHES = float(
    os.environ.get("HEADSWAP_SKIN_RESCUE_THR", "2.0")
)


def _semantic_category_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Raw per-pixel class ids from the multiclass selfie segmenter."""
    import os as _os  # noqa: PLC0415

    model_path = next(
        (q for q in _SEM_MODEL_PATHS
         if _os.path.exists(q) and _os.path.getsize(q) > 100_000),
        None,
    )
    if model_path is None:
        # Loud, not silent. A missing/truncated model made every semantic
        # call return None, which silently downgraded compositing to the
        # column fallback -- and with the LAB wash off that produced a run
        # where the skin changed by nothing at all and nothing said why.
        # Size-check too: a failed curl leaves a short HTML error body that
        # exists() would happily accept.
        _found = [(q, _os.path.getsize(q)) for q in _SEM_MODEL_PATHS if _os.path.exists(q)]
        print(
            "[skin_harm] semantic segmenter model NOT usable -- searched "
            f"{list(_SEM_MODEL_PATHS)}, found {_found or 'nothing'}. "
            "Skin-vs-clothes separation is unavailable this run; re-run "
            "scripts/setup_colab.sh to fetch selfie_multiclass_256x256.tflite.",
            flush=True,
        )
        return None
    try:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415

        opts = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
        )
        with mp_vision.ImageSegmenter.create_from_options(opts) as seg:
            res = seg.segment(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_np),
            ))
            # COPY inside the context. numpy_view() is a view into a buffer
            # owned by MediaPipe; leaving the `with` closes the segmenter and
            # frees it, so returning the view hands back freed memory. It
            # sometimes still held valid data and sometimes read as all
            # zeros -- which every caller then took at face value as "no skin
            # here" / "no clothes here". That is the intermittent failure
            # behind skin gate 112,842px -> 0, clothes 15,367px -> 0 and head
            # 18,678px -> 0 across otherwise identical runs.
            cat = np.array(res.category_mask.numpy_view(), copy=True)
        return cat
    except Exception as exc:  # noqa: BLE001
        print(f"[skin_harm] segmenter failed ({type(exc).__name__}: {exc})", flush=True)
        return None


_SEM_HAIR, _SEM_OTHERS = 1, 5


def semantic_head_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(head) = hair + face-skin + accessories (hats), or None.

    Lets compositing put the generated/original boundary along the HEAD
    SILHOUETTE instead of across the body. A straight line or rectangle must
    cross whatever lies at that coordinate -- measured on the athlete, both a
    horizontal ramp (y=354..498) and a head column (x=272..854) cut through
    the shoulders. A head-shaped boundary only ever runs through hair, hat
    edge and a short piece of neck.

    Accessories are included so a hat stays part of the head; without it the
    songkok would fall outside and be restored from the original, undoing the
    swap.
    """
    cat = _semantic_category_mask(rgb_np)
    if cat is None:
        return None
    m = np.isin(cat, (_SEM_HAIR, _SEM_FACE_SKIN, _SEM_OTHERS)).astype(np.float32)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def semantic_hair_accessories_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(hair OR accessories), EXCLUDING face skin, or None.

    ``semantic_head_mask`` returns hair ∪ face-skin ∪ accessories, which is
    the right region when the WHOLE head is being replaced -- a head swap
    wants the donor's hairstyle and treats a hat as part of the head.

    A FACE swap wants the opposite: the donor's face, but the target's own
    hair and headwear. Face-skin is therefore deliberately excluded here.
    Subtracting this mask from a "generated pixels win" region drops hair
    and hats out of it -- so they are restored from the original -- without
    touching the newly generated face.

    Measured need: with hair/accessories left in the keep-mask, a strong
    identity prompt produced a good donor face but also replaced the
    target's winged hat with a plain one and swapped the hairstyle.
    """
    cat = _semantic_category_mask(rgb_np)
    if cat is None:
        return None
    m = np.isin(cat, (_SEM_HAIR, _SEM_OTHERS)).astype(np.float32)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


_SEM_BACKGROUND = 0


def semantic_person_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(anything that is not background), or None.

    Used to clamp the restore mask to the person's silhouette. The head+skin
    mask gets dilated (to catch hair edges) and then blurred, which together
    push it ~23px PAST the hair on a small-face full-body frame. Those extra
    pixels are background, so the composite there blends the generated
    background against the original one -- and the model never reproduces a
    background bit-identically, so the mismatch shows up as a soft ring
    around the head. Clamping to this mask keeps background 100% original.
    """
    cat = _semantic_category_mask(rgb_np)
    if cat is None:
        return None
    m = (cat != _SEM_BACKGROUND).astype(np.float32)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def person_minus_clothes_mask(
    rgb_np: np.ndarray, pil_img: "Image.Image"
) -> np.ndarray | None:
    """rembg person matte MINUS the semantic clothes class, or None.

    The class-based skin mask depends on MediaPipe labelling each limb as
    body-skin, and it does not do that reliably: on the athlete it caught one
    arm and missed the other, so the missed arm fell outside the keep mask and
    was restored tan from the original while its pair came back pale. That
    asymmetry is a segmenter recall failure, not a tuning problem.

    rembg answers a much easier question -- "is this pixel the person?" -- and
    answers it far more reliably than the selfie segmenter's per-class call.
    Subtracting only the clothes class from it yields skin + hair +
    accessories without needing each limb recognised individually. Used as a
    UNION with the class mask, so it can only ever ADD coverage.
    """
    matte = _get_person_matte(pil_img)
    if matte is None:
        return None
    m = (matte.astype(np.float32) / 255.0)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    cloth = semantic_clothes_mask(rgb_np)
    if cloth is not None:
        m = m * (1.0 - np.clip(cloth, 0.0, 1.0))
    return np.clip(m, 0.0, 1.0)


def semantic_person_skin_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(head OR bare skin) = hair + face-skin + body-skin +
    accessories, or None.

    This is the region the GENERATED image should win, so the model's own
    skin rendering survives compositing.

    The head-silhouette restore kept the generated pixels only inside the
    head and restored the ORIGINAL everywhere else -- which discarded every
    bit of skin the model had just recoloured, then a LAB wash was painted
    onto those restored original pixels. Measured on the full-body case:
    by the time harmonization ran, the legs were still at their ORIGINAL
    L=74, because the model's version had already been thrown away one step
    earlier. That is why the result reads as tinted skin rather than skin --
    a mean-shift adds a uniform offset and preserves luminance, so dark legs
    stay dark and merely change hue. No strength setting can fix that; only
    keeping the model's render can, because it carries the right luminance,
    shadow terminators and subsurface look.

    Clothes (4) and background (0) are deliberately excluded, so garments,
    pose and background still come back from the original verbatim.
    """
    cat = _semantic_category_mask(rgb_np)
    if cat is None:
        return None
    m = np.isin(
        cat, (_SEM_HAIR, _SEM_FACE_SKIN, _SEM_BODY_SKIN, _SEM_OTHERS)
    ).astype(np.float32)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


_SEM_CLOTHES = 4


def semantic_clothes_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(clothes) from the multiclass selfie segmenter, or None.

    Exposed so compositing can refuse to paint over a garment. Geometry
    cannot make that decision: a horizontal ramp under the chin blends the
    generated neck against whatever the original has there, which on a
    high-collared robe is black fabric -- producing a tan-to-black gradient
    down the chest that reads as "the colour on the shirt is bad".
    """
    cat = _semantic_category_mask(rgb_np)
    if cat is None:
        return None
    m = (cat == _SEM_CLOTHES).astype(np.float32)
    if m.shape != rgb_np.shape[:2]:
        m = cv2.resize(m, (rgb_np.shape[1], rgb_np.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def _semantic_skin_mask(rgb_np: np.ndarray) -> np.ndarray | None:
    """Per-pixel P(skin) from a segmenter that knows skin from CLOTHES.

    Chroma alone cannot do this. Skin-coloured fabric -- a cream dress, a
    beige skirt, tan trousers -- sits at the same LAB chroma as skin, so a
    colour-only rule recolours the garment: measured on a hanfu dress, the
    "skin" region came back at L=197 (the fabric, not the wearer) and got
    darkened by 59 L-points. No threshold separates those two populations
    because they genuinely overlap; the discriminator has to be semantic.

    This model emits explicit body-skin / face-skin / clothes classes, so
    clothing is excluded by label rather than by hoping its colour differs.
    Returns None when unavailable, and the caller falls back to the
    colour-only path.
    """
    import os as _os  # noqa: PLC0415

    model_path = next((p for p in _SEM_MODEL_PATHS if _os.path.exists(p)), None)
    if model_path is None:
        print(
            "[skin_harm] semantic segmenter model not found -- falling back to "
            "colour-only skin detection (skin-coloured CLOTHING may be recoloured)",
            flush=True,
        )
        return None
    try:
        cat = _semantic_category_mask(rgb_np)
        if cat is None:
            return None
        skin = np.isin(cat, (_SEM_BODY_SKIN, _SEM_FACE_SKIN)).astype(np.float32)
        if skin.shape != rgb_np.shape[:2]:
            skin = cv2.resize(
                skin, (rgb_np.shape[1], rgb_np.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.clip(skin, 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[skin_harm] semantic segmenter failed ({type(exc).__name__}: {exc}) "
            "-- falling back to colour-only skin detection",
            flush=True,
        )
        return None


def _skin_likeness(
    rgb_np: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    *,
    tolerance: float = 2.6,
    knee: float = 0.22,
    confident: float = 0.35,
) -> np.ndarray:
    """Continuous 0..1 "how much does this pixel look like the reference skin".

    Replaces the old binary limb-mask ∩ HSV ∩ matte intersection. A binary
    mask ALWAYS has an edge, and a tone change that stops at that edge is
    visible as a seam no matter how wide the feather -- that is the
    "two layers / different mask beyond the chin" artifact. It also relied on
    fixed global HSV thresholds, which silently missed most exposed skin
    (measured coverage of the athlete's arms: mid=0.20, upper=0.00).

    Instead, score every pixel by how close its CHROMA is to a reference
    skin distribution measured from this specific photo. Chroma (LAB a/b)
    identifies skin far more reliably than absolute HSV cutoffs, and
    deliberately ignoring L means the same arm reads as skin in direct light
    and in shadow -- so shading survives instead of being masked out.
    """
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Floors on the reference spread. A cheek patch is uniform, so its own
    # MAD is tiny (measured a=+-4.5, b=+-7.4 on the athlete) -- but skin
    # chroma across a whole body varies far more than that, because strong
    # light desaturates it. Using the raw cheek MAD scored his sunlit arm
    # (12-20 LAB units away) as non-skin and left it untouched, which is the
    # "arms are still the wrong colour" symptom. The floors widen the
    # acceptance to a realistic whole-body spread while staying far tighter
    # than the distance to clothing (his blue vest sits ~37 units out on b).
    sa = max(float(ref_std[1]), 6.0)
    sb = max(float(ref_std[2]), 9.0)
    da = (lab[:, :, 1] - float(ref_mean[1])) / sa
    db = (lab[:, :, 2] - float(ref_mean[2])) / sb
    d2 = da * da + db * db
    w = np.exp(-0.5 * d2 / (tolerance * tolerance))
    # Near-black pixels (deep shadow, black clothing) carry no usable chroma
    # -- their a/b sit near neutral and can score as "skin" by accident.
    L = lab[:, :, 0]
    w *= np.clip((L - 8.0) / 18.0, 0.0, 1.0)
    # Map the raw score onto [0,1] between `knee` (definitely not skin) and
    # `confident` (definitely skin, transfer fully).
    #
    # This previously divided by (1.0 - knee), i.e. it only reached 1.0 for a
    # pixel scoring an exact 1.0 -- which exp(-d^2/..) essentially never
    # returns, since that needs chroma *identical* to the reference. Real skin
    # scored 0.3-0.6 and so only ever travelled 30-60% of the way to the donor
    # tone. That single line caused both remaining artifacts:
    #   * arms/hands stayed visibly near their original tone, and
    #   * a step at the head-exclusion line, because pixels above it are 100%
    #     generated donor face while pixels just below were only ~40% shifted.
    # The step was a difference in TRANSFER DISTANCE, not a spatial mask edge,
    # which is why widening feathers never removed it.
    #
    # Saturating at `confident` means the neck lands on the same tone as the
    # face it meets, so the two sides converge instead of stepping.
    #
    # `confident` also sets HOW MUCH of the transfer actually lands: with it
    # at 0.45 the mean applied weight measured 0.681, so a correct +38 L
    # target only delivered +26 and the arms read as under-corrected. At 0.35
    # a shadowed arm scoring 0.356 goes 0.64 -> 1.00. `knee` is unchanged, so
    # clothing is unaffected: the vest (0.06) and robe (0.00) still map to
    # exactly 0.00.
    w = (w - knee) / max(1e-6, confident - knee)
    w = np.clip(w, 0.0, 1.0)
    # Smoothstep: C1-continuous at both ends, so saturation adds no hard edge.
    return w * w * (3.0 - 2.0 * w)


def _robust_lab_stats(flat_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median_lab, robust_std_lab) via median + MAD, trimming the top/bottom
    10% per channel (highlights/shadows/outliers) before computing stats --
    a plain mean/std over a skin patch is easily dragged by a handful of
    specular-highlight or deep-shadow pixels."""
    med = np.median(flat_lab, axis=0)
    mad = np.median(np.abs(flat_lab - med), axis=0)
    robust_std = mad * 1.4826 + 1e-6  # MAD -> std-equivalent for a normal dist
    return med, robust_std


def _face_skin_lab_stats(
    rgb_np: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Robust skin tone from a face box, chosen by CONTENT not by geometry.

    A fixed rectangle inside the face box keeps landing on things that are
    not skin, and the failure is silent and directional: the old wide window
    caught mustache/lips/beard and read L=131 on a pale donor; the narrower
    lateral window then caught brow shadow or hair and read L=102 -- darker
    than the target's own face, which is impossible for a paler person under
    the same light. Every downstream fix then aims at a wrong target.

    Instead select pixels by skin-likeness within the box and read the tone
    off those. Facial hair, brows, nostrils, lips and cast shadow all fall
    out on chroma or luminance, so the answer no longer depends on guessing
    a rectangle that avoids them.
    """
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(rgb_np.shape[1], x1); y1 = min(rgb_np.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    box = rgb_np[y0:y1, x0:x1]
    lab = cv2.cvtColor(box, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    L, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    # Warm chroma is what separates skin from hair, brows and cloth; a wide
    # band keeps every complexion in while dropping neutral/cool pixels.
    warm = (a > 132.0) & (b > 132.0) & (a < 180.0) & (b < 190.0)
    if int(warm.sum()) < 64:
        return None
    Lw = L[warm]
    # Drop the darkest 10% (nostrils, cast shadow, hair edges) and the
    # brightest 5% (specular highlight), then read the tone off what's left.
    #
    # This cut used to be 35%, which is directionally unsafe. It assumes dark
    # pixels are beard/shadow -- true for a pale donor, false for a dark-
    # complexioned one, where the dark pixels ARE the skin. Discarding the
    # darkest third of a dark face biases the reported tone brighter, and
    # since this value is the TARGET of the transfer, the whole body then
    # aims too light and stops there.
    #
    # GPU-measured: a dark donor read L=106 here while the lateral-cheek
    # estimate of the same face read L=81. The body transferred to 106 and
    # remained permanently lighter than the face -- diagnosed repeatedly as a
    # mask-coverage problem, because a body that stops short looks identical
    # to a body that was never fully selected.
    #
    # 10% still removes nostrils and hard shadow, and _robust_lab_stats below
    # is median/MAD-based, so it is already outlier-resistant; the aggressive
    # extra haircut was redundant with it as well as biased.
    lo_c, hi_c = np.percentile(Lw, 10.0), np.percentile(Lw, 95.0)
    keep = warm & (L >= lo_c) & (L <= hi_c)
    if int(keep.sum()) < 32:
        keep = warm
    mean, std = _robust_lab_stats(flat[keep])
    # Cross-check against the lateral-cheek reading, which samples with an
    # explicit margin in from the box edge and cannot drift bright by
    # percentile choice. If the content-selected answer is much BRIGHTER, it
    # has eaten real skin; prefer the cheek reading. Only guards the bright
    # direction -- a darker answer here is legitimate (it means the cheek
    # rectangle caught background or hair).
    try:
        cheek_mean, cheek_std = _cheek_lab_stats(rgb_np, x0, y0, x1, y1)
        if float(mean[0]) - float(cheek_mean[0]) > 12.0:
            print(
                f"[skin_harm] face-skin tone L={float(mean[0]):.0f} is "
                f"{float(mean[0]) - float(cheek_mean[0]):.0f} brighter than the "
                f"cheek reading L={float(cheek_mean[0]):.0f}; using the cheek "
                "value so a dark complexion is not read as light",
                flush=True,
            )
            return cheek_mean, cheek_std
    except Exception:  # noqa: BLE001 — cross-check must never break the read
        pass
    return mean, std


def _cheek_lab_stats(
    result_np: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(median_lab, robust_std_lab) from the donor head's two lateral cheeks.

    The window used to span 0.40-0.72 of face height across nearly the full
    width, which covers the nose, nostrils, lips, mustache and beard shadow --
    not skin. On a stubbled or smiling donor that pulls the reference dark and
    warm, so the "donor skin tone" the whole transfer aims at was wrong at the
    source: measured tgt L=131 with chroma indistinguishable from tan skin,
    which makes a pale complexion unreachable no matter how the weights are
    tuned.

    Sample the two lateral cheeks instead -- below the eyes, above the mouth
    line, and skipping the central nose strip (specular highlight + nostril
    shadow). That band is the largest genuinely-skin, mostly-flat region of a
    face and is the standard place to read complexion from.
    """
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    py0 = y0 + int(fh * 0.34)
    py1 = y0 + int(fh * 0.54)
    def _stats(flat_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(
            flat_rgb.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        return _robust_lab_stats(lab.reshape(-1, 3))

    patches = []
    # Kept a margin in from the box edge: a face box that is even slightly
    # loose puts an edge-hugging patch on the BACKGROUND, and a donor shot on
    # a white backdrop then reads as L=255 neutral -- a "skin tone" that is
    # not skin at all and would drive the whole transfer to grey.
    for fx0, fx1 in ((0.16, 0.36), (0.64, 0.84)):
        cx0 = x0 + int(fw * fx0)
        cx1 = x0 + int(fw * fx1)
        p = result_np[py0:py1, cx0:cx1]
        if p.size:
            patches.append(p.reshape(-1, 3))

    if patches:
        mean, std = _stats(np.concatenate(patches, axis=0))
        # Plausibility guard: real skin always carries some warm chroma. A
        # near-neutral sample means the patches missed the face (backdrop,
        # hair, deep shadow), so fall back to the central band rather than
        # aiming the transfer at a colour no skin has.
        chroma = abs(float(mean[1]) - 128.0) + abs(float(mean[2]) - 128.0)
        if chroma >= 8.0:
            return mean, std

    central = result_np[py0:py1, x0 + int(fw * 0.18): x0 + int(fw * 0.82)]
    if central.size == 0:
        central = result_np[y0:y1, x0:x1]
    if central.size == 0:
        central = result_np
    return _stats(central.reshape(-1, 3))


def _reinhard_transfer(
    region_rgb: np.ndarray,
    src_mean: np.ndarray,
    src_std: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
    *,
    contrast_clamp: tuple[float, float] = (0.55, 1.30),
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

    The lower clamp matters more than it looks. The selected skin spans very
    different tones at once -- a lit chest around L=120 and tan arms down at
    L=60 -- while the donor reference is a single uniform cheek patch. A pure
    mean-shift (ratio 1.0) moves both ends by the same amount, so with
    src=96 -> tgt=124 the chest overshoots to 148 while the arm only reaches
    88 and still reads as its original tan. Allowing more compression pulls
    the two ends TOWARD the face tone instead of translating them past it:
        ratio 0.75 (old floor):  arm 60 -> 97,   chest 120 -> 142
        ratio 0.55 (now):        arm 60 -> 104,  chest 120 -> 139
    The cost is some flattening of shading contrast within skin, so this is
    a genuine trade -- lower it further only if arms still read too dark, and
    raise it if skin starts looking matte.
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
    luminance_preserve: float = 0.85,
    neck_ramp_frac: float = 0.35,
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
    # 2. Continuous skin-likeness weight (no binary mask -> no mask edge)
    # ------------------------------------------------------------------
    # Reference = the TARGET's own original face, which is definitionally
    # their skin tone in this photo's lighting. Scoring arms/hands against
    # that (rather than fixed HSV cutoffs) is what actually finds exposed
    # skin: a tanned arm is close to its owner's face in chroma, while a
    # black robe or a sand background is not.
    orig_np = np.asarray(body_full_pil.convert("RGB"), dtype=np.uint8)
    if orig_np.shape[:2] != (H, W):
        orig_np = cv2.resize(orig_np, (W, H), interpolation=cv2.INTER_LINEAR)
    # Measure BOTH ends of the transfer the same way, or the difference
    # between them is partly just a difference in how they were sampled.
    _ref = _face_skin_lab_stats(orig_np, head_x0, head_y0, head_x1, head_y1)
    if _ref is not None:
        ref_mean, ref_std = _ref
    else:
        ref_mean, ref_std = _cheek_lab_stats(
            orig_np, head_x0, head_y0, head_x1, head_y1
        )
    info["ref_lab_mean"] = [round(float(x), 2) for x in ref_mean]

    weight = _skin_likeness(result_np, ref_mean, ref_std)
    weight *= np.clip(person_matte.astype(np.float32) / 255.0, 0.0, 1.0)
    # Semantic gate: keep only what a skin/clothes-aware segmenter calls skin.
    # Without this the colour rule alone repaints skin-coloured garments.
    _sem = _semantic_skin_mask(result_np)
    # Union in rembg(person - clothes) - minus the head - as a skin FLOOR.
    #
    # The multiclass segmenter has to label each limb as body-skin and it does
    # not do so reliably: krea2's restore already works around exactly this
    # ("caught one arm and missed the other"). When it misses a limb here the
    # gate contributes nothing there, the pixel keeps only the graded colour
    # score, and a shaded leg ends up visibly half-corrected next to a
    # fully-corrected one -- measured w_full_frac=0.05 with the miss.
    #
    # rembg answers the much easier "is this the person?" question reliably,
    # and clothes are already subtracted from it. The head is subtracted too,
    # so hair overhanging a shoulder is never mistaken for skin and recoloured.
    if _sem is not None:
        try:
            _pmc = person_minus_clothes_mask(result_np, result_pil)
            if _pmc is not None:
                _pm_head = semantic_head_mask(result_np)
                if _pm_head is not None:
                    _pmc = _pmc * (1.0 - np.clip(_pm_head, 0.0, 1.0))
                _pre_union = int((_sem > 0.5).sum())
                _sem = np.maximum(_sem, _pmc)
                info["semantic_skin_pmc_union"] = [
                    _pre_union, int((_sem > 0.5).sum())
                ]
                print(
                    f"[skin_harm] skin gate widened by rembg(person-clothes-head): "
                    f"{_pre_union} -> {int((_sem > 0.5).sum())}px so a limb the class "
                    "segmenter missed is corrected as fully as its pair",
                    flush=True,
                )
        except Exception as _uexc:  # noqa: BLE001
            print(
                f"[skin_harm] person-minus-clothes union failed "
                f"({type(_uexc).__name__}: {_uexc}); using the class mask alone",
                flush=True,
            )
    # Rescue skin the CLOTHES class has vetoed.
    #
    # Observed as a hard vertical edge down one leg: pale/original on one side,
    # correctly toned on the other. A soft under-correction fades; a hard edge
    # is a mask boundary. A bare leg rendered in a pale cream tone reads as a
    # stocking or legging to the class segmenter, so it is labelled CLOTHES --
    # which both drops it from this gate and marks it for original-pixel
    # protection upstream. The limb is then frozen at its original tone with a
    # class boundary running down it, and no weighting change can reach it
    # because the veto happens before weighting.
    #
    # `weight` here is _skin_likeness scored against THIS person's own face in
    # THIS photo's light, which is the right discriminator: their leg is close
    # to their face in chroma, a genuine garment generally is not. The
    # threshold is deliberately high -- colour alone scored a blue jersey at
    # 1.00 in an earlier regression, so this only overrides the class veto
    # where the colour evidence is strong AND rembg agrees the pixel is the
    # person. Set skin_rescue_from_clothes to 0 to disable if a skin-coloured
    # garment starts being recoloured.
    _rescue_thr = _SKIN_RESCUE_FROM_CLOTHES
    if _sem is not None:
        try:
            _pm_norm = np.clip(person_matte.astype(np.float32) / 255.0, 0.0, 1.0)
            _rescue = ((weight >= _rescue_thr) & (_pm_norm > 0.5)).astype(np.float32)
            _rescued = int(((_rescue > 0.5) & (_sem <= 0.5)).sum())
            if _rescued > 0:
                _sem = np.maximum(_sem, _rescue)
                info["skin_rescued_from_clothes"] = _rescued
                print(
                    f"[skin_harm] rescued {_rescued}px the clothes class vetoed but "
                    f"that score >= {_rescue_thr:.2f} against this person's own face "
                    "and sit inside the rembg matte -- bare skin rendered in a "
                    "garment-like tone is not frozen at the original",
                    flush=True,
                )
        except Exception as _rexc:  # noqa: BLE001
            print(
                f"[skin_harm] clothes-veto rescue failed "
                f"({type(_rexc).__name__}: {_rexc}); class mask used unchanged",
                flush=True,
            )
    info["semantic_skin"] = _sem is not None
    if _sem is not None:
        # Sanity-check the gate before trusting it to veto. Measured on the
        # athlete: the segmenter returned 79k CLOTHES pixels but ZERO skin, so
        # multiplying by it zeroed the whole weight field and the recolour was
        # silently skipped ("too_few_skin_px") -- the arms simply never
        # changed. A gate that finds no skin at all on a person with a bare
        # face and arms is wrong, not authoritative, so fall back to the
        # colour-only weight rather than letting it suppress everything.
        _sem_px = int((_sem > 0.5).sum())
        _sem_frac = _sem_px / float(max(1, H * W))
        _min_frac = 0.005
        if _sem_frac < _min_frac:
            info["semantic_skin"] = False
            info["semantic_skin_rejected"] = round(_sem_frac, 5)
            # Colour-only weighting on its own is NOT a safe fallback: chroma
            # distance alone scores this athlete's blue/purple jersey at
            # weight 1.00 (its raw score, 0.46, still clears the knee). That
            # measurably wrecked the result -- mask area went 78,938 ->
            # 150,626 px, src b-channel 149 -> 130 as jersey pixels entered
            # the statistics, and the computed shift collapsed from +21.6 to
            # +5.9, so the body barely changed colour at all.
            #
            # Re-apply the HSV skin heuristic that the semantic gate had been
            # standing in for. It discriminates on HUE, which is exactly what
            # separates skin from a blue/purple garment: tan (h=12) and pale
            # (h=13) skin pass, blue (h=124) and purple (h=138) are rejected.
            _hsv = _hsv_skin_mask(result_np).astype(np.float32) / 255.0
            _hk = max(3, (int(max(H, W) * 0.01) * 2 + 1))
            _hsv = cv2.GaussianBlur(_hsv, (_hk, _hk), 0)
            weight *= np.clip(_hsv, 0.0, 1.0)
            info["semantic_skin_fallback"] = "hsv_hue"
            print(
                f"[skin_harm] semantic gate REJECTED - it found only "
                f"{_sem_px}px ({_sem_frac:.3%}) of skin, below {_min_frac:.1%}. "
                "Treating that as a segmenter failure, not as 'no skin here'. "
                "Falling back to colour + HSV-hue weighting (hue rejects the "
                "blue/purple jersey that chroma alone scored as skin).",
                flush=True,
            )
        else:
            # Soften the class boundary so the gate never contributes a hard
            # edge of its own, then keep a floor of 0 outside skin: clothing
            # must go to zero, not merely be attenuated.
            _k = max(3, (int(max(H, W) * 0.01) * 2 + 1))
            _sem_soft = cv2.GaussianBlur(_sem, (_k, _k), 0)
            # Promote to FULL weight inside validated semantic skin instead of
            # attenuating it by the colour score.
            #
            # `weight` above is _skin_likeness: a graded chroma distance from
            # the target's ORIGINAL face. As a skin FINDER that is sound, but
            # as a per-pixel STRENGTH it under-corrects exactly where skin is
            # shaded, because shadow moves a pixel away from the sunlit-face
            # reference and the score falls. Multiplying the two then caps the
            # correction at the colour score everywhere: GPU-measured
            # w_mean=0.727 with only w_full_frac=0.049 of selected pixels at
            # full strength, which is why one leg came back visibly
            # half-corrected while the rest of the body matched -- the darker
            # leg simply scored lower and got proportionally less of the
            # shift.
            #
            # Once the semantic gate has passed its own sanity check it has
            # already answered "is this skin" on class, not colour, so the
            # colour score has no remaining job. Take the max so semantic skin
            # gets the full transfer regardless of shading, while everything
            # the segmenter rejects still goes to zero (floor preserved) and
            # the colour score still carries any skin the segmenter missed.
            _sem_c = np.clip(_sem_soft, 0.0, 1.0)
            _matte = np.clip(person_matte.astype(np.float32) / 255.0, 0.0, 1.0)
            weight = np.maximum(weight * _sem_c, _sem_c * _matte)
            print(
                f"[skin_harm] semantic skin gate applied (skin px={_sem_px}); "
                "full transfer weight inside semantic skin so shaded limbs are "
                "not left half-corrected",
                flush=True,
            )
    weight[:head_excl] = 0.0
    # Smooth the weight field itself. This is the only "feather" now, and it
    # is applied to a continuous field rather than to a hard mask edge, so
    # there is no boundary for it to reveal.
    blur_k = max(3, (int(max(H, W) * 0.02) * 2 + 1))
    weight = cv2.GaussianBlur(weight, (blur_k, blur_k), 0)
    # Re-zero AFTER the blur. Blurring smears nonzero weight from just below
    # head_excl back up across the line -- the same "hard-zero, then blur"
    # ordering bug diagnosed earlier this session for the isolated-layer
    # composite mask. GPU-confirmed here too: head region max_diff was 19
    # (pair1) and 11 (pair2) before this reorder, violating the byte-
    # identical guarantee this function promises.
    weight[:head_excl] = 0.0
    # Ease the transfer in over the rows just below the head instead of
    # switching it on in a single row. The head region receives NO transfer by
    # design (it must stay byte-identical), so a hard cut puts full-strength
    # recolouring immediately against untouched pixels: with a large shift
    # (measured -27.6 L on the robed figure) that boundary renders as a dark
    # band under the chin. The ramp spans a fraction of head height, so it
    # scales with the subject rather than being a fixed pixel count.
    ramp_rows = max(1, int(head_h * float(neck_ramp_frac)))
    r0, r1 = head_excl, min(H, head_excl + ramp_rows)
    if r1 > r0:
        ramp = np.linspace(0.0, 1.0, r1 - r0, dtype=np.float32)
        # smoothstep so both ends meet their neighbours with zero slope
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        weight[r0:r1] *= ramp[:, None]
    info["neck_ramp_rows"] = int(ramp_rows)
    weight = np.clip(weight, 0.0, 1.0)

    # Report what the transfer will actually DO, not just how many pixels it
    # selected. A high skin_px with a near-zero mean weight means "selected a
    # lot, changed almost nothing" -- indistinguishable from "didn't run" in
    # the output image, which is exactly the state that kept being misread as
    # a masking failure. matte_frac catches the other candidate: on a photo
    # whose background is a blurred crowd (skin-coloured), a person matte that
    # leaks would drag src_mean toward the background and flatten the shift.
    _sel = weight > 0.15
    info["weight_mean_over_selected"] = (
        round(float(weight[_sel].mean()), 3) if _sel.any() else 0.0
    )
    info["weight_frac_full"] = round(float((weight > 0.9).mean()), 4)
    info["matte_frac"] = round(
        float((person_matte > 16).mean()), 4
    )

    skin_px = int((weight > 0.15).sum())
    info["skin_px"] = skin_px
    _t = H // 3
    for _name, _sl in (
        ("upper", slice(0, _t)),
        ("mid", slice(_t, 2 * _t)),
        ("lower", slice(2 * _t, H)),
    ):
        _band = weight[_sl]
        info[f"cover_{_name}"] = (
            round(float((_band > 0.15).mean()), 4) if _band.size else 0.0
        )
    print(
        f"[skin_harm] weighted skin_px={skin_px} coverage "
        f"upper={info['cover_upper']} mid={info['cover_mid']} "
        f"lower={info['cover_lower']} "
        f"w_mean={info['weight_mean_over_selected']} "
        f"w_full_frac={info['weight_frac_full']} "
        f"matte_frac={info['matte_frac']}",
        flush=True,
    )
    if skin_px < 300:
        info["skip_reason"] = "too_few_skin_px"
        print("[skin_harm] skipped — too few skin pixels", flush=True)
        return result_pil, info

    # ------------------------------------------------------------------
    # 3. Donor tone (target of the shift) + this body's current skin stats
    # ------------------------------------------------------------------
    _tgt = _face_skin_lab_stats(
        result_np, head_x0, head_y0, head_x1, head_y1
    )
    if _tgt is not None:
        tgt_mean, tgt_std = _tgt
        info["tgt_source"] = "face_skin_pixels"
    else:
        tgt_mean, tgt_std = _cheek_lab_stats(
            result_np, head_x0, head_y0, head_x1, head_y1
        )
        info["tgt_source"] = "cheek_rect_fallback"
    sel = weight > 0.35
    if int(sel.sum()) < 300:
        sel = weight > 0.15
    lab_all = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = _robust_lab_stats(lab_all[sel])
    info["src_lab_mean"] = [round(float(x), 2) for x in src_mean]
    info["tgt_lab_mean"] = [round(float(x), 2) for x in tgt_mean]
    # The L shift a fully-weighted pixel receives, scaled by the mean weight
    # actually applied. This is the single number that says whether the result
    # can look different: a large src->tgt gap still produces no visible change
    # if the mean weight is small, and that combination is precisely what has
    # been misdiagnosed as a mask problem repeatedly.
    info["effective_dL"] = round(
        float(tgt_mean[0] - src_mean[0])
        * float(info.get("weight_mean_over_selected", 0.0))
        * float(transfer_strength),
        2,
    )

    # ------------------------------------------------------------------
    # 4. Chroma-led transfer, modulated by the continuous weight
    # ------------------------------------------------------------------
    # NOTE: luminance_preserve is intentionally NOT forwarded. _reinhard_transfer
    # now uses a mean/deviation split, which protects shading via the deviation
    # term instead -- see its docstring for why holding L back actively broke
    # light<->dark skin transfer. The parameter is still accepted by this
    # function so existing callers keep working.
    matched_rgb = _reinhard_transfer(
        result_np, src_mean, src_std, tgt_mean, tgt_std,
    )
    w3 = (weight * float(transfer_strength))[..., None]
    result_np = np.clip(
        matched_rgb.astype(np.float32) * w3
        + result_np.astype(np.float32) * (1.0 - w3),
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
        f"[skin_harm] done — strength={transfer_strength} feather={feather_px}px "
        f"| L {info.get('src_lab_mean',[None])[0]} -> "
        f"{info.get('tgt_lab_mean',[None])[0]} "
        f"(effective dL={info.get('effective_dL')})",
        flush=True,
    )
    return Image.fromarray(result_np), info


def restore_stripped_garment(
    rendered: Image.Image,
    original: Image.Image,
    head_x0: int,
    head_y0: int,
    head_x1: int,
    head_y1: int,
    *,
    min_stripped_frac: float = 0.003,
    min_stripped_px: int = 400,
    open_px: int = 3,
    min_component_frac: float = 0.0015,
    feather_px: int = 12,
    neck_excl_frac: float = 0.04,
) -> tuple[Image.Image, dict]:
    """Paste back ORIGINAL garment pixels wherever the render turned fabric
    into skin, and nowhere else.

    Three prompt-text fixes for "a covered subject can come back with
    clothing stripped near the torso" were tried and rejected (see
    krea2.py's run_simple_full_body and DEFAULTS in chain.py): a
    do-not-expose prohibition moved the stripping to the sleeves instead of
    fixing it; naming garment types turned an unrelated bare-armed subject's
    polo shirt into a bathrobe ("on this route the model draws whatever the
    prompt names"); dropping or scoping the skin-recolour sentence for a
    "covered" subject fixed the stripping but left genuinely-visible skin
    (praying hands) with no tone instruction, rendering it visibly
    white/mismatched -- twice, under two different wordings.

    This is a different kind of fix: not a prompt change, a post-render
    diff. `clothes_before` (semantic_clothes_mask on the ORIGINAL photo) AND
    `skin_after` (a skin classifier on the RENDER) together identify exactly
    the pixels that were fabric and are now skin -- structurally, not by
    threshold tuning, so it can never touch skin that was already bare
    before the swap (hands, neck, face). That is what keeps it from
    reproducing the white-hands regression: those pixels were never
    "clothes_before", so they are never in the restore mask to begin with.

    Head/neck exclusion is BOTH semantic (semantic_head_mask) and geometric
    (a row-zero from the passed-in head box), not semantic alone --
    _semantic_category_mask has a documented failure mode where it silently
    returns an all-zero array (not None) on some frames, which would make a
    semantic-only exclusion evaluate to "no exclusion at all" and paste
    original pixels under the just-swapped neck. The geometric zero is
    applied both before AND after the Gaussian blur, mirroring
    extend_skin_harmonization's own fix for exactly this reorder bug
    (that function's docstring/comments measured max_diff 19 and 11 in the
    head region before the reorder -- same mistake, not repeated here).

    Speckle guard: cv2.INTER_LINEAR resizing inside the mask functions
    produces a thin fractional halo along EVERY garment edge on every
    render, not just a buggy one. Left unguarded this would "fix" the target
    case while adding a hairline fringe of restored-original pixels along
    every collar/cuff on cases that were already correct -- the same
    "fixed one case, broke a working one" trap as the earlier do-not-expose
    and garment-naming attempts. A morphological opening removes the
    speckle, then a connected-component AREA filter (not just a total pixel
    count) drops anything too small to be the real stripped region -- a
    total-pixel-count check alone can be fooled by a long thin halo summing
    past the floor while still being visually a hairline.

    Blends actual original garment pixels, not a LAB-shifted statistical
    correction -- extend_skin_harmonization's own docstring notes a mean-shift
    "cannot produce skin rendered under the scene's light"; restoring the
    real original pixels sidesteps that critique entirely, since they are
    already lit consistently with whatever of the same garment the render
    got right (img2img on this route is explicitly instructed to preserve
    clothing/lighting, and mostly does for the parts it does not strip).

    Fails closed: returns `rendered` unchanged whenever either classifier is
    unavailable, or the detected area is below the noise floor -- never
    breaks a render.
    """
    info: dict = {"applied": False}
    W, H = rendered.size
    rendered_np = np.asarray(rendered.convert("RGB"), dtype=np.uint8).copy()
    original_np_full = np.asarray(original.convert("RGB"), dtype=np.uint8)
    if original_np_full.shape[:2] != (H, W):
        original_np_full = cv2.resize(
            original_np_full, (W, H), interpolation=cv2.INTER_LINEAR
        )

    clothes_before = semantic_clothes_mask(original_np_full)
    if clothes_before is None:
        info["reason"] = "segmenter_unavailable_clothes_before"
        print(
            "[skin_harm] restore_stripped_garment skipped -- clothes "
            "segmenter unavailable on the original photo",
            flush=True,
        )
        return rendered, info

    skin_after = _semantic_skin_mask(rendered_np)
    if skin_after is None:
        info["reason"] = "segmenter_unavailable_skin_after"
        print(
            "[skin_harm] restore_stripped_garment skipped -- skin "
            "segmenter unavailable on the render",
            flush=True,
        )
        return rendered, info

    stripped = np.clip(clothes_before, 0.0, 1.0) * np.clip(skin_after, 0.0, 1.0)

    # Head/neck exclusion -- semantic AND geometric, see docstring.
    head_h = max(1, head_y1 - head_y0)
    head_excl = min(H, max(0, head_y1) + max(2, int(head_h * neck_excl_frac)))
    _head_sem = semantic_head_mask(rendered_np)
    if _head_sem is not None:
        stripped = stripped * (1.0 - np.clip(_head_sem, 0.0, 1.0))
    stripped[:head_excl] = 0.0

    raw_px = int((stripped > 0.5).sum())
    info["raw_stripped_px"] = raw_px
    if raw_px < min_stripped_px:
        info["reason"] = f"below_pixel_floor:{raw_px}<{min_stripped_px}"
        print(
            f"[skin_harm] restore_stripped_garment skipped -- only {raw_px}px "
            f"detected (< {min_stripped_px})",
            flush=True,
        )
        return rendered, info

    # Speckle guard: remove isolated noise, then keep only components large
    # enough to be a real stripped region rather than an edge-resample halo.
    binary = (stripped > 0.5).astype(np.uint8)
    k = max(1, int(open_px)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    frame_area = float(H * W)
    min_comp_px = max(1.0, min_component_frac * frame_area)
    kept = np.zeros_like(opened)
    kept_area = 0
    for comp_id in range(1, n_comp):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area >= min_comp_px:
            kept[labels == comp_id] = 1
            kept_area += int(area)
    info["kept_component_px"] = kept_area

    stripped_frac = kept_area / frame_area
    info["stripped_frac"] = round(stripped_frac, 5)
    if kept_area < min_stripped_px or stripped_frac < min_stripped_frac:
        info["reason"] = (
            f"below_area_floor:frac={stripped_frac:.5f}<{min_stripped_frac}"
        )
        print(
            f"[skin_harm] restore_stripped_garment skipped -- surviving area "
            f"{stripped_frac:.4%} below floor {min_stripped_frac:.4%} "
            "(likely edge-resample noise, not a real stripped region)",
            flush=True,
        )
        return rendered, info

    weight = kept.astype(np.float32)
    blur_k = max(3, (int(feather_px) // 2) * 2 + 1)
    weight = cv2.GaussianBlur(weight, (blur_k, blur_k), 0)
    # Re-zero AFTER the blur -- blurring smears nonzero weight back up across
    # the head-exclusion line otherwise. Same reorder fix as
    # extend_skin_harmonization; see this function's docstring.
    weight[:head_excl] = 0.0
    weight = np.clip(weight, 0.0, 1.0)

    # Tone-match the restored patch to the RENDER's own grade before
    # pasting, not the original photo's raw grade.
    #
    # Measured on GPU: pasting raw original pixels produced a visible
    # rectangular patch on both shoulders -- the diffusion pass shifts
    # global exposure/white-balance even at high denoise-preserve, so the
    # untouched original photo and the render are two different colour
    # grades, and a boundary between them is visible exactly where they
    # differ (the same "boundary is visible when the two sides differ"
    # mechanism _raw_model exists to avoid elsewhere in this file).
    #
    # This is NOT the LAB-wash critique repeated -- that critique is about
    # shifting SKIN tone, where "cannot produce skin rendered under the
    # scene's light" is a real problem because skin has to look alive under
    # specific lighting. This is a garment, and the reference sample is
    # GARMENT PIXELS ADJACENT TO THE PATCH, in the SAME render, at the SAME
    # photo coordinates -- so the transfer target is literally "how this
    # exact render already lit this exact fabric a few centimetres away",
    # not a generic donor-face colour. Falls back to unmatched original
    # pixels (the pre-fix behaviour) if too little reference garment
    # survives to measure a transform from.
    ref_mask = (
        (np.clip(clothes_before, 0.0, 1.0) > 0.5) & (kept < 1)
    )
    ref_px = int(ref_mask.sum())
    info["tone_ref_px"] = ref_px
    paste_source = original_np_full
    if ref_px >= 300:
        lab_orig = cv2.cvtColor(original_np_full, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_rend = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = _robust_lab_stats(lab_orig[ref_mask])
        tgt_mean, tgt_std = _robust_lab_stats(lab_rend[ref_mask])
        paste_source = _reinhard_transfer(
            original_np_full, src_mean, src_std, tgt_mean, tgt_std
        )
        info["tone_matched"] = True
    else:
        info["tone_matched"] = False
        print(
            f"[skin_harm] restore_stripped_garment -- only {ref_px}px of "
            "reference garment survived to tone-match against (< 300); "
            "pasting the original photo's own grade unmatched",
            flush=True,
        )

    w3 = weight[..., None]
    out_np = np.clip(
        paste_source.astype(np.float32) * w3
        + rendered_np.astype(np.float32) * (1.0 - w3),
        0, 255,
    ).astype(np.uint8)

    hx0, hy0 = max(0, head_x0), max(0, head_y0)
    hx1 = min(W, head_x1)
    head_before = rendered_np[hy0:head_excl, hx0:hx1]
    head_after = out_np[hy0:head_excl, hx0:hx1]
    info["head_identical"] = bool(np.array_equal(head_before, head_after))

    info["applied"] = True
    print(
        f"[skin_harm] restore_stripped_garment applied -- restored "
        f"{stripped_frac:.4%} of the frame ({kept_area}px) where the "
        "original was fabric and the render turned it into skin; "
        f"head_identical={info['head_identical']}",
        flush=True,
    )
    return Image.fromarray(out_np), info


def build_garment_containment_mask(
    body_full: Image.Image,
    head_mask: Image.Image,
    *,
    feather_px: int = 12,
) -> tuple[Image.Image | None, dict]:
    """Sampling-time containment mask: white (255) where the model may
    regenerate, black (0) where it must stay pinned to the source latent.

    A post-render pixel-diff patch (restore_stripped_garment, above) can
    restore COLOUR where a covered subject's clothing was stripped, but it
    cannot fix the render having drawn the arm/garment at a slightly
    different shape than the source photo in the first place -- there is no
    original-photo pixel that is "correct" for geometry the sampler never
    produced. Measured on GPU: garment colour and coverage were restored
    correctly, but a sleeve/cuff still showed a geometric misalignment the
    patch could not touch. The fix has to happen at sampling time, not
    after.

    ``head_mask`` MUST be a generously-oversized geometric mask (this
    codebase's own build_head_hair_mask(..., backend="ellipse"), NOT a
    silhouette/segmentation mask measured on the ORIGINAL photo. An earlier
    version of this function computed head coverage from
    semantic_person_skin_mask(body_full) alone, i.e. sized to the TARGET's
    own original hairstyle -- and a donor head/hairstyle is routinely a
    different shape and larger. Measured on GPU: a translucent halo with
    wing-shaped protrusions at ear level appeared around the new head,
    because containment re-pinned the region just outside that
    too-tight boundary back to the ORIGINAL frame, and the original frame
    there was the TARGET's own hair/ear silhouette showing through behind
    the new one -- the exact "translucent oval with wings at ear level"
    failure class this codebase already has a name for elsewhere
    (_append_headwear_policy's docstring, a different bug with the same
    root cause: a too-tight region boundary around a head that changes
    shape). A geometric ellipse sized for ANY head, not measured from this
    specific photo's silhouette, is what CHECKPOINT-17's containment
    attempt used for exactly this reason and did not exhibit this artifact
    -- its only measured problem was the temporal seam DifferentialDiffusion
    now fixes.

    "Free" (mask=1) is ``head_mask`` UNION genuinely-bare skin
    (semantic_person_skin_mask, silhouette-based -- fine for hands/arms
    since those don't change shape or position the way a swapped head
    does). "Protected" (mask=0) is everywhere else. semantic_clothes_mask
    acts as a HARD VETO, but ONLY on the skin contribution, never on
    ``head_mask`` -- the head region must always stay free regardless of
    segmenter noise near the collar, or a clothes misfire could partially
    veto the very region the swap needs to regenerate.

    Fails closed: returns (None, diag) if the skin/clothes segmenter is
    unavailable, so the caller can fall through to the current unmasked
    img2img path unchanged -- never breaks a render.
    """
    info: dict = {}
    rgb = np.asarray(body_full.convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]

    skin = semantic_person_skin_mask(rgb)
    if skin is None:
        info["reason"] = "segmenter_unavailable_person_skin"
        print(
            "[skin_harm] build_garment_containment_mask skipped -- "
            "person/skin segmenter unavailable",
            flush=True,
        )
        return None, info

    clothes = semantic_clothes_mask(rgb)
    if clothes is None:
        info["reason"] = "segmenter_unavailable_clothes"
        print(
            "[skin_harm] build_garment_containment_mask skipped -- "
            "clothes segmenter unavailable",
            flush=True,
        )
        return None, info

    head_arr = np.asarray(head_mask.convert("L"), dtype=np.float32) / 255.0
    if head_arr.shape[:2] != (H, W):
        head_arr = cv2.resize(head_arr, (W, H), interpolation=cv2.INTER_LINEAR)

    skin_free = np.clip(skin, 0.0, 1.0) * (1.0 - np.clip(clothes, 0.0, 1.0))
    free = np.maximum(np.clip(head_arr, 0.0, 1.0), skin_free)

    blur_k = max(3, (int(feather_px) // 2) * 2 + 1)
    free = cv2.GaussianBlur(free, (blur_k, blur_k), 0)
    free = np.clip(free, 0.0, 1.0)

    free_frac = float((free > 0.5).mean())
    info["free_frac"] = round(free_frac, 4)
    info["protected_frac"] = round(1.0 - free_frac, 4)
    print(
        f"[skin_harm] build_garment_containment_mask -- free={free_frac:.3%} "
        f"protected={1.0 - free_frac:.3%} of the frame",
        flush=True,
    )

    mask = Image.fromarray((free * 255.0).astype(np.uint8), mode="L")
    if mask.size != (W, H):
        mask = mask.resize((W, H), Image.Resampling.BILINEAR)
    return mask, info
