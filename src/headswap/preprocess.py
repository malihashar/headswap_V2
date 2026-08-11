from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class FaceBox:
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)


def evenify(x: int, div_by: int = 2) -> int:
    if div_by <= 1:
        return max(1, x)
    return max(div_by, (x // div_by) * div_by)


def resize_max_keep_ar(im: Image.Image, max_dim: int, div_by: int = 2) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = evenify(max(1, int(round(w * scale))), div_by)
    nh = evenify(max(1, int(round(h * scale))), div_by)
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def resize_to_megapixels(
    im: Image.Image,
    *,
    target_mp: float = 1.25,
    min_mp: float = 1.0,
    max_mp: float = 1.5,
    max_dim: int = 2048,
    div_by: int = 16,
    allow_upscale: bool = False,
) -> Image.Image:
    """Resize so pixel count lands near ``target_mp`` megapixels (AR preserved).

    Author Identity Edit guidance: generate around 1–1.5MP (≤2MP). Large
    sources are downscaled into that band; small sources are left at native
    resolution unless ``allow_upscale`` is set (avoids inventing detail).
    """
    im = im.convert("RGB")
    w, h = im.size
    cur_mp = (w * h) / 1_000_000.0
    goal = float(min(max(float(target_mp), float(min_mp)), float(max_mp)))
    if cur_mp <= 1e-9:
        return im
    if (not allow_upscale) and cur_mp <= goal:
        # Native is already ≤ target band — only enforce max_dim + divisibility.
        if max(w, h) > int(max_dim):
            return resize_max_keep_ar(im, int(max_dim), div_by=div_by)
        nw = evenify(w, div_by)
        nh = evenify(h, div_by)
        if (nw, nh) == (w, h):
            return im
        return im.resize((nw, nh), Image.Resampling.LANCZOS)

    scale = math.sqrt(goal / cur_mp)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    # Hard long-side cap (still keep AR).
    long = max(nw, nh)
    if long > int(max_dim):
        s2 = float(max_dim) / float(long)
        nw = max(1, int(round(nw * s2)))
        nh = max(1, int(round(nh * s2)))
    nw = evenify(nw, div_by)
    nh = evenify(nh, div_by)
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def resize_contain(
    im: Image.Image,
    size: tuple[int, int],
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Fit ``im`` inside ``size`` preserving aspect ratio; letterbox with fill.

    Avoids the non-uniform stretch from Image.resize(scene.size), which warps
    identity faces whenever the body crop AR differs from the face crop AR
    (common in multi-person wider isolates).
    """
    im = im.convert("RGB")
    tw, th = int(size[0]), int(size[1])
    if tw <= 0 or th <= 0:
        return im
    w, h = im.size
    scale = min(tw / max(1, w), th / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (tw, th), fill)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def identity_content_frac(im: Image.Image, *, thresh: float = 12.0) -> float:
    """Fraction of pixels that are not near-black (identity signal occupancy)."""
    arr = np.asarray(im.convert("RGB"))
    return float((arr.max(axis=-1) > thresh).mean())


def place_face_at_height_frac(
    face: Image.Image,
    canvas_size: tuple[int, int],
    *,
    height_frac: float,
    fill: tuple[int, int, int] = (0, 0, 0),
    max_height_frac: float = 0.72,
    min_height_frac: float = 0.18,
) -> Image.Image:
    """
    Place an identity face on a canvas so its height is ``height_frac`` of the
    canvas height. Used so the identity ref matches body-head scale in group
    shots (full-bleed identity refs make Krea2 enlarge the swapped head).
    """
    face = face.convert("RGB")
    tw, th = int(canvas_size[0]), int(canvas_size[1])
    frac = float(min(max_height_frac, max(min_height_frac, height_frac)))
    target_h = max(8, int(round(th * frac)))
    w, h = face.size
    scale = target_h / max(1, h)
    # Also keep width inside canvas.
    if w * scale > tw * 0.95:
        scale = (tw * 0.95) / max(1, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = face.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (tw, th), fill)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def clamp_edited_head_scale(
    original_scene: Image.Image,
    edited: Image.Image,
    cache_dir,
    *,
    max_height_ratio: float = 1.08,
    target_ratio: float = 0.98,
    min_height_ratio: float = 0.92,
    max_grow: float = 1.45,
    border_fill: Image.Image | None = None,
) -> tuple[Image.Image, dict[str, float]]:
    """
    Rescale ``edited`` about the face center so its face height matches
    ``original_scene``'s, when it drifts outside
    [``min_height_ratio``, ``max_height_ratio``] (cv2 warp, border replicate).

    Corrects BOTH directions: a generated head that came out too large
    (ratio > max_height_ratio) is shrunk, and one that came out too small
    (ratio < min_height_ratio) is grown, up to ``max_grow``. Growing was
    previously impossible (the scale factor was hard-clamped to <= 1.0), so
    an undersized generated head silently kept its wrong proportions.

    IMPORTANT: for the FACE region, never paste onto ``original_scene`` --
    that re-exposes the old face around a shrunk swap (double-face /
    ghosting in group shots). The exposed BORDER strip the shrink/grow warp
    itself uncovers is a separate, narrower case (see below) where a patch
    from ``original_scene`` is the least-bad of the options tried -- EXCEPT
    when the caller's crop is tight around the head itself (see
    ``border_fill``): there, ``original_scene`` IS mostly the old head/hair,
    so its border patch can leak a fragment of the ORIGINAL person's hair as
    a disconnected dark island wherever the exposed strip happens to
    overlap where their real head was (GPU-observed 2026-08-10, tight local
    head-clamp box with the face near the top of frame). ``border_fill``
    lets such a caller patch the border from its own already-correct
    result instead, which can never reintroduce the wrong identity.

    Three approaches were GPU-tried for that exposed border on 2026-08-09.
    ``cv2.BORDER_REPLICATE`` stretches the warped image's own edge
    row/column outward, producing a warped, vertically-striped duplicate of
    whatever was at that edge (a ghost copy of the subject's feet/shoes) --
    rejected outright. Leaving the gap flat black and relying on a
    downstream person-matte (e.g. ``restore_background``) to repair it
    sounded cleanest in principle, but measured: a solid black rectangle
    against sandy ground, directly below a pair of shoes, is genuinely
    ambiguous to rembg (46% of gap pixels scored "person", 54%
    "background" -- not a confident background read), so the matte left
    most of the black hole exactly as it was instead of repairing it --
    also rejected. Pasting from ``original_scene`` using the warp's own
    EXACT geometric coverage mask (zero ambiguity -- we know precisely
    which pixels the affine transform didn't reach, unlike a neural
    matte's guess) can reproduce a soft echo of the person's real,
    unshrunk limbs where the gap overlaps their original position, but
    this was the least-bad artifact of the three on GPU inspection, so it
    is what ships -- a visually minor, geometrically-principled compromise
    over a guaranteed-wrong flat black hole or a matte that measurably
    can't tell the difference either way.
    """
    info = {
        "clamped": 0.0,
        "ratio_before": 1.0,
        "ratio_after": 1.0,
        "shrink": 1.0,
    }
    if original_scene.size != edited.size:
        edited = edited.resize(original_scene.size, Image.Resampling.LANCZOS)
    fo = detect_best_face(pil_to_rgb_np(original_scene), cache_dir)
    fe = detect_best_face(pil_to_rgb_np(edited), cache_dir)
    if fo is None or fe is None or fo.height < 8 or fe.height < 8:
        return edited, info
    ratio = float(fe.height) / float(fo.height)
    info["ratio_before"] = ratio
    if float(min_height_ratio) <= ratio <= float(max_height_ratio):
        info["ratio_after"] = ratio
        return edited, info

    shrink = (float(fo.height) * float(target_ratio)) / float(fe.height)
    # Bounded both ways: shrink an oversized head, grow an undersized one.
    shrink = float(min(float(max_grow), max(0.55, shrink)))
    info["shrink"] = shrink
    info["clamped"] = 1.0

    arr = pil_to_rgb_np(edited)
    h, w = arr.shape[:2]
    # Scale about the *edited* face center, then translate so that center
    # lands on the original face center (keeps neck alignment).
    ecx = 0.5 * (fe.x0 + fe.x1)
    ecy = 0.5 * (fe.y0 + fe.y1)
    ocx = 0.5 * (fo.x0 + fo.x1)
    ocy = 0.5 * (fo.y0 + fo.y1)
    m = cv2.getRotationMatrix2D((ecx, ecy), 0.0, shrink)
    # After scale about (ecx,ecy), that point stays fixed; shift to (ocx,ocy).
    m[0, 2] += ocx - ecx
    m[1, 2] += ocy - ecy
    warped = cv2.warpAffine(
        arr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Patch the border the warp uncovers directly from original_scene, using
    # the EXACT geometric coverage mask (not a neural matte -- see below for
    # why that matters). Two other approaches were GPU-tried and rejected on
    # 2026-08-09: BORDER_REPLICATE stretched the warped image's own edge
    # outward into a striped duplicate of whatever was there (a ghost copy
    # of the subject's feet); leaving the gap flat black and relying on a
    # downstream person-matte (e.g. restore_background) to repair it from
    # original_scene sounded cleanest in principle, but measured: a solid
    # black rectangle against sandy ground, directly below a pair of shoes,
    # is genuinely ambiguous to rembg (46% of gap pixels scored "person",
    # 54% "background" -- not a confident background read), so the matte
    # left most of the black hole exactly as-is instead of repairing it.
    # Pasting here, using the warp's own exact coverage mask (zero
    # ambiguity -- we know precisely which pixels the affine transform
    # didn't reach), can reproduce a soft echo of the person's real,
    # unshrunk limbs where the gap overlaps their original position -- an
    # accepted, visually minor trade-off against a guaranteed-wrong flat
    # black hole or a neural matte that measurably can't tell the
    # difference either way.
    valid = cv2.warpAffine(
        np.full((h, w), 255, dtype=np.uint8),
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if valid.min() < 255:
        fill_src = border_fill if border_fill is not None else original_scene
        orig_arr = pil_to_rgb_np(fill_src)
        if orig_arr.shape[:2] != (h, w):
            orig_arr = np.asarray(
                fill_src.resize((w, h), Image.Resampling.LANCZOS)
            )
        alpha = (valid.astype(np.float32) / 255.0)[..., None]
        warped = (
            warped.astype(np.float32) * alpha
            + orig_arr.astype(np.float32) * (1.0 - alpha)
        )
        warped = np.clip(warped, 0, 255).astype(np.uint8)
    out = np_to_pil(warped)
    fe2 = detect_best_face(pil_to_rgb_np(out), cache_dir)
    if fe2 is not None and fo.height > 0:
        info["ratio_after"] = float(fe2.height) / float(fo.height)
    else:
        info["ratio_after"] = ratio * shrink
    return out, info


def clamp_edited_head_scale_full_frame(
    original_scene: Image.Image,
    edited: Image.Image,
    cache_dir,
    *,
    max_height_ratio: float = 1.08,
    target_ratio: float = 0.98,
    min_height_ratio: float = 0.92,
    max_grow: float = 1.45,
    # Head+hair only — deliberately tighter than the stitch mask (which
    # extends into neck/collar). bot_extend≈0.08 stops at the chin so donor
    # clothing never enters the clamp composite.
    mask_top_extend: float = 1.80,
    mask_side_extend: float = 0.35,
    mask_bot_extend: float = 0.08,
    mask_expand_px: int = 6,
    mask_feather_px: int = 48,
) -> tuple[Image.Image, dict[str, float], Image.Image | None, Image.Image | None]:
    """Full-frame similarity head-scale clamp with post-transform soft composite.

    Unlike the local-box clamp (which cropped a head sub-region, warped it, and
    paste-blended it back — producing rectangular seams, clipped hair, and
    double-head ghosts), this:

      1. Computes a similarity transform (uniform scale + x/y translation)
         mapping the generated face onto the original target face.
      2. Applies that transform to the **entire** generated frame — nothing
         can be clipped by a pre-sized box.
      3. Builds a **head+hair-only** compositing mask **after** the warp
         (tight bot/side extents — no shoulder/collar), then intersects it
         with the warp's coverage so ``BORDER_CONSTANT`` black pixels never
         enter the composite (fixes the hard black crown silhouette).
      4. Soft-composites over the full frame:
         ``final = mask * transformed + (1-mask) * original``
         with a generously Gaussian-feathered mask boundary.

    Returns ``(result, info, composite_mask_or_None, transformed_or_None)``.
    Both mask and transformed are returned so callers can dump them for
    seam/ghost/black-region diagnosis (``None`` when the clamp did not fire).
    """
    info: dict[str, float] = {
        "clamped": 0.0,
        "ratio_before": 1.0,
        "ratio_after": 1.0,
        "shrink": 1.0,
        "tx": 0.0,
        "ty": 0.0,
        "mask_feather_px": float(mask_feather_px),
        "mask_bot_extend": float(mask_bot_extend),
        "mask_top_extend": float(mask_top_extend),
        "coverage_frac": 1.0,
    }
    if original_scene.size != edited.size:
        edited = edited.resize(original_scene.size, Image.Resampling.LANCZOS)
    fo = detect_best_face(pil_to_rgb_np(original_scene), cache_dir)
    fe = detect_best_face(pil_to_rgb_np(edited), cache_dir)
    if fo is None or fe is None or fo.height < 8 or fe.height < 8:
        return edited, info, None, None
    ratio = float(fe.height) / float(fo.height)
    info["ratio_before"] = ratio
    if float(min_height_ratio) <= ratio <= float(max_height_ratio):
        info["ratio_after"] = ratio
        return edited, info, None, None

    shrink = (float(fo.height) * float(target_ratio)) / float(fe.height)
    shrink = float(min(float(max_grow), max(0.55, shrink)))
    info["shrink"] = shrink
    info["clamped"] = 1.0

    arr = pil_to_rgb_np(edited)
    h, w = arr.shape[:2]
    ecx = 0.5 * (fe.x0 + fe.x1)
    ecy = 0.5 * (fe.y0 + fe.y1)
    ocx = 0.5 * (fo.x0 + fo.x1)
    ocy = 0.5 * (fo.y0 + fo.y1)
    # Similarity: scale about edited face center, then translate to original.
    m = cv2.getRotationMatrix2D((ecx, ecy), 0.0, shrink)
    m[0, 2] += ocx - ecx
    m[1, 2] += ocy - ecy
    info["tx"] = float(m[0, 2])
    info["ty"] = float(m[1, 2])

    warped = cv2.warpAffine(
        arr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Exact geometric coverage of the warp — pixels the affine never reached
    # are 0 (black border). Intersecting the composite mask with this stops
    # solid-black crown/edge silhouettes from leaking into the result.
    coverage = cv2.warpAffine(
        np.full((h, w), 255, dtype=np.uint8),
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    info["coverage_frac"] = float(coverage.mean()) / 255.0

    # Where the transformed head lands: map the generated face box through m.
    # Analytic (not re-detect) so the mask is deterministic even if InsightFace
    # briefly misses the warped face.
    corners = np.array(
        [
            [fe.x0, fe.y0],
            [fe.x1, fe.y0],
            [fe.x1, fe.y1],
            [fe.x0, fe.y1],
        ],
        dtype=np.float32,
    )
    ones = np.ones((4, 1), dtype=np.float32)
    mapped = (m @ np.hstack([corners, ones]).T).T
    mx0 = int(np.floor(mapped[:, 0].min()))
    my0 = int(np.floor(mapped[:, 1].min()))
    mx1 = int(np.ceil(mapped[:, 0].max()))
    my1 = int(np.ceil(mapped[:, 1].max()))
    mx0, my0 = max(0, mx0), max(0, my0)
    mx1, my1 = min(w, mx1), min(h, my1)
    if mx1 - mx0 < 8 or my1 - my0 < 8:
        # Degenerate warp — fall back to original unscaled edit.
        return edited, info, None, None
    landed = FaceBox(mx0, my0, mx1, my1, float(fe.conf))

    # Mask AFTER transform, from where the head landed — head+hair only.
    warped_pil = np_to_pil(warped)
    composite_mask = head_hair_mask_from_face(
        warped_pil,
        cache_dir,
        expand_px=int(mask_expand_px),
        blur_px=0,  # feather separately below
        top_extend=float(mask_top_extend),
        side_extend=float(mask_side_extend),
        bot_extend=float(mask_bot_extend),
        face_box=landed,
    )
    mask_arr = np.asarray(composite_mask.convert("L"), dtype=np.float32)
    # Drop any mask mass that sits on empty warp border (would composite black).
    cov = coverage.astype(np.float32) / 255.0
    mask_arr = mask_arr * cov
    feather = max(0, int(mask_feather_px))
    if feather > 0:
        k = feather * 2 + 1
        mask_arr = cv2.GaussianBlur(mask_arr, (k, k), 0)
    # Re-apply coverage after feather so soft edges cannot reintroduce black.
    mask_arr = mask_arr * cov
    alpha = (mask_arr / 255.0)[..., None]
    orig_arr = pil_to_rgb_np(original_scene).astype(np.float32)
    if orig_arr.shape[:2] != (h, w):
        orig_arr = pil_to_rgb_np(
            original_scene.resize((w, h), Image.Resampling.LANCZOS)
        ).astype(np.float32)
    blended = warped.astype(np.float32) * alpha + orig_arr * (1.0 - alpha)
    out = np_to_pil(np.clip(blended, 0, 255).astype(np.uint8))
    mask_out = Image.fromarray(np.clip(mask_arr, 0, 255).astype(np.uint8))

    fe2 = detect_best_face(pil_to_rgb_np(out), cache_dir)
    if fe2 is not None and fo.height > 0:
        info["ratio_after"] = float(fe2.height) / float(fo.height)
    else:
        info["ratio_after"] = ratio * shrink
    return out, info, mask_out, warped_pil


def dilate_mask(mask: Image.Image, px: int) -> Image.Image:
    """Expand an L/RGBA mask so the opaque region covers more of the old head."""
    if px <= 0:
        return mask
    arr = np.asarray(mask.convert("L"))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    arr = cv2.dilate(arr, k)
    return Image.fromarray(arr)


def erode_mask(mask: Image.Image, px: int) -> Image.Image:
    """Shrink an L mask (used to build a seam annulus)."""
    if px <= 0:
        return mask
    arr = np.asarray(mask.convert("L"))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    arr = cv2.erode(arr, k)
    return Image.fromarray(arr)


def seam_annulus_mask(
    mask: Image.Image,
    *,
    erode_px: int = 8,
    dilate_px: int = 16,
) -> Image.Image:
    """Band around the stitch boundary: dilate(mask) − erode(mask)."""
    d = np.asarray(dilate_mask(mask, dilate_px).convert("L")).astype(np.int16)
    e = np.asarray(erode_mask(mask, erode_px).convert("L")).astype(np.int16)
    band = np.clip(d - e, 0, 255).astype(np.uint8)
    return Image.fromarray(band)


def narrow_band_seam_refine(
    original: Image.Image,
    stitched: Image.Image,
    mask: Image.Image,
    *,
    erode_px: int = 8,
    dilate_px: int = 16,
    strength: float = 0.85,
    blur_px: int = 6,
) -> Image.Image:
    """
    Stage E: re-blend only a narrow annulus around the stitch boundary.

    Face interior (eroded mask core) stays as stitched. Outside the dilated
    mask stays as original. The annulus pulls color/luma toward the original
    body so neck seams soften without re-running diffusion.
    """
    if strength <= 0:
        return stitched
    if stitched.size != original.size:
        stitched = stitched.resize(original.size, Image.Resampling.LANCZOS)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.BILINEAR)

    band = seam_annulus_mask(mask, erode_px=erode_px, dilate_px=dilate_px)
    if blur_px > 0:
        k = max(3, int(blur_px) * 2 + 1)
        band_arr = cv2.GaussianBlur(np.asarray(band.convert("L")), (k, k), 0)
        band = Image.fromarray(band_arr)

    # Color-match stitched toward original inside the band, then lerp.
    matched = lab_histogram_match_face(stitched, original, band, strength=1.0)
    a = pil_to_rgb_np(stitched).astype(np.float32)
    b = pil_to_rgb_np(matched).astype(np.float32)
    o = pil_to_rgb_np(original).astype(np.float32)
    w = (np.asarray(band.convert("L")).astype(np.float32) / 255.0) * float(strength)
    w = w[..., None]
    # In band: mix matched stitched with original; outside band keep stitched.
    out = a * (1.0 - w) + (0.55 * b + 0.45 * o) * w
    return np_to_pil(np.clip(out, 0, 255))


def resize_long_side(im: Image.Image, long_side: int, div_by: int = 16) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    scale = float(long_side) / float(max(w, h))
    nw = evenify(max(1, int(round(w * scale))), div_by)
    nh = evenify(max(1, int(round(h * scale))), div_by)
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def pil_to_rgb_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


_FACE_NET = None
_PROTO_MODEL: tuple[str, str] | None = None
_HAAR = None
_FACE_BACKEND: str | None = None
_INSIGHTFACE_APP = None
_INSIGHTFACE_INIT_ERROR: str | None = None


def ensure_insightface_app(cache_dir=None):
    """
    Lazily construct InsightFace FaceAnalysis (buffalo_l).

    Models auto-download into INSIGHTFACE_HOME / ~/.insightface on first use.
    Returns the app or None when insightface/onnxruntime is unavailable.
    """
    global _INSIGHTFACE_APP, _INSIGHTFACE_INIT_ERROR
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP
    if _INSIGHTFACE_INIT_ERROR is not None:
        return None
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _INSIGHTFACE_INIT_ERROR = f"insightface_import_failed:{exc}"
        return None
    try:
        import os
        from pathlib import Path

        if cache_dir is not None:
            root = Path(cache_dir) / "insightface"
            root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("INSIGHTFACE_HOME", str(root))
        # Prefer CUDA when available; otherwise CPU (avoid noisy provider warnings).
        providers: list[str] = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort  # type: ignore

            avail = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CoreMLExecutionProvider" in avail:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHTFACE_APP = app
        return _INSIGHTFACE_APP
    except Exception as exc:  # noqa: BLE001
        _INSIGHTFACE_INIT_ERROR = f"insightface_runtime_failed:{exc}"
        return None


def _iou_box(a: FaceBox, b0: float, b1: float, b2: float, b3: float) -> float:
    ix0, iy0 = max(a.x0, b0), max(a.y0, b1)
    ix1, iy1 = min(a.x1, b2), min(a.y1, b3)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = float(max(1, a.width) * max(1, a.height))
    area_b = float(max(1.0, (b2 - b0) * (b3 - b1)))
    return inter / max(1e-6, area_a + area_b - inter)


def _ensure_face_dnn(cache_dir) -> tuple[str, str]:
    global _PROTO_MODEL
    if _PROTO_MODEL is not None:
        return _PROTO_MODEL
    from pathlib import Path
    import urllib.request

    face_dir = Path(cache_dir)
    face_dir.mkdir(parents=True, exist_ok=True)
    proto = face_dir / "deploy.prototxt"
    model = face_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    if not proto.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            proto,
        )
    if not model.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model,
        )
    _PROTO_MODEL = (str(proto), str(model))
    return _PROTO_MODEL


def _haar_cascade():
    global _HAAR
    if _HAAR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR = cv2.CascadeClassifier(path)
    return _HAAR


def get_face_backend(cache_dir) -> str:
    """Prefer Caffe SSD when available; otherwise Haar; else geometric prior."""
    global _FACE_NET, _FACE_BACKEND
    if _FACE_BACKEND is not None:
        return _FACE_BACKEND
    if hasattr(cv2.dnn, "readNetFromCaffe"):
        try:
            proto, model = _ensure_face_dnn(cache_dir)
            _FACE_NET = cv2.dnn.readNetFromCaffe(proto, model)
            _FACE_BACKEND = "caffe"
            return _FACE_BACKEND
        except Exception:
            pass
    try:
        casc = _haar_cascade()
        if casc is not None and not casc.empty():
            _FACE_BACKEND = "haar"
            return _FACE_BACKEND
    except Exception:
        pass
    _FACE_BACKEND = "prior"
    return _FACE_BACKEND


def _nonblack_content_box(rgb: np.ndarray, thresh: float = 14.0) -> FaceBox | None:
    """Bounding box of non-black pixels — works for cutout faces on black backgrounds."""
    lum = rgb.astype(np.float32).mean(axis=2)
    ys, xs = np.where(lum > thresh)
    if len(xs) < 100:
        return None
    h, w = rgb.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # Reject if almost the full frame (not a cutout) or tiny.
    area = (x1 - x0) * (y1 - y0)
    if area < 0.05 * h * w or area > 0.98 * h * w:
        return None
    return FaceBox(x0, y0, x1, y1, 0.35)


def _face_box_from_cutout(rgb: np.ndarray) -> FaceBox | None:
    """
    For studio cutouts on black: content bbox, then keep the upper face
    (drop jersey/shoulders) and center horizontally.
    """
    content = _nonblack_content_box(rgb)
    if content is None:
        return None
    cw, ch = content.width, content.height
    # Upper ~58% is face+hair; lower is neck/jersey on typical athlete cutouts.
    face_h = max(32, int(ch * 0.58))
    # Horizontal: center 78% of content width to avoid arms/jersey edges.
    face_w = max(32, int(cw * 0.78))
    cx = (content.x0 + content.x1) / 2.0
    x0 = int(round(cx - face_w / 2.0))
    x1 = int(round(cx + face_w / 2.0))
    y0 = content.y0
    y1 = content.y0 + face_h
    h, w = rgb.shape[:2]
    return FaceBox(max(0, x0), max(0, y0), min(w, x1), min(h, y1), 0.35)


def detect_best_face(rgb: np.ndarray, cache_dir, conf_thresh: float = 0.30) -> FaceBox | None:
    faces = detect_faces(rgb, cache_dir, conf_thresh=conf_thresh, allow_prior=True)
    return faces[0] if faces else None


def detect_faces(
    rgb: np.ndarray,
    cache_dir,
    *,
    conf_thresh: float = 0.30,
    allow_prior: bool = False,
) -> list[FaceBox]:
    """
    Detect all faces, largest-first.

    Preference: InsightFace buffalo_l (when installed) → OpenCV SSD → Haar.
    When allow_prior=False (multi-person / validation), never invent a geometric
    face — return [] if detectors find nothing.
    """
    h, w = rgb.shape[:2]
    faces: list[FaceBox] = []

    app = ensure_insightface_app(cache_dir)
    if app is not None:
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for f in app.get(bgr):
                conf = float(getattr(f, "det_score", 0.9) or 0.9)
                if conf < conf_thresh:
                    continue
                x0, y0, x1, y1 = [int(v) for v in f.bbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 + 2 or y1 <= y0 + 2:
                    continue
                faces.append(FaceBox(x0, y0, x1, y1, conf))
        except Exception:
            faces = []

    backend = get_face_backend(cache_dir)

    if not faces and backend == "caffe":
        assert _FACE_NET is not None
        max_side = 640
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            small = cv2.resize(
                rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = rgb
        sh, sw = small.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        _FACE_NET.setInput(blob)
        det = _FACE_NET.forward()
        for min_conf in (conf_thresh, 0.15):
            batch: list[FaceBox] = []
            for i in range(det.shape[2]):
                conf = float(det[0, 0, i, 2])
                if conf < min_conf:
                    continue
                x0 = int(det[0, 0, i, 3] * sw)
                y0 = int(det[0, 0, i, 4] * sh)
                x1 = int(det[0, 0, i, 5] * sw)
                y1 = int(det[0, 0, i, 6] * sh)
                if scale < 1.0:
                    x0 = int(round(x0 / scale))
                    x1 = int(round(x1 / scale))
                    y0 = int(round(y0 / scale))
                    y1 = int(round(y1 / scale))
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 + 2 or y1 <= y0 + 2:
                    continue
                batch.append(FaceBox(x0, y0, x1, y1, conf))
            if batch:
                faces = batch
                break

    if not faces and backend in ("haar", "caffe"):
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            haar = _haar_cascade().detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
            )
            for x, y, fw, fh in haar:
                faces.append(FaceBox(int(x), int(y), int(x + fw), int(y + fh), 0.5))
        except Exception:
            pass

    if not faces and allow_prior:
        content = _face_box_from_cutout(rgb)
        if content is not None:
            return [content]
        face_h = int(h * 0.42)
        face_w = int(min(w * 0.55, face_h * 0.90))
        cx, cy = w // 2, int(h * 0.36)
        return [
            FaceBox(
                max(0, cx - face_w // 2),
                max(0, cy - face_h // 2),
                min(w, cx + face_w // 2),
                min(h, cy + face_h // 2),
                0.2,
            )
        ]

    # De-dupe overlaps, keep higher area*conf. IoU stays at the standard 0.5:
    # real adjacent faces in tight groups can overlap, and losing a real face
    # silently breaks the neighbor carve/clamp locality guarantees downstream.
    faces.sort(key=lambda b: b.width * b.height * max(0.05, b.conf), reverse=True)
    kept: list[FaceBox] = []
    for f in faces:
        dup = False
        for k in kept:
            ix0, iy0 = max(f.x0, k.x0), max(f.y0, k.y0)
            ix1, iy1 = min(f.x1, k.x1), min(f.y1, k.y1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = f.width * f.height + k.width * k.height - inter
            if union > 0 and inter / union > 0.5:
                dup = True
                break
            # Treat strong containment as a duplicate: the same face detected
            # at two scales nests almost fully, unlike two distinct faces.
            smaller = min(f.width * f.height, k.width * k.height)
            if smaller > 0 and inter / smaller > 0.7:
                dup = True
                break
        if not dup:
            kept.append(f)
    # NOTE: no relative-area filter here. Far-away faces in group photos are
    # legitimately much smaller than the closest face; dropping them would
    # remove real neighbors from the locality guarantees.
    return kept


def select_face_box(
    rgb: np.ndarray,
    cache_dir,
    *,
    index: int = 0,
    policy: str = "largest",
    conf_thresh: float = 0.30,
) -> tuple[FaceBox | None, list[FaceBox]]:
    """
    Pick which body face to swap in multi-person photos.

    policy:
      largest   — biggest face (default)
      rightmost — max center-x (common for 'person on the right')
      leftmost  — min center-x
      index     — use ``index`` into left-to-right list (matches Colab §3b
                  TARGET_HEAD / Face N numbering: Face 1 = leftmost)
    """
    faces = detect_faces(rgb, cache_dir, conf_thresh=conf_thresh, allow_prior=False)
    if not faces:
        # Fall back to legacy single-face path (may use prior).
        best = detect_best_face(rgb, cache_dir, conf_thresh=conf_thresh)
        return best, ([best] if best is not None else [])

    pol = (policy or "largest").strip().lower()
    ordered = list(faces)
    if pol == "rightmost":
        ordered = sorted(faces, key=lambda b: (b.x0 + b.x1) / 2.0, reverse=True)
    elif pol == "leftmost":
        ordered = sorted(faces, key=lambda b: (b.x0 + b.x1) / 2.0)
    elif pol == "index":
        # Left-to-right so TARGET_HEAD / Face N from §3b matches crop_stitch
        # and full_frame identically (Face 1 = leftmost, not largest).
        ordered = sorted(faces, key=lambda b: (b.x0 + b.x1) / 2.0)
    else:
        ordered = list(faces)

    idx = int(index) if pol == "index" else 0
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx], faces


def mean_hsv_v(
    rgb: np.ndarray,
    box: FaceBox | None = None,
) -> float:
    """Mean HSV Value (luminance proxy) over full frame or an optional face box.

    OpenCV HSV V is in [0, 255]. Lower = darker. Used by multi-person lighting
    route to classify dark vs normal scenes.
    """
    if rgb is None or rgb.size == 0:
        return 0.0
    region = rgb
    if box is not None:
        y0, y1 = max(0, int(box.y0)), max(0, int(box.y1))
        x0, x1 = max(0, int(box.x0)), max(0, int(box.x1))
        h, w = rgb.shape[:2]
        y1, x1 = min(h, y1), min(w, x1)
        if y1 > y0 + 1 and x1 > x0 + 1:
            region = rgb[y0:y1, x0:x1]
    if region.size == 0:
        region = rgb
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2]))


def classify_lighting(
    rgb: np.ndarray,
    *,
    threshold: float,
    box: FaceBox | None = None,
    boxes: list[FaceBox] | None = None,
) -> dict[str, Any]:
    """Compare mean HSV-V against ``threshold``; below => dark.

    Evaluates both face region(s) and whole frame so that dark night photos
    with flash-lit faces or underexposed faces in bright backgrounds are
    reliably classified as dark.
    """
    frame_metric = mean_hsv_v(rgb, None)

    face_metrics: list[float] = []
    if boxes:
        for b in boxes:
            if b is not None:
                m = mean_hsv_v(rgb, b)
                if m > 0:
                    face_metrics.append(m)
    elif box is not None:
        m = mean_hsv_v(rgb, box)
        if m > 0:
            face_metrics.append(m)

    face_metric = float(np.mean(face_metrics)) if face_metrics else None

    # Dark if EITHER face region OR overall frame falls below threshold.
    dark_frame = frame_metric < float(threshold)
    dark_face = (face_metric < float(threshold)) if face_metric is not None else False
    dark = bool(dark_frame or dark_face)

    # Primary metric reported is the minimum of face & frame luminance for conservative dark detection
    effective_metric = min(frame_metric, face_metric) if face_metric is not None else frame_metric

    return {
        "lighting_metric": round(effective_metric, 4),
        "face_luminance": round(face_metric, 4) if face_metric is not None else None,
        "whole_frame_luminance": round(frame_metric, 4),
        "lighting_metric_name": "mean_hsv_v",
        "lighting_region": "face+frame" if face_metric is not None else "full_frame",
        "dark_lighting_threshold": float(threshold),
        "is_dark": dark,
    }



def expand_box(
    box: FaceBox,
    img_w: int,
    img_h: int,
    top: float = 0.65,
    bot: float = 0.15,
    side: float = 0.35,
    shoulder_extra: float = 0.0,
) -> FaceBox:
    fw, fh = box.width, box.height
    xx0 = int(round(box.x0 - side * fw))
    xx1 = int(round(box.x1 + side * fw))
    yy0 = int(round(box.y0 - top * fh))
    yy1 = int(round(box.y1 + (bot + shoulder_extra) * fh))
    xx0, yy0 = max(0, xx0), max(0, yy0)
    xx1, yy1 = min(img_w, xx1), min(img_h, yy1)
    return FaceBox(xx0, yy0, xx1, yy1, box.conf)


def pad_to_square(
    im: Image.Image,
    *,
    fill: tuple[int, int, int] | str = "edge",
    div_by: int = 16,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Pad RGB image to a square (divisible by div_by). Returns (square, content_box).
    content_box = (ox, oy, w, h) of the original pixels inside the square.

    fill="edge" uses the median border color (avoids black/white pad halos).
    """
    im = im.convert("RGB")
    w, h = im.size
    # Round UP so the square never shrinks below the content size.
    side = max(w, h)
    if div_by > 1:
        side = max(div_by, ((side + div_by - 1) // div_by) * div_by)
    if fill == "edge":
        arr = np.asarray(im)
        border = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        fill_rgb = tuple(int(x) for x in np.median(border, axis=0))
    else:
        fill_rgb = fill  # type: ignore[assignment]
    out = Image.new("RGB", (side, side), fill_rgb)
    ox, oy = (side - w) // 2, (side - h) // 2
    out.paste(im, (ox, oy))
    return out, (ox, oy, w, h)


def crop_face_reference(
    face_pil: Image.Image,
    cache_dir,
    top: float = 0.65,
    bot: float = 0.25,
    side: float = 0.35,
    include_shoulders: bool = True,
    tight_identity_crop: bool = True,
) -> Image.Image:
    rgb = pil_to_rgb_np(face_pil)
    box = detect_best_face(rgb, cache_dir)
    if box is None:
        return face_pil.convert("RGB")
    # Cutout / content boxes are already face-sized — large pads pull in jersey/bg.
    if box.conf <= 0.40 and box.height >= 0.28 * face_pil.height:
        top = min(float(top), 0.18)
        bot = min(float(bot), 0.02)
        side = min(float(side), 0.18)
        include_shoulders = False

    if tight_identity_crop and not include_shoulders:
        lm, _, _ = get_face_landmarks5(rgb, cache_dir, prefer_box=box)
        if lm is not None:
            le, re, _, lmouth, rmouth = lm
            iod = float(np.linalg.norm(re - le))
            mouth_y = float(0.5 * (lmouth[1] + rmouth[1]))
            chin_y = mouth_y + 0.75 * iod
            # Stop at chin / upper jawline with buffer (chin_y + 0.65*iod), keeping a
            # clean neck/jaw transition rather than cutting mid-chin (0.15 was too tight).
            max_y1 = max(box.y1 + max(0.02, float(bot)) * box.height, chin_y + 0.65 * iod)
        else:
            max_y1 = box.y1 + max(0.02, float(bot)) * box.height

        fw, fh = box.width, box.height
        xx0 = int(round(box.x0 - side * fw))
        xx1 = int(round(box.x1 + side * fw))
        yy0 = int(round(box.y0 - top * fh))
        yy1 = int(round(max_y1))
        xx0, yy0 = max(0, xx0), max(0, yy0)
        xx1, yy1 = min(face_pil.width, xx1), min(face_pil.height, yy1)
        return face_pil.crop((xx0, yy0, xx1, yy1)).convert("RGB")

    expanded = expand_box(
        box,
        face_pil.width,
        face_pil.height,
        top=top,
        bot=bot,
        side=side,
        shoulder_extra=0.35 if include_shoulders else 0.0,
    )
    return face_pil.crop((expanded.x0, expanded.y0, expanded.x1, expanded.y1)).convert("RGB")


def _clamp_xyxy(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int
) -> tuple[int, int, int, int]:
    xx0 = int(max(0, min(w - 1, round(x0))))
    yy0 = int(max(0, min(h - 1, round(y0))))
    xx1 = int(max(xx0 + 1, min(w, round(x1))))
    yy1 = int(max(yy0 + 1, min(h, round(y1))))
    return xx0, yy0, xx1, yy1


def estimate_identity_head_box(
    rgb: np.ndarray,
    cache_dir,
    *,
    # Multiples of inter-ocular distance (IOD) — robust across face sizes.
    hair_above_eyes_iod: float = 2.15,
    below_mouth_iod: float = 2.05,
    side_beyond_eyes_iod: float = 1.55,
    # Fallback multiples of detector face height/width when landmarks weak.
    top_face_h: float = 1.15,
    bot_face_h: float = 0.70,
    side_face_w: float = 0.55,
) -> dict[str, Any]:
    """
    Estimate a full-head box (hair + beard + ears + upper neck) from landmarks.

    Uses InsightFace 5-point landmarks when available. Does **not** use the raw
    detector box as the crop — only as a prior / fallback scale.
    """
    h, w = rgb.shape[:2]
    det = detect_best_face(rgb, cache_dir)
    lm, lm_backend, lm_note = get_face_landmarks5(rgb, cache_dir, prefer_box=det)

    info: dict[str, Any] = {
        "landmark_backend": lm_backend,
        "landmark_note": lm_note,
        "detector_box": None if det is None else [det.x0, det.y0, det.x1, det.y1],
        "policy": "landmark_iod",
    }

    if lm is not None:
        # kps: left_eye, right_eye, nose, left_mouth, right_mouth
        le, re, nose, lmouth, rmouth = lm
        iod = float(np.linalg.norm(re - le))
        if iod < 1.0 and det is not None:
            iod = float(max(1.0, 0.35 * det.width))
        elif iod < 1.0:
            iod = float(max(1.0, 0.08 * min(w, h)))
        eye_y = float(0.5 * (le[1] + re[1]))
        eye_l = float(min(le[0], re[0]))
        eye_r = float(max(le[0], re[0]))
        mouth_y = float(0.5 * (lmouth[1] + rmouth[1]))
        # Chin proxy: below mouth; prefer detector bottom if it extends further
        # (beard often sits under the mouth landmarks).
        chin_y = mouth_y + 0.85 * iod
        if det is not None:
            chin_y = max(chin_y, float(det.y1))

        x0 = eye_l - side_beyond_eyes_iod * iod
        x1 = eye_r + side_beyond_eyes_iod * iod
        y0 = eye_y - hair_above_eyes_iod * iod
        y1 = chin_y + below_mouth_iod * iod * 0.55  # upper neck below chin
        # Keep nose / mouth inside with margin.
        x0 = min(x0, float(nose[0]) - 1.2 * iod, float(lmouth[0]) - 1.1 * iod)
        x1 = max(x1, float(nose[0]) + 1.2 * iod, float(rmouth[0]) + 1.1 * iod)
        info.update(
            {
                "iod": round(iod, 2),
                "eye_y": round(eye_y, 2),
                "mouth_y": round(mouth_y, 2),
                "chin_y": round(chin_y, 2),
            }
        )
    elif det is not None:
        info["policy"] = "detector_expanded_fallback"
        fw, fh = float(det.width), float(det.height)
        x0 = det.x0 - side_face_w * fw
        x1 = det.x1 + side_face_w * fw
        y0 = det.y0 - top_face_h * fh
        y1 = det.y1 + bot_face_h * fh
    else:
        info["policy"] = "full_frame_fallback"
        return {
            **info,
            "head_box": [0, 0, w, h],
            "head_box_clamped": [0, 0, w, h],
        }

    # Optionally enlarge with 106-pt landmarks (jaw / brow extremes) when present.
    app = ensure_insightface_app(cache_dir)
    if app is not None and lm is not None:
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            if faces:
                face = max(
                    faces,
                    key=lambda f: float(
                        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    ),
                )
                pts106 = getattr(face, "landmark_2d_106", None)
                if pts106 is not None:
                    pts = np.asarray(pts106, dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[0] >= 16:
                        x0 = min(x0, float(pts[:, 0].min()) - 0.15 * (
                            float(pts[:, 0].max() - pts[:, 0].min())
                        ))
                        x1 = max(x1, float(pts[:, 0].max()) + 0.15 * (
                            float(pts[:, 0].max() - pts[:, 0].min())
                        ))
                        y0 = min(y0, float(pts[:, 1].min()) - 0.35 * (
                            float(pts[:, 1].max() - pts[:, 1].min())
                        ))
                        y1 = max(y1, float(pts[:, 1].max()) + 0.25 * (
                            float(pts[:, 1].max() - pts[:, 1].min())
                        ))
                        info["used_landmark_2d_106"] = True
        except Exception:
            info["used_landmark_2d_106"] = False

    raw = [float(x0), float(y0), float(x1), float(y1)]
    clamped = _clamp_xyxy(x0, y0, x1, y1, w, h)
    info["head_box"] = [round(v, 2) for v in raw]
    info["head_box_clamped"] = list(clamped)
    return info


def crop_identity_head(
    face_pil: Image.Image,
    cache_dir,
    *,
    square: bool = True,
    head_fill: float = 0.88,
    div_by: int = 16,
    **head_box_kwargs: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Crop a complete head (hair, forehead, ears, beard, jaw, upper neck).

    Returns (crop_rgb, debug_info) where debug_info includes detector_box,
    head_box, and geometry used for consistent head scale.
    """
    im = face_pil.convert("RGB")
    w, h = im.size
    rgb = pil_to_rgb_np(im)
    geom = estimate_identity_head_box(rgb, cache_dir, **head_box_kwargs)
    hx0, hy0, hx1, hy1 = [int(v) for v in geom["head_box_clamped"]]
    hw, hh = max(1, hx1 - hx0), max(1, hy1 - hy0)

    if square:
        # Consistent head scale: square window with head filling ``head_fill``.
        fill = float(min(0.96, max(0.70, head_fill)))
        side = int(round(max(hw, hh) / fill))
        side = max(side, max(hw, hh) + 2)
        if div_by > 1:
            side = max(div_by, ((side + div_by - 1) // div_by) * div_by)
        cx = 0.5 * (hx0 + hx1)
        cy = 0.5 * (hy0 + hy1)
        # Bias slightly upward so hair isn't clipped when clamping to image.
        cy = cy - 0.04 * hh
        x0 = int(round(cx - side / 2.0))
        y0 = int(round(cy - side / 2.0))
        x1, y1 = x0 + side, y0 + side
        # Shift into frame without shrinking (prefer keep full head).
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > w:
            shift = x1 - w
            x0 = max(0, x0 - shift)
            x1 = w
        if y1 > h:
            shift = y1 - h
            y0 = max(0, y0 - shift)
            y1 = h
        # Re-evenify after clamp.
        x0, y0, x1, y1 = _clamp_xyxy(x0, y0, x1, y1, w, h)
        if div_by > 1:
            cw = evenify(x1 - x0, div_by)
            ch = evenify(y1 - y0, div_by)
            x1 = min(w, x0 + cw)
            y1 = min(h, y0 + ch)
    else:
        x0, y0, x1, y1 = hx0, hy0, hx1, hy1
        if div_by > 1:
            x1 = min(w, x0 + evenify(x1 - x0, div_by))
            y1 = min(h, y0 + evenify(y1 - y0, div_by))

    crop = im.crop((x0, y0, x1, y1)).convert("RGB")
    head_in_crop = [
        hx0 - x0,
        hy0 - y0,
        hx1 - x0,
        hy1 - y0,
    ]
    det = geom.get("detector_box")
    det_in_crop = None
    if det is not None:
        det_in_crop = [
            int(det[0]) - x0,
            int(det[1]) - y0,
            int(det[2]) - x0,
            int(det[3]) - y0,
        ]
    info = {
        **geom,
        "crop_box": [x0, y0, x1, y1],
        "crop_size": list(crop.size),
        "head_box_in_crop": head_in_crop,
        "detector_box_in_crop": det_in_crop,
        "square": bool(square),
        "head_fill": float(head_fill) if square else None,
        "head_area_frac_in_crop": round(
            (hw * hh) / float(max(1, crop.size[0] * crop.size[1])), 4
        ),
    }
    return crop, info


def draw_identity_head_debug(
    face_pil: Image.Image,
    info: dict[str, Any],
) -> Image.Image:
    """Overlay detector (red), head estimate (lime), final crop (cyan)."""
    vis = face_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    det = info.get("detector_box")
    if det is not None:
        draw.rectangle(det, outline=(255, 64, 64), width=3)
        draw.text((int(det[0]) + 2, max(0, int(det[1]) - 14)), "detector", fill=(255, 64, 64))
    head = info.get("head_box_clamped") or info.get("head_box")
    if head is not None:
        draw.rectangle(head, outline=(80, 255, 80), width=3)
        draw.text((int(head[0]) + 2, max(0, int(head[1]) - 14)), "head", fill=(80, 255, 80))
    crop = info.get("crop_box")
    if crop is not None:
        draw.rectangle(crop, outline=(64, 200, 255), width=3)
        draw.text((int(crop[0]) + 2, max(0, int(crop[1]) - 14)), "crop", fill=(64, 200, 255))
    return vis


def identity_face_only_matte(
    face_pil: Image.Image,
    cache_dir,
    *,
    top: float = 0.35,
    bot: float = 0.08,
    side: float = 0.12,
    force_ellipse: bool = True,
) -> tuple[Image.Image, dict]:
    """
    Clothing-free identity prep: tight face crop + ellipse/white matte.

    Strips collar/suit/jersey that otherwise leak through dual-ref / paste ID paths.
    """
    info: dict = {
        "identity_face_only": True,
        "face_matte_top": float(top),
        "face_matte_bot": float(bot),
        "face_matte_side": float(side),
    }
    crop = crop_face_reference(
        face_pil,
        cache_dir,
        top=top,
        bot=bot,
        side=side,
        include_shoulders=False,
    )
    matted = face_on_white_background(
        crop, cache_dir=cache_dir, force_ellipse=bool(force_ellipse)
    )
    info["matte_size"] = list(matted.size)
    return matted.convert("RGB"), info


def crop_around_face_box(
    image: Image.Image,
    face: FaceBox | None,
    *,
    pad_frac: float = 1.15,
    div_by: int = 16,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Crop a working window around ``face`` for align-paste (multi-person safe).

    Returns (crop_rgb, box_xyxy on the full image).
    """
    w, h = image.size
    if face is None:
        box = (0, 0, w, h)
        return image.convert("RGB"), box
    fw, fh = max(1, face.width), max(1, face.height)
    pad_x = int(pad_frac * fw)
    pad_y = int(pad_frac * fh)
    x0 = max(0, face.x0 - pad_x)
    y0 = max(0, face.y0 - pad_y)
    x1 = min(w, face.x1 + pad_x)
    y1 = min(h, face.y1 + pad_y)
    # Evenify for VAE-friendly sizes when used as a Krea2 crop later.
    cw, ch = x1 - x0, y1 - y0
    x1 = x0 + evenify(cw, div_by)
    y1 = y0 + evenify(ch, div_by)
    x1, y1 = min(w, x1), min(h, y1)
    x0 = max(0, x1 - evenify(x1 - x0, div_by))
    y0 = max(0, y1 - evenify(y1 - y0, div_by))
    box = (x0, y0, x1, y1)
    return image.crop(box).convert("RGB"), box


def pad_to_ar_blur(im: Image.Image, target_ar: float) -> Image.Image:
    """Legacy baseline helper — prefer not using for improved pipelines."""
    im = im.convert("RGB")
    w, h = im.size
    ar = w / h
    if abs(ar - target_ar) < 1e-6:
        return im
    if ar > target_ar:
        new_w, new_h = w, int(round(w / target_ar))
    else:
        new_h, new_w = h, int(round(h * target_ar))
    bg = im.resize((new_w, new_h), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(14))
    bg.paste(im, ((new_w - w) // 2, (new_h - h) // 2))
    return bg


def head_hair_mask_from_face(
    body_pil: Image.Image,
    cache_dir,
    expand_px: int = 18,
    blur_px: int = 12,
    top_extend: float = 1.1,
    side_extend: float = 0.55,
    bot_extend: float = 0.35,
    face_box: FaceBox | None = None,
) -> Image.Image:
    """
    Approximate head+hair mask without SAM (portable fallback).
    Uses face box expanded toward hair/neck. Good enough for crop locality;
    replace with SAM/BiRefNet in production GPU stacks when available.

    Pass ``face_box`` to target a specific person in multi-face photos.
    """
    rgb = pil_to_rgb_np(body_pil)
    h, w = rgb.shape[:2]
    box = face_box if face_box is not None else detect_best_face(rgb, cache_dir)
    mask = np.zeros((h, w), dtype=np.uint8)
    if box is None:
        # Center prior fallback
        cx, cy = w // 2, h // 3
        axes = (max(8, w // 5), max(8, h // 4))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    else:
        fw, fh = box.width, box.height
        x0 = int(box.x0 - side_extend * fw)
        x1 = int(box.x1 + side_extend * fw)
        y0 = int(box.y0 - top_extend * fh)
        y1 = int(box.y1 + bot_extend * fh)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        cv2.ellipse(
            mask,
            ((x0 + x1) // 2, (y0 + y1) // 2),
            (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
            0,
            0,
            360,
            255,
            -1,
        )
    if expand_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
        mask = cv2.dilate(mask, k)
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (blur_px * 2 + 1, blur_px * 2 + 1), 0)
    return Image.fromarray(mask)


def mask_bbox(mask_pil: Image.Image, pad: int = 8) -> tuple[int, int, int, int]:
    m = np.asarray(mask_pil.convert("L"))
    ys, xs = np.where(m > 16)
    if len(xs) == 0:
        w, h = mask_pil.size
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = mask_pil.size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def crop_with_mask(
    image: Image.Image, mask: Image.Image, pad: int = 8, div_by: int = 16
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = mask_bbox(mask, pad=pad)
    # Make crop dimensions divisible
    cw, ch = x1 - x0, y1 - y0
    x1 = x0 + evenify(cw, div_by)
    y1 = y0 + evenify(ch, div_by)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    # Re-align if clipping broke divisibility
    x0 = max(0, x1 - evenify(x1 - x0, div_by))
    y0 = max(0, y1 - evenify(y1 - y0, div_by))
    box = (x0, y0, x1, y1)
    return image.crop(box), mask.crop(box), box


def clamp_crop_away_neighbors(
    image_size: tuple[int, int],
    box: tuple[int, int, int, int],
    selected: FaceBox | None,
    other_faces: list[FaceBox],
    *,
    margin_frac: float = 0.18,
    protect_top_frac: float = 1.55,
    protect_side_frac: float = 0.60,
    protect_bot_frac: float = 0.40,
    protect_pad_px: int = 0,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """
    Shrink a crop window so neighboring face centers fall outside it.

    Production locality mechanism for multi-person Krea2: keep the diffusion
    canvas identical to the single-person recipe, but do not let a second
    person enter the scene tensor. Prefer this over post-hoc neighbor pixel
    freezes, which can erase the swapped head on tight groups.

    ``protect_*_frac`` must mirror the head/hair mask extents so the clamp can
    never slice into the stitch region (e.g. cut hair above the face). Pass
    the mask's ``expand_px + blur_px`` as ``protect_pad_px``.

    Neighbors whose center lies inside the protected head region cannot be
    excluded geometrically; they are counted in ``neighbors_unexcludable``
    and left to the mask carve / optional freeze.

    Box divisibility is intentionally NOT enforced here — the VAE input is
    evenified later by ``resize_long_side`` and stitching accepts any box.
    """
    iw, ih = image_size
    x0, y0, x1, y1 = [int(v) for v in box]
    info: dict[str, float] = {
        "clamped": 0.0,
        "neighbors_excluded": 0.0,
        "neighbors_unexcludable": 0.0,
        "crop_w_before": float(max(1, x1 - x0)),
        "crop_h_before": float(max(1, y1 - y0)),
    }
    if selected is None or not other_faces:
        info["crop_w_after"] = info["crop_w_before"]
        info["crop_h_after"] = info["crop_h_before"]
        return (x0, y0, x1, y1), info

    sx0, sy0, sx1, sy1 = selected.x0, selected.y0, selected.x1, selected.y1
    sw, sh = max(1, selected.width), max(1, selected.height)
    pad = max(0, int(protect_pad_px))
    # Protected region = selected head+hair mask extents (+mask dilate/blur pad).
    # The crop must always fully contain it, or the stitch loses coverage.
    p_x0 = max(0, int(sx0 - protect_side_frac * sw) - pad)
    p_y0 = max(0, int(sy0 - protect_top_frac * sh) - pad)
    p_x1 = min(iw, int(sx1 + protect_side_frac * sw) + pad)
    p_y1 = min(ih, int(sy1 + protect_bot_frac * sh) + pad)

    excluded = 0
    unexcludable = 0
    for face in other_faces:
        if (
            face.x0 == selected.x0
            and face.y0 == selected.y0
            and face.x1 == selected.x1
            and face.y1 == selected.y1
        ):
            continue
        # Near-duplicate of selected — ignore.
        ix0, iy0 = max(sx0, face.x0), max(sy0, face.y0)
        ix1, iy1 = min(sx1, face.x1), min(sy1, face.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter > 0.5 * face.width * face.height:
            continue
        cx = 0.5 * (face.x0 + face.x1)
        cy = 0.5 * (face.y0 + face.y1)
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        margin = int(margin_frac * max(1, face.width))
        # Candidate edges that exclude this neighbor. Feasible only when the
        # new edge stays outside the protected head region. Among feasible
        # sides keep the one preserving the most crop area (least context loss).
        candidates: list[tuple[str, int, int]] = []
        nx0 = int(cx + margin)
        if nx0 <= p_x0:
            candidates.append(("x0", nx0, (x1 - max(x0, nx0)) * (y1 - y0)))
        nx1 = int(cx - margin)
        if nx1 >= p_x1:
            candidates.append(("x1", nx1, (min(x1, nx1) - x0) * (y1 - y0)))
        ny0 = int(cy + margin)
        if ny0 <= p_y0:
            candidates.append(("y0", ny0, (x1 - x0) * (y1 - max(y0, ny0))))
        ny1 = int(cy - margin)
        if ny1 >= p_y1:
            candidates.append(("y1", ny1, (x1 - x0) * (min(y1, ny1) - y0)))
        if not candidates:
            unexcludable += 1
            continue
        edge, value, _area = max(candidates, key=lambda c: c[2])
        if edge == "x0":
            x0 = max(x0, value)
        elif edge == "x1":
            x1 = min(x1, value)
        elif edge == "y0":
            y0 = max(y0, value)
        else:
            y1 = min(y1, value)
        excluded += 1

    info["clamped"] = 1.0 if excluded else 0.0
    info["neighbors_excluded"] = float(excluded)
    info["neighbors_unexcludable"] = float(unexcludable)
    info["crop_w_after"] = float(max(1, x1 - x0))
    info["crop_h_after"] = float(max(1, y1 - y0))
    return (x0, y0, x1, y1), info


def suppress_neighbor_faces_in_mask(
    mask: Image.Image,
    selected: FaceBox | None,
    other_faces: list[FaceBox],
    *,
    shrink: float = 0.92,
) -> Image.Image:
    """
    Zero mask coverage over other detected faces so an expanded group-shot crop
    does not stitch neighbor heads when the edit canvas includes them.
    """
    if selected is None or not other_faces:
        return mask
    arr = np.asarray(mask.convert("L")).copy()
    sx0, sy0, sx1, sy1 = selected.x0, selected.y0, selected.x1, selected.y1
    for face in other_faces:
        if (
            face.x0 == selected.x0
            and face.y0 == selected.y0
            and face.x1 == selected.x1
            and face.y1 == selected.y1
        ):
            continue
        # Skip near-duplicates of the selected box.
        ix0, iy0 = max(sx0, face.x0), max(sy0, face.y0)
        ix1, iy1 = min(sx1, face.x1), min(sy1, face.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter > 0.5 * face.width * face.height:
            continue
        fw, fh = max(1, face.width), max(1, face.height)
        cx = (face.x0 + face.x1) // 2
        cy = (face.y0 + face.y1) // 2
        axes = (
            max(4, int(0.5 * shrink * fw)),
            max(4, int(0.55 * shrink * fh)),
        )
        cv2.ellipse(arr, (cx, cy), axes, 0, 0, 360, 0, -1)
    return Image.fromarray(arr)


def hard_freeze_neighbor_faces(
    result: Image.Image,
    original: Image.Image,
    selected: FaceBox | None,
    all_faces: list[FaceBox],
    *,
    pad_frac: float = 0.25,
    expand_top_frac: float = 0.40,
    protect_expand: float = 1.15,
) -> Image.Image:
    """
    Pixel-copy neighbor face (+hair) regions from ``original`` onto ``result``.

    Used after full-frame Krea2 (which has no latent inpaint mask) so neighboring
    people are restored exactly even if the model rewrote them during denoise.

    Critical: never overwrite the selected face/hair region. On tight group shots
    expanded neighbor boxes often overlap the swap target; pasting raw rectangles
    would erase the edit (looks like "no swap happened").
    """
    if selected is None or not all_faces:
        return result
    if result.size != original.size:
        original = original.resize(result.size, Image.Resampling.LANCZOS)
    w, h = result.size
    out_arr = np.asarray(result.convert("RGB")).copy()
    orig_arr = np.asarray(original.convert("RGB"))

    # Protect selected head+hair — neighbor restores must not touch this.
    protect = np.zeros((h, w), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = selected.x0, selected.y0, selected.x1, selected.y1
    sw, sh = max(1, selected.width), max(1, selected.height)
    cx = (sx0 + sx1) // 2
    cy = (sy0 + sy1) // 2
    # Shift center slightly up and enlarge vertically to cover hair.
    cy_p = max(0, cy - int(0.20 * sh))
    axes = (
        max(6, int(0.55 * protect_expand * sw)),
        max(6, int(0.75 * protect_expand * sh)),
    )
    cv2.ellipse(protect, (cx, cy_p), axes, 0, 0, 360, 255, -1)

    for face in all_faces:
        if (
            face.x0 == selected.x0
            and face.y0 == selected.y0
            and face.x1 == selected.x1
            and face.y1 == selected.y1
        ):
            continue
        fw, fh = max(1, face.width), max(1, face.height)
        pad_x = int(pad_frac * fw)
        pad_y = int(pad_frac * fh)
        top_extra = int(expand_top_frac * fh)
        x0 = max(0, face.x0 - pad_x)
        y0 = max(0, face.y0 - top_extra)
        x1 = min(w, face.x1 + pad_x)
        y1 = min(h, face.y1 + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue
        # Skip neighbors that heavily overlap the selected box (near-duplicates).
        ix0, iy0 = max(sx0, face.x0), max(sy0, face.y0)
        ix1, iy1 = min(sx1, face.x1), min(sy1, face.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter > 0.5 * face.width * face.height:
            continue
        keep = protect[y0:y1, x0:x1] == 0
        if not np.any(keep):
            continue
        patch = out_arr[y0:y1, x0:x1]
        patch[keep] = orig_arr[y0:y1, x0:x1][keep]
        out_arr[y0:y1, x0:x1] = patch
    return Image.fromarray(out_arr)


def ensure_selected_face_mask_coverage(
    mask: Image.Image,
    selected: FaceBox | None,
    *,
    min_frac: float = 0.35,
    top_extend: float = 1.2,
    side_extend: float = 0.45,
    bot_extend: float = 0.35,
) -> Image.Image:
    """
    If neighbor carve-outs wiped most of the selected head mask, redraw it.

    Prevents freeze soft-composite from becoming a no-op (original body only).
    """
    if selected is None:
        return mask
    arr = np.asarray(mask.convert("L")).copy()
    h, w = arr.shape
    x0 = max(0, selected.x0)
    y0 = max(0, selected.y0)
    x1 = min(w, selected.x1)
    y1 = min(h, selected.y1)
    if x1 <= x0 or y1 <= y0:
        return mask
    region = arr[y0:y1, x0:x1]
    frac = float((region > 127).mean()) if region.size else 0.0
    if frac >= min_frac:
        return mask
    fw, fh = max(1, selected.width), max(1, selected.height)
    ex0 = int(selected.x0 - side_extend * fw)
    ex1 = int(selected.x1 + side_extend * fw)
    ey0 = int(selected.y0 - top_extend * fh)
    ey1 = int(selected.y1 + bot_extend * fh)
    ex0, ey0 = max(0, ex0), max(0, ey0)
    ex1, ey1 = min(w, ex1), min(h, ey1)
    cv2.ellipse(
        arr,
        ((ex0 + ex1) // 2, (ey0 + ey1) // 2),
        (max(1, (ex1 - ex0) // 2), max(1, (ey1 - ey0) // 2)),
        0,
        0,
        360,
        255,
        -1,
    )
    return Image.fromarray(arr)


def identity_face_boost_mask(
    person: Image.Image,
    cache_dir,
    *,
    expand: float = 1.35,
    include_hair: bool = True,
) -> Image.Image:
    """
    L-mask over the identity head on the person/reference canvas.

    For Krea2 ``ref_boost_mask`` (last-ref attention boost only — not an edit
    freeze). When ``include_hair`` is True, expand upward/sideways so hair and
    beard stay inside the boosted region (tight face ellipses starved hair ID).
    """
    rgb = pil_to_rgb_np(person)
    h, w = rgb.shape[:2]
    box = detect_best_face(rgb, cache_dir)
    mask = np.zeros((h, w), dtype=np.uint8)
    if box is None:
        cx, cy = w // 2, int(h * 0.42)
        axes = (max(8, int(w * 0.38)), max(8, int(h * 0.46)))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    else:
        cx = (box.x0 + box.x1) // 2
        # Shift center slightly up so hair is covered.
        cy = int((box.y0 + box.y1) / 2 - (0.12 * box.height if include_hair else 0.0))
        ex = float(expand) * (1.25 if include_hair else 1.0)
        ey = float(expand) * (1.45 if include_hair else 1.0)
        axes = (
            max(4, int(0.55 * ex * box.width)),
            max(4, int(0.70 * ey * box.height)),
        )
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return Image.fromarray(mask)


def expand_crop_box_for_face_fill(
    image_size: tuple[int, int],
    box: tuple[int, int, int, int],
    face: FaceBox | None,
    *,
    target_face_area_frac: float = 0.16,
    min_long_side: int = 448,
    other_faces: list[FaceBox] | None = None,
    div_by: int = 16,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """
    Enlarge a head crop so the selected face occupies ~target_face_area_frac of
    the crop (similar to single-person crops) and the crop long side is at least
    min_long_side before the 768 upscale — avoids soft, over-upscaled group faces.

    Prefers vertical expansion (hair/neck) when other faces sit to the sides.
    """
    iw, ih = image_size
    x0, y0, x1, y1 = [int(v) for v in box]
    info: dict[str, float] = {
        "expanded": 0.0,
        "face_fill_before": 0.0,
        "face_fill_after": 0.0,
        "crop_long_before": float(max(x1 - x0, y1 - y0)),
        "crop_long_after": float(max(x1 - x0, y1 - y0)),
    }
    if face is None:
        return (x0, y0, x1, y1), info

    def _face_fill(a: int, b: int, c: int, d: int) -> float:
        area = max(1, (c - a) * (d - b))
        return float(face.width * face.height) / float(area)

    fill0 = _face_fill(x0, y0, x1, y1)
    info["face_fill_before"] = fill0
    long0 = max(x1 - x0, y1 - y0)
    need_fill = fill0 > float(target_face_area_frac) * 1.05
    need_size = long0 < int(min_long_side)
    if not need_fill and not need_size:
        info["face_fill_after"] = fill0
        return (x0, y0, x1, y1), info

    # Directional room: if another face is close on a side, expand less that way.
    left_block = right_block = 0.0
    if other_faces:
        fcx = 0.5 * (face.x0 + face.x1)
        for o in other_faces:
            if (
                o.x0 == face.x0
                and o.y0 == face.y0
                and o.x1 == face.x1
                and o.y1 == face.y1
            ):
                continue
            ocx = 0.5 * (o.x0 + o.x1)
            gap = abs(ocx - fcx) / max(1.0, float(face.width))
            if gap < 2.5:
                if ocx < fcx:
                    left_block = max(left_block, 1.0 - gap / 2.5)
                else:
                    right_block = max(right_block, 1.0 - gap / 2.5)

    # Grow until fill ≈ target and long side ≥ min (or image bounds).
    for _ in range(48):
        cw, ch = x1 - x0, y1 - y0
        fill = _face_fill(x0, y0, x1, y1)
        long_side = max(cw, ch)
        if fill <= target_face_area_frac and long_side >= min_long_side:
            break
        # Step size ~3% of current crop; bias vertical for neck/hair context.
        step_x = max(2, int(0.03 * cw))
        step_y = max(2, int(0.04 * ch))
        grow_l = int(step_x * (1.0 - 0.75 * left_block))
        grow_r = int(step_x * (1.0 - 0.75 * right_block))
        grow_t = step_y
        grow_b = int(step_y * 1.25)  # prefer neck room for lighting match
        nx0 = max(0, x0 - grow_l)
        nx1 = min(iw, x1 + grow_r)
        ny0 = max(0, y0 - grow_t)
        ny1 = min(ih, y1 + grow_b)
        if (nx0, ny0, nx1, ny1) == (x0, y0, x1, y1):
            break
        x0, y0, x1, y1 = nx0, ny0, nx1, ny1

    # Enforce divisibility like crop_with_mask.
    cw, ch = x1 - x0, y1 - y0
    x1 = min(iw, x0 + evenify(cw, div_by))
    y1 = min(ih, y0 + evenify(ch, div_by))
    x0 = max(0, x1 - evenify(x1 - x0, div_by))
    y0 = max(0, y1 - evenify(y1 - y0, div_by))
    info["expanded"] = 1.0
    info["face_fill_after"] = _face_fill(x0, y0, x1, y1)
    info["crop_long_after"] = float(max(x1 - x0, y1 - y0))
    return (x0, y0, x1, y1), info


def soft_composite(
    base: Image.Image,
    edit: Image.Image,
    mask: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Paste edited crop back using soft alpha; preserves unmasked body pixels."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    base = base.convert("RGBA")
    edit = edit.convert("RGBA").resize((bw, bh), Image.Resampling.LANCZOS)
    # Crop mask to the same box, then resize to match edit (handles working-res crops)
    alpha = mask.convert("L").crop(box).resize((bw, bh), Image.Resampling.BILINEAR)
    edit.putalpha(alpha)
    out = base.copy()
    out.alpha_composite(edit, dest=(x0, y0))
    return out.convert("RGB")


def face_on_white_background(
    face: Image.Image,
    *,
    black_thresh: float = 18.0,
    cache_dir=None,
    force_ellipse: bool = False,
) -> Image.Image:
    """
    Put a face onto a white canvas (BFS 'sticker' prep for identity ReferenceLatent).

    Preference:
      1. Black-studio cutout matte (non-black pixels kept)
      2. Face-box ellipse matte (kills jersey / busy background around the head)
      3. Unchanged RGB if neither applies
    """
    rgb = pil_to_rgb_np(face)
    lum = rgb.astype(np.float32).mean(axis=2)
    black_frac = float((lum <= black_thresh).mean())
    if black_frac >= 0.20 and not force_ellipse:
        alpha = (lum > black_thresh).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        white = np.full_like(rgb, 255, dtype=np.float32)
        out = rgb.astype(np.float32) * alpha[..., None] + white * (1.0 - alpha[..., None])
        return np_to_pil(np.clip(out, 0, 255))

    # Ellipse matte from face detection — stronger ID sticker for jersey cutouts.
    if cache_dir is not None:
        box = detect_best_face(rgb, cache_dir)
        if box is not None:
            h, w = rgb.shape[:2]
            fw, fh = box.width, box.height
            cx = int((box.x0 + box.x1) / 2)
            cy = int((box.y0 + box.y1) / 2)
            # Slightly generous so hair / jaw stay in the sticker.
            axes = (
                max(8, int(0.62 * fw)),
                max(8, int(0.78 * fh)),
            )
            alpha = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(alpha, (cx, cy), axes, 0, 0, 360, 1.0, -1)
            alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
            white = np.full_like(rgb, 255, dtype=np.float32)
            out = rgb.astype(np.float32) * alpha[..., None] + white * (1.0 - alpha[..., None])
            return np_to_pil(np.clip(out, 0, 255))
    return face.convert("RGB")


def prepare_krea2_identity_person(
    face: Image.Image,
    cache_dir,
    *,
    long_side: int = 768,
    div_by: int = 16,
    top: float = 0.70,
    bot: float = 0.12,
    side: float = 0.22,
    white_bg: bool = True,
    square_fill: bool = True,
    fill_frac: float = 0.92,
    use_head_crop: bool = True,
    head_fill: float = 0.88,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """
    Build a head-dominant identity ref for Krea2 dual-ref conditioning.

    Default policy: landmark-based **full head** crop (hair/beard/ears/neck),
    then scale to a consistent square. Do not letterbox into the scene canvas.
    """
    head_info: dict[str, Any] = {}
    if use_head_crop:
        face_crop, head_info = crop_identity_head(
            face,
            cache_dir,
            square=True,
            head_fill=float(head_fill),
            div_by=max(2, int(div_by)),
        )
    else:
        # Legacy tight face pads (kept for A/B only).
        face_crop = crop_face_reference(
            face,
            cache_dir,
            top=float(top),
            bot=float(bot),
            side=float(side),
            include_shoulders=False,
        )
        head_info = {"policy": "legacy_face_pads"}

    person = face_crop.convert("RGB")
    applied_white = False
    if white_bg:
        # Soft matte: black-studio cutout preferred. Avoid force_ellipse — it
        # clips hair/beard on full-head crops.
        person = face_on_white_background(
            person, cache_dir=cache_dir, force_ellipse=False
        )
        applied_white = True
    side_px = int(long_side)
    if square_fill:
        person = fit_face_on_square(
            person,
            side_px,
            fill_frac=float(fill_frac),
            bg=(255, 255, 255) if applied_white else (0, 0, 0),
            div_by=int(div_by),
        )
    else:
        person = resize_long_side(person, side_px, div_by=int(div_by))
    frac = identity_content_frac(person)
    rgb = pil_to_rgb_np(person)
    box = detect_best_face(rgb, cache_dir)
    face_area_frac = 0.0
    if box is not None:
        face_area_frac = float(box.width * box.height) / float(
            max(1, person.size[0] * person.size[1])
        )
    info: dict[str, Any] = {
        "person_size": list(person.size),
        "face_crop_size": list(face_crop.size),
        "identity_content_frac": round(frac, 4),
        "identity_face_area_frac": round(face_area_frac, 4),
        "identity_long_side": int(long_side),
        "white_bg": bool(applied_white),
        "square_fill": bool(square_fill),
        "use_head_crop": bool(use_head_crop),
        "letterboxed_to_scene": False,
        "identity_starved": bool(face_area_frac < 0.12 and frac < 0.40),
        "head_crop": head_info,
    }
    return person, face_crop, info


def fit_face_on_square(
    im: Image.Image,
    side: int,
    *,
    fill_frac: float = 0.90,
    bg: tuple[int, int, int] = (255, 255, 255),
    div_by: int = 16,
) -> Image.Image:
    """
    Place face on a square canvas filling ~fill_frac of the side (stronger ID signal).
    """
    im = im.convert("RGB")
    if div_by > 1:
        side = max(div_by, ((side + div_by - 1) // div_by) * div_by)
    fill_frac = float(min(0.98, max(0.55, fill_frac)))
    target = max(div_by, int(side * fill_frac))
    w, h = im.size
    scale = target / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if div_by > 1:
        nw = max(div_by, (nw // div_by) * div_by)
        nh = max(div_by, (nh // div_by) * div_by)
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (side, side), bg)
    out.paste(resized, ((side - nw) // 2, (side - nh) // 2))
    return out


def clean_alpha_tails(mask: Image.Image, *, floor: int = 10, ceil: int = 245) -> Image.Image:
    """Snap a blurred alpha mask's faint tails to 0/255, keep the mid-transition soft.

    The head+hair mask is Gaussian-blurred at least twice before it reaches a
    composite (once building the mask, once again as stitch feathering). Two
    stacked blurs produce a long, very-low-alpha tail well outside the
    intended silhouette. At composite time that tail is a translucent ghost
    of the generated crop layered over the original (visible as a soft halo
    around the head/shoulders, especially where the two layers aren't
    pixel-aligned) rather than a clean fade to fully-original pixels.
    Rescaling so alpha < ``floor`` snaps to 0 and > ``ceil`` snaps to 255
    removes that faint tail while leaving the actual transition band (which
    is what keeps the seam from looking hard-edged) untouched.
    """
    floor = max(0, min(254, int(floor)))
    ceil = max(floor + 1, min(255, int(ceil)))
    arr = np.asarray(mask.convert("L")).astype(np.float32)
    arr = (arr - floor) * (255.0 / float(ceil - floor))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _skin_likelihood_strict(rgb_u8: np.ndarray) -> np.ndarray:
    """Strict [0,1] skin score; rejects cream cloth and dyed fabric.

    Dark donor faces (L≈25–40) still need non-zero scores so cheek sampling
    and under-chin gates work — only cream/dye reject stays hard.
    """
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    cr_score = np.clip(1.0 - np.abs(cr - 148.0) / 36.0, 0.0, 1.0)
    cb_score = np.clip(1.0 - np.abs(cb - 115.0) / 34.0, 0.0, 1.0)
    # Allow darker skin (Y down to ~20) without collapsing the score to 0.
    y_score = np.clip((y - 18.0) / 28.0, 0.0, 1.0) * np.clip(
        (245.0 - y) / 35.0, 0.0, 1.0
    )
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1] / 255.0
    # Dark skin often has low HSV sat in 8-bit; keep a floor so cheeks score.
    sat_score = np.clip((sat - 0.05) / 0.14, 0.0, 1.0) * np.clip(
        (0.70 - sat) / 0.20, 0.0, 1.0
    )
    hue_ok = ((hue <= 30.0) | (hue >= 155.0)).astype(np.float32)
    rgb_f = rgb_u8.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    fabric_red = (
        (r > 1.85 * np.maximum(g, 1.0))
        & (r > 1.85 * np.maximum(b, 1.0))
        & (sat > 0.55)
        & (y < 100)
    ).astype(np.float32)
    non_skin_hue = ((hue > 32.0) & (hue < 150.0)).astype(np.float32)
    return (
        cr_score * cb_score * y_score * sat_score * hue_ok
        * (1.0 - fabric_red)
        * (1.0 - non_skin_hue * np.clip(sat / 0.25, 0.0, 1.0))
    ).astype(np.float32)


def collapse_soft_chin_ghost(
    result: Image.Image,
    plate: Image.Image,
    edited: Image.Image,
    soft_alpha: Image.Image,
    *,
    mid_lo: float = 0.12,
    mid_hi: float = 0.88,
    chin_start_frac: float = 0.42,
) -> Image.Image:
    """Prefer edited pixels in soft mid-alpha under-chin skin (kills double neck).

    Soft stitch mid-band mixes the original pale neck stub under the new jaw,
    reading as a translucent duplicate / pale V. Where alpha is mid-range and
    the edited pixel is skin-like, take edited content instead of the blend.
    Cloth, sky, and the opaque face interior stay untouched.
    """
    if result.size != plate.size:
        plate = plate.resize(result.size, Image.Resampling.LANCZOS)
    if edited.size != result.size:
        edited = edited.resize(result.size, Image.Resampling.LANCZOS)
    a_img = soft_alpha.convert("L")
    if a_img.size != result.size:
        a_img = a_img.resize(result.size, Image.Resampling.BILINEAR)
    a = np.asarray(a_img, dtype=np.float32) / 255.0
    mid = (a > float(mid_lo)) & (a < float(mid_hi))
    if int(mid.sum()) < 20:
        return result

    core = a > 0.50
    ys, xs = np.where(core)
    if ys.size < 30:
        ys, xs = np.where(a > float(mid_lo))
    if ys.size < 30:
        return result
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    below = np.zeros_like(a, dtype=bool)
    below[y0 + int(float(chin_start_frac) * max(1, y1 - y0)) :, x0:x1] = True

    ed_u8 = pil_to_rgb_np(edited)
    pl_u8 = pil_to_rgb_np(plate)
    skin_ed = _skin_likelihood_strict(ed_u8)
    skin_pl = _skin_likelihood_strict(pl_u8)
    ed_lab = cv2.cvtColor(ed_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    pl_lab = cv2.cvtColor(pl_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    ed_chroma = np.sqrt(ed_lab[:, :, 1] ** 2 + ed_lab[:, :, 2] ** 2)
    pl_chroma = np.sqrt(pl_lab[:, :, 1] ** 2 + pl_lab[:, :, 2] ** 2)
    ed_hsv = cv2.cvtColor(ed_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    pl_hsv = cv2.cvtColor(pl_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    ed_sat = ed_hsv[:, :, 1] / 255.0
    pl_sat = pl_hsv[:, :, 1] / 255.0
    cloth = (
        ((ed_lab[:, :, 0] > 78.0) & (ed_chroma < 14.0))
        | ((ed_lab[:, :, 0] > 72.0) & (ed_sat < 0.16))
        | ((pl_lab[:, :, 0] > 78.0) & (pl_chroma < 14.0))
        | ((pl_lab[:, :, 0] > 72.0) & (pl_sat < 0.16))
        | ((ed_hsv[:, :, 0] > 32.0) & (ed_hsv[:, :, 0] < 150.0) & (ed_sat > 0.22))
        | ((pl_hsv[:, :, 0] > 32.0) & (pl_hsv[:, :, 0] < 150.0) & (pl_sat > 0.22))
    )
    # Ghost signature: mid-alpha under chin where plate is pale skin OR much
    # lighter than edited (dark donor necks fail strict skin_ed alone).
    pale_plate = (skin_pl >= 0.22) | (
        (pl_lab[:, :, 0] - ed_lab[:, :, 0] > 10.0) & (pl_chroma > 6.0) & (pl_sat > 0.08)
    )
    pick = mid & below & pale_plate & (~cloth) & (
        (skin_ed >= 0.12) | (ed_lab[:, :, 0] < pl_lab[:, :, 0] - 6.0)
    )
    if int(pick.sum()) < 15:
        return result
    res = pil_to_rgb_np(result).astype(np.float32)
    ed = ed_u8.astype(np.float32)
    # Soft ramp: more opaque mid-alpha → more edited; never paint outside pick.
    w = np.clip((a - float(mid_lo)) / max(float(mid_hi) - float(mid_lo), 1e-3), 0.0, 1.0)
    w = np.clip(0.55 + 0.45 * w, 0.0, 1.0) * pick.astype(np.float32)
    out = res * (1.0 - w[..., None]) + ed * w[..., None]
    return np_to_pil(np.clip(out, 0, 255))


def feathered_soft_composite(
    base: Image.Image,
    edit: Image.Image,
    mask: Image.Image,
    box: tuple[int, int, int, int],
    *,
    extra_blur_px: int = 8,
    alpha_floor: int = 10,
    alpha_ceil: int = 245,
) -> Image.Image:
    """Soft stitch with an extra blur on the alpha for less obvious jaw seams."""
    m = mask.convert("L")
    if extra_blur_px > 0:
        k = max(3, int(extra_blur_px) * 2 + 1)
        arr = np.asarray(m)
        arr = cv2.GaussianBlur(arr, (k, k), 0)
        m = Image.fromarray(arr)
    if alpha_floor > 0 or alpha_ceil < 255:
        m = clean_alpha_tails(m, floor=alpha_floor, ceil=alpha_ceil)
    return soft_composite(base, edit, m, box)

def describe_hair_length_hint(body: Image.Image, face: Image.Image, cache_dir) -> str:
    """Heuristic prompt add-on when body hair is longer than face crop."""
    br = pil_to_rgb_np(body)
    bbox = detect_best_face(br, cache_dir)
    if bbox is None:
        return ""
    # Measure vertical extent above face vs face height as crude long-hair proxy
    top_room = bbox.y0 / max(1, br.shape[0])
    face_frac = bbox.height / max(1, br.shape[0])
    if top_room > 0.18 and face_frac < 0.28:
        return " Specifically remove the long hair from Picture 1 completely."
    return ""


def get_face_landmarks5(
    rgb: np.ndarray,
    cache_dir,
    prefer_box: FaceBox | None = None,
) -> tuple[np.ndarray | None, str, str | None]:
    """
    Return 5 face landmarks as float32 (5, 2) in image XY order.

    Preference order:
      1. InsightFace (buffalo_l) — production geometry lock
      2. OpenCV box corners derived from detect_best_face — weak fallback

    When ``prefer_box`` is set (multi-person), pick the InsightFace detection
    with highest IoU against that box instead of the largest face.

    Returns (landmarks_or_None, backend_name, skip_reason_or_None).
    """
    insight_err = _INSIGHTFACE_INIT_ERROR
    app = ensure_insightface_app(cache_dir)
    if app is not None:
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            if faces:
                if prefer_box is not None:
                    scored = []
                    for f in faces:
                        x0, y0, x1, y1 = [float(v) for v in f.bbox]
                        scored.append((_iou_box(prefer_box, x0, y0, x1, y1), f))
                    scored.sort(key=lambda t: t[0], reverse=True)
                    face = scored[0][1] if scored[0][0] > 0.05 else max(
                        faces,
                        key=lambda f: float(
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        ),
                    )
                else:
                    face = max(
                        faces,
                        key=lambda f: float(
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        ),
                    )
                kps = np.asarray(face.kps, dtype=np.float32)
                if kps.shape != (5, 2):
                    insight_err = f"insightface_bad_kps_shape:{kps.shape}"
                else:
                    return kps, "insightface", None
            else:
                insight_err = "insightface_no_face_detected"
        except Exception as exc:
            insight_err = f"insightface_runtime_failed:{exc}"

    # OpenCV box → synthetic 5-point layout (eyes / nose / mouth corners)
    if prefer_box is not None:
        box = prefer_box
    else:
        box = detect_best_face(rgb, cache_dir)
    if box is None:
        reason = insight_err or "no_face_for_landmarks"
        return None, "none", reason
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    # Approximate 5-point template inside the box (not as good as InsightFace).
    pts = np.array(
        [
            [x0 + 0.30 * w, y0 + 0.38 * h],  # left eye
            [x0 + 0.70 * w, y0 + 0.38 * h],  # right eye
            [x0 + 0.50 * w, y0 + 0.55 * h],  # nose
            [x0 + 0.35 * w, y0 + 0.75 * h],  # left mouth
            [x0 + 0.65 * w, y0 + 0.75 * h],  # right mouth
        ],
        dtype=np.float32,
    )
    note = insight_err or "insightface_unavailable_used_box_prior"
    return pts, "box_prior", note


def estimate_head_direction_label(
    landmarks5: np.ndarray,
    *,
    roll_deg_thresh: float = 3.0,
    yaw_thresh: float = 0.12,
    yaw_strong_thresh: float = 0.30,
) -> dict[str, Any]:
    """Coarse, self-referential head yaw/roll description from 5 landmarks.

    Used to give the model a CONCRETE, image-specific direction cue in the
    prompt instead of a generic "keep the same direction" instruction, which
    on its own doesn't reliably stop head yaw / eye gaze from drifting
    toward the donor face or a frontal default. Unlike a pixel-level warp
    (which repeatedly produced compositing artifacts -- ghosting, streaking
    -- across several attempts), this only changes prompt text, so it cannot
    introduce a visual defect by construction; worst case is an unhelpful
    hint, never a broken pixel.

    Landmark order follows ``get_face_landmarks5``'s convention: [eye_a,
    eye_b, nose, mouth_a, mouth_b] in image (x, y) pixel coordinates. Index
    0/1 are NOT guaranteed to be screen-left/right (that depends on the
    detector, not on this function), so eyes are re-sorted by x here before
    anything is labeled "left"/"right" -- the labels below are always
    defined relative to that local sort, so they are self-consistent and
    verifiable by looking at the image, never dependent on an assumed
    anatomical left/right convention that could be backwards.

    Roll (in-plane tilt) is described directly and unambiguously ("the eye
    on the left/right side sits higher"), since that's derived purely from
    which sorted landmark has the smaller y -- no directional judgement
    call involved. Yaw (turn) is deliberately described only by MAGNITUDE
    ("turned moderately/strongly off-center"), not by an asserted left/right
    turn direction: a 2D nose-offset-from-eye-midpoint proxy is too noisy to
    confidently commit to a turn direction, and telling the model the wrong
    direction would be worse than not claiming one -- the model can still
    look at image 1 itself to see which way the offset goes.
    """
    lm = np.asarray(landmarks5, dtype=np.float64)
    info: dict[str, Any] = {"label": "", "roll_deg": None, "yaw_offset": None}
    if lm.shape[0] < 3:
        info["reason"] = "insufficient_landmarks"
        return info

    eyes = lm[:2]
    nose = lm[2]
    eye_left, eye_right = (eyes[0], eyes[1]) if eyes[0][0] <= eyes[1][0] else (eyes[1], eyes[0])
    eye_span = max(1.0, float(eye_right[0] - eye_left[0]))
    roll_deg = math.degrees(
        math.atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
    )
    eye_mid_x = 0.5 * (eye_left[0] + eye_right[0])
    yaw_offset = float(nose[0] - eye_mid_x) / eye_span
    info["roll_deg"] = round(roll_deg, 2)
    info["yaw_offset"] = round(yaw_offset, 4)

    parts: list[str] = []
    if abs(roll_deg) > roll_deg_thresh:
        higher = "left" if eye_left[1] < eye_right[1] else "right"
        parts.append(
            f"the eye on the {higher} side of the frame sits noticeably "
            f"higher than the other (about a {abs(round(roll_deg))}-degree "
            "tilt) -- match this exact tilt, do not level the head"
        )
    yaw_mag = abs(yaw_offset)
    if yaw_mag > yaw_thresh:
        strength = "strongly" if yaw_mag > yaw_strong_thresh else "moderately"
        parts.append(
            f"the head/nose is {strength} turned off-center relative to the "
            "eyes, not facing straight at the camera -- study the exact "
            "eye/nose/mouth alignment visible in the first image and "
            "reproduce that same off-center angle, do not straighten or "
            "frontalize it"
        )

    info["label"] = "; ".join(parts)
    return info


def _ellipse_alpha(
    h: int,
    w: int,
    cx: float,
    cy: float,
    span_x: float,
    span_y: float,
    *,
    core_min_alpha: float,
    feather_px: int,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - cx) / max(1e-3, span_x / 2.0)
    ny = (yy - cy) / max(1e-3, span_y / 2.0)
    r2 = nx * nx + ny * ny
    alpha = np.clip(1.0 - (r2 - 0.35) / 0.65, 0.0, 1.0)
    core = r2 <= 0.70
    alpha = np.where(core, np.maximum(alpha, float(core_min_alpha)), alpha)
    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    k = max(3, int(feather_px) | 1)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (k, k), 0)
    alpha_f = alpha_u8.astype(np.float32) / 255.0
    alpha_f = np.where(core, np.maximum(alpha_f, float(core_min_alpha)), alpha_f)
    return (np.clip(alpha_f, 0.0, 1.0) * 255.0).astype(np.uint8), core, alpha_f


def _box_paste_rgba(
    source_face: Image.Image,
    destination: Image.Image,
    cache_dir,
    *,
    core_min_alpha: float,
    feather_px: int,
    scale: float = 1.12,
) -> tuple[Image.Image | None, dict]:
    """
    Resize donor face into the destination face box (no affine).

    Used when InsightFace landmarks are unavailable — synthetic box_prior
    landmarks produce black/garbage warps that Kontext then "heals" back to
    the original identity.
    """
    info: dict = {
        "face_alignment": False,
        "face_alignment_backend": "box_paste",
        "face_alignment_skip_reason": None,
        "paste_core_min_alpha": float(core_min_alpha),
    }
    dest_rgb = pil_to_rgb_np(destination)
    src_rgb = pil_to_rgb_np(source_face)
    dest_box = detect_best_face(dest_rgb, cache_dir)
    if dest_box is None:
        info["face_alignment_skip_reason"] = "dest_face_box_missing"
        return None, info

    h, w = dest_rgb.shape[:2]
    # Expand dest box upward/sideways so glasses + forehead are covered.
    fw = max(8, int(dest_box.width * scale))
    fh = max(8, int(dest_box.height * scale * 1.08))
    cx = (dest_box.x0 + dest_box.x1) / 2.0
    # Bias upward: top of paste near top of dest face box (not centered, which
    # leaves original hair/glasses above the donor).
    top = float(dest_box.y0) - 0.25 * dest_box.height
    x0 = int(round(cx - fw / 2.0))
    y0 = int(round(top))
    x1, y1 = x0 + fw, y0 + fh
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c - x0c < 8 or y1c - y0c < 8:
        info["face_alignment_skip_reason"] = "dest_face_box_too_small"
        return None, info

    donor = cv2.resize(src_rgb, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    dx0, dy0 = x0c - x0, y0c - y0
    canvas[y0c:y1c, x0c:x1c] = donor[dy0 : dy0 + (y1c - y0c), dx0 : dx0 + (x1c - x0c)]

    alpha_u8, core, alpha_f = _ellipse_alpha(
        h,
        w,
        (x0c + x1c) / 2.0,
        (y0c + y1c) / 2.0,
        float(x1c - x0c) * 1.02,
        float(y1c - y0c) * 1.05,
        core_min_alpha=core_min_alpha,
        feather_px=feather_px,
    )
    placed = np.zeros((h, w), dtype=np.uint8)
    placed[y0c:y1c, x0c:x1c] = 1
    lum = canvas.astype(np.float32).mean(axis=2)
    valid = (placed > 0) & (lum > 12.0)
    alpha_u8 = (alpha_u8.astype(np.float32) * valid.astype(np.float32)).astype(np.uint8)
    alpha_f = alpha_u8.astype(np.float32) / 255.0
    core = core & valid

    rgba = np.dstack([canvas, alpha_u8])
    info["face_alignment"] = True
    info["face_alignment_backend"] = "box_paste"
    info["paste_mean_alpha"] = float(alpha_f[core].mean()) if core.any() else 0.0
    info["dest_face_box"] = [dest_box.x0, dest_box.y0, dest_box.x1, dest_box.y1]
    info["paste_valid_px"] = int(valid.sum())
    return Image.fromarray(rgba, mode="RGBA"), info


def align_face_to_destination(
    source_face: Image.Image,
    destination: Image.Image,
    cache_dir,
    *,
    core_min_alpha: float = 0.92,
    ellipse_scale_x: float = 2.05,
    ellipse_scale_y: float = 2.55,
    feather_px: int = 21,
    use_full_affine: bool = True,
    prefer_dest_box: FaceBox | None = None,
) -> tuple[Image.Image | None, dict]:
    """
    Warp source face onto destination face geometry.

    Default uses full 6-DOF affine (not similarity-only) so head yaw / looking
    direction can approximate the destination more closely.

    Returns (aligned_rgba_or_None, info_dict).
    Falls back to box-paste when InsightFace is missing or the affine warp is
    low-quality (black/empty), because bad warps make Kontext regenerate the
    destination identity.
    """
    info: dict = {
        "face_alignment": False,
        "face_alignment_backend": None,
        "face_alignment_skip_reason": None,
        "dest_landmarks_backend": None,
        "src_landmarks_backend": None,
        "paste_core_min_alpha": float(core_min_alpha),
        "use_full_affine": bool(use_full_affine),
    }
    dest_rgb = pil_to_rgb_np(destination)
    src_rgb = pil_to_rgb_np(source_face)

    dest_lm, dest_backend, dest_note = get_face_landmarks5(
        dest_rgb, cache_dir, prefer_box=prefer_dest_box
    )
    src_lm, src_backend, src_note = get_face_landmarks5(src_rgb, cache_dir)
    info["dest_landmarks_backend"] = dest_backend
    info["src_landmarks_backend"] = src_backend

    use_affine = (
        dest_lm is not None
        and src_lm is not None
        and dest_backend == "insightface"
        and src_backend == "insightface"
    )

    if not use_affine:
        # box_prior affine is unreliable — use explicit box paste instead.
        boxed, box_info = _box_paste_rgba(
            source_face,
            destination,
            cache_dir,
            core_min_alpha=core_min_alpha,
            feather_px=feather_px,
        )
        box_info["dest_landmarks_backend"] = dest_backend
        box_info["src_landmarks_backend"] = src_backend
        box_info["affine_skipped_reason"] = (
            dest_note or src_note or "insightface_required_for_affine"
        )
        if boxed is not None:
            return boxed, box_info
        info["face_alignment_skip_reason"] = box_info.get(
            "face_alignment_skip_reason"
        ) or (dest_note or src_note or "align_failed")
        return None, info

    # Full 6-DOF affine (not similarity-only): non-uniform scale/shear better
    # approximates head yaw so the pasted face looks the same direction as dest.
    if use_full_affine:
        matrix, inliers = cv2.estimateAffine2D(src_lm, dest_lm, method=cv2.LMEDS)
        affine_name = "estimateAffine2D"
    else:
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_lm, dest_lm, method=cv2.LMEDS
        )
        affine_name = "estimateAffinePartial2D"
    if matrix is None:
        boxed, box_info = _box_paste_rgba(
            source_face,
            destination,
            cache_dir,
            core_min_alpha=core_min_alpha,
            feather_px=feather_px,
        )
        if boxed is not None:
            box_info["affine_skipped_reason"] = f"{affine_name}_failed"
            return boxed, box_info
        info["face_alignment_skip_reason"] = f"{affine_name}_failed"
        return None, info
    info["affine_estimator"] = affine_name

    h, w = dest_rgb.shape[:2]
    warped = cv2.warpAffine(
        src_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    xs, ys = dest_lm[:, 0], dest_lm[:, 1]
    # Landmark-anchored face oval: cover cheeks/jaw, stop before hair silhouette
    # and dark night background (oversized ellipses cause flash halos).
    eye_span = float(max(1.0, xs.max() - xs.min()))
    mouth_y = float(ys[3:5].mean()) if ys.shape[0] >= 5 else float(ys.max())
    eye_y = float(ys[0:2].mean()) if ys.shape[0] >= 2 else float(ys.min())
    cx = float(xs.mean())
    cy = float(0.45 * eye_y + 0.55 * mouth_y)
    span_x = float(max(48.0, eye_span * float(ellipse_scale_x)))
    span_y = float(max(60.0, (mouth_y - eye_y) * float(ellipse_scale_y) * 2.2))
    # Cap vertical extent so top stays near brows, bottom near chin.
    top_limit = eye_y - 0.55 * eye_span
    bot_limit = mouth_y + 0.95 * eye_span
    cy = float(0.5 * (top_limit + bot_limit))
    span_y = float(min(span_y, max(60.0, bot_limit - top_limit)))
    alpha_u8, core, alpha_f = _ellipse_alpha(
        h,
        w,
        cx,
        cy,
        span_x,
        span_y,
        core_min_alpha=core_min_alpha,
        feather_px=feather_px,
    )

    # Reject black/empty warps (landmark failure residue).
    lum = warped.astype(np.float32).mean(axis=2)
    core_lum = float(lum[core].mean()) if core.any() else 0.0
    if core_lum < 18.0:
        boxed, box_info = _box_paste_rgba(
            source_face,
            destination,
            cache_dir,
            core_min_alpha=core_min_alpha,
            feather_px=feather_px,
        )
        if boxed is not None:
            box_info["affine_skipped_reason"] = f"warp_too_dark:core_lum={core_lum:.1f}"
            return boxed, box_info
        info["face_alignment_skip_reason"] = f"warp_too_dark:core_lum={core_lum:.1f}"
        return None, info

    rgba = np.dstack([warped, alpha_u8])
    info["face_alignment"] = True
    info["face_alignment_backend"] = f"src={src_backend}+dest={dest_backend}"
    if inliers is not None:
        info["face_alignment_inliers"] = int(np.asarray(inliers).sum())
    info["paste_mean_alpha"] = float(alpha_f[core].mean()) if core.any() else 0.0
    info["warp_core_luminance"] = core_lum
    return Image.fromarray(rgba, mode="RGBA"), info


def _landmark_centroid_in_box(
    lm: np.ndarray, box: FaceBox | None, *, tolerance: float = 0.75
) -> bool:
    """True when the landmark centroid sits inside ``box`` expanded by tolerance.

    ``get_face_landmarks5`` silently falls back to the largest face when the
    ``prefer_box`` IoU is poor, so a successful "insightface" return does not
    guarantee it locked onto the intended face. This is that missing check.
    """
    if box is None:
        return True
    cx = float(np.mean(lm[:, 0]))
    cy = float(np.mean(lm[:, 1]))
    mx = tolerance * float(max(1, box.width))
    my = tolerance * float(max(1, box.height))
    return (
        (float(box.x0) - mx) <= cx <= (float(box.x1) + mx)
        and (float(box.y0) - my) <= cy <= (float(box.y1) + my)
    )


def _insightface_pick_face(
    rgb: np.ndarray,
    cache_dir,
    prefer_box: FaceBox | None = None,
):
    """Return best InsightFace detection or None."""
    app = ensure_insightface_app(cache_dir)
    if app is None:
        return None
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            return None
        if prefer_box is not None:
            scored = [
                (_iou_box(prefer_box, *map(float, f.bbox)), f) for f in faces
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            if scored[0][0] > 0.05:
                return scored[0][1]
        return max(
            faces,
            key=lambda f: float(
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ),
        )
    except Exception:
        return None


def get_jaw_alignment_points(
    rgb: np.ndarray,
    cache_dir,
    prefer_box: FaceBox | None = None,
) -> tuple[np.ndarray | None, str, str | None]:
    """Jaw/chin + eye alignment points for Procrustes (6, 2) float32."""
    face = _insightface_pick_face(rgb, cache_dir, prefer_box)
    if face is not None:
        kps = np.asarray(getattr(face, "kps", None), dtype=np.float32)
        if kps.shape == (5, 2):
            return _jaw_points_from_landmarks5(kps), "insightface_jaw5", None

    lm, backend, note = get_face_landmarks5(rgb, cache_dir, prefer_box=prefer_box)
    if lm is None:
        return None, backend, note or "landmarks_missing"
    return _jaw_points_from_landmarks5(lm), backend, note


def _jaw_points_from_landmarks5(lm: np.ndarray) -> np.ndarray:
    """Derive chin + jaw corners from 5-point layout."""
    le, re, nose, lmouth, rmouth = lm[:5]
    mouth = 0.5 * (lmouth + rmouth)
    iod = float(math.hypot(re[0] - le[0], re[1] - le[1]))
    iod = max(iod, 1.0)
    chin = mouth + np.array([0.0, 0.38 * iod], dtype=np.float32)
    jaw_l = chin + np.array([-0.55 * iod, -0.08 * iod], dtype=np.float32)
    jaw_r = chin + np.array([0.55 * iod, -0.08 * iod], dtype=np.float32)
    return np.stack([le, re, nose, chin, jaw_l, jaw_r], axis=0).astype(np.float32)


def dump_neck_seam_debug(
    image: Image.Image,
    gen_mask_full: Image.Image,
    body: Image.Image,
    cache_dir,
    *,
    prefer_box: FaceBox | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """RCA overlays: gen-mask boundary vs jaw landmarks + neck seam crop."""
    info: dict[str, Any] = {}
    img = image.convert("RGB").copy()
    w, h = img.size
    if gen_mask_full.size != (w, h):
        gen_mask_full = gen_mask_full.convert("L").resize((w, h), Image.Resampling.NEAREST)

    m = (np.asarray(gen_mask_full.convert("L")) > 128).astype(np.uint8)
    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.where(boundary > 0)
    draw = ImageDraw.Draw(img)
    for x, y in zip(xs[:: max(1, len(xs) // 800)], ys[:: max(1, len(ys) // 800)]):
        draw.point((int(x), int(y)), fill=(0, 200, 255))

    body_rgb = pil_to_rgb_np(body.convert("RGB"))
    if body.size != (w, h):
        body_rgb = cv2.resize(body_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    box = prefer_box or detect_best_face(body_rgb, cache_dir)
    jaw_pts, backend, _ = get_jaw_alignment_points(body_rgb, cache_dir, prefer_box=box)
    info["jaw_backend"] = backend
    chin_y = None
    if jaw_pts is not None:
        for x, y in jaw_pts:
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 0))
        chin_y = float(np.max(jaw_pts[:, 1]))

    if box is not None:
        fh = max(1, box.height)
        cy = chin_y if chin_y is not None else float(box.y1)
        y0 = int(max(0, cy - 0.05 * fh))
        y1 = int(min(h, cy + 0.35 * fh))
        cx = int(0.5 * (box.x0 + box.x1))
        half_w = int(max(40, 0.55 * box.width))
        x0 = max(0, cx - half_w)
        x1 = min(w, cx + half_w)
        info["neck_crop_box"] = [x0, y0, x1, y1]
        neck_crop = img.crop((x0, y0, x1, y1))
    else:
        info["neck_crop_box"] = None
        neck_crop = img.crop((0, int(0.25 * h), w, int(0.45 * h)))

    return img, neck_crop, info


def procrustes_align_edited_crop_to_body_box(
    edited: Image.Image,
    body: Image.Image,
    box: tuple[int, int, int, int],
    cache_dir,
    *,
    crop_content_box: tuple[int, int, int, int] | None = None,
    prefer_body_box: FaceBox | None = None,
    alignment: str = "jaw",
    min_inliers: int = 3,
    min_scale: float = 0.40,
    max_scale: float = 2.50,
    max_rotation_deg: float = 12.0,
    max_translate_frac: float = 0.25,
    max_residual_frac: float = 0.06,
) -> tuple[Image.Image, dict[str, Any]]:
    """Procrustes (similarity) on a crop_stitch edited crop before stitch.

    Every coordinate lives in **content-local** space: the sub-rectangle of
    ``edited`` that maps 1:1 onto ``box`` in ``body``. When ``square_crop`` pads
    the scene, ``crop_content_box`` = ``(ox, oy, w, h)`` describes that
    sub-rectangle; otherwise the whole ``edited`` image is the content.

      body pixel (bx, by)
        -> content pixel (ox + (bx - x0) * cw_c / box_w,
                          oy + (by - y0) * ch_c / box_h)

    Source landmarks are detected on the content crop (not the padded image) so
    detection and warp share one frame. The similarity transform is applied once
    to the whole content crop, then pasted back at the pad offset.

    On any failure the original ``edited`` is returned unchanged with a reason.
    """
    info: dict[str, Any] = {
        "procrustes": False,
        "procrustes_reason": None,
        "procrustes_alignment": str(alignment),
        "scale": None,
        "rotation_deg": None,
        "translation": None,
        "inliers": None,
        "residual_px": None,
        "chin_shift_px": None,
        "content_box": None,
    }
    edited_rgb = edited.convert("RGB")
    body_rgb = body.convert("RGB")
    x0, y0, x1, y1 = [int(v) for v in box]
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    ew, eh = edited_rgb.size

    # Content sub-rectangle of `edited` that corresponds to `box` in body space.
    if crop_content_box is not None:
        ox, oy, cw_c, ch_c = [int(v) for v in crop_content_box]
    else:
        ox, oy, cw_c, ch_c = 0, 0, ew, eh
    ox = max(0, min(ox, max(0, ew - 1)))
    oy = max(0, min(oy, max(0, eh - 1)))
    cw_c = max(1, min(cw_c, ew - ox))
    ch_c = max(1, min(ch_c, eh - oy))
    info["content_box"] = [ox, oy, cw_c, ch_c]

    # Single body -> content-local scale used for BOTH prefer box and landmarks.
    sx = float(cw_c) / float(box_w)
    sy = float(ch_c) / float(box_h)

    content = edited_rgb.crop((ox, oy, ox + cw_c, oy + ch_c))

    def _to_content_xy(bx: float, by: float) -> tuple[float, float]:
        return ((float(bx) - x0) * sx, (float(by) - y0) * sy)

    # Expected face location in content-local coords (clamped to the content).
    prefer_gen: FaceBox | None = None
    if prefer_body_box is not None:
        gx0, gy0 = _to_content_xy(prefer_body_box.x0, prefer_body_box.y0)
        gx1, gy1 = _to_content_xy(prefer_body_box.x1, prefer_body_box.y1)
        gx0 = max(0.0, min(gx0, cw_c - 1.0))
        gy0 = max(0.0, min(gy0, ch_c - 1.0))
        gx1 = max(1.0, min(gx1, float(cw_c)))
        gy1 = max(1.0, min(gy1, float(ch_c)))
        if (gx1 - gx0) < 8.0 or (gy1 - gy0) < 8.0:
            info["procrustes_reason"] = "selected_face_outside_crop"
            return edited_rgb, info
        prefer_gen = FaceBox(
            int(gx0), int(gy0), int(gx1), int(gy1), float(prefer_body_box.conf)
        )

    use_jaw = str(alignment).lower() == "jaw"
    if use_jaw:
        src_lm, src_backend, src_note = get_jaw_alignment_points(
            pil_to_rgb_np(content), cache_dir, prefer_box=prefer_gen
        )
        dst_lm_body, dst_backend, dst_note = get_jaw_alignment_points(
            pil_to_rgb_np(body_rgb), cache_dir, prefer_box=prefer_body_box
        )
    else:
        src_lm, src_backend, src_note = get_face_landmarks5(
            pil_to_rgb_np(content), cache_dir, prefer_box=prefer_gen
        )
        dst_lm_body, dst_backend, dst_note = get_face_landmarks5(
            pil_to_rgb_np(body_rgb), cache_dir, prefer_box=prefer_body_box
        )
    info["src_landmarks_backend"] = src_backend
    info["dst_landmarks_backend"] = dst_backend
    if src_lm is None or dst_lm_body is None:
        info["procrustes_reason"] = src_note or dst_note or "landmarks_missing"
        return edited_rgb, info
    if not use_jaw and (
        src_backend != "insightface" or dst_backend != "insightface"
    ):
        info["procrustes_reason"] = "low_confidence_need_insightface"
        return edited_rgb, info

    # Reject a silent lock onto a neighbouring face on either side.
    if not _landmark_centroid_in_box(src_lm, prefer_gen):
        info["procrustes_reason"] = "src_landmarks_wrong_face"
        return edited_rgb, info
    if not _landmark_centroid_in_box(dst_lm_body, prefer_body_box):
        info["procrustes_reason"] = "dst_landmarks_wrong_face"
        return edited_rgb, info

    # Map body landmarks into the same content-local frame as `src_lm`.
    dst_lm = np.stack(
        [
            (dst_lm_body[:, 0] - float(x0)) * sx,
            (dst_lm_body[:, 1] - float(y0)) * sy,
        ],
        axis=1,
    ).astype(np.float32)

    if src_lm.shape != dst_lm.shape or src_lm.shape[0] < 3:
        info["procrustes_reason"] = f"jaw_point_mismatch:{src_lm.shape} vs {dst_lm.shape}"
        return edited_rgb, info

    try:
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_lm.astype(np.float32), dst_lm, method=cv2.LMEDS
        )
    except cv2.error as exc:
        info["procrustes_reason"] = f"estimateAffinePartial2D_cv_error:{exc}"
        return edited_rgb, info
    if matrix is None:
        info["procrustes_reason"] = "estimateAffinePartial2D_failed"
        return edited_rgb, info
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(math.hypot(a, b))
    rot = float(math.degrees(math.atan2(b, a))) if scale > 1e-8 else 0.0
    n_in = int(np.asarray(inliers).sum()) if inliers is not None else 0

    # Residual of the fit itself, in content-local pixels.
    proj = (matrix[:, :2] @ src_lm.astype(np.float64).T).T + matrix[:, 2]
    residual = float(np.sqrt(np.mean(np.sum((proj - dst_lm) ** 2, axis=1))))
    long_side = float(max(cw_c, ch_c))
    # Face-centre displacement, the number that actually moves the head.
    src_c = src_lm.astype(np.float64).mean(axis=0)
    dst_c = dst_lm.astype(np.float64).mean(axis=0)
    shift = float(math.hypot(*(dst_c - src_c)))

    info["scale"] = round(scale, 4)
    info["rotation_deg"] = round(rot, 3)
    info["translation"] = [round(float(matrix[0, 2]), 2), round(float(matrix[1, 2]), 2)]
    info["inliers"] = n_in
    info["residual_px"] = round(residual, 3)
    info["face_shift_px"] = round(shift, 2)
    src_chin_y = float(np.max(src_lm[:, 1]))
    dst_chin_y = float(np.max(dst_lm[:, 1]))
    info["chin_shift_px"] = round(abs(dst_chin_y - src_chin_y), 2)

    if n_in < int(min_inliers):
        info["procrustes_reason"] = f"low_inliers:{n_in}"
        return edited_rgb, info
    if not (float(min_scale) <= scale <= float(max_scale)):
        info["procrustes_reason"] = f"scale_out_of_range:{scale:.3f}"
        return edited_rgb, info
    if abs(rot) > float(max_rotation_deg):
        info["procrustes_reason"] = f"rotation_out_of_range:{rot:.2f}"
        return edited_rgb, info
    if shift > float(max_translate_frac) * long_side:
        info["procrustes_reason"] = f"translation_too_large:{shift:.1f}px"
        return edited_rgb, info
    if residual > float(max_residual_frac) * long_side:
        info["procrustes_reason"] = f"fit_residual_too_large:{residual:.1f}px"
        return edited_rgb, info

    # One warp of the whole content crop. BORDER_REPLICATE (never REFLECT) so a
    # translation cannot mirror the head back into frame as a ghost fragment.
    warped = cv2.warpAffine(
        pil_to_rgb_np(content),
        matrix,
        (cw_c, ch_c),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    out = edited_rgb.copy()
    out.paste(Image.fromarray(warped, mode="RGB"), (ox, oy))
    info["procrustes"] = True
    return out, info


def expand_crop_box_wide(
    image_size: tuple[int, int],
    selected: FaceBox,
    *,
    pad_top_frac: float = 1.20,
    pad_side_frac: float = 1.80,
    pad_bot_frac: float = 2.50,
    target_mp: float = 1.0,
    div_by: int = 16,
) -> tuple[int, int, int, int]:
    """Larger head+shoulder crop from the face box (not an upscale of a tight crop).

    Starts from face-relative pads, then grows uniformly toward image bounds until
    the native crop is near ``target_mp`` megapixels (or the frame is exhausted).
    """
    iw, ih = image_size
    fw = max(1, selected.width)
    fh = max(1, selected.height)
    cx = 0.5 * (selected.x0 + selected.x1)
    cy = 0.5 * (selected.y0 + selected.y1)

    x0 = int(selected.x0 - pad_side_frac * fw)
    y0 = int(selected.y0 - pad_top_frac * fh)
    x1 = int(selected.x1 + pad_side_frac * fw)
    y1 = int(selected.y1 + pad_bot_frac * fh)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)

    goal_px = max(1.0, float(target_mp) * 1_000_000.0)
    # Grow box toward image edges while keeping the face roughly centered.
    for _ in range(48):
        area = float(max(1, (x1 - x0) * (y1 - y0)))
        if area >= goal_px * 0.92:
            break
        # Expand by ~8% of current size each side, clamped to image.
        dx = max(8, int(0.08 * (x1 - x0)))
        dy = max(8, int(0.08 * (y1 - y0)))
        nx0, ny0 = max(0, x0 - dx), max(0, y0 - dy)
        nx1, ny1 = min(iw, x1 + dx), min(ih, y1 + dy)
        if (nx0, ny0, nx1, ny1) == (x0, y0, x1, y1):
            break
        x0, y0, x1, y1 = nx0, ny0, nx1, ny1

    # Keep face inside the crop (safety).
    x0 = min(x0, int(selected.x0))
    y0 = min(y0, int(selected.y0))
    x1 = max(x1, int(selected.x1))
    y1 = max(y1, int(selected.y1))

    cw = evenify(max(div_by, x1 - x0), div_by)
    ch = evenify(max(div_by, y1 - y0), div_by)
    # Re-center on face if possible.
    x0 = int(max(0, min(iw - cw, round(cx - 0.5 * cw))))
    y0 = int(max(0, min(ih - ch, round(cy - 0.45 * ch))))
    x1 = min(iw, x0 + cw)
    y1 = min(ih, y0 + ch)
    return (x0, y0, x1, y1)


def procrustes_align_generated_to_body(
    generated: Image.Image,
    body: Image.Image,
    cache_dir,
    *,
    prefer_body_box: FaceBox | None = None,
    prefer_gen_box: FaceBox | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Similarity transform (Procrustes): map generated face landmarks → body.

    Uses ``cv2.estimateAffinePartial2D`` (rotation + uniform scale + translation)
    so head size/pose can be locked to the original photo after a full-frame
    Krea2 sample. Returns ``(aligned, info)``; on failure returns ``generated``.
    """
    info: dict[str, Any] = {
        "procrustes": False,
        "procrustes_reason": None,
        "scale": None,
        "rotation_deg": None,
        "translation": None,
    }
    gen = generated.convert("RGB")
    bod = body.convert("RGB")
    if gen.size != bod.size:
        gen = gen.resize(bod.size, Image.Resampling.LANCZOS)
    gen_rgb = pil_to_rgb_np(gen)
    bod_rgb = pil_to_rgb_np(bod)
    src_lm, src_backend, src_note = get_face_landmarks5(
        gen_rgb, cache_dir, prefer_box=prefer_gen_box
    )
    dst_lm, dst_backend, dst_note = get_face_landmarks5(
        bod_rgb, cache_dir, prefer_box=prefer_body_box
    )
    info["src_landmarks_backend"] = src_backend
    info["dst_landmarks_backend"] = dst_backend
    if src_lm is None or dst_lm is None:
        info["procrustes_reason"] = src_note or dst_note or "landmarks_missing"
        return gen, info
    if src_backend != "insightface" or dst_backend != "insightface":
        info["procrustes_reason"] = "need_insightface_landmarks"
        return gen, info
    matrix, inliers = cv2.estimateAffinePartial2D(src_lm, dst_lm, method=cv2.LMEDS)
    if matrix is None:
        info["procrustes_reason"] = "estimateAffinePartial2D_failed"
        return gen, info
    # Recover similarity params from the 2x3 matrix.
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(math.hypot(a, b))
    rot = float(math.degrees(math.atan2(b, a))) if scale > 1e-8 else 0.0
    info["scale"] = round(scale, 4)
    info["rotation_deg"] = round(rot, 3)
    info["translation"] = [round(float(matrix[0, 2]), 2), round(float(matrix[1, 2]), 2)]
    if inliers is not None:
        info["inliers"] = int(np.asarray(inliers).sum())
    h, w = bod_rgb.shape[:2]
    warped = cv2.warpAffine(
        gen_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    info["procrustes"] = True
    return Image.fromarray(warped, mode="RGB"), info


def relock_pose_to_destination(
    generated: Image.Image,
    destination: Image.Image,
    cache_dir,
    *,
    face_mask: Image.Image | None = None,
    use_full_affine: bool = True,
    core_min_alpha: float = 0.90,
    ellipse_scale_x: float = 2.05,
    ellipse_scale_y: float = 2.55,
    feather_px: int = 21,
    stitch_feather_px: int = 8,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
    max_rotation_deg: float = 20.0,
    max_translate_frac: float = 0.20,
) -> tuple[Image.Image, dict]:
    """
    Force ``generated`` face orientation onto ``destination`` landmarks.

    After generative refine, head yaw / eye gaze often drift toward a frontal
    identity. Re-aligning locks looking direction to the original photo.

    ``min_scale``/``max_scale``/``max_rotation_deg``/``max_translate_frac``
    bound the recovered similarity/affine transform in the ``face_mask``
    branch: a bad landmark match can send the estimator to an implausible
    transform, and warping+compositing that produces a large, misplaced
    duplicate of image content instead of a subtle pose correction. Anything
    outside these bounds is rejected and ``generated`` is returned unchanged.
    """
    info: dict = {"pose_relock": False, "pose_relock_reason": None}
    gen = generated.convert("RGB")
    dest = destination.convert("RGB")
    if gen.size != dest.size:
        gen = gen.resize(dest.size, Image.Resampling.LANCZOS)

    aligned_rgba, align_info = align_face_to_destination(
        gen,
        dest,
        cache_dir,
        core_min_alpha=core_min_alpha,
        ellipse_scale_x=ellipse_scale_x,
        ellipse_scale_y=ellipse_scale_y,
        feather_px=feather_px,
        use_full_affine=use_full_affine,
    )
    info["align_info"] = align_info
    if aligned_rgba is None:
        info["pose_relock_reason"] = align_info.get("face_alignment_skip_reason") or "align_failed"
        return gen, info

    if face_mask is not None:
        # Warp full generated crop via the same landmark matrix, then mask-blend.
        dest_rgb = pil_to_rgb_np(dest)
        gen_rgb = pil_to_rgb_np(gen)
        dest_lm, dest_backend, _ = get_face_landmarks5(dest_rgb, cache_dir)
        src_lm, src_backend, _ = get_face_landmarks5(gen_rgb, cache_dir)
        if (
            dest_lm is not None
            and src_lm is not None
            and dest_backend == "insightface"
            and src_backend == "insightface"
        ):
            if use_full_affine:
                matrix, _ = cv2.estimateAffine2D(src_lm, dest_lm, method=cv2.LMEDS)
            else:
                matrix, _ = cv2.estimateAffinePartial2D(
                    src_lm, dest_lm, method=cv2.LMEDS
                )
            if matrix is not None:
                h, w = dest_rgb.shape[:2]
                # Sanity-bound the recovered transform before ever warping pixels.
                # A bad landmark match (stylized lighting, wrong face, low-conf
                # detection) can send estimateAffine[Partial]2D to a wild
                # scale/rotation/translation; warping+compositing that produces a
                # large, misplaced duplicate of image content (a "ghost" head)
                # rather than a subtle pose correction. Reject anything outside a
                # plausible single-photo pose-correction range and keep the
                # unwarped generated image instead.
                a, b = float(matrix[0, 0]), float(matrix[0, 1])
                scale = float(math.hypot(a, b))
                rot = float(math.degrees(math.atan2(b, a))) if scale > 1e-8 else 0.0
                tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
                diag = max(1.0, float(math.hypot(w, h)))
                translate_frac = float(math.hypot(tx, ty)) / diag
                info["scale"] = round(scale, 4)
                info["rotation_deg"] = round(rot, 3)
                info["translation"] = [round(tx, 2), round(ty, 2)]
                out_of_bounds = (
                    scale < min_scale
                    or scale > max_scale
                    or abs(rot) > max_rotation_deg
                    or translate_frac > max_translate_frac
                )
                if out_of_bounds:
                    info["pose_relock_reason"] = (
                        f"transform_out_of_bounds(scale={scale:.3f} "
                        f"rot_deg={rot:.1f} translate_frac={translate_frac:.3f})"
                    )
                    return gen, info
                warped = cv2.warpAffine(
                    gen_rgb,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                out = feathered_soft_composite(
                    dest,
                    Image.fromarray(warped, mode="RGB"),
                    face_mask,
                    (0, 0, w, h),
                    extra_blur_px=int(stitch_feather_px),
                )
                info["pose_relock"] = True
                info["pose_relock_backend"] = "affine_mask_blend"
                info["affine_estimator"] = (
                    "estimateAffine2D" if use_full_affine else "estimateAffinePartial2D"
                )
                return out, info

    pasted, paste_info = paste_aligned_face(dest, aligned_rgba)
    info.update(paste_info)
    info["pose_relock"] = bool(paste_info.get("composite_paste"))
    info["pose_relock_backend"] = "rgba_paste"
    if not info["pose_relock"]:
        info["pose_relock_reason"] = paste_info.get("composite_paste_skip_reason") or "paste_failed"
        return gen, info
    return pasted, info


def color_match_rgba_to_destination(
    aligned_rgba: Image.Image,
    destination: Image.Image,
    strength: float = 0.55,
) -> Image.Image:
    """
    Shift pasted-face LAB toward the destination face being replaced.

    Target stats come from destination pixels *under* the paste alpha (the
    original face), not a dilated ring outside it. Night/group shots put
    jacket/sky/bushes in that ring and pull skin toward dark reddish smears.
    """
    if aligned_rgba is None or strength <= 0:
        return aligned_rgba
    if aligned_rgba.size != destination.size:
        aligned_rgba = aligned_rgba.resize(destination.size, Image.Resampling.LANCZOS)
    rgba = np.asarray(aligned_rgba.convert("RGBA")).astype(np.float32)
    bod = pil_to_rgb_np(destination).astype(np.float32)
    alpha = rgba[:, :, 3] / 255.0
    core = alpha > 0.55
    if int(core.sum()) < 50:
        return aligned_rgba
    face_lab = cv2.cvtColor(rgba[:, :, :3] / 255.0, cv2.COLOR_RGB2LAB)
    bod_lab = cv2.cvtColor(bod / 255.0, cv2.COLOR_RGB2LAB)
    for c in range(3):
        src_vals = face_lab[:, :, c][core]
        tgt_vals = bod_lab[:, :, c][core]
        if src_vals.size == 0 or tgt_vals.size == 0:
            continue
        shift = float(tgt_vals.mean() - src_vals.mean()) * float(strength)
        face_lab[:, :, c] = face_lab[:, :, c] + shift * alpha
    matched = cv2.cvtColor(face_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    out = np.dstack(
        [
            np.clip(matched * 255.0, 0, 255).astype(np.uint8),
            rgba[:, :, 3].astype(np.uint8),
        ]
    )
    return Image.fromarray(out, mode="RGBA")


def paste_aligned_face(
    destination: Image.Image,
    aligned_rgba: Image.Image,
    *,
    seamless: bool = True,
    clone_mode: str = "normal",
) -> tuple[Image.Image, dict]:
    """
    Composite aligned RGBA face onto destination RGB.

    When ``seamless`` is True, follow alpha composite with OpenCV
    ``seamlessClone``. Default mode is ``NORMAL_CLONE`` (Poisson lighting).

    ``MIXED_CLONE`` is available but unsafe on night/flash group photos — it
    often paints dark reddish cheek/jaw smears. If the clone drifts too far
    from the alpha composite, we discard it and keep the LAB-matched paste.
    """
    info: dict = {
        "composite_paste": False,
        "composite_paste_skip_reason": None,
        "seamless_clone": False,
        "seamless_clone_mode": None,
    }
    if aligned_rgba is None:
        info["composite_paste_skip_reason"] = "aligned_rgba_none"
        return destination.convert("RGB"), info
    if aligned_rgba.size != destination.size:
        aligned_rgba = aligned_rgba.resize(destination.size, Image.Resampling.LANCZOS)
    base = destination.convert("RGBA")
    out = Image.alpha_composite(base, aligned_rgba.convert("RGBA")).convert("RGB")
    info["composite_paste"] = True

    if not seamless:
        return out, info

    mode = str(clone_mode or "normal").strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return out, info
    clone_flag = cv2.NORMAL_CLONE if mode != "mixed" else cv2.MIXED_CLONE
    info["seamless_clone_mode"] = "mixed" if clone_flag == cv2.MIXED_CLONE else "normal"

    try:
        dest_rgb = pil_to_rgb_np(destination)
        face_rgb = pil_to_rgb_np(out)
        alpha = np.asarray(aligned_rgba.convert("RGBA"))[:, :, 3]
        # Inner face only — exclude soft feather halo that bleeds into night sky.
        mask = (alpha >= 120).astype(np.uint8) * 255
        if int(mask.sum()) < 500:
            return out, info
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_core = cv2.erode(mask, k, iterations=2)
        if int(mask_core.sum()) < 200:
            mask_core = cv2.erode(mask, k, iterations=1)
        if int(mask_core.sum()) < 200:
            mask_core = mask
        ys, xs = np.where(mask_core > 0)
        if len(xs) < 20:
            return out, info
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        src_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(dest_rgb, cv2.COLOR_RGB2BGR)
        cloned = cv2.seamlessClone(src_bgr, dst_bgr, mask_core, (cx, cy), clone_flag)
        cloned_rgb = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB).astype(np.float32)
        a = (alpha.astype(np.float32) / 255.0)[..., None]
        dest_f = dest_rgb.astype(np.float32)
        alpha_rgb = face_rgb.astype(np.float32)
        blended = cloned_rgb * a + dest_f * (1.0 - a)

        # Guard: reject smear/halo clones that diverge hard from alpha paste.
        core = alpha >= 120
        if int(core.sum()) >= 80:
            drift = float(
                np.mean(np.abs(blended[core] - alpha_rgb[core]))
            )
            info["seamless_clone_drift"] = round(drift, 2)
            if drift > 28.0:
                info["seamless_clone_rejected"] = "drift_too_high"
                return out, info

        info["seamless_clone"] = True
        return np_to_pil(np.clip(blended, 0, 255)), info
    except Exception as exc:  # noqa: BLE001 — alpha composite already succeeded
        info["seamless_clone_error"] = str(exc)
        return out, info


def lab_histogram_match_face(
    result: Image.Image,
    body: Image.Image,
    mask: Image.Image,
    strength: float = 0.35,
    edge_only_px: int = 0,
) -> Image.Image:
    """Mild LAB mean match inside mask to reduce neck/skin discontinuity.

    Target stats = original body pixels under the mask (the face being replaced),
    not a dilated exterior ring (which on night shots is jacket/sky).

    ``edge_only_px`` restricts WHERE the shift is applied, not the shift
    itself: with it set, the mean color delta is still measured over the
    whole masked region, but only painted back within ``edge_only_px`` of the
    mask boundary -- the solid interior (face/forehead/cheeks) is left alone.
    Without this, the full-mask weighting (``shift * m``) pulls the donor's
    identity skin tone partway back toward the target's original skin tone
    everywhere under the mask, including deep in the face -- GPU-observed as
    the swapped face reading as a blend toward the wrong person's skin color
    rather than the donor's own tone (2026-08-10). A seam only needs
    correcting at the seam.
    """
    if strength <= 0:
        return result
    res = pil_to_rgb_np(result).astype(np.float32)
    bod = pil_to_rgb_np(body).astype(np.float32)
    m = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    if m.shape[:2] != res.shape[:2]:
        m = cv2.resize(m, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
    core = m > 0.5
    if int(core.sum()) < 50:
        return result
    weight = m
    if edge_only_px > 0:
        k = edge_only_px * 2 + 1
        eroded = cv2.erode(
            core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
        weight = m * (1.0 - eroded.astype(np.float32))
    res_lab = cv2.cvtColor(res / 255.0, cv2.COLOR_RGB2LAB)
    bod_lab = cv2.cvtColor(bod / 255.0, cv2.COLOR_RGB2LAB)
    for c in range(3):
        src_vals = res_lab[:, :, c][core]
        tgt_vals = bod_lab[:, :, c][core]
        if src_vals.size == 0 or tgt_vals.size == 0:
            continue
        shift = float(tgt_vals.mean() - src_vals.mean()) * strength
        res_lab[:, :, c] = res_lab[:, :, c] + shift * weight
    out = cv2.cvtColor(res_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np_to_pil(np.clip(out * 255.0, 0, 255))


def match_neck_stub_to_head_tone(
    result: Image.Image,
    mask: Image.Image,
    *,
    ring_px: int = 70,
    interior_erode_px: int = 20,
    strength: float = 0.65,
    l_strength: float | None = 0.90,
    ab_strength: float | None = 1.0,
    passes: int = 2,
    ab_gate_scale: float = 22.0,
    blur_frac: float = 0.12,
) -> Image.Image:
    """Tint the exposed original neck just outside the paste mask toward the
    donor's own skin tone -- color only, no new content is pasted.

    The paste mask (``mask``) has to stay tight: widening it to reach the
    real collarline pastes the edited crop's own (donor) clothing/necklace
    content over the target's real clothes -- GPU-observed as a double
    necklace (2026-08-10). But that means the sliver of neck just past the
    mask edge is still 100% the TARGET's original, un-swapped skin tone,
    which reads as a seam against the donor-toned face right next to it.

    This shifts ONLY that thin exterior ring's LAB mean toward the mean tone
    sampled from deep inside the mask (a heavily-eroded core, so the sample
    is pure donor face, not itself edge-blended) -- the opposite direction
    from ``lab_histogram_match_face``, which pulls edited pixels toward the
    original. No pixels inside the mask are touched.

    ``l_strength``/``ab_strength`` override ``strength`` per-channel when
    given (L = lightness, a/b = hue/chroma). A single weak pass left a pale
    collar-V under dark donor faces (2026-08-10): donor-relative ab-distance
    gating alone half-rejected pale skin, blur leaked into the face, and one
    mean-shift couldn't close a ~50 L gap. Fixes: absolute skin likelihood *
    softer ab gate, cream/cloth reject, re-zero weight inside the mask after
    blur, mean from gated ring pixels only, and optional second pass.

    The ring is clipped to the BOTTOM half of the mask's bounding box
    (chin/neck/chest side only). Un-clipped, ``dilate(mask) - mask`` forms a
    band around the mask's ENTIRE perimeter, including above the hair --
    which pulled sky toward donor skin (dark halo above head).
    """
    if ring_px <= 0:
        return result
    if strength <= 0 and (l_strength or 0) <= 0 and (ab_strength or 0) <= 0:
        return result
    n_passes = max(1, int(passes))
    res_u8 = pil_to_rgb_np(result)
    res = res_u8.astype(np.float32)
    m = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    if m.shape[:2] != res.shape[:2]:
        m = cv2.resize(m, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
    core = (m > 0.5).astype(np.uint8)
    k_deep = interior_erode_px * 2 + 1
    deep_core = cv2.erode(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_deep, k_deep))
    )
    if int(deep_core.sum()) < 50:
        return result
    k_ring = ring_px * 2 + 1
    dilated = cv2.dilate(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ring, k_ring))
    )
    ring = dilated.astype(bool) & (~core.astype(bool))
    ys, _xs = np.where(core.astype(bool))
    if ys.size > 0:
        y0, y1 = int(ys.min()), int(ys.max())
        below_chin = np.zeros_like(core, dtype=bool)
        below_chin[y0 + int(0.5 * (y1 - y0)) :, :] = True
        ring = ring & below_chin
    if int(ring.sum()) < 50:
        return result
    res_lab = cv2.cvtColor(res / 255.0, cv2.COLOR_RGB2LAB)
    per_channel_strength = [
        l_strength if l_strength is not None else strength,
        ab_strength if ab_strength is not None else strength,
        ab_strength if ab_strength is not None else strength,
    ]
    deep_bool = deep_core.astype(bool)
    for _ in range(n_passes):
        # Skin gate: absolute skin likelihood (pale necks pass) * softer
        # donor-ab proximity (rejects white trim / saturated fabric). Hard
        # cream/cloth reject keeps floral dresses and jerseys untouched.
        donor_a = float(res_lab[:, :, 1][deep_bool].mean())
        donor_b = float(res_lab[:, :, 2][deep_bool].mean())
        ab_dist = np.sqrt(
            (res_lab[:, :, 1] - donor_a) ** 2 + (res_lab[:, :, 2] - donor_b) ** 2
        )
        ab_gate = np.clip(1.0 - ab_dist / max(float(ab_gate_scale), 1e-3), 0.0, 1.0)
        cur_rgb = np.clip(
            cv2.cvtColor(res_lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        skin_like = _skin_likelihood_strict(cur_rgb)
        chroma = np.sqrt(res_lab[:, :, 1] ** 2 + res_lab[:, :, 2] ** 2)
        hsv = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        cloth_reject = (
            ((res_lab[:, :, 0] > 78.0) & (chroma < 14.0))
            | ((res_lab[:, :, 0] > 72.0) & (sat < 0.16))
            | ((res_lab[:, :, 0] > 68.0) & (np.abs(res_lab[:, :, 1]) < 4.0) & (chroma < 18.0))
        )
        skin_gate = np.clip(0.25 + 0.75 * skin_like, 0.0, 1.0) * ab_gate
        skin_gate = skin_gate * (~cloth_reject).astype(np.float32)

        ring_weight = np.zeros(res.shape[:2], dtype=np.float32)
        ring_weight[ring] = 1.0
        ring_weight = cv2.GaussianBlur(
            ring_weight, (0, 0), sigmaX=max(1.0, ring_px * float(blur_frac))
        )
        # Blur would otherwise leak into the face interior — lock mask pixels.
        ring_weight = ring_weight * (1.0 - m) * skin_gate

        gated = ring & (skin_gate > 0.20)
        if int(gated.sum()) < 50:
            break
        for c in range(3):
            ch_strength = per_channel_strength[c]
            if ch_strength <= 0:
                continue
            donor_mean = float(res_lab[:, :, c][deep_bool].mean())
            ring_vals = res_lab[:, :, c][gated]
            if ring_vals.size == 0:
                continue
            shift = (donor_mean - float(ring_vals.mean())) * ch_strength
            res_lab[:, :, c] = res_lab[:, :, c] + shift * ring_weight
    out = cv2.cvtColor(res_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np_to_pil(np.clip(out * 255.0, 0, 255))


def build_harmonization_mask(
    body: Image.Image,
    gen_mask: Image.Image,
    *,
    dilate_px: int = 24,
    bot_extend_frac: float = 0.15,
    skin_thresh: float = 0.45,
    include_arms: bool = True,
) -> tuple[Image.Image, Image.Image, dict[str, float | int | bool]]:
    """Build full harmonization mask (gen ⊂ harm) and color-only harm ring.

    Returns ``(full_harm_mask, harm_ring_mask, info)``. The generation mask
    is always a subset of ``full_harm_mask``. ``harm_ring_mask`` is the band
    where only original-body pixels get tone-matched (never generation).
    """
    info: dict[str, float | int | bool] = {
        "harmonization_dilate_px": int(dilate_px),
        "harmonization_bot_extend_frac": float(bot_extend_frac),
        "harmonization_skin_thresh": float(skin_thresh),
        "harmonization_include_arms": bool(include_arms),
    }
    w, h = body.size
    m = np.asarray(gen_mask.convert("L")).astype(np.float32) / 255.0
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    core = (m > 0.5).astype(np.uint8)
    empty = Image.new("L", (w, h), 0)
    if int(core.sum()) < 20:
        info.update(
            harm_area_px=0,
            harm_ring_area_px=0,
            full_harm_area_px=0,
            harm_skin_frac=0.0,
            gen_subset_of_harm=True,
        )
        return empty, empty, info

    dilated = np.asarray(
        dilate_mask(Image.fromarray(core * 255), max(0, int(dilate_px))).convert("L")
    )
    dilated_bool = dilated > 128

    ys, xs = np.where(core.astype(bool))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    mh = max(1, y1 - y0)
    chin_y = y0 + int(0.45 * mh)
    extend_y = min(h, int(y1 + max(1.0, float(bot_extend_frac)) * mh))
    lateral = int(0.85 * mh) if include_arms else int(0.35 * mh)
    x0a = max(0, x0 - lateral)
    x1a = min(w, x1 + lateral)
    below = np.zeros((h, w), dtype=bool)
    below[chin_y:extend_y, x0a:x1a] = True
    extension = dilated_bool | below

    bod = pil_to_rgb_np(body).astype(np.uint8)
    skin_like = _skin_likelihood_strict(bod)
    skin_mask = skin_like >= float(skin_thresh)
    harm_ring_bool = extension & skin_mask & (~core.astype(bool))
    full_harm_bool = core.astype(bool) | harm_ring_bool

    harm_ring_img = Image.fromarray(harm_ring_bool.astype(np.uint8) * 255)
    full_harm_img = Image.fromarray(full_harm_bool.astype(np.uint8) * 255)

    harm_ring_area = int(harm_ring_bool.sum())
    gen_area = int(core.sum())
    full_harm_area = int(full_harm_bool.sum())
    skin_in_extension = int((extension & skin_mask).sum())
    info["harm_area_px"] = harm_ring_area
    info["harm_ring_area_px"] = harm_ring_area
    info["gen_area_px"] = gen_area
    info["full_harm_area_px"] = full_harm_area
    info["harm_skin_frac"] = (
        float(harm_ring_area) / float(max(1, skin_in_extension))
        if skin_in_extension
        else 0.0
    )
    info["gen_subset_of_harm"] = bool(np.all(~core.astype(bool) | full_harm_bool))

    if harm_ring_area > 0:
        hy, hx = np.where(harm_ring_bool)
        info["harm_bbox"] = (
            int(hx.min()),
            int(hy.min()),
            int(hx.max()) + 1,
            int(hy.max()) + 1,
        )
    else:
        info["harm_bbox"] = (0, 0, 0, 0)
    gy, gx = np.where(core.astype(bool))
    info["gen_bbox"] = (
        int(gx.min()),
        int(gy.min()),
        int(gx.max()) + 1,
        int(gy.max()) + 1,
    )
    if full_harm_area > 0:
        fy, fx = np.where(full_harm_bool)
        info["full_harm_bbox"] = (
            int(fx.min()),
            int(fy.min()),
            int(fx.max()) + 1,
            int(fy.max()) + 1,
        )
    else:
        info["full_harm_bbox"] = (0, 0, 0, 0)
    return full_harm_img, harm_ring_img, info


def assert_no_gen_interior_leak(
    pre_harm: Image.Image,
    post_harm: Image.Image,
    gen_mask: Image.Image,
    *,
    interior_erode_px: int = 20,
    epsilon: float = 1.0,
) -> dict[str, float | bool]:
    """Verify generation-mask interior is unchanged after harmonization."""
    pre = pil_to_rgb_np(pre_harm).astype(np.float32)
    post = pil_to_rgb_np(post_harm).astype(np.float32)
    gm = np.asarray(gen_mask.convert("L")).astype(np.float32) / 255.0
    h, w = pre.shape[:2]
    if gm.shape[:2] != (h, w):
        gm = cv2.resize(gm, (w, h), interpolation=cv2.INTER_LINEAR)
    core = (gm > 0.5).astype(np.uint8)
    k = interior_erode_px * 2 + 1
    deep = cv2.erode(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if int(deep.sum()) < 1:
        deep = core
    mask = deep.astype(bool)
    if not mask.any():
        return {"gen_interior_max_diff": 0.0, "gen_interior_ok": True}
    diff = np.abs(post - pre)[mask]
    max_diff = float(diff.max()) if diff.size else 0.0
    return {
        "gen_interior_max_diff": max_diff,
        "gen_interior_ok": max_diff <= epsilon,
    }


def harmonize_skin_tone(
    pre_harm: Image.Image,
    body: Image.Image,
    gen_mask: Image.Image,
    full_harm_mask: Image.Image,
    harm_ring_mask: Image.Image,
    *,
    strength: float = 0.55,
    feather_px: int = 16,
    interior_erode_px: int = 20,
    gen_lock_epsilon: float = 1.0,
) -> tuple[Image.Image, dict[str, float | str | bool]]:
    """Tone-match original body in harm ring only; gen interior locked to pre_harm."""
    info: dict[str, float | str | bool] = {
        "harmonization_method": "lab_mean_body_ring",
        "harmonization_strength": float(strength),
        "harmonization_feather_px": float(feather_px),
        "harmonization_applied": False,
    }
    if strength <= 0:
        info["harmonization_reason"] = "strength_zero"
        return pre_harm, info

    pre = pil_to_rgb_np(pre_harm).astype(np.float32)
    bod = pil_to_rgb_np(body).astype(np.float32)
    gm = np.asarray(gen_mask.convert("L")).astype(np.float32) / 255.0
    full_m = np.asarray(full_harm_mask.convert("L")).astype(np.float32) / 255.0
    ring_m = np.asarray(harm_ring_mask.convert("L")).astype(np.float32) / 255.0
    h, w = pre.shape[:2]
    for arr in (gm, full_m, ring_m):
        if arr.shape[:2] != (h, w):
            pass
    if gm.shape[:2] != (h, w):
        gm = cv2.resize(gm, (w, h), interpolation=cv2.INTER_LINEAR)
    if full_m.shape[:2] != (h, w):
        full_m = cv2.resize(full_m, (w, h), interpolation=cv2.INTER_LINEAR)
    if ring_m.shape[:2] != (h, w):
        ring_m = cv2.resize(ring_m, (w, h), interpolation=cv2.INTER_LINEAR)

    core = (gm > 0.5).astype(np.uint8)
    ring = (ring_m > 0.5) & (~core.astype(bool))
    full_harm = (full_m > 0.5).astype(np.uint8)
    k_deep = interior_erode_px * 2 + 1
    deep_core = cv2.erode(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_deep, k_deep))
    )
    if int(deep_core.sum()) < 50 or int(ring.sum()) < 30:
        info["harmonization_reason"] = "insufficient_pixels"
        return pre_harm, info

    # Outer-edge feather only: weight rises inward from full_harm boundary.
    dist_in = cv2.distanceTransform(full_harm, cv2.DIST_L2, 5).astype(np.float32)
    if feather_px > 0:
        weight = np.clip(dist_in / float(feather_px), 0.0, 1.0)
    else:
        weight = (full_harm > 0).astype(np.float32)
    weight = weight * ring.astype(np.float32)
    weight[core.astype(bool)] = 0.0

    pre_lab = cv2.cvtColor(pre / 255.0, cv2.COLOR_RGB2LAB)
    bod_lab = cv2.cvtColor(bod / 255.0, cv2.COLOR_RGB2LAB)
    deep_bool = deep_core.astype(bool)
    tone_lab = bod_lab.copy()
    for c in range(3):
        ref_mean = float(pre_lab[:, :, c][deep_bool].mean())
        ring_vals = bod_lab[:, :, c][ring]
        if ring_vals.size == 0:
            continue
        shift = (ref_mean - float(ring_vals.mean())) * strength
        tone_lab[:, :, c] = tone_lab[:, :, c] + shift * weight

    tone_rgb = cv2.cvtColor(tone_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    tone_rgb = np.clip(tone_rgb * 255.0, 0, 255)
    w3 = weight[:, :, np.newaxis]
    out = pre * (1.0 - w3) + tone_rgb * w3

    out_img = np_to_pil(out.astype(np.uint8))
    leak = assert_no_gen_interior_leak(
        pre_harm,
        out_img,
        gen_mask,
        interior_erode_px=interior_erode_px,
        epsilon=gen_lock_epsilon,
    )
    info.update(leak)
    if not leak["gen_interior_ok"]:
        info["harmonization_reason"] = "gen_interior_leak_fail_safe"
        info["harmonization_applied"] = False
        # Restore gen interior from pre_harm verbatim.
        core_bool = core.astype(bool)
        safe = pre.copy()
        post_arr = pil_to_rgb_np(out_img).astype(np.float32)
        safe[core_bool] = pre[core_bool]
        # Keep ring adjustments outside core only.
        ring_w = weight[:, :, np.newaxis]
        blended = pre * (1.0 - ring_w) + tone_rgb * ring_w
        safe[~core_bool] = blended[~core_bool]
        out_img = np_to_pil(np.clip(safe, 0, 255).astype(np.uint8))
        leak2 = assert_no_gen_interior_leak(
            pre_harm, out_img, gen_mask, interior_erode_px=interior_erode_px
        )
        info["gen_interior_max_diff"] = leak2["gen_interior_max_diff"]
        info["gen_interior_ok"] = leak2["gen_interior_ok"]
        return out_img, info

    info["harmonization_applied"] = True
    info["harmonization_reason"] = "ok"
    return out_img, info
