from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter


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
) -> tuple[Image.Image, dict[str, float]]:
    """
    If the face in ``edited`` is larger than in ``original_scene``, shrink the
    *edited* image about the face center (cv2 warp, border replicate).

    IMPORTANT: never paste onto ``original_scene`` — that re-exposes the old
    face around a shrunk swap (double-face / ghosting in group shots).
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
    if ratio <= float(max_height_ratio):
        info["ratio_after"] = ratio
        return edited, info

    shrink = (float(fo.height) * float(target_ratio)) / float(fe.height)
    shrink = float(min(1.0, max(0.55, shrink)))
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
        borderMode=cv2.BORDER_REPLICATE,
    )
    out = np_to_pil(warped)
    fe2 = detect_best_face(pil_to_rgb_np(out), cache_dir)
    if fe2 is not None and fo.height > 0:
        info["ratio_after"] = float(fe2.height) / float(fo.height)
    else:
        info["ratio_after"] = ratio * shrink
    return out, info


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

    When allow_prior=False (multi-person / validation), never invent a geometric
    face — return [] if OpenCV finds nothing.
    """
    backend = get_face_backend(cache_dir)
    h, w = rgb.shape[:2]
    faces: list[FaceBox] = []

    if backend == "caffe":
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

    # De-dupe overlaps, keep higher area*conf. IoU 0.4 catches near-duplicate
    # detector boxes that inflate multi-person face counts (e.g. 5 on 3 people).
    faces.sort(key=lambda b: b.width * b.height * max(0.05, b.conf), reverse=True)
    kept: list[FaceBox] = []
    for f in faces:
        dup = False
        for k in kept:
            ix0, iy0 = max(f.x0, k.x0), max(f.y0, k.y0)
            ix1, iy1 = min(f.x1, k.x1), min(f.y1, k.y1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = f.width * f.height + k.width * k.height - inter
            if union > 0 and inter / union > 0.4:
                dup = True
                break
            # Also treat strong containment as a duplicate (nested boxes).
            smaller = min(f.width * f.height, k.width * k.height)
            if smaller > 0 and inter / smaller > 0.7:
                dup = True
                break
        if not dup:
            kept.append(f)
    # Drop tiny false positives relative to the largest face (Haar/SSD noise).
    if kept:
        max_area = max(f.width * f.height for f in kept)
        kept = [f for f in kept if f.width * f.height >= 0.12 * max_area]
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
      index     — use ``index`` into largest-first list
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
        ordered = list(faces)  # already largest-first
    else:
        ordered = list(faces)

    idx = int(index) if pol == "index" else 0
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx], faces


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
) -> Image.Image:
    rgb = pil_to_rgb_np(face_pil)
    box = detect_best_face(rgb, cache_dir)
    if box is None:
        return face_pil.convert("RGB")
    # Cutout / content boxes are already face-sized — large pads pull in jersey/bg.
    if box.conf <= 0.40 and box.height >= 0.28 * face_pil.height:
        top = min(float(top), 0.18)
        bot = min(float(bot), 0.22)
        side = min(float(side), 0.18)
        include_shoulders = False
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
    min_face_margin_frac: float = 0.35,
    div_by: int = 16,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """
    Shrink a crop window so neighboring face centers fall outside it.

    This is the production locality mechanism for multi-person Krea2: keep the
    diffusion canvas identical to the single-person recipe, but do not let a
    second person enter the scene tensor. Prefer this over post-hoc neighbor
    pixel freezes, which can erase the swapped head on tight groups.
    """
    iw, ih = image_size
    x0, y0, x1, y1 = [int(v) for v in box]
    info: dict[str, float] = {
        "clamped": 0.0,
        "neighbors_excluded": 0.0,
        "crop_w_before": float(max(1, x1 - x0)),
        "crop_h_before": float(max(1, y1 - y0)),
    }
    if selected is None or not other_faces:
        info["crop_w_after"] = info["crop_w_before"]
        info["crop_h_after"] = info["crop_h_before"]
        return (x0, y0, x1, y1), info

    sx0, sy0, sx1, sy1 = selected.x0, selected.y0, selected.x1, selected.y1
    sw, sh = max(1, selected.width), max(1, selected.height)
    # Never shrink past a margin around the selected face.
    min_x0 = max(0, int(sx0 - min_face_margin_frac * sw))
    min_y0 = max(0, int(sy0 - min_face_margin_frac * sh))
    max_x1 = min(iw, int(sx1 + min_face_margin_frac * sw))
    max_y1 = min(ih, int(sy1 + min_face_margin_frac * sh))

    excluded = 0
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
        # Push the nearer crop edge inward past this neighbor center.
        dist_l = cx - x0
        dist_r = x1 - cx
        dist_t = cy - y0
        dist_b = y1 - cy
        side = min(
            (dist_l, "l"),
            (dist_r, "r"),
            (dist_t, "t"),
            (dist_b, "b"),
            key=lambda t: t[0],
        )[1]
        if side == "l":
            x0 = min(max(x0, int(cx + margin)), max_x1 - 8)
            x0 = max(x0, min_x0)
        elif side == "r":
            x1 = max(min(x1, int(cx - margin)), min_x0 + 8)
            x1 = min(x1, max_x1)
        elif side == "t":
            y0 = min(max(y0, int(cy + margin)), max_y1 - 8)
            y0 = max(y0, min_y0)
        else:
            y1 = max(min(y1, int(cy - margin)), min_y0 + 8)
            y1 = min(y1, max_y1)
        excluded += 1

    # Keep VAE-friendly sizes. Prefer shrinking (never grow back toward neighbors).
    cw, ch = x1 - x0, y1 - y0
    if cw < 16 or ch < 16:
        info["crop_w_after"] = info["crop_w_before"]
        info["crop_h_after"] = info["crop_h_before"]
        return box, info
    x1 = min(iw, x0 + evenify(cw, div_by))
    y1 = min(ih, y0 + evenify(ch, div_by))
    # If evenify clipped, shrink origin — do not expand outward.
    x0 = max(0, x1 - evenify(x1 - x0, div_by))
    y0 = max(0, y1 - evenify(y1 - y0, div_by))
    # Hard requirement: selected face must remain inside (no margin re-expansion).
    x0 = min(x0, sx0)
    y0 = min(y0, sy0)
    x1 = max(x1, sx1)
    y1 = max(y1, sy1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)

    info["clamped"] = 1.0 if excluded else 0.0
    info["neighbors_excluded"] = float(excluded)
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
) -> Image.Image:
    """
    L-mask over the identity face on the person/reference canvas.

    For Krea2 ``ref_boost_mask`` (last-ref attention boost only — not an edit
    freeze). Comfy MASK expects float later; this returns an 8-bit L image.
    """
    rgb = pil_to_rgb_np(person)
    h, w = rgb.shape[:2]
    box = detect_best_face(rgb, cache_dir)
    mask = np.zeros((h, w), dtype=np.uint8)
    if box is None:
        cx, cy = w // 2, int(h * 0.35)
        axes = (max(8, int(w * 0.22)), max(8, int(h * 0.28)))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    else:
        cx = (box.x0 + box.x1) // 2
        cy = (box.y0 + box.y1) // 2
        axes = (
            max(4, int(0.5 * expand * box.width)),
            max(4, int(0.55 * expand * box.height)),
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


def feathered_soft_composite(
    base: Image.Image,
    edit: Image.Image,
    mask: Image.Image,
    box: tuple[int, int, int, int],
    *,
    extra_blur_px: int = 8,
) -> Image.Image:
    """Soft stitch with an extra blur on the alpha for less obvious jaw seams."""
    if extra_blur_px > 0:
        m = mask.convert("L")
        k = max(3, int(extra_blur_px) * 2 + 1)
        arr = np.asarray(m)
        arr = cv2.GaussianBlur(arr, (k, k), 0)
        mask = Image.fromarray(arr)
    return soft_composite(base, edit, mask, box)

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
    rgb: np.ndarray, cache_dir
) -> tuple[np.ndarray | None, str, str | None]:
    """
    Return 5 face landmarks as float32 (5, 2) in image XY order.

    Preference order:
      1. InsightFace (buffalo_l / default) — community Align→Paste path
      2. OpenCV box corners derived from detect_best_face — weak fallback

    Returns (landmarks_or_None, backend_name, skip_reason_or_None).
    """
    # InsightFace (optional GPU extra)
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:
        insight_err = f"insightface_import_failed:{exc}"
    else:
        insight_err = None
        global _INSIGHTFACE_APP
        try:
            if _INSIGHTFACE_APP is None:
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                _INSIGHTFACE_APP = app
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _INSIGHTFACE_APP.get(bgr)
            if not faces:
                return None, "insightface", "insightface_no_face_detected"
            face = max(
                faces,
                key=lambda f: float(
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                ),
            )
            kps = np.asarray(face.kps, dtype=np.float32)
            if kps.shape != (5, 2):
                return None, "insightface", f"insightface_bad_kps_shape:{kps.shape}"
            return kps, "insightface", None
        except Exception as exc:
            insight_err = f"insightface_runtime_failed:{exc}"

    # OpenCV box → synthetic 5-point layout (eyes / nose / mouth corners)
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

    dest_lm, dest_backend, dest_note = get_face_landmarks5(dest_rgb, cache_dir)
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
    cx, cy = float(xs.mean()), float(ys.mean() - 0.05 * (ys.max() - ys.min()))
    span_x = float(max(48.0, (xs.max() - xs.min()) * float(ellipse_scale_x)))
    span_y = float(max(60.0, (ys.max() - ys.min()) * float(ellipse_scale_y)))
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
) -> tuple[Image.Image, dict]:
    """
    Force ``generated`` face orientation onto ``destination`` landmarks.

    After generative refine, head yaw / eye gaze often drift toward a frontal
    identity. Re-aligning locks looking direction to the original photo.
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
    Shift pasted-face LAB toward destination skin under the RGBA alpha.

    Done *before* Kontext refine so the model blends lighting instead of
    fighting a strong color mismatch (which often reverts toward the original face).
    """
    if aligned_rgba is None or strength <= 0:
        return aligned_rgba
    if aligned_rgba.size != destination.size:
        aligned_rgba = aligned_rgba.resize(destination.size, Image.Resampling.LANCZOS)
    rgba = np.asarray(aligned_rgba.convert("RGBA")).astype(np.float32)
    bod = pil_to_rgb_np(destination).astype(np.float32)
    alpha = rgba[:, :, 3] / 255.0
    if (alpha > 0.4).sum() < 50:
        return aligned_rgba
    # Destination ring just outside the paste for target skin stats.
    core = (alpha > 0.45).astype(np.uint8)
    ring = cv2.dilate(core, np.ones((25, 25), np.uint8)) - core
    if ring.sum() < 40:
        return aligned_rgba
    face_lab = cv2.cvtColor(rgba[:, :, :3] / 255.0, cv2.COLOR_RGB2LAB)
    bod_lab = cv2.cvtColor(bod / 255.0, cv2.COLOR_RGB2LAB)
    for c in range(3):
        src_vals = face_lab[:, :, c][alpha > 0.5]
        tgt_vals = bod_lab[:, :, c][ring > 0]
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
) -> tuple[Image.Image, dict]:
    """Alpha-composite aligned RGBA face onto destination RGB. Returns (RGB, info)."""
    info = {"composite_paste": False, "composite_paste_skip_reason": None}
    if aligned_rgba is None:
        info["composite_paste_skip_reason"] = "aligned_rgba_none"
        return destination.convert("RGB"), info
    if aligned_rgba.size != destination.size:
        aligned_rgba = aligned_rgba.resize(destination.size, Image.Resampling.LANCZOS)
    base = destination.convert("RGBA")
    out = Image.alpha_composite(base, aligned_rgba.convert("RGBA"))
    info["composite_paste"] = True
    return out.convert("RGB"), info


def lab_histogram_match_face(
    result: Image.Image, body: Image.Image, mask: Image.Image, strength: float = 0.35
) -> Image.Image:
    """Mild LAB mean match inside mask to reduce neck/skin discontinuity."""
    if strength <= 0:
        return result
    res = pil_to_rgb_np(result).astype(np.float32)
    bod = pil_to_rgb_np(body).astype(np.float32)
    m = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    if m.shape[:2] != res.shape[:2]:
        m = cv2.resize(m, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Ring just outside mask for target skin stats
    ring = cv2.dilate((m > 0.4).astype(np.uint8), np.ones((21, 21), np.uint8)) - (m > 0.4).astype(
        np.uint8
    )
    if ring.sum() < 50 or (m > 0.5).sum() < 50:
        return result
    res_lab = cv2.cvtColor(res / 255.0, cv2.COLOR_RGB2LAB)
    bod_lab = cv2.cvtColor(bod / 255.0, cv2.COLOR_RGB2LAB)
    for c in range(3):
        src_vals = res_lab[:, :, c][m > 0.5]
        tgt_vals = bod_lab[:, :, c][ring > 0]
        if src_vals.size == 0 or tgt_vals.size == 0:
            continue
        shift = float(tgt_vals.mean() - src_vals.mean()) * strength
        res_lab[:, :, c] = res_lab[:, :, c] + shift * m
    out = cv2.cvtColor(res_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np_to_pil(np.clip(out * 255.0, 0, 255))
