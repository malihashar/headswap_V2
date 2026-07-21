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
    backend = get_face_backend(cache_dir)
    h, w = rgb.shape[:2]

    if backend == "caffe":
        assert _FACE_NET is not None
        max_side = 640
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        sh, sw = small.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        _FACE_NET.setInput(blob)
        det = _FACE_NET.forward()
        best: FaceBox | None = None
        best_score = -1.0
        # Retry with a lower conf floor — glossy studio faces on black often score ~0.2–0.3.
        for min_conf in (conf_thresh, 0.15):
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
                area = (x1 - x0) * (y1 - y0)
                score = conf * area
                if score > best_score:
                    best_score = score
                    best = FaceBox(x0, y0, x1, y1, conf)
            if best is not None:
                return best

    if backend == "haar" or backend == "caffe":
        # Haar as secondary when caffe misses (common on black-bg cutouts).
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = _haar_cascade().detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
            )
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                return FaceBox(int(x), int(y), int(x + fw), int(y + fh), 0.5)
        except Exception:
            pass

    content = _face_box_from_cutout(rgb)
    if content is not None:
        return content

    # Geometric prior for portrait selfies: large upper-center face.
    # Used when OpenCV DNN/Haar are unavailable in the environment.
    face_h = int(h * 0.42)
    face_w = int(min(w * 0.55, face_h * 0.90))
    cx, cy = w // 2, int(h * 0.36)
    return FaceBox(
        max(0, cx - face_w // 2),
        max(0, cy - face_h // 2),
        min(w, cx + face_w // 2),
        min(h, cy + face_h // 2),
        0.2,
    )


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
    fill: tuple[int, int, int] = (0, 0, 0),
    div_by: int = 16,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Pad RGB image to a square (divisible by div_by). Returns (square, content_box).
    content_box = (ox, oy, w, h) of the original pixels inside the square.
    """
    im = im.convert("RGB")
    w, h = im.size
    # Round UP so the square never shrinks below the content size.
    side = max(w, h)
    if div_by > 1:
        side = max(div_by, ((side + div_by - 1) // div_by) * div_by)
    out = Image.new("RGB", (side, side), fill)
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
) -> Image.Image:
    """
    Approximate head+hair mask without SAM (portable fallback).
    Uses face box expanded toward hair/neck. Good enough for crop locality;
    replace with SAM/BiRefNet in production GPU stacks when available.
    """
    rgb = pil_to_rgb_np(body_pil)
    h, w = rgb.shape[:2]
    box = detect_best_face(rgb, cache_dir)
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
) -> tuple[Image.Image | None, dict]:
    """
    Warp source face onto destination face geometry (similarity transform).

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

    matrix, inliers = cv2.estimateAffinePartial2D(src_lm, dest_lm, method=cv2.LMEDS)
    if matrix is None:
        boxed, box_info = _box_paste_rgba(
            source_face,
            destination,
            cache_dir,
            core_min_alpha=core_min_alpha,
            feather_px=feather_px,
        )
        if boxed is not None:
            box_info["affine_skipped_reason"] = "estimateAffinePartial2D_failed"
            return boxed, box_info
        info["face_alignment_skip_reason"] = "estimateAffinePartial2D_failed"
        return None, info

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
