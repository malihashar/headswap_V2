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
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < conf_thresh:
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
        return best

    if backend == "haar":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = _haar_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return FaceBox(int(x), int(y), int(x + fw), int(y + fh), 0.5)

    # Geometric prior: upper-center ellipse face box (works for our synthetic set)
    cx, cy = w // 2, int(h * 0.28)
    hr = int(min(w, h) * 0.16)
    return FaceBox(cx - hr, cy - hr, cx + hr, cy + hr, 0.2)


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
