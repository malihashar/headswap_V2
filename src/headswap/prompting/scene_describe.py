"""Automatic scene description + identity-edit instruction builder.

Produces a natural-language prompt from the body image and selected face
geometry. Uses InsightFace attributes when available, plus photometric /
composition heuristics. Optional transformers VLM captioning if configured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from headswap.preprocess import FaceBox, detect_faces, pil_to_rgb_np, select_face_box


@dataclass
class SceneDescription:
    """Structured scene facts used to assemble the edit prompt."""

    n_faces: int
    selected_index: int
    selected_role: str
    lighting: str
    composition: str
    camera: str
    clothing_guess: str
    expression_guess: str
    pose_guess: str
    background_guess: str
    hair_guess: str
    other_people: str
    vlm_caption: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _brightness_stats(rgb: np.ndarray) -> tuple[float, float, float]:
    """Return (mean_luma, warm_bias, saturation)."""
    arr = rgb.astype(np.float32)
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    mean_luma = float(luma.mean())
    r, g, b = float(arr[..., 0].mean()), float(arr[..., 1].mean()), float(arr[..., 2].mean())
    warm = (r - b) / max(1.0, (r + g + b) / 3.0)
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = float(((mx - mn) / np.maximum(mx, 1.0)).mean())
    return mean_luma, warm, sat


def _describe_lighting(mean_luma: float, warm: float, sat: float) -> str:
    if mean_luma < 55:
        base = "low-light nighttime scene with soft frontal fill and visible grain"
    elif mean_luma < 95:
        base = "dim indoor or evening lighting"
    elif mean_luma < 150:
        base = "natural ambient daylight"
    else:
        base = "bright well-lit scene"
    if warm > 0.12:
        base += ", warm color cast"
    elif warm < -0.08:
        base += ", cool color cast"
    if sat < 0.12 and mean_luma < 80:
        base += ", muted desaturated tones"
    return base


def _describe_composition(w: int, h: int, faces: list[FaceBox], selected: FaceBox) -> str:
    aspect = w / max(1, h)
    if aspect > 1.25:
        framing = "wide horizontal group composition"
    elif aspect < 0.85:
        framing = "vertical portrait framing"
    else:
        framing = "balanced near-square framing"

    if len(faces) >= 3:
        people = f"{len(faces)} people arranged side-by-side"
    elif len(faces) == 2:
        people = "two people sharing the frame"
    else:
        people = "a single subject"

    cx = (selected.x0 + selected.x1) / 2.0 / w
    if cx < 0.33:
        pos = "selected subject on the left"
    elif cx > 0.66:
        pos = "selected subject on the right"
    else:
        pos = "selected subject near the center"
    return f"{framing}; {people}; {pos}"


def _describe_camera(h: int, selected: FaceBox) -> str:
    face_frac = selected.height / max(1, h)
    cy = (selected.y0 + selected.y1) / 2.0 / max(1, h)
    if face_frac > 0.35:
        dist = "close-up head-and-shoulders distance"
    elif face_frac > 0.18:
        dist = "medium portrait distance"
    else:
        dist = "environmental distance with smaller faces"
    if cy < 0.40:
        angle = "slightly low camera angle"
    elif cy > 0.62:
        angle = "slightly high camera angle"
    else:
        angle = "eye-level camera"
    return f"{angle}, {dist}"


def _role_for_face(faces: list[FaceBox], selected: FaceBox) -> str:
    if len(faces) <= 1:
        return "the only person"
    ordered = sorted(faces, key=lambda b: (b.x0 + b.x1) / 2.0)
    try:
        idx = next(i for i, f in enumerate(ordered) if f is selected or (
            f.x0 == selected.x0 and f.y0 == selected.y0 and f.x1 == selected.x1 and f.y1 == selected.y1
        ))
    except StopIteration:
        # Match by IoU-ish center distance.
        scx = (selected.x0 + selected.x1) / 2.0
        idx = int(np.argmin([abs(((f.x0 + f.x1) / 2.0) - scx) for f in ordered]))
    if len(ordered) == 2:
        return "the person on the left" if idx == 0 else "the person on the right"
    if idx == 0:
        return "the leftmost person"
    if idx == len(ordered) - 1:
        return "the rightmost person"
    if len(ordered) == 3 and idx == 1:
        return "the person in the center"
    return f"person {idx + 1} from the left"


def _clothing_guess(rgb: np.ndarray, face: FaceBox) -> str:
    """Sample torso band below the face for a coarse clothing color/tone."""
    h, w = rgb.shape[:2]
    fh = max(1, face.height)
    y0 = min(h - 1, face.y1 + int(0.05 * fh))
    y1 = min(h, face.y1 + int(1.35 * fh))
    x0 = max(0, face.x0 - int(0.15 * face.width))
    x1 = min(w, face.x1 + int(0.15 * face.width))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        return "clothing matching the original outfit"
    patch = rgb[y0:y1, x0:x1].astype(np.float32)
    mean = patch.reshape(-1, 3).mean(axis=0)
    luma = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
    r, g, b = mean
    if luma < 55:
        tone = "dark"
    elif luma > 170:
        tone = "light"
    else:
        tone = "medium-toned"
    if abs(r - g) < 12 and abs(g - b) < 12:
        color = "neutral gray/black/white"
    elif r > g + 15 and r > b + 15:
        color = "warm reddish/brown"
    elif b > r + 15 and b > g + 10:
        color = "cool blue"
    elif g > r + 10 and g > b + 10:
        color = "greenish"
    else:
        color = "mixed"
    return f"{tone} {color} clothing on the torso"


def _expression_guess(rgb: np.ndarray, face: FaceBox, cache_dir) -> str:
    """Mouth openness / smile proxy from landmarks when InsightFace is present."""
    try:
        from headswap.preprocess import ensure_insightface_app, get_face_landmarks5

        app = ensure_insightface_app(cache_dir)
        lm, _backend, _skip = get_face_landmarks5(rgb, cache_dir, prefer_box=face)
        if lm is None:
            return "the original facial expression"
        # landmarks5: left_eye, right_eye, nose, left_mouth, right_mouth
        mouth_w = float(np.linalg.norm(lm[3] - lm[4]))
        eye_w = float(np.linalg.norm(lm[0] - lm[1]))
        ratio = mouth_w / max(eye_w, 1e-3)
        # Vertical mouth openness if we can find insightface face with more kps.
        open_hint = ""
        if app is not None:
            import cv2

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            best = None
            best_iou = 0.0
            for f in faces or []:
                x0, y0, x1, y1 = [int(v) for v in f.bbox]
                ix0, iy0 = max(x0, face.x0), max(y0, face.y0)
                ix1, iy1 = min(x1, face.x1), min(y1, face.y1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = face.width * face.height + (x1 - x0) * (y1 - y0) - inter
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou, best = iou, f
            if best is not None and getattr(best, "kps", None) is not None:
                kps = np.asarray(best.kps, dtype=np.float32)
                if kps.shape[0] >= 5:
                    # Approximate: nose-to-mouth vertical vs face height
                    nose_y = float(kps[2, 1])
                    mouth_y = float(0.5 * (kps[3, 1] + kps[4, 1]))
                    open_ratio = abs(mouth_y - nose_y) / max(1.0, float(face.height))
                    if open_ratio > 0.42:
                        open_hint = ", mouth slightly open"
                    else:
                        open_hint = ", mouth closed"
        if ratio > 1.05:
            return f"a mild smile or wider mouth{open_hint}"
        return f"a neutral closed-mouth expression{open_hint}"
    except Exception:
        return "the original facial expression"


def _pose_guess(face: FaceBox, w: int, h: int) -> str:
    cx = (face.x0 + face.x1) / 2.0 / max(1, w)
    # Without full 3D, assume mostly frontal upright for group photos.
    lean = ""
    if cx < 0.35:
        lean = ", body angled slightly toward frame center"
    elif cx > 0.65:
        lean = ", body angled slightly toward frame center"
    return f"upright head facing the camera{lean}"


def _background_guess(rgb: np.ndarray, faces: list[FaceBox]) -> str:
    h, w = rgb.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    for f in faces:
        y0 = max(0, f.y0 - f.height)
        y1 = min(h, f.y1 + int(1.6 * f.height))
        x0 = max(0, f.x0 - int(0.3 * f.width))
        x1 = min(w, f.x1 + int(0.3 * f.width))
        mask[y0:y1, x0:x1] = False
    if mask.sum() < 500:
        return "the original background"
    bg = rgb[mask].astype(np.float32)
    mean_luma = float(
        (0.2126 * bg[:, 0] + 0.7152 * bg[:, 1] + 0.0722 * bg[:, 2]).mean()
    )
    # Upper third often carries sky / distant scene.
    top = rgb[: max(1, h // 3)]
    top_luma = float(
        (
            0.2126 * top[..., 0] + 0.7152 * top[..., 1] + 0.0722 * top[..., 2]
        ).mean()
    )
    if mean_luma < 50 and top_luma < 70:
        return "dark nighttime background, possibly city lights or night sky"
    if mean_luma < 50:
        return "dark indoor or nighttime background"
    if top_luma > mean_luma + 25:
        return "brighter upper background / sky region behind the subjects"
    return "the original environmental background"


def _hair_guess(rgb: np.ndarray, face: FaceBox) -> str:
    h, w = rgb.shape[:2]
    y0 = max(0, face.y0 - int(0.55 * face.height))
    y1 = max(y0 + 2, face.y0 + int(0.15 * face.height))
    x0 = max(0, face.x0 - int(0.1 * face.width))
    x1 = min(w, face.x1 + int(0.1 * face.width))
    patch = rgb[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 10:
        return "the original hairstyle when possible"
    luma = float(
        (0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]).mean()
    )
    # Cap / accessory cue: very dark flat band above forehead.
    if luma < 40:
        return "dark hair or a dark hat/cap above the forehead — preserve that silhouette"
    if luma > 160:
        return "light hair or bright headwear — preserve that silhouette"
    return "the original hairstyle and head silhouette when possible"


def _other_people(n: int, role: str) -> str:
    if n <= 1:
        return "no other people in the frame"
    return (
        f"{n - 1} other person(s) remain in the frame beside {role}; "
        "their faces, hair, clothing, and poses must stay identical"
    )


def _try_vlm_caption(body: Image.Image, cfg: dict[str, Any]) -> str | None:
    """Optional transformers caption. Disabled unless cfg enables it."""
    if not bool(cfg.get("use_vlm_caption", False)):
        return None
    model_id = str(cfg.get("vlm_caption_model", "Salesforce/blip-image-captioning-base"))
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        cap = pipeline("image-to-text", model=model_id, device=device)
        out = cap(body.convert("RGB"))
        if isinstance(out, list) and out:
            text = out[0].get("generated_text") or out[0].get("caption")
            if text:
                return str(text).strip()
    except Exception:
        return None
    return None


def describe_scene(
    body: Image.Image,
    cache_dir: Path | str,
    *,
    face_index: int = 0,
    face_policy: str = "largest",
    cfg: dict[str, Any] | None = None,
) -> tuple[SceneDescription, FaceBox | None, list[FaceBox]]:
    """Analyze body image and return structured scene facts + face boxes."""
    cfg = cfg or {}
    rgb = pil_to_rgb_np(body)
    h, w = rgb.shape[:2]
    selected, faces = select_face_box(
        rgb,
        cache_dir,
        index=int(face_index),
        policy=str(face_policy or "largest"),
    )
    mean_luma, warm, sat = _brightness_stats(rgb)
    if selected is None:
        desc = SceneDescription(
            n_faces=0,
            selected_index=0,
            selected_role="the main subject",
            lighting=_describe_lighting(mean_luma, warm, sat),
            composition="full-frame photograph",
            camera="eye-level camera",
            clothing_guess="original clothing",
            expression_guess="original facial expression",
            pose_guess="original pose",
            background_guess=_background_guess(rgb, []),
            hair_guess="original hairstyle when possible",
            other_people="preserve everyone else unchanged",
            vlm_caption=_try_vlm_caption(body, cfg),
        )
        return desc, None, faces

    role = _role_for_face(faces, selected)
    # index among left-to-right for meta
    ordered = sorted(faces, key=lambda b: (b.x0 + b.x1) / 2.0)
    try:
        sel_i = next(
            i
            for i, f in enumerate(ordered)
            if f.x0 == selected.x0 and f.y0 == selected.y0
        )
    except StopIteration:
        sel_i = 0

    desc = SceneDescription(
        n_faces=len(faces),
        selected_index=sel_i,
        selected_role=role,
        lighting=_describe_lighting(mean_luma, warm, sat),
        composition=_describe_composition(w, h, faces, selected),
        camera=_describe_camera(h, selected),
        clothing_guess=_clothing_guess(rgb, selected),
        expression_guess=_expression_guess(rgb, selected, cache_dir),
        pose_guess=_pose_guess(selected, w, h),
        background_guess=_background_guess(rgb, faces),
        hair_guess=_hair_guess(rgb, selected),
        other_people=_other_people(len(faces), role),
        vlm_caption=_try_vlm_caption(body, cfg),
        extras={
            "image_size": [w, h],
            "selected_box": [selected.x0, selected.y0, selected.x1, selected.y1],
            "mean_luma": round(mean_luma, 2),
            "warm_bias": round(warm, 3),
        },
    )
    return desc, selected, faces


def build_identity_edit_prompt(
    desc: SceneDescription,
    *,
    instruction_suffix: str | None = None,
) -> str:
    """Assemble a full-image head/face/hair *replacement* prompt.

    Matches the trained Identity Edit swap vocabulary (replace face/hair/head
    from image 2) and the production ``krea2_identity_edit.yaml`` wording.
    Intentionally does **not** describe the target person's hair/beard/jaw as
    things to preserve — that locks scene identity geometry.
    """
    scene_bits = [
        f"Photograph description: {desc.composition}.",
        f"Lighting: {desc.lighting}.",
        f"Camera: {desc.camera}.",
        f"Background: {desc.background_guess}.",
        f"Edit target: {desc.selected_role} — keep {desc.pose_guess} and "
        f"{desc.expression_guess}; keep body clothing ({desc.clothing_guess}).",
        f"Other people: {desc.other_people}.",
    ]
    if desc.vlm_caption:
        scene_bits.insert(0, f"Scene caption: {desc.vlm_caption}.")

    suffix = instruction_suffix or (
        f"Replace the face, hair, and head of {desc.selected_role} in image 1 "
        "with the person from image 2. Preserve the exact facial identity of "
        "image 2 — bone structure, jawline, facial hair / beard or clean-shaven "
        "look, eyebrows, eyes, nose, mouth, skin, hairline, hairstyle, hair "
        "length, and hair color. Completely remove the original face, facial "
        "hair, and hair of that person in image 1; do not blend them with image 2. "
        "CRITICAL: copy the facial expression from image 1 only — smile / no-smile, "
        "mouth shape, eye gaze, and micro-expressions must stay from image 1, "
        "never from image 2. Keep body, clothing, pose, head rotation, head size, "
        "camera angle, lighting, and background from image 1. If other people are "
        "visible, leave them completely unchanged. Photorealistic, natural skin, "
        "lighting matched to image 1."
    )
    target_line = (
        "Image 1 is the full scene. Image 2 is the identity reference. "
        f"Perform a head/face/hair swap on {desc.selected_role} only."
    )
    return " ".join(scene_bits + [target_line, suffix]).strip()
