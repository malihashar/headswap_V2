#!/usr/bin/env python3
"""Production-demo helpers for the Krea2 Colab notebook.

Keeps notebook cells thin: uploads, face checks, preflight, clean errors,
progress lines, and a short run summary. Not used by the API/service path.
"""
from __future__ import annotations

import io
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


REQUIRED_KREA2_WEIGHTS = (
    ("diffusion_models", "krea2_turbo_fp8_scaled.safetensors"),
    ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
    ("vae", "qwen_image_vae.safetensors"),
    ("loras", "krea2_identity_edit_v1_2_r64.safetensors"),
)

STABLE_RESULT_NAME = "HEADSWAP_RESULT.png"
DEFAULT_OUTPUT_DIR = Path("/content/headswap_outputs")


class DemoError(Exception):
    """User-facing failure — notebook should show the message, not a traceback."""


def progress(msg: str) -> None:
    print(f"→ {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"✓ {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"✗ {msg}", flush=True)


def load_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise DemoError(
            f"Could not read image ({exc}). Use JPG, PNG, or WEBP."
        ) from None


def save_upload(data: bytes, dest: Path) -> Image.Image:
    """Decode upload bytes (incl. webp) and save as RGB PNG."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = load_image_bytes(data)
    im.save(dest, format="PNG")
    return im


def require_path(path: Path | str, label: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise DemoError(f"{label} not found: {p}. Upload or run the earlier cells first.")
    return p


def require_face(image: Image.Image, cache_dir: Path | str, label: str) -> dict[str, Any]:
    """Fail early if OpenCV cannot find a face (same detector the pipeline uses)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from headswap.preprocess import detect_best_face, pil_to_rgb_np

    box = detect_best_face(pil_to_rgb_np(image), Path(cache_dir))
    if box is None:
        raise DemoError(
            f"No face detected in the {label} image. "
            "Use a clear, front-facing photo with a visible face."
        )
    return {
        "box": [box.x0, box.y0, box.x1, box.y1],
        "confidence": round(float(box.conf), 3),
        "size": list(image.size),
    }


def required_model_paths(store: Path | str | None = None) -> list[Path]:
    root = Path(store or os.environ.get("HEADSWAP_MODEL_STORE", "/content/models"))
    return [root / sub / name for sub, name in REQUIRED_KREA2_WEIGHTS]


def verify_models(store: Path | str | None = None) -> list[Path]:
    paths = required_model_paths(store)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise DemoError(
            f"Missing model file(s): {names}. Re-run the Download models cell."
        )
    return paths


def verify_gpu() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise DemoError(f"PyTorch is not available ({exc}).") from None
    if not torch.cuda.is_available():
        raise DemoError(
            "No GPU detected. Runtime → Change runtime type → GPU "
            "(prefer A100), then Runtime → Run all."
        )
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        ),
        "torch": torch.__version__,
    }


def environment_summary(
    *,
    paths: dict[str, Path] | None = None,
    seed: int | None = None,
    stitch: bool | None = None,
    debug: bool | None = None,
) -> dict[str, Any]:
    gpu = verify_gpu()
    store = Path(
        (paths or {}).get("model_store")
        or os.environ.get("HEADSWAP_MODEL_STORE", "/content/models")
    )
    models = required_model_paths(store)
    present = sum(1 for p in models if p.is_file())
    info = {
        "gpu": gpu["name"],
        "vram_gb": gpu["vram_gb"],
        "torch": gpu["torch"],
        "model_store": str(store),
        "models_present": f"{present}/{len(models)}",
        "seed": seed,
        "stitch": stitch,
        "debug": debug,
    }
    print("Environment")
    for k, v in info.items():
        if v is None:
            continue
        print(f"  {k:<16} {v}")
    return info


def preflight(
    *,
    body_path: Path | str,
    face_path: Path | str,
    cache_dir: Path | str,
    model_store: Path | str | None = None,
) -> dict[str, Any]:
    """Validate GPU, weights, and both images before inference."""
    progress("Preflight checks…")
    gpu = verify_gpu()
    models = verify_models(model_store)
    body_p = require_path(body_path, "Body image")
    face_p = require_path(face_path, "Face image")
    body = Image.open(body_p).convert("RGB")
    face = Image.open(face_p).convert("RGB")
    body_face = require_face(body, cache_dir, "body")
    face_face = require_face(face, cache_dir, "face")
    ok(
        f"Preflight passed — GPU={gpu['name']}, "
        f"models={len(models)}, body face conf={body_face['confidence']}, "
        f"face conf={face_face['confidence']}"
    )
    return {
        "gpu": gpu,
        "models": [str(p) for p in models],
        "body": body_face,
        "face": face_face,
        "body_path": str(body_p),
        "face_path": str(face_p),
    }


def apply_user_knobs(
    cfg: dict[str, Any],
    *,
    seed: int,
    prompt: str | None,
    steps: int | None,
    cfg_scale: float | None,
    output_long_side: int | None,
    stitch: bool,
    debug: bool,
) -> dict[str, Any]:
    """Map the small user-facing knob set onto pipeline config."""
    out = dict(cfg)
    out["seed"] = int(seed)
    out["mask_crop_stitch"] = bool(stitch)
    out["save_debug"] = bool(debug)
    out["verbose"] = bool(debug)
    if prompt is not None and str(prompt).strip():
        out["prompt"] = str(prompt).strip()
    if steps is not None:
        out["steps"] = int(steps)
    if cfg_scale is not None:
        out["cfg"] = float(cfg_scale)
    if output_long_side is not None:
        side = int(output_long_side)
        # Final canvas / body size; edit crop stays ≤ this.
        out["max_body_dim"] = side
        out["max_dim"] = side
        out["crop_long_side"] = min(side, int(out.get("crop_long_side", 768) or 768))
        if not stitch:
            out["max_dim"] = side
    return out


def write_stable_result(
    result_png: Path | str,
    output_dir: Path | str | None = None,
) -> Path:
    src = Path(result_png)
    if not src.is_file():
        raise DemoError(f"Result image missing: {src}")
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stable = out_dir / STABLE_RESULT_NAME
    stable.write_bytes(src.read_bytes())
    return stable


def print_run_summary(
    *,
    success: bool,
    total_s: float,
    sampling_s: float | None,
    steps: int | None,
    seed: int | None,
    gpu: str | None,
    resolution: tuple[int, int] | list[int] | None,
    output_path: Path | str | None,
    cache_hit: bool | None = None,
    error: str | None = None,
) -> None:
    print()
    if success:
        ok("Head swap complete")
    else:
        fail(error or "Head swap failed")
    print("Run summary")
    print(f"  total_runtime_s   {total_s:.2f}")
    if sampling_s is not None:
        print(f"  sampling_s        {sampling_s:.2f}")
    if steps is not None:
        print(f"  steps             {steps}")
    if seed is not None:
        print(f"  seed              {seed}")
    if gpu:
        print(f"  gpu               {gpu}")
    if resolution is not None:
        print(f"  output_resolution {list(resolution)}")
    if cache_hit is not None:
        print(f"  model_cache_hit   {cache_hit}")
    if output_path:
        print(f"  output_path       {output_path}")


@contextmanager
def quiet_logs(*, debug: bool) -> Iterator[None]:
    """Hide pipeline/Comfy chatter unless DEBUG is on. Progress lines use progress()."""
    if debug:
        yield
        return
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


def show_side_by_side(
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    *,
    height: int = 320,
) -> Image.Image:
    imgs = [body.convert("RGB"), face.convert("RGB"), result.convert("RGB")]
    resized = []
    for im in imgs:
        w = max(1, int(im.width * (height / im.height)))
        resized.append(im.resize((w, height)))
    gap = 16
    canvas_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (canvas_w, height), (255, 255, 255))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def elapsed(t0: float) -> float:
    return time.perf_counter() - t0
