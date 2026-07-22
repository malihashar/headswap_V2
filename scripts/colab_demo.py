#!/usr/bin/env python3
"""Production-demo helpers for the Krea2 Colab notebook.

Keeps notebook cells thin: version pinning, uploads, face checks, model
integrity, timestamped output packages, quality gates, and clean errors.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "scripts" / "models.json"

REQUIRED_KREA2_WEIGHTS = (
    ("diffusion_models", "krea2_turbo_fp8_scaled.safetensors"),
    ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
    ("vae", "qwen_image_vae.safetensors"),
    ("loras", "krea2_identity_edit_v1_2_r64.safetensors"),
)

STABLE_RESULT_NAME = "HEADSWAP_RESULT.png"
DEFAULT_OUTPUT_DIR = Path("/content/headswap_outputs")
MIN_SIDE_WARN = 256
MAX_SIDE_WARN = 4096
DOWNLOADER_HINT = "python scripts/download_krea2.py"


class DemoError(Exception):
    """User-facing failure — notebook should show the message, not a traceback."""


def progress(msg: str) -> None:
    print(f"→ {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"✓ {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"⚠ {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"✗ {msg}", flush=True)


def _git(cwd: Path | str, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def repo_git_info(repo: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo or ROOT)
    commit = _git(root, "rev-parse", "HEAD")
    short = _git(root, "rev-parse", "--short", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    return {
        "repo": str(root),
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(dirty),
    }


def ensure_pinned_commit(repo: Path | str, pinned: str | None) -> dict[str, Any]:
    """If pinned is set, checkout that commit (detached OK). Always return git info."""
    root = Path(repo)
    info = repo_git_info(root)
    pin = (pinned or "").strip() or None
    if pin and info.get("commit") and not (
        info["commit"].startswith(pin) or (info.get("commit_short") or "").startswith(pin)
    ):
        progress(f"Checking out pinned commit {pin}…")
        try:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--depth", "1", "origin", pin],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "checkout", "--detach", pin],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            raise DemoError(
                f"Could not checkout pinned commit {pin}: {exc}. "
                "Unset PINNED_COMMIT or fix network/git access."
            ) from None
        info = repo_git_info(root)
    info["pinned"] = pin
    return info


def comfy_versions(comfyui: Path | str | None = None) -> dict[str, Any]:
    comfy = Path(comfyui or os.environ.get("COMFYUI_PATH", "/content/ComfyUI"))
    nodes = comfy / "custom_nodes" / "comfyui-krea2edit"
    info = {
        "comfyui_path": str(comfy),
        "comfyui_commit": _git(comfy, "rev-parse", "--short", "HEAD") if comfy.is_dir() else None,
        "comfyui_exists": comfy.is_dir(),
        "krea2edit_path": str(nodes),
        "krea2edit_commit": _git(nodes, "rev-parse", "--short", "HEAD") if nodes.is_dir() else None,
        "krea2edit_exists": nodes.is_dir(),
    }
    return info


def torch_versions() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise DemoError(f"PyTorch is not available ({exc}).") from None
    cuda = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "gpu": None,
        "vram_gb": None,
    }
    if cuda:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    return info


def collect_versions(
    *,
    repo: Path | str | None = None,
    comfyui: Path | str | None = None,
    pinned_commit: str | None = None,
) -> dict[str, Any]:
    git = ensure_pinned_commit(repo or ROOT, pinned_commit) if pinned_commit else repo_git_info(repo)
    torch_info = torch_versions()
    comfy = comfy_versions(comfyui)
    versions = {
        "git": git,
        "torch": torch_info,
        "comfy": comfy,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    return versions


def print_versions(versions: dict[str, Any]) -> None:
    git = versions.get("git") or {}
    torch_info = versions.get("torch") or {}
    comfy = versions.get("comfy") or {}
    print("Versions")
    print(
        f"  repo_commit     {git.get('commit_short')} ({git.get('branch')})"
        f"{'  DIRTY' if git.get('dirty') else ''}"
    )
    if git.get("pinned"):
        print(f"  pinned_commit   {git.get('pinned')}")
    print(f"  torch           {torch_info.get('torch')}")
    print(f"  cuda            {torch_info.get('cuda_runtime')}  available={torch_info.get('cuda_available')}")
    print(f"  gpu             {torch_info.get('gpu')} ({torch_info.get('vram_gb')} GiB)")
    print(f"  comfyui         {comfy.get('comfyui_commit') or ('missing' if not comfy.get('comfyui_exists') else 'unknown')}")
    print(
        f"  krea2edit       {comfy.get('krea2edit_commit') or ('missing' if not comfy.get('krea2edit_exists') else 'unknown')}"
    )


def load_manifest_sizes() -> dict[str, int]:
    if not MANIFEST_PATH.is_file():
        return {}
    data = json.loads(MANIFEST_PATH.read_text())
    return {name: int(meta["size"]) for name, meta in data.items() if "size" in meta}


def required_model_paths(store: Path | str | None = None) -> list[Path]:
    root = Path(store or os.environ.get("HEADSWAP_MODEL_STORE", "/content/models"))
    return [root / sub / name for sub, name in REQUIRED_KREA2_WEIGHTS]


def verify_models(
    store: Path | str | None = None,
    *,
    check_sizes: bool = True,
) -> list[dict[str, Any]]:
    """Verify required Krea2 weights exist and match models.json sizes."""
    sizes = load_manifest_sizes() if check_sizes else {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    bad_size: list[str] = []
    for path in required_model_paths(store):
        row: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "expected_size": sizes.get(path.name),
            "size_ok": None,
        }
        if not row["exists"]:
            missing.append(path.name)
        elif row["expected_size"] is not None:
            row["size_ok"] = int(row["size"]) == int(row["expected_size"])
            if not row["size_ok"]:
                bad_size.append(
                    f"{path.name} (got {row['size']}, expected {row['expected_size']})"
                )
        rows.append(row)

    if missing:
        raise DemoError(
            f"Missing model file(s): {', '.join(missing)}. "
            f"Re-run: {DOWNLOADER_HINT}"
        )
    if bad_size:
        raise DemoError(
            "Incomplete or corrupt model download(s): "
            + "; ".join(bad_size)
            + f". Delete the bad file(s) and re-run: {DOWNLOADER_HINT}"
        )
    return rows


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


def check_image_geometry(image: Image.Image, label: str) -> list[str]:
    """Return warning strings for extreme sizes (does not fail the run)."""
    w, h = image.size
    warnings: list[str] = []
    if min(w, h) < MIN_SIDE_WARN:
        warnings.append(
            f"{label} is very small ({w}×{h}). Quality may be poor; prefer ≥{MIN_SIDE_WARN}px on the short side."
        )
    if max(w, h) > MAX_SIDE_WARN:
        warnings.append(
            f"{label} is very large ({w}×{h}). It will be downscaled; prefer ≤{MAX_SIDE_WARN}px on the long side."
        )
    return warnings


def _detect_faces_strict(image: Image.Image, cache_dir: Path | str) -> list[dict[str, Any]]:
    """List faces from OpenCV only — no geometric prior fallback."""
    sys.path.insert(0, str(ROOT / "src"))
    from headswap.preprocess import (  # noqa: WPS433
        _FACE_NET,
        _haar_cascade,
        get_face_backend,
        pil_to_rgb_np,
    )
    import cv2
    import numpy as np

    rgb = pil_to_rgb_np(image)
    h, w = rgb.shape[:2]
    backend = get_face_backend(cache_dir)
    faces: list[dict[str, Any]] = []

    if backend == "caffe" and _FACE_NET is not None:
        max_side = 640
        scale = min(1.0, max_side / float(max(h, w)))
        small = (
            cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else rgb
        )
        sh, sw = small.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        _FACE_NET.setInput(blob)
        det = _FACE_NET.forward()
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < 0.30:
                continue
            x0 = int(det[0, 0, i, 3] * sw)
            y0 = int(det[0, 0, i, 4] * sh)
            x1 = int(det[0, 0, i, 5] * sw)
            y1 = int(det[0, 0, i, 6] * sh)
            if scale < 1.0:
                x0, x1 = int(round(x0 / scale)), int(round(x1 / scale))
                y0, y1 = int(round(y0 / scale)), int(round(y1 / scale))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                continue
            area = (x1 - x0) * (y1 - y0)
            faces.append(
                {
                    "box": [x0, y0, x1, y1],
                    "confidence": round(conf, 3),
                    "area": int(area),
                    "backend": "caffe",
                }
            )

    if not faces:
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            haar = _haar_cascade().detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32)
            )
            for x, y, fw, fh in haar:
                faces.append(
                    {
                        "box": [int(x), int(y), int(x + fw), int(y + fh)],
                        "confidence": 0.5,
                        "area": int(fw * fh),
                        "backend": "haar",
                    }
                )
        except Exception:
            pass

    faces.sort(key=lambda f: f["area"] * f["confidence"], reverse=True)
    # De-dupe heavily overlapping boxes (keep larger/higher score first).
    kept: list[dict[str, Any]] = []
    for f in faces:
        x0, y0, x1, y1 = f["box"]
        duplicate = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["box"]
            ix0, iy0 = max(x0, kx0), max(y0, ky0)
            ix1, iy1 = min(x1, kx1), min(y1, ky1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = f["area"] + k["area"] - inter
            if union > 0 and inter / union > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(f)
    return kept


def require_face(image: Image.Image, cache_dir: Path | str, label: str) -> dict[str, Any]:
    """Fail if no real face is found; warn when multiple faces (largest used)."""
    faces = _detect_faces_strict(image, cache_dir)
    if not faces:
        raise DemoError(
            f"No face detected in the {label} image. "
            "Use a clear, front-facing photo with a visible face."
        )
    chosen = faces[0]
    if len(faces) > 1:
        warn(
            f"{label}: detected {len(faces)} faces — using the largest "
            f"(conf={chosen['confidence']}, box={chosen['box']}). "
            "Crop to a single subject for more reliable swaps."
        )
    return {
        "box": chosen["box"],
        "confidence": chosen["confidence"],
        "size": list(image.size),
        "face_count": len(faces),
        "all_faces": faces,
        "selected": "largest",
    }


def verify_gpu() -> dict[str, Any]:
    info = torch_versions()
    if not info["cuda_available"]:
        raise DemoError(
            "No GPU detected. Runtime → Change runtime type → GPU "
            "(prefer A100), then Runtime → Run all."
        )
    return {
        "name": info["gpu"],
        "vram_gb": info["vram_gb"],
        "torch": info["torch"],
        "cuda_runtime": info["cuda_runtime"],
    }


def print_run_parameters(params: dict[str, Any]) -> None:
    print("Run parameters")
    for key in (
        "seed",
        "steps",
        "cfg",
        "output_long_side",
        "stitch",
        "debug",
        "identity_thresh",
        "body_psnr_thresh",
        "prompt",
        "repo_commit",
    ):
        if key in params and params[key] is not None:
            val = params[key]
            if key == "prompt" and isinstance(val, str) and len(val) > 100:
                val = val[:100] + "…"
            print(f"  {key:<20} {val}")


def environment_summary(
    *,
    paths: dict[str, Path] | None = None,
    versions: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    # Back-compat kwargs from earlier notebook cells
    seed: int | None = None,
    stitch: bool | None = None,
    debug: bool | None = None,
) -> dict[str, Any]:
    if versions is None:
        versions = collect_versions(
            repo=(paths or {}).get("repo"),
            comfyui=(paths or {}).get("comfyui"),
        )
    print_versions(versions)
    # Merge legacy kwargs into params if the notebook still passes seed=/stitch=/debug=
    merged = dict(params or {})
    if seed is not None:
        merged.setdefault("seed", seed)
    if stitch is not None:
        merged.setdefault("stitch", stitch)
    if debug is not None:
        merged.setdefault("debug", debug)
    if merged:
        print_run_parameters(merged)
    store = Path(
        (paths or {}).get("model_store")
        or os.environ.get("HEADSWAP_MODEL_STORE", "/content/models")
    )
    models = required_model_paths(store)
    present = sum(1 for p in models if p.is_file())
    print(f"  model_store       {store}")
    print(f"  models_present    {present}/{len(models)}")
    return {"versions": versions, "params": merged, "model_store": str(store)}


def preflight(
    *,
    body_path: Path | str,
    face_path: Path | str,
    cache_dir: Path | str,
    model_store: Path | str | None = None,
) -> dict[str, Any]:
    """Validate GPU, weights (+sizes), and both images before inference."""
    progress("Preflight checks…")
    gpu = verify_gpu()
    models = verify_models(model_store, check_sizes=True)
    body_p = require_path(body_path, "Body image")
    face_p = require_path(face_path, "Face image")
    body = Image.open(body_p).convert("RGB")
    face = Image.open(face_p).convert("RGB")
    for msg in check_image_geometry(body, "Body") + check_image_geometry(face, "Face"):
        warn(msg)
    body_face = require_face(body, cache_dir, "body")
    face_face = require_face(face, cache_dir, "face")
    ok(
        f"Preflight passed — GPU={gpu['name']}, models={len(models)}, "
        f"body faces={body_face['face_count']} (using largest, conf={body_face['confidence']}), "
        f"face faces={face_face['face_count']} (conf={face_face['confidence']})"
    )
    return {
        "gpu": gpu,
        "models": models,
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
        out["max_body_dim"] = side
        out["max_dim"] = side
        out["crop_long_side"] = min(side, int(out.get("crop_long_side", 768) or 768))
        if not stitch:
            out["max_dim"] = side
    return out


def knobs_from_run_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Extract notebook knobs from a saved run_config.json for replay."""
    knobs = run_config.get("knobs") or run_config
    return {
        "SEED": int(knobs["seed"]),
        "PROMPT": knobs.get("prompt"),
        "STEPS": int(knobs["steps"]),
        "CFG": float(knobs["cfg"]),
        "OUTPUT_LONG_SIDE": int(knobs.get("output_long_side") or knobs.get("max_body_dim") or 1024),
        "STITCH": bool(knobs.get("stitch", knobs.get("mask_crop_stitch", True))),
        "DEBUG": bool(knobs.get("debug", False)),
        "IDENTITY_THRESH": float(knobs.get("identity_thresh", 0.35)),
        "BODY_PSNR_THRESH": float(knobs.get("body_psnr_thresh", 28.0)),
        "PINNED_COMMIT": (run_config.get("versions") or {}).get("git", {}).get("commit"),
    }


def make_run_dir(output_root: Path | str | None = None) -> Path:
    root = Path(output_root or DEFAULT_OUTPUT_DIR)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "debug").mkdir(exist_ok=True)
    return run_dir


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


def save_output_package(
    run_dir: Path,
    *,
    result_image: Image.Image,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    timing: dict[str, Any] | None,
    debug_paths: dict[str, str] | None = None,
    save_debug: bool = False,
) -> dict[str, str]:
    """Write final image + config + metrics + timing (+ optional debug) into run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.png"
    result_image.save(result_path)

    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2, default=str))

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    timing_path = run_dir / "timing.json"
    timing_path.write_text(json.dumps(timing or {}, indent=2, default=str))

    copied_debug: dict[str, str] = {}
    if save_debug and debug_paths:
        dbg_dir = run_dir / "debug"
        dbg_dir.mkdir(exist_ok=True)
        for key, src in debug_paths.items():
            sp = Path(src)
            if sp.is_file():
                dest = dbg_dir / sp.name
                shutil.copy2(sp, dest)
                copied_debug[key] = str(dest)

    # Also refresh the stable shortcut at the outputs root.
    stable = write_stable_result(result_path, run_dir.parent)
    return {
        "run_dir": str(run_dir),
        "result": str(result_path),
        "run_config": str(config_path),
        "metrics": str(metrics_path),
        "timing": str(timing_path),
        "stable": str(stable),
        **{f"debug_{k}": v for k, v in copied_debug.items()},
    }


def score_result(
    *,
    body: Image.Image,
    face: Image.Image,
    result: Image.Image,
    latency_s: float,
    cache_dir: Path | str,
    pipeline: str = "krea2_identity_edit",
    pair_id: str = "custom_001",
    identity_thresh: float = 0.35,
    body_psnr_thresh: float = 28.0,
    stitch: bool = True,
) -> dict[str, Any]:
    """Run existing eval metrics and apply configurable quality thresholds."""
    sys.path.insert(0, str(ROOT / "src"))
    from headswap.metrics.scoring import score_pair
    from headswap.preprocess import head_hair_mask_from_face

    head_mask = None
    if stitch:
        try:
            head_mask = head_hair_mask_from_face(body, Path(cache_dir))
            head_mask = head_mask.resize(result.size)
        except Exception:
            head_mask = None

    body_r = body.resize(result.size, Image.Resampling.LANCZOS)
    pm = score_pair(
        pair_id=pair_id,
        pipeline=pipeline,
        body=body_r,
        face=face,
        result=result,
        latency_s=latency_s,
        head_mask=head_mask,
        cache_dir=Path(cache_dir),
        identity_thresh=identity_thresh,
        body_psnr_thresh=body_psnr_thresh,
    )
    data = pm.to_dict()
    warnings: list[str] = []
    id_cos = data.get("identity_cosine")
    body_psnr = data.get("body_preserve_psnr")
    if id_cos is None:
        warnings.append(
            "Identity score unavailable (insightface not installed). "
            "Body/face detection metrics still apply."
        )
    elif id_cos < identity_thresh:
        warnings.append(
            f"Identity score {id_cos:.3f} is below threshold {identity_thresh}."
        )
    if body_psnr is not None and body_psnr < body_psnr_thresh:
        warnings.append(
            f"Body preservation PSNR {body_psnr:.1f} is below threshold {body_psnr_thresh}."
        )
    if not data.get("success"):
        warnings.append(
            "Automatic quality check marked this run as FAIL: "
            + ", ".join(data.get("fail_reasons") or ["unknown"])
        )
    data["quality_warnings"] = warnings
    data["thresholds"] = {
        "identity_thresh": identity_thresh,
        "body_psnr_thresh": body_psnr_thresh,
    }
    return data


def print_quality_report(quality: dict[str, Any]) -> None:
    print("Quality check")
    print(f"  success           {quality.get('success')}")
    print(f"  identity_cosine   {quality.get('identity_cosine')}")
    print(f"  body_preserve_psnr {quality.get('body_preserve_psnr')}")
    print(f"  face_detected     {quality.get('face_detected')}")
    print(f"  fail_reasons      {quality.get('fail_reasons') or []}")
    for w in quality.get("quality_warnings") or []:
        warn(w)
    if quality.get("success") and not quality.get("quality_warnings"):
        ok("Quality check passed")


def print_eval_card(
    *,
    gpu: str | None,
    commit: str | None,
    identity: float | None,
    body_psnr: float | None,
    sampling_s: float | None,
    total_s: float | None,
    output_path: Path | str | None,
    model: str = "Krea2 Identity Edit",
    success: bool | None = None,
) -> None:
    """One-screen summary for stakeholders — no scroll required."""
    def _fmt(v: float | None, digits: int = 2) -> str:
        if v is None:
            return "n/a"
        return f"{v:.{digits}f}"

    print("Run Summary")
    print("-----------")
    print(f"Model: {model}")
    print(f"GPU: {gpu or 'n/a'}")
    print(f"Commit: {commit or 'n/a'}")
    print()
    if success is not None:
        print(f"Quality gate: {'PASS' if success else 'FAIL'}")
    print(f"Identity score: {_fmt(identity, 3)}")
    print(f"Body PSNR: {_fmt(body_psnr, 1)}")
    print(f"Sampling: {_fmt(sampling_s)} s")
    print(f"Total runtime: {_fmt(total_s)} s")
    print()
    print("Output:")
    print(str(output_path) if output_path else "n/a")


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
    quality: dict[str, Any] | None = None,
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
    if quality:
        print_quality_report(quality)


@contextmanager
def quiet_logs(*, debug: bool) -> Iterator[None]:
    """Hide pipeline/Comfy chatter unless DEBUG is on."""
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
