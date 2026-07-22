#!/usr/bin/env python3
"""Colab path / environment helpers for headswap_V2 demo notebooks.

Keeps notebook cells thin: detect Colab, mount-aware model store, print a
short environment summary. Does not run pipelines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return Path("/content").exists() and "COLAB_GPU" in os.environ


def default_paths(*, use_drive: bool = True) -> dict[str, Path]:
    """Return canonical Colab (or local fallback) paths."""
    if is_colab():
        drive_root = Path("/content/drive/MyDrive/headswap_V2")
        content = Path("/content")
        if use_drive and (Path("/content/drive/MyDrive").exists()):
            model_store = drive_root / "models"
        else:
            model_store = content / "models"
        return {
            "repo": content / "headswap_V2",
            "comfyui": content / "ComfyUI",
            "model_store": model_store,
            "staging": content / "_hf_dl_staging",
            "outputs": content / "headswap_outputs",
            "drive_root": drive_root,
        }
    # Local / non-Colab fallback (dev only)
    root = Path(__file__).resolve().parents[1]
    return {
        "repo": root,
        "comfyui": Path(os.environ.get("COMFYUI_PATH", str(root.parent / "ComfyUI"))),
        "model_store": Path(os.environ.get("HEADSWAP_MODEL_STORE", str(root / "models"))),
        "staging": Path(os.environ.get("HEADSWAP_STAGING_DIR", str(root / ".cache" / "hf_staging"))),
        "outputs": root / "results",
        "drive_root": root,
    }


def apply_env(paths: dict[str, Path] | None = None) -> dict[str, Path]:
    """Export COMFYUI_PATH / HEADSWAP_* and ensure directories exist."""
    paths = paths or default_paths()
    os.environ["COMFYUI_PATH"] = str(paths["comfyui"])
    os.environ["HEADSWAP_MODEL_STORE"] = str(paths["model_store"])
    os.environ["HEADSWAP_STAGING_DIR"] = str(paths["staging"])
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for key in ("model_store", "staging", "outputs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    repo = paths["repo"]
    if repo.exists():
        os.chdir(repo)
        src = str(repo / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    return paths


def gpu_summary() -> dict:
    info: dict = {"cuda": False, "name": None, "total_gb": None}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            info["torch"] = torch.__version__
    except Exception as exc:
        info["error"] = str(exc)
    return info


def print_banner(paths: dict[str, Path] | None = None) -> None:
    paths = paths or apply_env()
    gpu = gpu_summary()
    print("headswap_V2 · Colab environment")
    print(f"  colab:       {is_colab()}")
    print(f"  repo:        {paths['repo']}")
    print(f"  comfyui:     {paths['comfyui']}")
    print(f"  model_store: {paths['model_store']}")
    print(f"  staging:     {paths['staging']}")
    print(f"  outputs:     {paths['outputs']}")
    if gpu.get("cuda"):
        print(f"  gpu:         {gpu.get('name')} ({gpu.get('total_gb')} GiB)")
        print(f"  torch:       {gpu.get('torch')}")
    else:
        print("  gpu:         NOT AVAILABLE — Runtime → Change runtime type → GPU (A100 preferred)")


if __name__ == "__main__":
    print_banner(apply_env())
