#!/usr/bin/env python3
"""Report whether this runtime can evaluate the FLUX Fill identity-edit spike.

This tool never downloads models or accepts a license on the user's behalf.
It reports the exact missing prerequisite into a JSON artifact.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


def _module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("FLUX_FILL_MODEL_DIR", "/content/models/flux-fill")),
    )
    args = parser.parse_args()

    diffusers = _module_version("diffusers")
    flux_fill_importable = False
    flux_fill_error = None
    if diffusers["available"]:
        try:
            from diffusers import FluxFillPipeline  # noqa: F401

            flux_fill_importable = True
        except Exception as exc:
            flux_fill_error = f"{type(exc).__name__}: {exc}"

    torch = _module_version("torch")
    cuda = False
    vram_gb = None
    if torch["available"]:
        try:
            import torch as torch_module

            cuda = bool(torch_module.cuda.is_available())
            if cuda:
                vram_gb = round(
                    torch_module.cuda.get_device_properties(0).total_memory / 1024**3, 2
                )
        except Exception:
            pass

    local_weights = list(args.model_dir.glob("**/*.safetensors")) if args.model_dir.exists() else []
    blockers: list[str] = []
    if not flux_fill_importable:
        blockers.append("FluxFillPipeline unavailable in installed diffusers")
    if not cuda:
        blockers.append("CUDA unavailable")
    if not local_weights:
        blockers.append(f"no local FLUX Fill weights under {args.model_dir}")
    blockers.append(
        "license approval and identity-adapter compatibility must be confirmed before download/integration"
    )

    report = {
        "spike": "flux_fill_identity_local",
        "diffusers": diffusers,
        "flux_fill_importable": flux_fill_importable,
        "flux_fill_import_error": flux_fill_error,
        "torch": torch,
        "cuda_available": cuda,
        "vram_gb": vram_gb,
        "model_dir": str(args.model_dir),
        "local_weight_count": len(local_weights),
        "blockers": blockers,
        "next_validation": [
            "Load FLUX Fill with a consented local model after license approval.",
            "Prove outside-mask pixels are byte-identical before adding identity conditioning.",
            "Evaluate a license-compatible FLUX identity adapter against the frozen benchmark.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
