#!/usr/bin/env python3
"""Ensure REFace can actually import its inference stack on Colab.

Whack-a-mole pip installs with ``|| true`` left ``taming`` / ``clip`` missing.
This script:
  1. patches REFace for Py3.12 / modern transformers / Lightning
  2. clones CompVis/taming-transformers + openai/CLIP next to REFace if needed
  3. smoke-imports the real module chain used by inference_swap_selected.py
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


VENDOR_SPECS = (
    {
        "name": "taming-transformers",
        "url": "https://github.com/CompVis/taming-transformers.git",
        "import_name": "taming",
        "probe": "taming.modules.vqvae.quantize",
    },
    {
        "name": "CLIP",
        "url": "https://github.com/openai/CLIP.git",
        "import_name": "clip",
        "probe": "clip",
    },
)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("→", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def vendor_root(reface_root: Path) -> Path:
    env = os.environ.get("REFACE_VENDOR_ROOT")
    if env:
        return Path(env)
    # Prefer /content on Colab; otherwise sit beside REFace.
    content = Path("/content")
    if content.is_dir() and os.access(content, os.W_OK):
        return content / "reface_vendor"
    return reface_root.parent / "reface_vendor"


def ensure_vendors(reface_root: Path) -> list[Path]:
    root = vendor_root(reface_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in VENDOR_SPECS:
        dest = root / spec["name"]
        if not (dest / ".git").is_dir():
            if dest.exists():
                # Incomplete leftover — re-clone.
                import shutil

                shutil.rmtree(dest)
            _run(["git", "clone", "--depth", "1", spec["url"], str(dest)])
        else:
            print(f"✓ vendor present: {dest}", flush=True)
        paths.append(dest)
        # Best-effort editable install (PYTHONPATH is the source of truth).
        try:
            _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(dest)])
        except subprocess.CalledProcessError as exc:
            print(f"⚠ pip -e {dest.name} failed ({exc}); relying on PYTHONPATH", flush=True)
    return paths


def prepend_pythonpath(paths: list[Path], reface_root: Path) -> str:
    parts = [str(reface_root), *[str(p) for p in paths]]
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged: list[str] = []
    seen: set[str] = set()
    for p in parts + existing:
        if p and p not in seen:
            seen.add(p)
            merged.append(p)
            if p not in sys.path:
                sys.path.insert(0, p)
    value = os.pathsep.join(merged)
    os.environ["PYTHONPATH"] = value
    return value


def patch_reface(reface_root: Path) -> None:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from patch_reface_py312 import patch_reface_tree

    touched = patch_reface_tree(reface_root)
    if touched:
        print(f"✓ patched {len(touched)} REFace file(s)", flush=True)
    else:
        print("✓ REFace patches already applied", flush=True)


def smoke_imports(reface_root: Path) -> None:
    # Keep probes aligned with the real inference import chain.
    probes = [
        "torch",
        "pytorch_lightning",
        "omegaconf",
        "einops",
        "albumentations",
        "taming.modules.vqvae.quantize",
        "clip",
        "ldm.util",
        "ldm.models.autoencoder",
        "ldm.models.diffusion.ddpm",
    ]
    missing: list[str] = []
    # Ensure REFace root is importable for ``ldm.*``.
    if str(reface_root) not in sys.path:
        sys.path.insert(0, str(reface_root))
    for name in probes:
        try:
            importlib.import_module(name)
            print(f"✓ import {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name}: {exc}")
    if missing:
        raise SystemExit(
            "REFace runtime imports FAILED:\n  - "
            + "\n  - ".join(missing)
            + "\n\nRe-run: bash /content/headswap_V2/scripts/setup_reface_colab.sh"
        )
    print("✓ REFace inference import chain OK", flush=True)


def ensure(reface_root: Path, *, do_smoke: bool = True) -> str:
    reface_root = reface_root.resolve()
    if not reface_root.is_dir():
        raise SystemExit(f"REFACE_ROOT not found: {reface_root}")
    patch_reface(reface_root)
    vendors = ensure_vendors(reface_root)
    pp = prepend_pythonpath(vendors, reface_root)
    if do_smoke:
        smoke_imports(reface_root)
    return pp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reface-root",
        type=Path,
        default=Path(os.environ.get("REFACE_ROOT", "/content/REFace")),
    )
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    pp = ensure(args.reface_root, do_smoke=not args.no_smoke)
    print(f"✓ PYTHONPATH ready\n  {pp}", flush=True)


if __name__ == "__main__":
    main()
