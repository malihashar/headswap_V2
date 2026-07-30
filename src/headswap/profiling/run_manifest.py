"""Immutable provenance manifests for identity-edit evaluation runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_image(image: Image.Image) -> str:
    """Hash decoded RGB pixels so file format metadata cannot hide a mismatch."""
    rgb = np.asarray(image.convert("RGB"))
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def git_commit(repo: Path | str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_artifact(path: Path | str, image: Image.Image) -> dict[str, Any]:
    """Describe one source image both as a file and as decoded pixels."""
    p = Path(path)
    return {
        "path": str(p.resolve()),
        "file_sha256": sha256_file(p),
        "decoded_rgb_sha256": sha256_image(image),
        "size": list(image.size),
        "mode": image.mode,
    }


def build_manifest(
    *,
    repo: Path | str,
    body_path: Path | str,
    face_path: Path | str,
    body: Image.Image,
    face: Image.Image,
    config: dict[str, Any],
    detections: list[dict[str, Any]],
    selected_index: int | None,
    selected_box: list[int] | None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the immutable run contract written before inference.

    The image hashes prove the actual decoded pair, while the config hash proves
    that a displayed output came from the stated routing/conditioning settings.
    """
    cfg = dict(config)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inputs": {
            "body": input_artifact(body_path, body),
            "face": input_artifact(face_path, face),
        },
        "config_sha256": _json_sha256(cfg),
        "effective_config": cfg,
        "target_selection": {
            "detections": detections,
            "selected_index": selected_index,
            "selected_box": selected_box,
        },
        "runtime": dict(runtime or {}),
    }


def write_manifest(run_dir: Path | str, manifest: dict[str, Any]) -> Path:
    """Write once into a newly-created run directory."""
    path = Path(run_dir) / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path
