"""Regression: HF snapshot relative symlinks must be resolved before place."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _load_download_models():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
    name = "download_models"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required before exec for @dataclass on 3.14
    spec.loader.exec_module(mod)
    return mod


dm = _load_download_models()


def _make_hf_style_layout(tmp_path: Path, payload: bytes) -> Path:
    """
    Mimic huggingface_hub cache layout:

      snapshots/<rev>/model.safetensors -> ../../blobs/<hash>
    """
    hub = tmp_path / "_hub_cache" / "models--org--repo"
    blobs = hub / "blobs"
    snap = hub / "snapshots" / "main"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(payload)
    link = snap / "model.safetensors"
    # Relative target exactly like HF: from snapshots/main → ../../blobs/deadbeef
    os.symlink("../../blobs/deadbeef", link)
    assert link.is_symlink()
    assert link.resolve(strict=True) == blob.resolve()
    return link


def test_relative_snapshot_symlink_in_staging_breaks(tmp_path: Path) -> None:
    """
    Document the Linux failure mode: hardlinking an HF snapshot symlink preserves
    the relative target (../../blobs/<hash>). Under _hf_dl_staging that target
    does not resolve, so dest.stat() raises FileNotFoundError.
    (macOS os.link follows symlinks; Linux does not — recreate the Linux result.)
    """
    payload = b"x" * 128
    _make_hf_style_layout(tmp_path, payload)
    staging = tmp_path / "_hf_dl_staging"
    staging.mkdir()
    dest = staging / "model.safetensors"
    os.symlink("../../blobs/deadbeef", dest)
    assert dest.is_symlink()
    assert not dest.exists()
    with pytest.raises(FileNotFoundError):
        dest.stat()


def test_place_resolves_hf_snapshot_symlink(tmp_path: Path) -> None:
    payload = b"weights-payload-" + b"0" * 64
    src_link = _make_hf_style_layout(tmp_path, payload)
    staging = tmp_path / "_hf_dl_staging"
    staging.mkdir()
    dest = staging / "model.safetensors"

    dm.place_file_without_doubling(src_link, dest, expected_size=len(payload))

    dm._verify_placed_dest(dest, len(payload))
    assert not dest.is_symlink()
    assert dest.is_file()
    assert dest.read_bytes() == payload
    # dest.stat() must succeed after verification.
    assert dest.stat().st_size == len(payload)


def test_verify_rejects_broken_symlink(tmp_path: Path) -> None:
    dest = tmp_path / "broken"
    os.symlink("does-not-exist", dest)
    with pytest.raises(FileNotFoundError, match="symlink"):
        dm._verify_placed_dest(dest, expected_size=1)
