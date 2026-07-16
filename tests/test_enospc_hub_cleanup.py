"""Regression: ENOSPC recovery and per-model hub-cache cleanup."""

from __future__ import annotations

import errno
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_download_models():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
    name = "download_models_enospc"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dm = _load_download_models()


def _artifact(
    *,
    filename: str = "qwen_image_edit_2511_fp8mixed.safetensors",
    repo_id: str = "Qwen/Qwen-Image-Edit",
    size: int = 1_000_000,
    download_repo_id: str | None = None,
) -> dm.Artifact:
    return dm.Artifact(
        filename=filename,
        size=size,
        path="diffusion_models",
        repo_id=repo_id,
        repo_path=filename,
        url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
        required=True,
        set_name="qwen",
        download_repo_id=download_repo_id,
    )


def _plant_partial_hub_cache(
    hub_cache: Path,
    repo_id: str,
    *,
    incomplete_name: str = "abc123.incomplete",
    incomplete_bytes: bytes = b"partial" * 1024,
) -> Path:
    """Create models--org--name/blobs/*.incomplete like a failed hf_hub_download."""
    repo_dir = dm.hub_repo_cache_dir(hub_cache, repo_id)
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True)
    incomplete = blobs / incomplete_name
    incomplete.write_bytes(incomplete_bytes)
    (repo_dir / "refs" / "main").parent.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text("deadbeef\n")
    return incomplete


def test_hub_repo_cache_dir_naming():
    assert (
        dm.hub_repo_cache_dir(Path("/c"), "Qwen/Qwen-Image-Edit").name
        == "models--Qwen--Qwen-Image-Edit"
    )


def test_cleanup_failed_model_hub_cache_removes_only_that_repo(tmp_path: Path):
    hub_cache = tmp_path / "_hub_cache"
    staging = tmp_path / "_hf_dl_staging"
    staging.mkdir()
    store = tmp_path / "models"  # promoted store — must not be touched
    store.mkdir()
    promoted = store / "other_model.safetensors"
    promoted.write_bytes(b"promoted-ok")

    art = _artifact(repo_id="Qwen/Qwen-Image-Edit")
    other = _artifact(
        filename="klein.safetensors",
        repo_id="org/klein-repo",
        size=100,
    )

    failed_incomplete = _plant_partial_hub_cache(hub_cache, art.repo_id)
    other_incomplete = _plant_partial_hub_cache(
        hub_cache, other.repo_id, incomplete_name="other.incomplete"
    )
    staging_partial = staging / art.filename
    staging_partial.write_bytes(b"flat-partial")

    freed = dm.cleanup_failed_model_hub_cache(
        art, hub_cache=hub_cache, staging_dir=staging
    )

    assert freed > 0
    assert not failed_incomplete.exists()
    assert not dm.hub_repo_cache_dir(hub_cache, art.repo_id).exists()
    # Other model's cache left intact
    assert other_incomplete.exists()
    assert dm.hub_repo_cache_dir(hub_cache, other.repo_id).exists()
    # Staging partial for failed model removed
    assert not staging_partial.exists()
    # Promoted store untouched
    assert promoted.read_bytes() == b"promoted-ok"


def test_cleanup_removes_incomplete_blobs_under_download_repo(tmp_path: Path):
    hub_cache = tmp_path / "_hub_cache"
    art = _artifact(
        repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
        download_repo_id="Qwen/Qwen-Image-Edit",
    )
    a = _plant_partial_hub_cache(hub_cache, art.repo_id)
    b = _plant_partial_hub_cache(hub_cache, art.download_repo_id)

    dm.cleanup_failed_model_hub_cache(art, hub_cache=hub_cache)

    assert not a.exists()
    assert not b.exists()
    assert not dm.hub_repo_cache_dir(hub_cache, art.repo_id).exists()
    assert not dm.hub_repo_cache_dir(hub_cache, art.download_repo_id).exists()


def test_require_free_space_aborts_when_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        dm.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=10_000, used=9_500, free=500),
    )
    with pytest.raises(dm.NoSpaceError, match="Not enough free space"):
        dm.require_free_space(
            tmp_path,
            needed_bytes=1_000,
            safety_margin_bytes=2_000,
        )


def test_require_free_space_passes_when_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        dm.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=10_000, used=1_000, free=9_000),
    )
    dm.require_free_space(
        tmp_path, needed_bytes=1_000, safety_margin_bytes=2_000
    )


def test_is_enospc_detects_oserror_and_message():
    assert dm._is_enospc(OSError(errno.ENOSPC, "No space left on device"))
    assert dm._is_enospc(RuntimeError("OSError: [Errno 28] No space left on device"))
    assert dm._is_enospc(dm.NoSpaceError("boom"))
    assert not dm._is_enospc(RuntimeError("connection reset"))


def test_enospc_recovery_cleans_before_retry_logic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Simulate the Kaggle failure loop: ENOSPC leaves a .incomplete blob; cleanup
    must remove it so a subsequent free-space check can succeed.
    """
    hub_cache = tmp_path / "_hub_cache"
    staging = tmp_path / "_hf_dl_staging"
    staging.mkdir()
    art = _artifact(size=100)

    incomplete = _plant_partial_hub_cache(
        hub_cache, art.repo_id, incomplete_bytes=b"x" * 50_000
    )
    assert incomplete.exists()

    # First free-space check fails (tight disk).
    frees = iter([100, 50_000 + dm.DEFAULT_FREE_SPACE_MARGIN_BYTES + 200])

    def fake_disk_usage(_p):
        free = next(frees)
        return SimpleNamespace(total=free + 1, used=1, free=free)

    monkeypatch.setattr(dm.shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(dm.NoSpaceError):
        dm.require_free_space(
            staging, art.size, safety_margin_bytes=dm.DEFAULT_FREE_SPACE_MARGIN_BYTES
        )

    dm.cleanup_failed_model_hub_cache(
        art, hub_cache=hub_cache, staging_dir=staging
    )
    assert not incomplete.exists()
    assert not dm.hub_repo_cache_dir(hub_cache, art.repo_id).exists()

    # After cleanup, next disk_usage value is large enough → download may proceed.
    dm.require_free_space(
        staging, art.size, safety_margin_bytes=dm.DEFAULT_FREE_SPACE_MARGIN_BYTES
    )


def test_hub_worker_enospc_exits_with_code_28(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    hub_cache = tmp_path / "_hub_cache"
    hub_cache.mkdir()
    staging_file = tmp_path / "model.safetensors"

    def boom(**_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    fake_hub = SimpleNamespace(hf_hub_download=boom)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(SystemExit) as ei:
        dm._hub_worker(
            "org/repo",
            "model.safetensors",
            "main",
            str(hub_cache),
            str(staging_file),
            100,
        )
    assert ei.value.code == errno.ENOSPC
    # Worker cleanup should have removed the planted cache if any; create one first
    # via a second call after planting.
    incomplete = _plant_partial_hub_cache(hub_cache, "org/repo")
    with pytest.raises(SystemExit) as ei2:
        dm._hub_worker(
            "org/repo",
            "model.safetensors",
            "main",
            str(hub_cache),
            str(staging_file),
            100,
        )
    assert ei2.value.code == errno.ENOSPC
    assert not incomplete.exists()
