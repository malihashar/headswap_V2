"""Unit tests for download_models path defaults (Colab / Kaggle / local)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DL_PATH = REPO / "scripts" / "download_models.py"


def _load_download_models():
    spec = importlib.util.spec_from_file_location("download_models", DL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def dl():
    return _load_download_models()


def test_kaggle_defaults_use_tmp_not_working(dl, monkeypatch, tmp_path):
    monkeypatch.delenv("HEADSWAP_MODEL_STORE", raising=False)
    monkeypatch.delenv("HEADSWAP_STAGING_DIR", raising=False)
    monkeypatch.setattr(dl, "_on_kaggle", lambda: True)
    monkeypatch.setattr(dl, "_on_colab", lambda: False)
    # Drive must not hijack Kaggle defaults.
    monkeypatch.setattr(Path, "exists", lambda self: False)

    store = dl.default_store_dir(Path("/kaggle/working/ComfyUI"))
    staging = dl.default_staging_dir()
    assert store == Path("/tmp/models")
    assert staging == Path("/tmp/_hf_dl_staging")
    assert not str(store).startswith("/kaggle/working")
    assert not str(staging).startswith("/kaggle/working")


def test_kaggle_env_overrides(dl, monkeypatch):
    monkeypatch.setenv("HEADSWAP_MODEL_STORE", "/tmp/custom_models")
    monkeypatch.setenv("HEADSWAP_STAGING_DIR", "/tmp/custom_staging")
    monkeypatch.setattr(dl, "_on_kaggle", lambda: True)
    assert dl.default_store_dir(Path("/x")) == Path("/tmp/custom_models")
    assert dl.default_staging_dir() == Path("/tmp/custom_staging")


def test_colab_defaults_unchanged(dl, monkeypatch):
    monkeypatch.delenv("HEADSWAP_MODEL_STORE", raising=False)
    monkeypatch.delenv("HEADSWAP_STAGING_DIR", raising=False)
    monkeypatch.setattr(dl, "_on_kaggle", lambda: False)
    monkeypatch.setattr(dl, "_on_colab", lambda: True)

    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self) == "/content/drive/MyDrive":
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert dl.default_store_dir(Path("/content/ComfyUI")) == dl.DEFAULT_DRIVE_STORE
    assert dl.default_staging_dir() == dl.DEFAULT_STAGING


def test_local_defaults(dl, monkeypatch):
    monkeypatch.delenv("HEADSWAP_MODEL_STORE", raising=False)
    monkeypatch.delenv("HEADSWAP_STAGING_DIR", raising=False)
    monkeypatch.setattr(dl, "_on_kaggle", lambda: False)
    monkeypatch.setattr(dl, "_on_colab", lambda: False)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    comfy = Path("/opt/ComfyUI")
    assert dl.default_store_dir(comfy) == comfy / "models"
    assert dl.default_staging_dir() == Path("/tmp/headswap_hf_staging")
