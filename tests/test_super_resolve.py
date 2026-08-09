"""Super-resolve the pre-swap plate without hallucinating background detail.

Real-ESRGAN x4 on the WHOLE frame was measured (GPU, reference case) to
hallucinate a row of arc/bird-like shapes into a smooth gradient sunset sky --
absent from the source and from a plain Lanczos upsample of the same region.
Only a padded crop around the face needs super-resolution (that's the only
region identity/quality depends on); the rest of the frame uses a plain
Lanczos upsample instead, which cannot hallucinate detail that isn't there.

The super-resolution backend is a third-party model and is exercised on GPU,
not here -- these tests cover the compositing geometry and graceful
degradation when the optional dependency is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.preprocess import FaceBox
from headswap.super_resolve import sr_backend_available, super_resolve_plate

W, H = 120, 180
FACE = FaceBox(45, 60, 75, 100, 0.9)
SCALE = 4


def _plate():
    return Image.new("RGB", (W, H), (240, 210, 160))


def test_backend_unavailable_falls_back_to_lanczos(monkeypatch):
    monkeypatch.setitem(sys.modules, "RealESRGAN", None)
    monkeypatch.setattr(
        "headswap.super_resolve.sr_backend_available", lambda: False
    )
    out, info = super_resolve_plate(_plate(), FACE, scale=SCALE)
    assert info["applied"] is False
    assert info["reason"] == "realesrgan_missing"
    assert out.size == (W * SCALE, H * SCALE)


def test_applied_only_changes_face_crop_region(monkeypatch):
    plate = _plate()

    class _FakeModel:
        def predict(self, im):
            # Distinct fill so the pasted region is unmistakable.
            return Image.new("RGB", (im.width * SCALE, im.height * SCALE), (1, 2, 3))

    monkeypatch.setattr("headswap.super_resolve.sr_backend_available", lambda: True)
    monkeypatch.setattr(
        "headswap.super_resolve._get_sr_model", lambda scale: _FakeModel()
    )
    out, info = super_resolve_plate(plate, FACE, scale=SCALE)
    assert info["applied"] is True
    assert out.size == (W * SCALE, H * SCALE)

    arr = np.asarray(out)
    x0, y0, x1, y1 = info["face_crop_box_hr"]
    inside = arr[y0:y1, x0:x1]
    assert np.all(inside == (1, 2, 3))

    # Outside the face-crop box: plain Lanczos upsample of a flat-color plate
    # is that same flat color everywhere -- nothing hallucinated in.
    outside_mask = np.ones(arr.shape[:2], dtype=bool)
    outside_mask[y0:y1, x0:x1] = False
    outside = arr[outside_mask]
    assert np.all(outside == (240, 210, 160))


def test_sr_failure_falls_back_to_lanczos(monkeypatch):
    monkeypatch.setattr("headswap.super_resolve.sr_backend_available", lambda: True)

    def _boom(scale):
        raise RuntimeError("boom")

    monkeypatch.setattr("headswap.super_resolve._get_sr_model", _boom)
    out, info = super_resolve_plate(_plate(), FACE, scale=SCALE)
    assert info["applied"] is False
    assert "sr_failed" in info["reason"]
    assert out.size == (W * SCALE, H * SCALE)


def test_sr_backend_available_reflects_import_success():
    # Whatever the actual environment has installed -- just must not raise.
    assert isinstance(sr_backend_available(), bool)


def test_backend_available_shims_before_import_attempt(monkeypatch):
    """Regression test for a real GPU-caught bug: RealESRGAN/__init__.py
    imports the removed huggingface_hub.cached_download API at MODULE load
    time, so the shim must run before ANY `import RealESRGAN`, including
    this availability probe -- not just the model-loading path. A test that
    only mocks `_get_sr_model` (as the other tests here do) cannot catch
    this: it never exercises the real, unshimmed import order that broke on
    GPU (sr_backend_available() reported the backend missing despite it
    being installed).
    """
    calls = []
    monkeypatch.setattr(
        "headswap.super_resolve._shim_huggingface_hub", lambda: calls.append("shim")
    )
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "RealESRGAN":
            calls.append("import_realesrgan")
            raise ImportError("cannot import name 'cached_download'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert sr_backend_available() is False
    assert calls == ["shim", "import_realesrgan"]
