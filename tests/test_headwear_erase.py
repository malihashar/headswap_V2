"""Headwear erase stage: make a real HEAD swap possible on a hatted target.

A hat that covers the skull makes a head swap impossible -- preserve it and
the donor's hair is hidden, so only the face changes and the result reads as
a face swap. Prompt-only removal fails: the edit model cannot reconstruct the
background behind a large opaque object and emits a bright blob instead
(observed on every removal arm). So the hat is erased deterministically by an
inpainting model FIRST, and the swap runs on a bare-headed plate.

These tests cover mask construction and graceful degradation. The inpainting
itself is a third-party model and is exercised on GPU, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.headwear_erase import erase_headwear, headwear_mask
from headswap.preprocess import FaceBox

W, H = 300, 460
FACE = FaceBox(120, 150, 180, 230, 0.95)


def _scene():
    """Bright sky + dark hat (crown above the face, flaps beside it) + body."""
    im = Image.new("RGB", (W, H), (240, 210, 160))
    d = ImageDraw.Draw(im)
    d.ellipse([118, 60, 182, 160], fill=(20, 20, 22))      # crown, above brow
    d.ellipse([88, 165, 120, 195], fill=(20, 20, 22))      # left flap
    d.ellipse([180, 165, 212, 195], fill=(20, 20, 22))     # right flap
    d.ellipse(list(FACE.__dict__.values())[:4] if False else [120, 150, 180, 230],
              fill=(200, 165, 140))                        # face (must survive)
    d.rectangle([90, 235, 210, H], fill=(30, 30, 35))      # body
    return im


def _matte():
    m = np.zeros((H, W), np.uint8)
    m[55:H, 85:215] = 255
    return m


def test_mask_covers_crown_and_flaps():
    hat = headwear_mask(_scene(), FACE, _matte())
    assert hat[100, 150] > 0, "crown must be masked"
    assert hat[180, 100] > 0, "left flap must be masked"
    assert hat[180, 200] > 0, "right flap must be masked"


def test_mask_never_erases_the_face():
    hat = headwear_mask(_scene(), FACE, _matte())
    cy = int((FACE.y0 + FACE.y1) / 2); cx = int((FACE.x0 + FACE.x1) / 2)
    assert hat[cy, cx] == 0, "face centre must be protected"


def test_mask_stays_out_of_the_body():
    hat = headwear_mask(_scene(), FACE, _matte())
    assert hat[H - 20, 150] == 0, "torso must not be erased"


def test_erase_degrades_gracefully_without_backend(monkeypatch):
    """A missing optional dep must return the image untouched, never raise."""
    import builtins
    real = builtins.__import__

    def _blocked(name, *a, **k):
        if name.startswith("simple_lama"):
            raise ImportError("blocked for test")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    src = _scene()
    out, info = erase_headwear(src, headwear_mask(src, FACE, _matte()))
    assert out is src
    assert info["applied"] is False
    assert "simple_lama_missing" in info["reason"]


def test_erase_skips_an_empty_mask():
    src = _scene()
    out, info = erase_headwear(src, np.zeros((H, W), np.uint8))
    assert out is src
    assert info["applied"] is False
    assert info["reason"] == "empty_headwear_mask"


def test_erase_calls_backend_and_returns_plate(monkeypatch):
    src = _scene()
    plate = Image.new("RGB", (W, H), (1, 2, 3))
    fake = mock.MagicMock(); fake.return_value.return_value = plate
    monkeypatch.setitem(sys.modules, "simple_lama_inpainting",
                        mock.MagicMock(SimpleLama=fake))
    mask = headwear_mask(src, FACE, _matte())
    out, info = erase_headwear(src, mask, feather_px=0)
    assert info["applied"] is True
    out_arr = np.asarray(out)
    # Only pixels inside the mask may take the inpainter's fill; LaMa returns
    # a re-encoded copy of the WHOLE frame, and returning that verbatim was
    # measured (GPU, reference case) to silently alter 40% of background
    # pixels outside the mask -- so erase_headwear now composites the fill
    # back through the mask instead of replacing the frame outright.
    assert np.array_equal(out_arr[mask > 0], np.asarray(plate)[mask > 0])
    assert np.array_equal(out_arr[mask == 0], np.asarray(src)[mask == 0])


def test_restore_background_copies_background_verbatim(monkeypatch):
    """Only the PERSON may differ from the source; background must be exact."""
    from headswap.headwear_erase import restore_background
    import headswap.segmentation as seg

    plate = Image.new("RGB", (W, H), (200, 180, 150))
    ImageDraw.Draw(plate).rectangle([100, 200, 200, H], fill=(30, 30, 35))
    # result: same person, but the sky has drifted brighter (the ghost-oval bug)
    result = Image.new("RGB", (W, H), (245, 225, 195))
    ImageDraw.Draw(result).rectangle([100, 200, 200, H], fill=(40, 40, 48))

    person = np.zeros((H, W), np.uint8); person[200:H, 100:200] = 255
    monkeypatch.setattr(seg, "_person_matte", lambda im: (person.copy(), None))

    out, info = restore_background(result, plate, dilate_px=0, blur_px=0)
    assert info["applied"] is True
    o = np.asarray(out).astype(int); p = np.asarray(plate).astype(int)
    bg = person == 0
    assert np.abs(o - p).mean(axis=2)[bg].max() == 0, "background must be verbatim"
    # and the person region still carries the generated pixels
    assert o[H - 5, 150, 0] > p[H - 5, 150, 0]


def test_restore_background_degrades_without_matte(monkeypatch):
    from headswap.headwear_erase import restore_background
    import headswap.segmentation as seg

    monkeypatch.setattr(seg, "_person_matte", lambda im: (None, "missing"))
    src = _scene()
    out, info = restore_background(src, src)
    assert info["applied"] is False
    assert info["reason"] == "no_matte_backend"
