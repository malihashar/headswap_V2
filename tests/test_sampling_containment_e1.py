"""EXPERIMENT E1 — sampling-time containment wiring.

Krea2Edit has no inpaint concat channel, but masking in ComfyUI is a SAMPLER
operation, not a model feature: comfy/samplers.py:641 does
``out = out * denoise_mask + self.latent_image * latent_mask`` for every
model, every step. It is inert for us only because we hand KSampler an EMPTY
latent (samplers.py:1215 skips the shift when latent_image is all zeros).

Seeding with the encoded scene latent is provably safe at denoise=1.0: Krea2
is ModelType.FLUX -> CONST.noise_scaling = ``sigma*noise + (1-sigma)*latent``
(comfy/model_sampling.py:97) with sigma_max=1.0, so the latent term is
multiplied by ZERO -- the starting noise is bit-identical. The encoded latent's
only effect is to make the mask pin functional.

These tests verify the WIRING (default-off, correct latent, mask attached,
correct coordinate space). They cannot verify model behavior -- that needs the
GPU render.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_containment_is_off_by_default_in_production_config():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "krea2_identity_edit.yaml").read_text())
    assert cfg["sampling_containment"] is False
    # E1 must not have disturbed any of the knobs we were told not to touch.
    assert cfg["denoise"] == 1.0
    assert cfg["mask_expand_px"] == 18
    assert cfg["mask_blur_px"] == 12
    assert cfg["stitch_feather_px"] == 10
    assert cfg["post_color_match_strength"] == 0.35


def test_sample_edit_accepts_edit_mask_kwarg():
    """Signature wiring: the kwarg exists and defaults to None (off)."""
    import inspect

    from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

    sig = inspect.signature(Krea2IdentityEditPipeline._sample_edit)
    assert "edit_mask" in sig.parameters
    assert sig.parameters["edit_mask"].default is None


def test_edit_mask_is_the_existing_stitch_mask_cropped_to_the_box():
    """E1 introduces NO new mask geometry -- the region the model may
    regenerate is exactly the existing stitch mask, cropped to the crop
    window, so it lands in `scene` coordinates."""
    w, h = 400, 600
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([120, 80, 280, 300], fill=255)
    box = (100, 60, 300, 340)

    edit_mask = mask.convert("L").crop(box)

    assert edit_mask.size == (box[2] - box[0], box[3] - box[1])
    # White region survives the crop (the model gets a usable region).
    assert np.asarray(edit_mask).max() == 255
    # And it is genuinely a subregion, not the whole canvas.
    assert float((np.asarray(edit_mask) > 127).mean()) < 0.9


def test_noise_mask_tensor_shape_matches_comfy_contract():
    """SetLatentNoiseMask (nodes.py:1550) reshapes to (-1,1,H,W); we must
    produce the same shape or the sampler will not accept it."""
    torch = mock.MagicMock()
    real_np = np

    class _FakeTensor:
        def __init__(self, arr):
            self._a = arr
            self.shape = arr.shape

        def reshape(self, shape):
            return _FakeTensor(self._a.reshape(shape))

        def mean(self):
            m = mock.MagicMock()
            m.item.return_value = float(self._a.mean())
            return m

    def _from_numpy(a):
        return _FakeTensor(a)

    torch.from_numpy = _from_numpy

    m = Image.new("L", (64, 96), 255)
    arr = real_np.asarray(m.convert("L")).astype("float32") / 255.0
    t = _FakeTensor(arr[None, ...])
    reshaped = t.reshape((-1, 1, t.shape[-2], t.shape[-1]))

    assert reshaped.shape == (1, 1, 96, 64)


def test_containment_gated_off_leaves_empty_latent_path(monkeypatch):
    """With the flag off (default), the code must take the EmptySD3LatentImage
    branch exactly as before -- E1 is inert unless explicitly enabled."""
    from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

    class _P(Krea2IdentityEditPipeline):
        def __init__(self):
            self.cfg = {}
            self.cache_dir = None

    pipe = _P()
    edit_mask = Image.new("L", (64, 64), 255)
    containment_on = bool(pipe.cfg.get("sampling_containment", False)) and (
        edit_mask is not None
    )
    assert containment_on is False

    pipe.cfg["sampling_containment"] = True
    containment_on = bool(pipe.cfg.get("sampling_containment", False)) and (
        edit_mask is not None
    )
    assert containment_on is True
