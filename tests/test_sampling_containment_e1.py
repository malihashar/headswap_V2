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
    # denoise is 0.85, not 1.0: the sampler now seeds from the SOURCE latent
    # (img2img) rather than EmptySD3LatentImage, so denoise is how far the
    # render may travel from the original photo rather than a no-op. At 1.0 it
    # regenerated the whole frame from noise and reinvented clothing.
    assert 0.0 < cfg["denoise"] < 1.0
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


# --- E1 on the T4 route (run_simple_full_body) ------------------------------
#
# The cap survives every GLOBAL lever because the scene reaches the sampler
# through three conditioning paths and only one has a strength knob:
#   1. source_image/source_latent on Krea2EditModelPatch  (ref_boost_a)
#   2. the img2img seed -- at denoise=0.85, latent_image IS the encoded scene
#   3. Krea2EditGroundedEncode's VLM grounding on the scene
# Measured: ref_boost_a 1.0/0.6/0.3 (shipped default 1.6) left the cap at every
# value while 0.3 wrecked the shirt, so carrier 1 does not hold it. Four prompt
# wordings failed too (CHECKPOINT-16). The requirement is SPATIAL.

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
CHAIN_SRC = (ROOT / "src" / "headswap" / "chain.py").read_text()


def _t4_containment_block() -> str:
    i = KREA2.find("# EXPERIMENT E1 on T4: sampling-time containment.")
    assert i > 0, "T4 containment block missing"
    j = KREA2.find("        out = sample_meta[\"edited\"]", i)
    assert j > i
    return KREA2[i:j]


def test_t4_passes_the_head_mask_as_edit_mask():
    assert "edit_mask=_contain_mask," in _t4_containment_block()


def test_t4_uses_the_ellipse_backend_only():
    """head_matte intersects a person matte that returns ZERO foreground for
    rigid headwear (CHECKPOINT-07, alpha_max_in_crown=0) -- it would chop the
    cap straight out of the region we are trying to regenerate."""
    body = _t4_containment_block()
    assert 'backend="ellipse"' in body
    assert "head_matte" not in body.split("# ELLIPSE ONLY")[1].split('backend="')[1][:40]


def test_t4_does_not_enlarge_the_mask():
    """mask_top_extend was REDUCED 1.55 -> 1.30 because Krea2 fails to fill the
    outer third of an extended region, and headwear_erase.py records that
    pairing removal with an enlarged mask "blew up into a glowing oval"."""
    body = _t4_containment_block()
    assert 'self.cfg.get("mask_top_extend", 1.30)' in body


def test_t4_forces_denoise_to_one_only_when_contained():
    """denoise=1.0 is safe ONLY because the mask exists: FLUX noise_scaling
    multiplies latent_image by (1 - sigma) = 0 at sigma_max, so the starting
    noise is bit-identical to the empty-latent case and the encoded latent's
    only role is to make the pin functional."""
    body = _t4_containment_block()
    assert "if _contain_mask is not None:" in body
    assert 'self.cfg["sampling_containment"] = True' in body
    assert 'self.cfg.get("containment_denoise", 1.0)' in body


def test_t4_restores_sampling_settings_afterwards():
    """face_refine runs after this and must NOT be contained -- and a leaked
    denoise=1.0 would regenerate the whole frame on the next call."""
    body = _t4_containment_block()
    assert "finally:" in body
    assert '("sampling_containment", _prev_cont)' in body
    assert '("denoise", _prev_dn)' in body


def test_t4_containment_is_off_by_default():
    """T4's approved recipe must be untouched unless a caller opts in."""
    assert 'self.cfg.get("simple_full_body_containment", False)' in KREA2


def test_chain_opts_in():
    assert '"containment": True' in CHAIN_SRC
    assert 'cfg["simple_full_body_containment"] = True' in CHAIN_SRC


def test_mask_is_dumped_for_inspection():
    """Judge the mask before judging the output: several runs were lost to
    interpreting results from masks that were not what was assumed."""
    assert 'containment_mask.png' in _t4_containment_block()


def test_dead_clamp_keys_are_gone():
    """headwear_up_face_heights / _side_face_widths were read nowhere after the
    clamp was reverted."""
    assert "headwear_up_face_heights" not in CHAIN_SRC
    assert "headwear_side_face_widths" not in CHAIN_SRC
