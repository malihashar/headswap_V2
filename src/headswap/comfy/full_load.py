"""
Force full GPU residency for the diffusion UNet during sampling.

Why this exists
---------------
ComfyUI's default NORMAL_VRAM path in load_models_gpu computes::

    lowvram_model_memory = free_vram - minimum_memory_required

``minimum_memory_required`` comes from estimate_memory(), which for Qwen Image
Edit includes dual ~1024² reference latents from TextEncodeQwenImageEditPlus.
On a 16GB T4 that reservation often leaves only a few GB for weights, so
ModelPatcher.partially_load streams layers CPU↔GPU every denoising step
(~minutes/step instead of ~seconds/step).

Colab Cell 5 (~2.8 s/step) runs with the UNet fully resident. Forcing
``force_full_load=True`` on prepare_sampling skips the partial-load branch
(model_load uses the full-weight path) without changing weights, dtype,
sampler, or graph — quality is unchanged; only residency changes.

We also free currently-loaded models (CLIP/VAE) before sampling so the UNet
has room to fully reside on a 16GB card.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def force_sampling_full_load() -> Iterator[dict]:
    """
    Context manager: unload idle GPU models, then force full UNet load for sample.

    Patches ``comfy.sampler_helpers.prepare_sampling`` to pass
    ``force_full_load=True``. Restores the original on exit.
    """
    info: dict = {
        "enabled": False,
        "freed_before_sample": False,
        "force_full_load": False,
        "error": None,
    }
    try:
        import comfy.model_management as mm
        import comfy.sampler_helpers as sh
    except Exception as exc:
        info["error"] = f"import_failed:{exc}"
        print(f"[full_load] skip — ComfyUI not available: {exc}")
        yield info
        return

    # Offload whatever is currently on GPU (typically CLIP/VAE after text encode)
    # so free_vram can hold the full UNet. Does not delete ModelPatcher objects.
    try:
        device = mm.get_torch_device()
        free_before = None
        try:
            free_before = round(mm.get_free_memory(device) / (1024**2), 1)
        except Exception:
            pass
        mm.free_memory(1e30, device)
        free_after = None
        try:
            free_after = round(mm.get_free_memory(device) / (1024**2), 1)
        except Exception:
            pass
        info["freed_before_sample"] = True
        info["free_vram_mb_before_free"] = free_before
        info["free_vram_mb_after_free"] = free_after
        print(
            f"[full_load] freed GPU residents before sample "
            f"(free_vram_mb {free_before} → {free_after})"
        )
        try:
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as exc:
        info["error"] = f"free_memory_failed:{exc}"
        print(f"[full_load] free_memory warning: {exc}")

    orig = sh.prepare_sampling

    def prepare_sampling_full(
        model,
        noise_shape,
        conds,
        model_options=None,
        force_full_load=False,
        force_offload=False,
    ):
        # Always force full residency for this sampling call — ignore caller flag.
        return orig(
            model,
            noise_shape,
            conds,
            model_options=model_options,
            force_full_load=True,
            force_offload=force_offload,
        )

    sh.prepare_sampling = prepare_sampling_full
    info["enabled"] = True
    info["force_full_load"] = True
    print("[full_load] prepare_sampling patched with force_full_load=True")
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        yield info
    finally:
        sh.prepare_sampling = orig
        print("[full_load] prepare_sampling patch restored")
        try:
            sys.stdout.flush()
        except Exception:
            pass
