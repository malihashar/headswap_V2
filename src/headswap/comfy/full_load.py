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

Lifecycle (matches ComfyUI's intended load/unload rhythm)
---------------------------------------------------------
1. Before sample: free currently-loaded models (CLIP/VAE) so the UNet fits.
2. During sample: force full UNet residency.
3. After sample: genuinely release the sampling ModelPatcher(s):
   ``unload_all_models()`` alone is not enough — after CFGGuider finishes,
   the UNet may already be absent from ``current_loaded_models`` while our
   pipeline still holds the same ModelPatcher (via ``bundle["model"]`` /
   the guider). Weights can remain on GPU behind that Python reference.
   Comfy's ``get_free_memory`` (= total − active) can also look high while
   ``torch.cuda.mem_get_info`` free stays low because the caching allocator
   still holds reserved blocks. The correct release is Comfy's own
   ``LoadedModel.model_unload()`` / ``ModelPatcher.detach()`` on the patcher
   we own, then ``unload_all_models()`` mop-up (which calls
   ``soft_empty_cache`` when models actually unload).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator


def _free_mb(mm: Any, device: Any) -> float | None:
    try:
        return round(mm.get_free_memory(device) / (1024**2), 1)
    except Exception:
        return None


def _cuda_memory_stats() -> dict[str, float | None]:
    """Allocator truth — distinct from Comfy get_free_memory (total − active)."""
    stats: dict[str, float | None] = {
        "cuda_free_mb": None,
        "cuda_total_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return stats
        free_b, total_b = torch.cuda.mem_get_info()
        stats["cuda_free_mb"] = round(free_b / (1024**2), 1)
        stats["cuda_total_mb"] = round(total_b / (1024**2), 1)
        stats["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        stats["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
    except Exception:
        pass
    return stats


def _loaded_models_snapshot(mm: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lm in list(getattr(mm, "current_loaded_models", []) or []):
        row: dict[str, Any] = {}
        try:
            row["dead"] = bool(lm.is_dead()) if hasattr(lm, "is_dead") else None
            patcher = lm.model
            if patcher is None:
                row["patcher"] = None
            else:
                inner = getattr(patcher, "model", None)
                row["patcher"] = type(patcher).__name__
                row["model_class"] = type(inner).__name__ if inner is not None else None
                try:
                    row["loaded_mb"] = round(lm.model_loaded_memory() / (1024**2), 1)
                except Exception:
                    row["loaded_mb"] = None
                try:
                    row["patcher_id"] = id(patcher)
                except Exception:
                    pass
        except Exception as exc:
            row["error"] = str(exc)
        out.append(row)
    return out


def _detach_patcher(patcher: Any) -> str:
    """Move a ModelPatcher's weights to its offload device (Comfy detach path)."""
    if patcher is None:
        return "skip_none"
    if hasattr(patcher, "detach"):
        patcher.detach(True)
        return "detach"
    if hasattr(patcher, "unpatch_model") and hasattr(patcher, "offload_device"):
        patcher.unpatch_model(patcher.offload_device, unpatch_weights=True)
        return "unpatch_model"
    return "no_detach_api"


def _scrub_broken_loaded_models(mm: Any) -> int:
    """
    Drop LoadedModel entries whose weakref/real_model is already gone.

    After a failed detach under torch.inference_mode, Comfy can leave entries
    where ``real_model`` is None (not a weakref). Later ``cleanup_models_gc``
    then crashes with ``'NoneType' object is not callable``.
    """
    kept: list[Any] = []
    dropped = 0
    for lm in list(getattr(mm, "current_loaded_models", []) or []):
        try:
            rm = getattr(lm, "real_model", None)
            if rm is None or not callable(rm):
                dropped += 1
                continue
            # Touch is_dead so we detect half-detached entries early.
            if hasattr(lm, "is_dead") and lm.is_dead():
                dropped += 1
                continue
            kept.append(lm)
        except Exception:
            dropped += 1
    mm.current_loaded_models[:] = kept
    return dropped


def _force_unload_matching_loaded_models(mm: Any, patchers: list[Any]) -> int:
    """
    Fully unload LoadedModel entries that wrap our patchers.

    free_memory() always passes a numeric memory_to_free, which can take the
    partially_unload branch. Calling model_unload() with no memory budget forces
    the full detach path used when Comfy wants the model completely off GPU.
    """
    if not patchers:
        return 0
    patcher_ids = {id(p) for p in patchers if p is not None}
    unloaded = 0
    kept: list[Any] = []
    for lm in list(getattr(mm, "current_loaded_models", []) or []):
        patcher = None
        try:
            patcher = lm.model
        except Exception:
            patcher = None
        match = patcher is not None and id(patcher) in patcher_ids
        if not match and patcher is not None:
            for p in patchers:
                try:
                    if hasattr(patcher, "is_clone") and patcher.is_clone(p):
                        match = True
                        break
                    if hasattr(p, "is_clone") and p.is_clone(patcher):
                        match = True
                        break
                except Exception:
                    continue
        if match:
            try:
                # memory_to_free=None → skip partial path → full detach
                lm.model_unload()
                unloaded += 1
                continue
            except Exception as exc:
                print(f"[full_load] LoadedModel.model_unload failed: {exc}")
                try:
                    _detach_patcher(patcher)
                    unloaded += 1
                    continue
                except Exception as exc2:
                    print(f"[full_load] patcher detach fallback failed: {exc2}")
        kept.append(lm)
    mm.current_loaded_models[:] = kept
    return unloaded


def offload_gpu_models(
    *,
    reason: str,
    patchers: Iterable[Any] | None = None,
) -> dict:
    """
    Offload GPU-resident models via ComfyUI model_management.

    When ``patchers`` is provided (the sampling UNet ModelPatcher we keep in
    ``bundle["model"]``), those are detached explicitly — this is required when
    they are no longer listed in ``current_loaded_models`` after sampling.
    """
    patcher_list = [p for p in (list(patchers) if patchers is not None else []) if p is not None]
    info: dict = {
        "ok": False,
        "reason": reason,
        "api": None,
        "n_patchers": len(patcher_list),
        "n_forced_loadedmodel_unloads": 0,
        "patcher_detach": [],
        "loaded_before": [],
        "loaded_after": [],
        "error": None,
    }
    info.update({f"before_{k}": v for k, v in _cuda_memory_stats().items()})

    try:
        import comfy.model_management as mm
    except Exception as exc:
        info["error"] = f"import_failed:{exc}"
        print(f"[full_load] offload skip ({reason}) — ComfyUI not available: {exc}")
        return info

    try:
        import gc

        import torch

        device = mm.get_torch_device()
        info["comfy_free_mb_before"] = _free_mb(mm, device)
        info["n_scrubbed_before"] = _scrub_broken_loaded_models(mm)
        info["loaded_before"] = _loaded_models_snapshot(mm)

        # Detach/unload must NOT run under inference_mode (version_counter).
        # Exit any ambient inference_mode for the duration of this cleanup.
        with torch.inference_mode(False):
            # 1) Registry-aware unload of the specific sampling patchers / clones.
            if patcher_list and hasattr(mm, "unload_model_and_clones"):
                for p in patcher_list:
                    try:
                        mm.unload_model_and_clones(p)
                    except Exception as exc:
                        print(f"[full_load] unload_model_and_clones warning: {exc}")
                info["api"] = "unload_model_and_clones+detach"

            # 2) Force full LoadedModel.model_unload() for matching registry entries.
            info["n_forced_loadedmodel_unloads"] = _force_unload_matching_loaded_models(
                mm, patcher_list
            )

            # 3) Always detach the patchers we own — covers the case where the UNet
            #    is no longer in current_loaded_models but weights remain on GPU.
            for p in patcher_list:
                try:
                    info["patcher_detach"].append(
                        {"patcher_id": id(p), "method": _detach_patcher(p)}
                    )
                except Exception as exc:
                    info["patcher_detach"].append(
                        {"patcher_id": id(p), "error": str(exc)}
                    )

            # 4) Mop up anything else still registered (CLIP/VAE leftovers, etc.).
            if hasattr(mm, "unload_all_models"):
                mm.unload_all_models()
                info["api"] = (info.get("api") or "unload_all_models") + "+unload_all_models"
            else:
                mm.free_memory(1e30, device)
                info["api"] = (info.get("api") or "free_memory") + "+free_memory"

            info["n_scrubbed_after"] = _scrub_broken_loaded_models(mm)

            # soft_empty_cache is what free_memory runs after a real unload — needed
            # so reserved CUDA blocks from the fully-resident UNet return to the driver.
            if hasattr(mm, "soft_empty_cache"):
                mm.soft_empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        info["comfy_free_mb_after"] = _free_mb(mm, device)
        info["loaded_after"] = _loaded_models_snapshot(mm)
        after_cuda = _cuda_memory_stats()
        info.update({f"after_{k}": v for k, v in after_cuda.items()})
        info["ok"] = True

        print(
            f"[full_load] offloaded GPU models ({reason}, api={info['api']}) "
            f"comfy_free_mb {info['comfy_free_mb_before']} → {info['comfy_free_mb_after']} | "
            f"cuda_free_mb {info.get('before_cuda_free_mb')} → {info.get('after_cuda_free_mb')} | "
            f"cuda_alloc_mb {info.get('before_cuda_allocated_mb')} → {info.get('after_cuda_allocated_mb')} | "
            f"cuda_reserved_mb {info.get('before_cuda_reserved_mb')} → {info.get('after_cuda_reserved_mb')} | "
            f"loaded_models {len(info['loaded_before'])} → {len(info['loaded_after'])} | "
            f"forced_unloads={info['n_forced_loadedmodel_unloads']} "
            f"scrubbed={info.get('n_scrubbed_before', 0)}+{info.get('n_scrubbed_after', 0)} "
            f"detach={info['patcher_detach']}"
        )
        try:
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as exc:
        info["error"] = str(exc)
        print(f"[full_load] offload warning ({reason}): {exc}")
        # Last-ditch: drop broken registry entries so VAEDecode's cleanup_models_gc
        # does not crash on None real_model.
        try:
            n = _scrub_broken_loaded_models(mm)
            if n:
                print(f"[full_load] scrubbed {n} broken LoadedModel entries after failure")
            if hasattr(mm, "soft_empty_cache"):
                mm.soft_empty_cache()
        except Exception as scrub_exc:
            print(f"[full_load] scrub fallback failed: {scrub_exc}")
    return info


@contextmanager
def force_sampling_full_load(
    models: Iterable[Any] | None = None,
    *,
    enabled: bool = True,
) -> Iterator[dict]:
    """
    Context manager: free idle GPU models, optionally force full UNet load for
    sample, then release the sampling ModelPatcher(s) so VAE decode can allocate.

    Pass the diffusion ModelPatcher(s) as ``models`` (e.g. ``bundle["model"]``)
    so release works even when they are not in ``current_loaded_models``.

    Set ``enabled=False`` on 15GB T4 / fragmented Kaggle sessions — Comfy's
    partial-load path is slower but avoids OOM when UNet+activations cannot
    fully reside (Qwen Image Edit reports ~20GB model_size on T4).
    """
    patchers = [p for p in (list(models) if models is not None else []) if p is not None]
    info: dict = {
        "enabled": False,
        "freed_before_sample": False,
        "freed_after_sample": False,
        "force_full_load": False,
        "n_release_patchers": len(patchers),
        "error": None,
    }

    # Always free CLIP/VAE before sampling for headroom.
    before = offload_gpu_models(reason="before_sample", patchers=None)
    info["freed_before_sample"] = bool(before.get("ok"))
    info["offload_before"] = before
    if before.get("error") and not info.get("error"):
        info["error"] = f"before_sample:{before['error']}"

    if not enabled:
        info["enabled"] = False
        info["force_full_load"] = False
        print(
            "[full_load] skipped (force_full_load=false) — "
            "Comfy partial/low-VRAM path (slower, T4-safe)"
        )
        try:
            yield info
        finally:
            after = offload_gpu_models(reason="after_sample", patchers=patchers)
            info["freed_after_sample"] = bool(after.get("ok"))
            info["offload_after"] = after
            if after.get("error") and not info.get("error"):
                info["error"] = f"after_sample:{after['error']}"
        return

    try:
        import comfy.sampler_helpers as sh
    except Exception as exc:
        info["error"] = f"import_failed:{exc}"
        print(f"[full_load] skip — ComfyUI not available: {exc}")
        yield info
        return

    orig = sh.prepare_sampling
    partial_hits = {"n": 0}

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

    # Belt-and-suspenders: force load_models_gpu(..., force_full_load=True).
    # NOTE: Comfy still *implements* full residency via ModelPatcher.partially_load
    # (one call with a huge memory budget). Do NOT raise on partially_load — that
    # aborts the initial GPU load. Count calls instead: 1 ≈ OK initial load;
    # >> steps ≈ per-step CPU↔GPU streaming.
    mm = None
    orig_load = None
    orig_partial = None
    patcher_cls = None
    prev_vram_state = None
    try:
        import comfy.model_management as mm
        import comfy.model_patcher as model_patcher

        orig_load = mm.load_models_gpu

        def load_models_gpu_full(*args, **kwargs):
            kwargs["force_full_load"] = True
            return orig_load(*args, **kwargs)

        mm.load_models_gpu = load_models_gpu_full

        patcher_cls = model_patcher.ModelPatcher
        orig_partial = patcher_cls.partially_load

        def partially_load_counted(self, *args, **kwargs):
            partial_hits["n"] += 1
            return orig_partial(self, *args, **kwargs)

        patcher_cls.partially_load = partially_load_counted

        # Prefer HIGH_VRAM for the sample window so Comfy's heuristics do not
        # budget for thin partial loads even before our prepare_sampling patch.
        if hasattr(mm, "VRAMState") and hasattr(mm, "vram_state"):
            prev_vram_state = mm.vram_state
            high = getattr(mm.VRAMState, "HIGH_VRAM", None)
            if high is not None:
                mm.vram_state = high
    except Exception as exc:
        print(f"[full_load] extra residency guards skipped: {exc}")

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
        n_partial = int(partial_hits["n"])
        info["partial_load_calls"] = n_partial
        if n_partial <= 2:
            print(
                f"[full_load] residency OK: partially_load called {n_partial}× "
                f"(Comfy's full-load entrypoint; not per-step streaming)"
            )
        else:
            print(
                f"[full_load] WARNING: partially_load called {n_partial}× during "
                f"sample — possible per-step CPU↔GPU streaming"
            )
        sh.prepare_sampling = orig
        if mm is not None and orig_load is not None:
            mm.load_models_gpu = orig_load
        if patcher_cls is not None and orig_partial is not None:
            patcher_cls.partially_load = orig_partial
        if mm is not None and prev_vram_state is not None:
            mm.vram_state = prev_vram_state
        print("[full_load] prepare_sampling patch restored")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        after = offload_gpu_models(reason="after_sample", patchers=patchers)
        info["freed_after_sample"] = bool(after.get("ok"))
        info["offload_after"] = after
        if after.get("error") and not info.get("error"):
            info["error"] = f"after_sample:{after['error']}"
