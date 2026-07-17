"""
VAE decode path probe — must print BEFORE any ModelPatcher weight inspection.

Prior probe built a heavy payload (param iteration / model_size) before printing,
which can reallocate ~14GB and OOM with zero log lines. This module only:

1. Prints CUDA allocator stats (safe)
2. Prints current_loaded_models ids/class names (no weight walks)
3. Monkey-patches the real call chain around VAEDecode
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from typing import Any, Iterator


def _flush() -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def _cuda_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "allocated_mb": None,
        "reserved_mb": None,
        "free_mb": None,
        "total_mb": None,
        "device": None,
        "device_index": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            out["device"] = "cpu"
            return out
        # Do NOT synchronize here — sync can surface deferred CUDA errors/allocs.
        idx = int(torch.cuda.current_device())
        out["device_index"] = idx
        out["device"] = torch.cuda.get_device_name(idx)
        free_b, total_b = torch.cuda.mem_get_info()
        out["free_mb"] = round(free_b / (1024**2), 1)
        out["total_mb"] = round(total_b / (1024**2), 1)
        out["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        out["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _loaded_models_light() -> list[dict[str, Any]]:
    """Ids + class names only — never iterate parameters or call model_size()."""
    rows: list[dict[str, Any]] = []
    try:
        import comfy.model_management as mm
    except Exception as exc:
        return [{"error": str(exc)}]
    for i, lm in enumerate(list(getattr(mm, "current_loaded_models", []) or [])):
        row: dict[str, Any] = {"index": i}
        try:
            row["loaded_model_id"] = id(lm)
            row["dead"] = bool(lm.is_dead()) if hasattr(lm, "is_dead") else None
            row["currently_used"] = getattr(lm, "currently_used", None)
            patcher = lm.model
            row["patcher_id"] = id(patcher) if patcher is not None else None
            row["patcher_type"] = type(patcher).__name__ if patcher is not None else None
            inner = getattr(patcher, "model", None) if patcher is not None else None
            row["model_class"] = type(inner).__name__ if inner is not None else None
            row["model_id"] = id(inner) if inner is not None else None
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def print_vae_probe(label: str, *, bundle: dict | None = None, extra: dict | None = None) -> None:
    """
    Print allocator + loaded-model identity.

    CRITICAL: emit the label line FIRST, before any CUDA API call. A deferred
    CUDA OOM can be raised by mem_get_info/memory_allocated; if we query CUDA
    before printing, the probe appears to "never run".
    """
    # --- always visible, even if the next CUDA call aborts the process ---
    print("", flush=True)
    print(f"[vae_probe] === {label} ===", flush=True)
    if extra is not None:
        try:
            print(f"[vae_probe] extra={extra}", flush=True)
        except Exception as exc:
            print(f"[vae_probe] extra_print_failed={exc}", flush=True)
    _flush()

    try:
        stats = _cuda_stats()
        print(
            f"[vae_probe] cuda device={stats.get('device')!r} idx={stats.get('device_index')} "
            f"allocated_mb={stats.get('allocated_mb')} reserved_mb={stats.get('reserved_mb')} "
            f"free_mb={stats.get('free_mb')} total_mb={stats.get('total_mb')}",
            flush=True,
        )
        if stats.get("error"):
            print(f"[vae_probe] cuda_stats_error={stats['error']}", flush=True)
    except BaseException as exc:
        print(
            f"[vae_probe] cuda_stats RAISED {type(exc).__name__}: {exc}",
            flush=True,
        )
        _flush()
        return

    try:
        loaded = _loaded_models_light()
        print(f"[vae_probe] current_loaded_models count={len(loaded)}", flush=True)
        for row in loaded:
            print(f"[vae_probe]   loaded: {row}", flush=True)
    except BaseException as exc:
        print(
            f"[vae_probe] loaded_models RAISED {type(exc).__name__}: {exc}",
            flush=True,
        )

    if bundle is not None:
        try:
            model = bundle.get("model")
            clip = bundle.get("clip")
            vae = bundle.get("vae")
            vae_patcher = getattr(vae, "patcher", None)
            print(
                f"[vae_probe] bundle ids model={id(model)} clip={id(clip)} vae={id(vae)} "
                f"vae_patcher={id(vae_patcher) if vae_patcher is not None else None}",
                flush=True,
            )
            print(
                f"[vae_probe] types model={type(model).__name__} clip={type(clip).__name__} "
                f"vae={type(vae).__name__} "
                f"vae_patcher={type(vae_patcher).__name__ if vae_patcher else None}",
                flush=True,
            )
            print(
                f"[vae_probe] vae.patcher is bundle['model']: {vae_patcher is model}",
                flush=True,
            )
        except BaseException as exc:
            print(
                f"[vae_probe] bundle_ids RAISED {type(exc).__name__}: {exc}",
                flush=True,
            )

    try:
        import torch

        if torch.cuda.is_available():
            print("[vae_probe] --- memory_summary(abbreviated=True) ---", flush=True)
            print(torch.cuda.memory_summary(abbreviated=True), flush=True)
            print("[vae_probe] --- end memory_summary ---", flush=True)
    except BaseException as exc:
        print(
            f"[vae_probe] memory_summary RAISED {type(exc).__name__}: {exc}",
            flush=True,
        )
    _flush()


def _describe_models_arg(models: Any) -> list[str]:
    labels: list[str] = []
    for m in models or []:
        try:
            inner = getattr(m, "model", None)
            labels.append(
                f"{type(m).__name__}(id={id(m)}, inner={type(inner).__name__ if inner else None}, "
                f"inner_id={id(inner) if inner is not None else None})"
            )
        except Exception as exc:
            labels.append(f"<err:{exc}>")
    return labels


@contextmanager
def install_vae_decode_probes(
    *, bundle: dict | None = None, runtime: Any = None
) -> Iterator[dict]:
    """
    Monkey-patch the live call chain:

      rt.call("VAEDecode")
        -> nodes.VAEDecode.decode
          -> comfy.sd.VAE.decode
            -> model_management.load_models_gpu

    Restores originals on exit.
    """
    state: dict[str, Any] = {"events": [], "installed": False}
    print_vae_probe("install_vae_decode_probes:begin", bundle=bundle)

    try:
        import nodes
        import comfy.sd as comfy_sd
        import comfy.model_management as mm
    except Exception as exc:
        print(f"[vae_probe] install FAILED import: {exc}", flush=True)
        traceback.print_exc()
        _flush()
        yield state
        return

    node_cls = nodes.VAEDecode
    if runtime is not None:
        mapped = getattr(runtime, "mappings", {}).get("VAEDecode")
        if mapped is not None:
            node_cls = mapped
            print(
                f"[vae_probe] patching runtime.mappings['VAEDecode']="
                f"{getattr(mapped, '__name__', mapped)} id={id(mapped)} "
                f"(nodes.VAEDecode id={id(nodes.VAEDecode)})",
                flush=True,
            )

    orig_node_decode = node_cls.decode
    orig_vae_decode = comfy_sd.VAE.decode
    orig_lmg = mm.load_models_gpu

    def wrapped_lmg(models, *args, **kwargs):
        labels = _describe_models_arg(models)
        force_full = kwargs.get("force_full_load", False)
        mem_req = kwargs.get("memory_required", args[0] if args else 0)
        min_mem = kwargs.get("minimum_memory_required")
        print_vae_probe(
            "load_models_gpu ENTER (inside VAE.decode path)",
            bundle=bundle,
            extra={
                "models": labels,
                "force_full_load": force_full,
                "memory_required_mb": (
                    None if mem_req is None else round(float(mem_req) / (1024**2), 1)
                ),
                "minimum_memory_required_mb": (
                    None if min_mem is None else round(float(min_mem) / (1024**2), 1)
                ),
            },
        )
        print("[vae_probe] load_models_gpu stack:", flush=True)
        print("".join(traceback.format_stack(limit=20)), flush=True)
        try:
            out = orig_lmg(models, *args, **kwargs)
        except BaseException as exc:
            print_vae_probe(
                "load_models_gpu RAISED (likely OOM while loading)",
                bundle=bundle,
                extra={
                    "models": labels,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            state["events"].append(
                {"kind": "load_models_gpu_error", "models": labels, "error": str(exc)}
            )
            raise
        print_vae_probe(
            "load_models_gpu EXIT (returned OK)",
            bundle=bundle,
            extra={"models": labels},
        )
        state["events"].append({"kind": "load_models_gpu_ok", "models": labels})
        return out

    def wrapped_node_decode(self, vae, samples):
        print_vae_probe(
            "nodes.VAEDecode.decode ENTER (first line)",
            bundle=bundle,
            extra={
                "vae_id": id(vae),
                "vae_type": type(vae).__name__,
                "samples_type": type(samples).__name__,
            },
        )
        try:
            return orig_node_decode(self, vae, samples)
        except BaseException as exc:
            print_vae_probe(
                "nodes.VAEDecode.decode RAISED",
                bundle=bundle,
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    def wrapped_vae_decode(self, samples_in, vae_options=None):
        if vae_options is None:
            vae_options = {}
        print_vae_probe(
            "comfy.sd.VAE.decode ENTER (first line)",
            bundle=bundle,
            extra={
                "vae_self_id": id(self),
                "patcher_id": id(getattr(self, "patcher", None)),
                "samples_in_type": type(samples_in).__name__,
                "samples_in_shape": list(samples_in.shape)
                if hasattr(samples_in, "shape")
                else None,
                "samples_in_device": str(getattr(samples_in, "device", None)),
            },
        )
        # Keep load_models_gpu wrapped for the duration of this decode.
        prev_lmg = mm.load_models_gpu
        mm.load_models_gpu = wrapped_lmg
        try:
            return orig_vae_decode(self, samples_in, vae_options)
        except BaseException as exc:
            print_vae_probe(
                "comfy.sd.VAE.decode RAISED",
                bundle=bundle,
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            mm.load_models_gpu = prev_lmg

    node_cls.decode = wrapped_node_decode
    comfy_sd.VAE.decode = wrapped_vae_decode
    mm.load_models_gpu = wrapped_lmg
    state["installed"] = True
    print(
        "[vae_probe] monkey-patches installed: VAEDecode.decode, VAE.decode, load_models_gpu",
        flush=True,
    )
    _flush()

    try:
        yield state
    finally:
        node_cls.decode = orig_node_decode
        comfy_sd.VAE.decode = orig_vae_decode
        mm.load_models_gpu = orig_lmg
        print("[vae_probe] monkey-patches restored", flush=True)
        _flush()
