"""
Bridge probe: find what reacquires VRAM between sampling unload and VAEDecode.

Diagnostic only — does not change model settings or sampling behavior.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from typing import Any, Iterator


def _flush() -> None:
    try:
        sys.stdout.flush()
    except Exception:
        pass


def cuda_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cuda_available": False,
        "device": None,
        "device_index": None,
        "allocated_mb": None,
        "reserved_mb": None,
        "free_mb": None,
        "total_mb": None,
        "max_allocated_mb": None,
    }
    try:
        import torch

        out["cuda_available"] = bool(torch.cuda.is_available())
        if not out["cuda_available"]:
            return out
        idx = torch.cuda.current_device()
        out["device_index"] = idx
        out["device"] = torch.cuda.get_device_name(idx)
        free_b, total_b = torch.cuda.mem_get_info()
        out["free_mb"] = round(free_b / (1024**2), 1)
        out["total_mb"] = round(total_b / (1024**2), 1)
        out["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        out["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 1)
        out["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def memory_summary_abbrev() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_summary(abbreviated=True)
    except Exception as exc:
        return f"<memory_summary failed: {exc}>"


def patcher_weight_devices(patcher: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "patcher_type": type(patcher).__name__ if patcher is not None else None,
        "patcher_id": id(patcher) if patcher is not None else None,
        "model_class": None,
        "model_device_attr": None,
        "param_devices": {},
        "n_params": 0,
        "n_cuda_params": 0,
        "loaded_weight_mb": None,
        "model_size_mb": None,
        "parent_id": None,
        "clone_base_uuid": None,
    }
    if patcher is None:
        return info
    try:
        inner = getattr(patcher, "model", None)
        info["model_class"] = type(inner).__name__ if inner is not None else None
        info["model_device_attr"] = str(getattr(inner, "device", None))
        info["parent_id"] = id(getattr(patcher, "parent", None)) if getattr(patcher, "parent", None) else None
        info["clone_base_uuid"] = str(getattr(patcher, "clone_base_uuid", None))
        try:
            info["loaded_weight_mb"] = round(
                float(getattr(inner, "model_loaded_weight_memory", 0) or 0) / (1024**2), 1
            )
        except Exception:
            pass
        try:
            info["model_size_mb"] = round(float(patcher.model_size()) / (1024**2), 1)
        except Exception:
            pass
        counts: dict[str, int] = {}
        n_cuda = 0
        n = 0
        if inner is not None:
            for p in inner.parameters():
                n += 1
                key = str(p.device)
                counts[key] = counts.get(key, 0) + 1
                if getattr(p.device, "type", None) == "cuda":
                    n_cuda += 1
        info["param_devices"] = counts
        info["n_params"] = n
        info["n_cuda_params"] = n_cuda
    except Exception as exc:
        info["error"] = str(exc)
    return info


def loaded_models_detail() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import comfy.model_management as mm
    except Exception as exc:
        return [{"error": f"import_failed:{exc}"}]
    for i, lm in enumerate(list(getattr(mm, "current_loaded_models", []) or [])):
        row: dict[str, Any] = {"index": i}
        try:
            row["dead"] = bool(lm.is_dead()) if hasattr(lm, "is_dead") else None
            row["currently_used"] = bool(getattr(lm, "currently_used", None))
            patcher = lm.model
            row["patcher_id"] = id(patcher) if patcher is not None else None
            row.update(patcher_weight_devices(patcher))
            try:
                row["loaded_mb"] = round(lm.model_loaded_memory() / (1024**2), 1)
            except Exception as exc:
                row["loaded_mb_error"] = str(exc)
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def describe_models_arg(models: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in models or []:
        out.append(patcher_weight_devices(m))
    return out


def print_bridge_checkpoint(label: str, *, bundle: dict | None = None, extra: dict | None = None) -> dict:
    """Print a full CUDA / ModelPatcher checkpoint and return the payload."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    snap = cuda_snapshot()
    payload: dict[str, Any] = {
        "label": label,
        "cuda": snap,
        "loaded_models_count": None,
        "loaded_models": [],
        "bundle_ids": None,
        "unet": None,
        "clip": None,
        "vae": None,
        "vae_patcher": None,
        "extra": extra or {},
    }
    try:
        import comfy.model_management as mm

        payload["loaded_models"] = loaded_models_detail()
        payload["loaded_models_count"] = len(payload["loaded_models"])
        payload["comfy_free_mb"] = round(mm.get_free_memory(mm.get_torch_device()) / (1024**2), 1)
    except Exception as exc:
        payload["loaded_models_error"] = str(exc)

    if bundle is not None:
        payload["bundle_ids"] = {
            "model": id(bundle.get("model")),
            "clip": id(bundle.get("clip")),
            "vae": id(bundle.get("vae")),
        }
        payload["unet"] = patcher_weight_devices(bundle.get("model"))
        clip = bundle.get("clip")
        payload["clip"] = {
            "id": id(clip) if clip is not None else None,
            "type": type(clip).__name__ if clip is not None else None,
            "patcher": patcher_weight_devices(getattr(clip, "patcher", None)),
        }
        vae = bundle.get("vae")
        payload["vae"] = {
            "id": id(vae) if vae is not None else None,
            "type": type(vae).__name__ if vae is not None else None,
        }
        payload["vae_patcher"] = patcher_weight_devices(getattr(vae, "patcher", None))

    print()
    print("=" * 72)
    print(f"[vae_bridge] CHECKPOINT: {label}")
    print(
        f"  cuda device={snap.get('device')!r} idx={snap.get('device_index')} | "
        f"allocated_mb={snap.get('allocated_mb')} reserved_mb={snap.get('reserved_mb')} | "
        f"free_mb={snap.get('free_mb')} total_mb={snap.get('total_mb')}"
    )
    if "comfy_free_mb" in payload:
        print(f"  comfy_free_mb={payload.get('comfy_free_mb')}")
    print(f"  current_loaded_models count={payload.get('loaded_models_count')}")
    for row in payload.get("loaded_models") or []:
        print(
            f"    [{row.get('index')}] class={row.get('model_class')} "
            f"patcher_id={row.get('patcher_id')} loaded_mb={row.get('loaded_mb')} "
            f"n_cuda_params={row.get('n_cuda_params')}/{row.get('n_params')} "
            f"param_devices={row.get('param_devices')} dead={row.get('dead')} "
            f"used={row.get('currently_used')}"
        )
    if payload.get("bundle_ids"):
        print(f"  bundle ids: {payload['bundle_ids']}")
    if payload.get("unet"):
        u = payload["unet"]
        print(
            f"  UNet patcher: class={u.get('model_class')} id={u.get('patcher_id')} "
            f"size_mb={u.get('model_size_mb')} loaded_weight_mb={u.get('loaded_weight_mb')} "
            f"n_cuda_params={u.get('n_cuda_params')}/{u.get('n_params')} "
            f"param_devices={u.get('param_devices')} parent_id={u.get('parent_id')}"
        )
        on_gpu = (u.get("n_cuda_params") or 0) > 0
        print(f"  UNet still has CUDA params: {on_gpu}")
    if payload.get("vae_patcher"):
        v = payload["vae_patcher"]
        print(
            f"  VAE patcher: class={v.get('model_class')} id={v.get('patcher_id')} "
            f"n_cuda_params={v.get('n_cuda_params')}/{v.get('n_params')} "
            f"param_devices={v.get('param_devices')}"
        )
    if extra:
        print(f"  extra: {extra}")
    summary = memory_summary_abbrev()
    if summary:
        print("  --- torch.cuda.memory_summary(abbreviated=True) ---")
        print(summary)
        print("  --- end memory_summary ---")
    print("=" * 72)
    _flush()
    return payload


@contextmanager
def watch_load_models_gpu(label: str = "vae_decode") -> Iterator[list[dict]]:
    """
    Wrap comfy.model_management.load_models_gpu and ModelPatcher.partially_load
    to log every load during VAE decode, with stack traces and CUDA deltas.
    """
    events: list[dict] = []
    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except Exception as exc:
        print(f"[vae_bridge] watch_load_models_gpu skip: {exc}")
        yield events
        return

    orig_lmg = mm.load_models_gpu
    orig_pl = mp.ModelPatcher.partially_load
    # Dynamic patcher subclass may have its own partially_load
    orig_dyn_pl = getattr(getattr(mp, "CoreModelPatcher", object), "partially_load", None)

    def _record(kind: str, **fields: Any) -> None:
        ev = {"kind": kind, "label": label, **fields}
        events.append(ev)

    def wrapped_lmg(models, *args, **kwargs):
        before = cuda_snapshot()
        model_info = describe_models_arg(models)
        force_full = kwargs.get("force_full_load", False)
        mem_req = kwargs.get("memory_required", args[0] if args else 0)
        min_mem = kwargs.get("minimum_memory_required")
        stack = "".join(traceback.format_stack(limit=18))
        print()
        print(f"[vae_bridge] load_models_gpu ENTER ({label})")
        print(
            f"  force_full_load={force_full} "
            f"memory_required_mb={None if mem_req is None else round(float(mem_req)/(1024**2),1)} "
            f"minimum_memory_required_mb="
            f"{None if min_mem is None else round(float(min_mem)/(1024**2),1)}"
        )
        print(f"  cuda before: alloc={before.get('allocated_mb')} free={before.get('free_mb')} "
              f"reserved={before.get('reserved_mb')}")
        for i, mi in enumerate(model_info):
            print(
                f"  model[{i}]: class={mi.get('model_class')} patcher_id={mi.get('patcher_id')} "
                f"size_mb={mi.get('model_size_mb')} n_cuda_params={mi.get('n_cuda_params')}/"
                f"{mi.get('n_params')} param_devices={mi.get('param_devices')}"
            )
        print("  --- stack ---")
        print(stack)
        print("  --- end stack ---")
        _flush()
        try:
            out = orig_lmg(models, *args, **kwargs)
        except Exception as exc:
            after_e = cuda_snapshot()
            print(f"[vae_bridge] load_models_gpu RAISED {type(exc).__name__}: {exc}")
            print(
                f"  cuda after raise: alloc={after_e.get('allocated_mb')} "
                f"free={after_e.get('free_mb')} reserved={after_e.get('reserved_mb')}"
            )
            _flush()
            _record(
                "load_models_gpu_error",
                models=model_info,
                before=before,
                after=after_e,
                error=str(exc),
                stack=stack,
            )
            raise
        after = cuda_snapshot()
        delta = None
        try:
            if before.get("allocated_mb") is not None and after.get("allocated_mb") is not None:
                delta = round(after["allocated_mb"] - before["allocated_mb"], 1)
        except Exception:
            pass
        print(f"[vae_bridge] load_models_gpu EXIT ({label}) delta_alloc_mb={delta}")
        print(
            f"  cuda after: alloc={after.get('allocated_mb')} free={after.get('free_mb')} "
            f"reserved={after.get('reserved_mb')}"
        )
        print(f"  loaded_models now: {len(getattr(mm, 'current_loaded_models', []) or [])}")
        for row in loaded_models_detail():
            print(
                f"    class={row.get('model_class')} patcher_id={row.get('patcher_id')} "
                f"loaded_mb={row.get('loaded_mb')} n_cuda_params={row.get('n_cuda_params')}"
            )
        _flush()
        _record(
            "load_models_gpu",
            models=model_info,
            before=before,
            after=after,
            delta_alloc_mb=delta,
            force_full_load=force_full,
            stack=stack,
        )
        return out

    def wrapped_pl(self_mp, device_to, extra_memory=0, force_patch_weights=False):
        before = cuda_snapshot()
        info = patcher_weight_devices(self_mp)
        stack = "".join(traceback.format_stack(limit=16))
        print()
        print(f"[vae_bridge] ModelPatcher.partially_load ENTER ({label})")
        print(
            f"  class={info.get('model_class')} patcher_id={info.get('patcher_id')} "
            f"device_to={device_to} extra_memory_mb="
            f"{round(float(extra_memory)/(1024**2),1) if extra_memory else 0} "
            f"size_mb={info.get('model_size_mb')} "
            f"n_cuda_params={info.get('n_cuda_params')}/{info.get('n_params')}"
        )
        print(
            f"  cuda before: alloc={before.get('allocated_mb')} free={before.get('free_mb')} "
            f"reserved={before.get('reserved_mb')}"
        )
        print("  --- stack ---")
        print(stack)
        print("  --- end stack ---")
        _flush()
        result = orig_pl(self_mp, device_to, extra_memory, force_patch_weights)
        after = cuda_snapshot()
        after_info = patcher_weight_devices(self_mp)
        delta = None
        try:
            if before.get("allocated_mb") is not None and after.get("allocated_mb") is not None:
                delta = round(after["allocated_mb"] - before["allocated_mb"], 1)
        except Exception:
            pass
        print(
            f"[vae_bridge] ModelPatcher.partially_load EXIT ({label}) delta_alloc_mb={delta} "
            f"result={result} n_cuda_params={after_info.get('n_cuda_params')}/"
            f"{after_info.get('n_params')}"
        )
        print(
            f"  cuda after: alloc={after.get('allocated_mb')} free={after.get('free_mb')} "
            f"reserved={after.get('reserved_mb')}"
        )
        _flush()
        _record(
            "partially_load",
            model=info,
            after_model=after_info,
            before=before,
            after=after,
            delta_alloc_mb=delta,
            extra_memory=extra_memory,
            result=result,
            stack=stack,
        )
        return result

    mm.load_models_gpu = wrapped_lmg
    mp.ModelPatcher.partially_load = wrapped_pl
    core = getattr(mp, "CoreModelPatcher", None)
    if core is not None and orig_dyn_pl is not None and core.partially_load is not orig_pl:
        # Keep dynamic path visible too if it overrides.
        def wrapped_dyn_pl(self_mp, device_to, extra_memory=0, force_patch_weights=False):
            return wrapped_pl(self_mp, device_to, extra_memory, force_patch_weights)

        # Prefer wrapping via calling orig_dyn through a thin logger
        def wrapped_dyn(self_mp, device_to, extra_memory=0, force_patch_weights=False):
            before = cuda_snapshot()
            info = patcher_weight_devices(self_mp)
            stack = "".join(traceback.format_stack(limit=16))
            print()
            print(f"[vae_bridge] CoreModelPatcher.partially_load ENTER ({label})")
            print(
                f"  class={info.get('model_class')} patcher_id={info.get('patcher_id')} "
                f"device_to={device_to} extra_mb="
                f"{round(float(extra_memory)/(1024**2),1) if extra_memory else 0}"
            )
            print(f"  cuda before alloc={before.get('allocated_mb')} free={before.get('free_mb')}")
            print(stack)
            _flush()
            result = orig_dyn_pl(self_mp, device_to, extra_memory, force_patch_weights)
            after = cuda_snapshot()
            print(
                f"[vae_bridge] CoreModelPatcher.partially_load EXIT alloc={after.get('allocated_mb')} "
                f"free={after.get('free_mb')}"
            )
            _flush()
            return result

        core.partially_load = wrapped_dyn

    print(f"[vae_bridge] watching load_models_gpu/partially_load for '{label}'")
    _flush()
    try:
        yield events
    finally:
        mm.load_models_gpu = orig_lmg
        mp.ModelPatcher.partially_load = orig_pl
        if core is not None and orig_dyn_pl is not None:
            core.partially_load = orig_dyn_pl
        print(f"[vae_bridge] watch restored ({label}), events={len(events)}")
        _flush()
