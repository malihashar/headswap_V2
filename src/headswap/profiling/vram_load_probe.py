"""
Verification-only probe: detect ComfyUI full vs partial UNet residency.

Does not change VRAM policy, force_full_load, attention backends, or sampler
settings. Only observes load_models_gpu / ModelPatcher.partially_load / load
and captures ComfyUI's own "loaded partially/completely" log lines.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


_PARTIAL_RE = re.compile(
    r"loaded partially;\s*(?:([\d.]+)\s*MB usable,)?\s*"
    r"([\d.]+)\s*MB loaded,\s*([\d.]+)\s*MB offloaded"
    r"(?:,\s*([\d.]+)\s*MB buffer reserved)?(?:,\s*lowvram patches:\s*(\d+))?",
    re.IGNORECASE,
)
_COMPLETE_RE = re.compile(
    r"loaded completely;\s*(?:([\d.]+)\s*MB usable,)?\s*"
    r"([\d.]+)\s*MB loaded,\s*full load:\s*(\w+)",
    re.IGNORECASE,
)


def _free_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return round(free / (1024**2), 1)
    except Exception:
        return None


def _alloc_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / (1024**2), 1)
    except Exception:
        return None


@dataclass
class LoadEvent:
    kind: str  # "partial" | "complete" | "load_models_gpu" | "partially_load"
    message: str = ""
    mb_loaded: float | None = None
    mb_offloaded: float | None = None
    mb_usable: float | None = None
    mb_buffer: float | None = None
    lowvram_patches: int | None = None
    full_load_flag: bool | None = None
    free_vram_mb_before: float | None = None
    free_vram_mb_after: float | None = None
    alloc_vram_mb_before: float | None = None
    alloc_vram_mb_after: float | None = None
    lowvram_model_memory_mb: float | None = None
    model_class: str | None = None
    code_path: str = ""
    t_s: float = 0.0


@dataclass
class VramLoadProbeReport:
    events: list[LoadEvent] = field(default_factory=list)
    free_vram_mb_at_install: float | None = None
    sampling_load_mode: str | None = None  # "partial" | "complete" | "unknown"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_vram_mb_at_install": self.free_vram_mb_at_install,
            "sampling_load_mode": self.sampling_load_mode,
            "summary": self.summary,
            "events": [asdict(e) for e in self.events],
        }


class _ComfyLoadLogHandler(logging.Handler):
    def __init__(self, probe: "VramLoadProbe") -> None:
        super().__init__(level=logging.INFO)
        self.probe = probe

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "loaded partially" not in msg and "loaded completely" not in msg:
            return
        self.probe._ingest_log_line(msg)


class VramLoadProbe:
    def __init__(self) -> None:
        self.report = VramLoadProbeReport()
        self._handler: _ComfyLoadLogHandler | None = None
        self._orig_load_models_gpu = None
        self._orig_partially_load = None
        self._orig_mp_load = None
        self._installed = False

    def _ingest_log_line(self, msg: str) -> None:
        m = _PARTIAL_RE.search(msg)
        if m:
            ev = LoadEvent(
                kind="partial",
                message=msg.strip(),
                mb_usable=float(m.group(1)) if m.group(1) else None,
                mb_loaded=float(m.group(2)),
                mb_offloaded=float(m.group(3)),
                mb_buffer=float(m.group(4)) if m.group(4) else None,
                lowvram_patches=int(m.group(5)) if m.group(5) else None,
                free_vram_mb_after=_free_vram_mb(),
                alloc_vram_mb_after=_alloc_vram_mb(),
                code_path="ModelPatcher.load → logging.info(loaded partially)",
                t_s=time.perf_counter(),
            )
            self.report.events.append(ev)
            self.report.sampling_load_mode = "partial"
            self._print_event(ev)
            return
        m = _COMPLETE_RE.search(msg)
        if m:
            full_flag = str(m.group(3)).lower() in ("true", "1", "yes")
            ev = LoadEvent(
                kind="complete",
                message=msg.strip(),
                mb_usable=float(m.group(1)) if m.group(1) else None,
                mb_loaded=float(m.group(2)),
                mb_offloaded=0.0,
                full_load_flag=full_flag,
                free_vram_mb_after=_free_vram_mb(),
                alloc_vram_mb_after=_alloc_vram_mb(),
                code_path="ModelPatcher.load → logging.info(loaded completely)",
                t_s=time.perf_counter(),
            )
            self.report.events.append(ev)
            # Prefer partial if we already saw one; else complete.
            if self.report.sampling_load_mode != "partial":
                self.report.sampling_load_mode = "complete"
            self._print_event(ev)

    @staticmethod
    def _print_event(ev: LoadEvent) -> None:
        print()
        print("=" * 72)
        print("[vram_load_probe] ComfyUI model residency")
        if ev.kind == "partial":
            print("  mode:           PARTIAL LOAD (CPU↔GPU weight streaming)")
            print(f"  MB loaded:      {ev.mb_loaded}")
            print(f"  MB offloaded:   {ev.mb_offloaded}")
            if ev.mb_usable is not None:
                print(f"  MB usable budg: {ev.mb_usable}")
            if ev.mb_buffer is not None:
                print(f"  MB buffer:      {ev.mb_buffer}")
            if ev.lowvram_patches is not None:
                print(f"  lowvram patches:{ev.lowvram_patches}")
        elif ev.kind == "complete":
            print("  mode:           FULL / COMPLETE LOAD")
            print(f"  MB loaded:      {ev.mb_loaded}")
            print(f"  MB offloaded:   {ev.mb_offloaded}")
            print(f"  full_load flag: {ev.full_load_flag}")
        else:
            print(f"  kind:           {ev.kind}")
        if ev.free_vram_mb_before is not None:
            print(f"  free VRAM before: {ev.free_vram_mb_before} MB")
        if ev.free_vram_mb_after is not None:
            print(f"  free VRAM after:  {ev.free_vram_mb_after} MB")
        if ev.alloc_vram_mb_after is not None:
            print(f"  alloc VRAM after: {ev.alloc_vram_mb_after} MB")
        if ev.lowvram_model_memory_mb is not None:
            print(f"  lowvram_model_memory arg: {ev.lowvram_model_memory_mb} MB")
        if ev.model_class:
            print(f"  model class:    {ev.model_class}")
        print(f"  code path:      {ev.code_path}")
        if ev.message:
            print(f"  raw log:        {ev.message}")
        print("=" * 72)
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def install(self) -> bool:
        if self._installed:
            return True
        self.report.free_vram_mb_at_install = _free_vram_mb()
        print()
        print(
            f"[vram_load_probe] installed — free VRAM now: "
            f"{self.report.free_vram_mb_at_install} MB "
            "(will report full vs partial on next UNet load)"
        )
        try:
            sys.stdout.flush()
        except Exception:
            pass

        # Capture ComfyUI's own log lines (primary signal).
        self._handler = _ComfyLoadLogHandler(self)
        logging.getLogger().addHandler(self._handler)
        # Comfy may log under its own loggers; attach broadly.
        for name in ("comfy", "comfy.model_patcher", "comfy.model_management"):
            logging.getLogger(name).addHandler(self._handler)
            logging.getLogger(name).setLevel(logging.INFO)

        try:
            import comfy.model_management as mm
            import comfy.model_patcher as mp

            probe = self
            orig_lmg = mm.load_models_gpu
            self._orig_load_models_gpu = orig_lmg

            def wrapped_load_models_gpu(*args, **kwargs):
                free_b = _free_vram_mb()
                alloc_b = _alloc_vram_mb()
                models = args[0] if args else kwargs.get("models")
                mem_req = kwargs.get("memory_required", args[1] if len(args) > 1 else 0)
                min_mem = kwargs.get("minimum_memory_required")
                force_full = kwargs.get("force_full_load", False)
                class_names: list[str] = []
                try:
                    for m in models or []:
                        inner = getattr(m, "model", m)
                        class_names.append(type(inner).__name__)
                except Exception:
                    pass
                print(
                    f"[vram_load_probe] load_models_gpu enter "
                    f"models={class_names} free_vram_mb={free_b} "
                    f"alloc_mb={alloc_b} memory_required_mb="
                    f"{None if mem_req is None else round(float(mem_req)/(1024**2),1)} "
                    f"minimum_memory_required_mb="
                    f"{None if min_mem is None else round(float(min_mem)/(1024**2),1)} "
                    f"force_full_load={force_full}"
                )
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                t0 = time.perf_counter()
                out = orig_lmg(*args, **kwargs)
                ev = LoadEvent(
                    kind="load_models_gpu",
                    message="load_models_gpu returned",
                    free_vram_mb_before=free_b,
                    free_vram_mb_after=_free_vram_mb(),
                    alloc_vram_mb_before=alloc_b,
                    alloc_vram_mb_after=_alloc_vram_mb(),
                    model_class=",".join(class_names) if class_names else None,
                    code_path="comfy.model_management.load_models_gpu",
                    t_s=time.perf_counter() - t0,
                )
                probe.report.events.append(ev)
                print(
                    f"[vram_load_probe] load_models_gpu exit "
                    f"free_vram_mb={ev.free_vram_mb_after} "
                    f"alloc_mb={ev.alloc_vram_mb_after} "
                    f"dt={ev.t_s:.2f}s"
                )
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                return out

            mm.load_models_gpu = wrapped_load_models_gpu

            orig_pl = mp.ModelPatcher.partially_load
            self._orig_partially_load = orig_pl

            def wrapped_partially_load(self_mp, device_to, extra_memory=0, force_patch_weights=False):
                free_b = _free_vram_mb()
                model_cls = type(getattr(self_mp, "model", self_mp)).__name__
                size_mb = None
                try:
                    size_mb = round(self_mp.model_size() / (1024**2), 1)
                except Exception:
                    pass
                loaded_mb = None
                try:
                    loaded_mb = round(
                        float(self_mp.model.model_loaded_weight_memory) / (1024**2), 1
                    )
                except Exception:
                    pass
                print(
                    f"[vram_load_probe] ModelPatcher.partially_load enter "
                    f"model={model_cls} device={device_to} "
                    f"extra_memory_mb={round(float(extra_memory)/(1024**2),1) if extra_memory else 0} "
                    f"model_size_mb={size_mb} already_loaded_mb={loaded_mb} "
                    f"free_vram_mb={free_b}"
                )
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                t0 = time.perf_counter()
                result = orig_pl(self_mp, device_to, extra_memory, force_patch_weights)
                after_loaded = None
                lowvram_flag = None
                try:
                    after_loaded = round(
                        float(self_mp.model.model_loaded_weight_memory) / (1024**2), 1
                    )
                    lowvram_flag = bool(getattr(self_mp.model, "model_lowvram", False))
                except Exception:
                    pass
                ev = LoadEvent(
                    kind="partially_load",
                    message=f"partially_load returned delta_bytes={result}",
                    mb_loaded=after_loaded,
                    free_vram_mb_before=free_b,
                    free_vram_mb_after=_free_vram_mb(),
                    alloc_vram_mb_after=_alloc_vram_mb(),
                    lowvram_model_memory_mb=(
                        round(float(extra_memory) / (1024**2), 1) if extra_memory else None
                    ),
                    model_class=model_cls,
                    full_load_flag=(False if lowvram_flag else True) if lowvram_flag is not None else None,
                    code_path="ModelPatcher.partially_load",
                    t_s=time.perf_counter() - t0,
                )
                if lowvram_flag is True:
                    probe.report.sampling_load_mode = "partial"
                elif lowvram_flag is False and probe.report.sampling_load_mode is None:
                    probe.report.sampling_load_mode = "complete"
                probe.report.events.append(ev)
                print(
                    f"[vram_load_probe] ModelPatcher.partially_load exit "
                    f"model_lowvram={lowvram_flag} loaded_weight_mb={after_loaded} "
                    f"free_vram_mb={ev.free_vram_mb_after} dt={ev.t_s:.2f}s"
                )
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                return result

            mp.ModelPatcher.partially_load = wrapped_partially_load
            self._installed = True
            return True
        except Exception as exc:
            print(f"[vram_load_probe] install failed: {exc}")
            self.uninstall()
            return False

    def uninstall(self) -> None:
        if self._handler is not None:
            try:
                logging.getLogger().removeHandler(self._handler)
            except Exception:
                pass
            for name in ("comfy", "comfy.model_patcher", "comfy.model_management"):
                try:
                    logging.getLogger(name).removeHandler(self._handler)
                except Exception:
                    pass
            self._handler = None
        try:
            if self._orig_load_models_gpu is not None:
                import comfy.model_management as mm

                mm.load_models_gpu = self._orig_load_models_gpu
        except Exception:
            pass
        try:
            if self._orig_partially_load is not None:
                import comfy.model_patcher as mp

                mp.ModelPatcher.partially_load = self._orig_partially_load
        except Exception:
            pass
        self._orig_load_models_gpu = None
        self._orig_partially_load = None
        self._installed = False

    def finalize(self) -> VramLoadProbeReport:
        partial_ev = next((e for e in self.report.events if e.kind == "partial"), None)
        complete_ev = next((e for e in self.report.events if e.kind == "complete"), None)
        pl_ev = next((e for e in reversed(self.report.events) if e.kind == "partially_load"), None)
        lmg_ev = next((e for e in self.report.events if e.kind == "load_models_gpu"), None)

        mode = self.report.sampling_load_mode or "unknown"
        if mode == "unknown" and pl_ev is not None and pl_ev.full_load_flag is False:
            mode = "partial"
        if mode == "unknown" and complete_ev is not None:
            mode = "complete"
        self.report.sampling_load_mode = mode

        summary = {
            "sampling_load_mode": mode,
            "free_vram_mb_at_probe_install": self.report.free_vram_mb_at_install,
            "free_vram_mb_at_load_models_gpu": (
                lmg_ev.free_vram_mb_before if lmg_ev else None
            ),
            "mb_loaded": (
                partial_ev.mb_loaded
                if partial_ev
                else (complete_ev.mb_loaded if complete_ev else (pl_ev.mb_loaded if pl_ev else None))
            ),
            "mb_offloaded": (
                partial_ev.mb_offloaded
                if partial_ev
                else (0.0 if complete_ev else None)
            ),
            "full_load_flag": (
                False
                if partial_ev
                else (complete_ev.full_load_flag if complete_ev else (pl_ev.full_load_flag if pl_ev else None))
            ),
            "code_paths_seen": sorted({e.code_path for e in self.report.events if e.code_path}),
            "comfy_log_captured": bool(partial_ev or complete_ev),
        }
        self.report.summary = summary

        print()
        print("=" * 72)
        print("[vram_load_probe] SUMMARY (verify partial-load hypothesis)")
        print(f"  sampling_load_mode:     {mode}")
        print(f"  free VRAM at install:   {summary['free_vram_mb_at_probe_install']} MB")
        print(f"  free VRAM at load_models_gpu: {summary['free_vram_mb_at_load_models_gpu']} MB")
        print(f"  MB loaded:              {summary['mb_loaded']}")
        print(f"  MB offloaded:           {summary['mb_offloaded']}")
        print(f"  full_load_flag:         {summary['full_load_flag']}")
        print(f"  comfy log captured:     {summary['comfy_log_captured']}")
        print(f"  code paths:             {summary['code_paths_seen']}")
        if mode == "partial":
            print("  conclusion:            PARTIAL LOAD confirmed — weight streaming active")
        elif mode == "complete":
            print("  conclusion:            FULL LOAD — partial-load hypothesis NOT supported")
        else:
            print("  conclusion:            UNCLEAR — check events list / ComfyUI logging config")
        print("=" * 72)
        try:
            sys.stdout.flush()
        except Exception:
            pass
        return self.report


@contextmanager
def vram_load_probe() -> Iterator[VramLoadProbe]:
    probe = VramLoadProbe()
    probe.install()
    try:
        yield probe
    finally:
        probe.finalize()
        probe.uninstall()
