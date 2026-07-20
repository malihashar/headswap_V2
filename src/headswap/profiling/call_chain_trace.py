"""
Post-sampling call-chain tracer (CUDA-free).

Starts after diffusion sampling finishes and logs every relevant function
entry with ``module.__file__`` until stopped (PNG write).

Also installs a thin wrapper on the live latent→image decode path so the
first CUDA allocation site (``load_models_gpu`` inside ``VAE.decode``) is
logged once the path is proven.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

_TRACE_PREFIXES = (
    "headswap.",
    "nodes",
    "comfy.sd",
    "comfy.model_management",
    "comfy.sample",
    "comfy.samplers",
    "PIL.",
)

# Function names that matter for latent→image / PNG write.
_ALWAYS_LOG_NAMES = frozenset(
    {
        "_sample_qwen",
        "run",
        "call",
        "invoke_node",
        "decode",
        "load_models_gpu",
        "comfy_tensor_to_pil",
        "save",
        "enter",
        "dump_identity",
        "install_decode_chain_enters",
    }
)

_tls = threading.local()
_lock = threading.Lock()


def _log_path() -> Path:
    env = os.environ.get("HEADSWAP_CALL_CHAIN_LOG")
    if env:
        return Path(env)
    try:
        repo = Path(__file__).resolve().parents[3]
        return repo / "results" / "call_chain.log"
    except Exception:
        return Path("results") / "call_chain.log"


def _emit(line: str) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            print(line, file=stream, flush=True)
        except Exception:
            pass
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


def _module_file(frame) -> str:
    mod_name = frame.f_globals.get("__name__")
    mod = sys.modules.get(mod_name) if mod_name else None
    f = getattr(mod, "__file__", None) if mod is not None else None
    if f:
        try:
            return str(Path(f).resolve())
        except Exception:
            return str(f)
    # Fallback to code object filename
    co = frame.f_code.co_filename
    try:
        return str(Path(co).resolve())
    except Exception:
        return str(co)


def _should_log(mod_name: str | None, func_name: str) -> bool:
    if func_name in _ALWAYS_LOG_NAMES:
        return True
    if not mod_name:
        return False
    if mod_name.startswith(_TRACE_PREFIXES) or mod_name in _TRACE_PREFIXES:
        return True
    # nodes is top-level in ComfyUI
    if mod_name == "nodes" or mod_name.startswith("nodes."):
        return True
    if mod_name.startswith("comfy."):
        # Only decode / load related comfy modules to limit noise
        if any(
            x in mod_name
            for x in ("sd", "model_management", "sample", "samplers", "model_patcher")
        ):
            return True
    return False


def _tracer(frame, event, arg):  # noqa: ANN001
    if event != "call":
        return _tracer
    if not getattr(_tls, "active", False):
        return None
    mod_name = frame.f_globals.get("__name__")
    func_name = frame.f_code.co_name
    if not _should_log(mod_name, func_name):
        return _tracer
    depth = getattr(_tls, "depth", 0)
    _tls.depth = depth + 1
    try:
        _emit(
            f"[call_chain] {'  ' * min(depth, 16)}ENTER {mod_name}.{func_name} "
            f"file={_module_file(frame)}:{frame.f_lineno}"
        )
    except Exception:
        pass
    return _tracer


def start_post_sample_trace(*, reason: str = "") -> None:
    """Begin tracing after sampling. Idempotent per thread."""
    with _lock:
        if getattr(_tls, "active", False):
            _emit(f"[call_chain] trace already active ({reason})")
            return
        _tls.active = True
        _tls.depth = 0
        _tls.reason = reason
        _emit("")
        _emit(f"[call_chain] === START post-sample trace {reason} ===")
        _emit(f"[call_chain] log_file={_log_path().resolve()}")
        _emit(
            "[call_chain] EXPECTED latent→image chain: "
            "_sample_qwen → NodeRuntime.call(VAEDecode) → invoke_node → "
            "nodes.VAEDecode.decode → comfy.sd.VAE.decode → "
            "model_management.load_models_gpu → … → comfy_tensor_to_pil → "
            "Image.save(result.png)"
        )
        sys.settrace(_tracer)
        # Also set on current thread's profile hook used by some runners
        threading.settrace(_tracer)


def stop_post_sample_trace(*, reason: str = "") -> None:
    with _lock:
        if not getattr(_tls, "active", False):
            return
        sys.settrace(None)
        threading.settrace(None)
        _tls.active = False
        _emit(f"[call_chain] === STOP post-sample trace {reason} ===")
        _emit("")


def install_live_decode_cuda_log(runtime: Any = None) -> Callable[[], None]:
    """
    Monkey-patch the live decode path.

    Latent→image is performed by ``comfy.sd.VAE.decode``. The first CUDA
    allocation on that path is ``model_management.load_models_gpu([vae.patcher], …)``.

    Logs CUDA memory immediately before that call. Returns a restore callback.
    """
    _emit("[call_chain] install_live_decode_cuda_log begin")
    try:
        import nodes
        import comfy.sd as comfy_sd
        import comfy.model_management as mm
        import torch
    except Exception as exc:
        _emit(f"[call_chain] install_live_decode_cuda_log FAILED import: {exc}")
        traceback.print_exc()
        return lambda: None

    node_cls = nodes.VAEDecode
    if runtime is not None:
        mapped = getattr(runtime, "mappings", {}).get("VAEDecode")
        if mapped is not None:
            node_cls = mapped
            _emit(
                f"[call_chain] VAEDecode mapping={getattr(mapped, '__name__', mapped)} "
                f"id={id(mapped)} file={getattr(sys.modules.get(mapped.__module__), '__file__', None)}"
            )

    try:
        import inspect

        src = inspect.getsourcefile(node_cls) or inspect.getfile(node_cls)
        _emit(f"[call_chain] VAEDecode.getsourcefile={Path(src).resolve()}")
        src_v = inspect.getsourcefile(comfy_sd.VAE.decode) or inspect.getfile(
            comfy_sd.VAE.decode
        )
        _emit(f"[call_chain] comfy.sd.VAE.decode.getsourcefile={Path(src_v).resolve()}")
    except Exception as exc:
        _emit(f"[call_chain] getsourcefile failed: {exc}")

    orig_node = node_cls.decode
    orig_vae = comfy_sd.VAE.decode
    orig_lmg = mm.load_models_gpu

    def _cuda_line(tag: str) -> None:
        try:
            if not torch.cuda.is_available():
                _emit(f"[decode_cuda] {tag} cuda=unavailable")
                return
            free_b, total_b = torch.cuda.mem_get_info()
            _emit(
                f"[decode_cuda] {tag} "
                f"allocated_mb={torch.cuda.memory_allocated() / 1024**2:.1f} "
                f"reserved_mb={torch.cuda.memory_reserved() / 1024**2:.1f} "
                f"free_mb={free_b / 1024**2:.1f} "
                f"total_mb={total_b / 1024**2:.1f}"
            )
        except Exception as exc:
            _emit(f"[decode_cuda] {tag} RAISED {type(exc).__name__}: {exc}")

    def wrapped_lmg(models, *args, **kwargs):
        labels = []
        for m in models or []:
            inner = getattr(m, "model", None)
            labels.append(
                f"{type(m).__name__}/inner={type(inner).__name__ if inner else None}"
            )
        _emit(f"[decode_cuda] load_models_gpu ENTER models={labels}")
        _cuda_line("IMMEDIATELY_BEFORE_load_models_gpu")
        return orig_lmg(models, *args, **kwargs)

    def wrapped_node_decode(self, vae, samples):
        _emit(
            f"[call_chain] LIVE nodes.VAEDecode.decode "
            f"vae={type(vae).__name__} file={getattr(sys.modules.get(type(vae).__module__), '__file__', None)}"
        )
        return orig_node(self, vae, samples)

    def wrapped_vae_decode(self, samples_in, vae_options=None):
        if vae_options is None:
            vae_options = {}
        shape = list(samples_in.shape) if hasattr(samples_in, "shape") else None
        _emit(
            f"[call_chain] LIVE comfy.sd.VAE.decode "
            f"(THIS is latent→image) shape={shape} "
            f"patcher_id={id(getattr(self, 'patcher', None))}"
        )
        _cuda_line("ENTER_VAE.decode_before_any_alloc")
        prev = mm.load_models_gpu
        mm.load_models_gpu = wrapped_lmg
        try:
            return orig_vae(self, samples_in, vae_options)
        finally:
            mm.load_models_gpu = prev

    node_cls.decode = wrapped_node_decode
    comfy_sd.VAE.decode = wrapped_vae_decode
    mm.load_models_gpu = wrapped_lmg
    _emit("[call_chain] live decode wrappers installed (VAEDecode.decode / VAE.decode / load_models_gpu)")

    def restore() -> None:
        node_cls.decode = orig_node
        comfy_sd.VAE.decode = orig_vae
        mm.load_models_gpu = orig_lmg
        _emit("[call_chain] live decode wrappers restored")

    return restore
