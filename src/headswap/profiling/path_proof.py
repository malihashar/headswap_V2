"""
Runtime path proof for qwen_baseline → VAEDecode.

CUDA-free. Prints to stdout, stderr, and an on-disk log so Kaggle OOM
truncation cannot hide whether this package's source actually ran.

Token: PATH_PROOF_V1 — if this never appears, the process is not executing
this checkout (stale editable install, wrong cwd, different package).
"""
from __future__ import annotations

import inspect
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PATH_PROOF_TOKEN = "PATH_PROOF_V1"


def _log_path() -> Path:
    env = os.environ.get("HEADSWAP_PATH_PROOF_LOG")
    if env:
        return Path(env)
    # Prefer results/ under the repo when possible; fall back to cwd.
    try:
        here = Path(__file__).resolve()
        repo = here.parents[3]  # .../src/headswap/profiling/path_proof.py
        return repo / "results" / "path_proof.log"
    except Exception:
        return Path("results") / "path_proof.log"


def _emit(line: str) -> None:
    """Always flush to stdout + stderr + file."""
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


def enter(letter: str, detail: str = "") -> None:
    """Unique ENTER marker. letter is A/B/C/... detail is free text."""
    suffix = f" {detail}" if detail else ""
    _emit(f"[path_proof] ENTER {letter}{suffix}")


def dump_identity(tag: str = "identity") -> None:
    """Print absolute paths of every module on the VAEDecode call chain."""
    _emit(f"[path_proof] === {tag} token={PATH_PROOF_TOKEN} ===")
    _emit(f"[path_proof] path_proof.__file__={Path(__file__).resolve()}")
    _emit(f"[path_proof] cwd={Path.cwd()}")
    _emit(f"[path_proof] sys.executable={sys.executable}")
    _emit(f"[path_proof] log_file={_log_path().resolve()}")

    modules = [
        "headswap",
        "headswap.cli",
        "headswap.comfy.runtime",
        "headswap.pipelines.qwen",
        "headswap.profiling.path_proof",
        "headswap.profiling.vae_bridge_probe",
        "nodes",
        "comfy.sd",
        "comfy.model_management",
    ]
    for name in modules:
        mod = sys.modules.get(name)
        if mod is None:
            _emit(f"[path_proof] module {name}: NOT_IN_sys.modules")
            continue
        f = getattr(mod, "__file__", None)
        _emit(
            f"[path_proof] module {name}: __file__="
            f"{Path(f).resolve() if f else None}"
        )

    # Editable-install / duplicate-package check
    try:
        import headswap

        _emit(f"[path_proof] headswap package dir={Path(headswap.__file__).resolve().parent}")
    except Exception as exc:
        _emit(f"[path_proof] headswap import failed: {exc}")

    try:
        import importlib.metadata as md

        dist = md.distribution("headswap")
        _emit(f"[path_proof] dist.files_root={dist.locate_file('')}")
        _emit(f"[path_proof] dist.version={dist.version}")
    except Exception as exc:
        _emit(f"[path_proof] importlib.metadata headswap: {exc}")


def dump_vaedecode_impl(runtime: Any) -> None:
    """Print inspect.getsourcefile for the VAEDecode class actually invoked."""
    enter("D1", "dump_vaedecode_impl")
    try:
        cls = None
        if runtime is not None:
            cls = getattr(runtime, "mappings", {}).get("VAEDecode")
        if cls is None:
            import nodes

            cls = getattr(nodes, "VAEDecode", None)
        if cls is None:
            _emit("[path_proof] VAEDecode class: NOT FOUND")
            return
        _emit(f"[path_proof] VAEDecode class={cls!r} id={id(cls)}")
        _emit(f"[path_proof] VAEDecode.__module__={getattr(cls, '__module__', None)}")
        _emit(f"[path_proof] VAEDecode.FUNCTION={getattr(cls, 'FUNCTION', None)}")
        try:
            src = inspect.getsourcefile(cls) or inspect.getfile(cls)
            _emit(f"[path_proof] VAEDecode getsourcefile={Path(src).resolve()}")
        except Exception as exc:
            _emit(f"[path_proof] VAEDecode getsourcefile FAILED: {exc}")
        method = getattr(cls, "decode", None)
        if method is not None:
            try:
                src_m = inspect.getsourcefile(method) or inspect.getfile(method)
                _emit(f"[path_proof] VAEDecode.decode getsourcefile={Path(src_m).resolve()}")
            except Exception as exc:
                _emit(f"[path_proof] VAEDecode.decode getsourcefile FAILED: {exc}")
            _emit(f"[path_proof] VAEDecode.decode is={method}")
    except Exception as exc:
        _emit(f"[path_proof] dump_vaedecode_impl RAISED: {exc}")
        traceback.print_exc()


@contextmanager
def install_decode_chain_enters(runtime: Any = None) -> Iterator[dict]:
    """
    Monkey-patch the live Comfy chain with ENTER markers only (no CUDA queries):

      F  nodes.VAEDecode.decode
      G  comfy.sd.VAE.decode
      H  comfy.model_management.load_models_gpu  (only while inside G)
    """
    state: dict[str, Any] = {"installed": False}
    enter("E0", "install_decode_chain_enters begin")
    try:
        import nodes
        import comfy.sd as comfy_sd
        import comfy.model_management as mm
    except Exception as exc:
        _emit(f"[path_proof] install FAILED import: {exc}")
        traceback.print_exc()
        yield state
        return

    node_cls = nodes.VAEDecode
    if runtime is not None:
        mapped = getattr(runtime, "mappings", {}).get("VAEDecode")
        if mapped is not None:
            node_cls = mapped
            _emit(
                f"[path_proof] using runtime.mappings['VAEDecode']="
                f"{getattr(mapped, '__name__', mapped)} id={id(mapped)} "
                f"(nodes.VAEDecode id={id(nodes.VAEDecode)})"
            )

    dump_vaedecode_impl(runtime)

    orig_node_decode = node_cls.decode
    orig_vae_decode = comfy_sd.VAE.decode
    orig_lmg = mm.load_models_gpu

    def wrapped_lmg(*args, **kwargs):
        enter("H", f"load_models_gpu models={len(args[0]) if args else '?'} kwargs={list(kwargs)}")
        return orig_lmg(*args, **kwargs)

    def wrapped_node_decode(self, vae, samples):
        enter(
            "F",
            f"nodes.VAEDecode.decode vae={type(vae).__name__} id={id(vae)}",
        )
        return orig_node_decode(self, vae, samples)

    def wrapped_vae_decode(self, samples_in, vae_options=None):
        if vae_options is None:
            vae_options = {}
        shape = list(samples_in.shape) if hasattr(samples_in, "shape") else None
        enter(
            "G",
            f"comfy.sd.VAE.decode patcher_id={id(getattr(self, 'patcher', None))} "
            f"shape={shape}",
        )
        prev = mm.load_models_gpu
        mm.load_models_gpu = wrapped_lmg
        try:
            return orig_vae_decode(self, samples_in, vae_options)
        finally:
            mm.load_models_gpu = prev

    node_cls.decode = wrapped_node_decode
    comfy_sd.VAE.decode = wrapped_vae_decode
    # Also wrap load_models_gpu for the whole window so we see it even if
    # decode path differs; ENTER H still unique.
    mm.load_models_gpu = wrapped_lmg
    state["installed"] = True
    enter("E1", "monkey-patches installed F/G/H")
    try:
        yield state
    finally:
        node_cls.decode = orig_node_decode
        comfy_sd.VAE.decode = orig_vae_decode
        mm.load_models_gpu = orig_lmg
        enter("E2", "monkey-patches restored")
