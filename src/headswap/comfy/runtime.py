from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Any


def detect_base_path() -> Path:
    if Path("/workspace/ComfyUI").exists():
        return Path("/workspace")
    if Path("/content/ComfyUI").exists():
        return Path("/content")
    env = os.environ.get("COMFYUI_PATH")
    if env:
        return Path(env).parent if Path(env).name == "ComfyUI" else Path(env)
    # Local default beside project
    return Path(os.environ.get("HEADSWAP_BASE", str(Path.home() / "ComfyUI_headswap"))).parent


def comfyui_path() -> Path:
    env = os.environ.get("COMFYUI_PATH")
    if env and Path(env).exists():
        return Path(env)
    base = detect_base_path()
    return base / "ComfyUI"


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _ensure_prompt_server(loop: asyncio.AbstractEventLoop) -> Any:
    """
    Instantiate PromptServer exactly as ComfyUI main.start_comfyui does.

    Custom nodes and some built-in extras register routes on PromptServer.instance.
    That attribute only exists after PromptServer(loop) runs — importing nodes without
    constructing the server produces: PromptServer has no attribute 'instance'.
    """
    import server

    instance = getattr(server.PromptServer, "instance", None)
    if instance is not None:
        return instance
    return server.PromptServer(loop)


def bootstrap_comfy(init_custom_nodes: bool | None = None) -> dict[str, Any]:
    """
    Import ComfyUI node mappings into this process (Colab/RunPod style).

    Boot order matches ComfyUI's start_comfyui():
      1. create asyncio loop
      2. construct PromptServer(loop)  → sets PromptServer.instance
      3. nodes.init_extra_nodes(...)

    Custom nodes (e.g. comfyui-krea2edit) load only when init_custom_nodes=True
    or HEADSWAP_INIT_CUSTOM_NODES=1.
    """
    if init_custom_nodes is None:
        init_custom_nodes = os.environ.get("HEADSWAP_INIT_CUSTOM_NODES", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    path = comfyui_path()
    if not path.exists():
        raise FileNotFoundError(
            f"ComfyUI not found at {path}. Set COMFYUI_PATH or run scripts/setup_comfyui.sh"
        )

    # Drop conflicting modules from a previous import / previous chdir
    for k in list(sys.modules.keys()):
        if k == "utils" or k.startswith("utils.") or k == "comfy" or k.startswith("comfy."):
            del sys.modules[k]

    sys.path = [p for p in sys.path if "custom_nodes" not in p]
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
    os.chdir(path)

    loop = _ensure_event_loop()
    _ensure_prompt_server(loop)

    import nodes

    loop.run_until_complete(
        nodes.init_extra_nodes(init_custom_nodes=init_custom_nodes, init_api_nodes=False)
    )
    return nodes.NODE_CLASS_MAPPINGS


def invoke_node(node_cls: type, **kwargs: Any) -> Any:
    """
    Call a ComfyUI node class the way execution.py does.

    Current ComfyUI mixes two node schemas:

    * **V1** (legacy): ``FUNCTION = "method_name"`` → instantiate, call instance method.
      Returns a plain ``tuple`` (or dict with ``result``).
    * **V3** (``io.ComfyNode``): ``FUNCTION`` is the classproperty ``"EXECUTE_NORMALIZED"``
      → call ``cls.EXECUTE_NORMALIZED(**kwargs)`` (wraps ``@classmethod execute``).
      Returns ``io.NodeOutput`` (``.args`` / ``.result``).

    This repository previously assumed every mapping had a classmethod ``.execute()``
    (true for V3, false for V1 nodes like ``ConditioningZeroOut`` whose FUNCTION is
    ``zero_out``). Always go through this helper (or ``NodeRuntime.call``).
    """
    # Path proof: only noise on VAEDecode (the node that OOMs).
    if getattr(node_cls, "__name__", None) == "VAEDecode" or (
        getattr(node_cls, "FUNCTION", None) == "decode"
        and "VAEDecode" in getattr(node_cls, "__qualname__", "")
    ):
        try:
            from headswap.profiling.path_proof import enter

            enter(
                "E",
                f"invoke_node cls={getattr(node_cls, '__name__', node_cls)} "
                f"FUNCTION={getattr(node_cls, 'FUNCTION', None)} "
                f"module={getattr(node_cls, '__module__', None)}",
            )
        except Exception:
            pass

    func_name = getattr(node_cls, "FUNCTION", None)
    if func_name is None:
        raise TypeError(
            f"Node class {getattr(node_cls, '__name__', node_cls)!r} has no FUNCTION; "
            "not a ComfyUI node mapping"
        )

    # V3: FUNCTION resolves to EXECUTE_NORMALIZED / _ASYNC (see comfy_api.latest._io)
    if func_name == "EXECUTE_NORMALIZED":
        if hasattr(node_cls, "VALIDATE_CLASS"):
            node_cls.VALIDATE_CLASS()
        return node_cls.EXECUTE_NORMALIZED(**kwargs)
    if func_name == "EXECUTE_NORMALIZED_ASYNC":
        if hasattr(node_cls, "VALIDATE_CLASS"):
            node_cls.VALIDATE_CLASS()
        coro = node_cls.EXECUTE_NORMALIZED_ASYNC(**kwargs)
        loop = _ensure_event_loop()
        return loop.run_until_complete(coro)

    # V1: FUNCTION is the instance method name (encode, zero_out, load_unet, ...)
    instance = node_cls()
    method = getattr(instance, func_name)
    if inspect.iscoroutinefunction(method):
        loop = _ensure_event_loop()
        return loop.run_until_complete(method(**kwargs))
    return method(**kwargs)


class NodeRuntime:
    def __init__(
        self,
        mappings: dict[str, Any] | None = None,
        *,
        init_custom_nodes: bool | None = None,
    ):
        self.mappings = mappings or bootstrap_comfy(init_custom_nodes=init_custom_nodes)
        self._cache: dict[str, Any] = {}
        self.models: dict[str, Any] = {}

    def get_node(self, name: str):
        """
        Cached instance of a node class.

        Prefer ``call()`` for execution. ``get_node`` remains useful when you need a
        stable instance (rare); V3 nodes are classmethod-based and do not need one.
        """
        if name not in self._cache:
            if name not in self.mappings:
                raise KeyError(f"ComfyUI node '{name}' not found. Update ComfyUI?")
            self._cache[name] = self.mappings[name]()
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self.mappings

    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke node ``name`` with kwargs matching its INPUT_TYPES / schema."""
        if name not in self.mappings:
            raise KeyError(f"ComfyUI node '{name}' not found. Update ComfyUI?")
        if name == "VAEDecode":
            try:
                from headswap.profiling.path_proof import dump_vaedecode_impl, enter

                enter(
                    "D",
                    f"NodeRuntime.call(VAEDecode) mapping={self.mappings[name]!r} "
                    f"id={id(self.mappings[name])}",
                )
                dump_vaedecode_impl(self)
            except Exception:
                pass
        return invoke_node(self.mappings[name], **kwargs)


def get_value_at_index(obj: Any, index: int):
    """
    Unwrap a node return value to output slot ``index``.

    Handles V1 tuples, V1 ``{"result": (...)}`` dicts, and V3 ``NodeOutput`` (``.args``).
    """
    if obj is None:
        raise TypeError("node returned None")

    # V3 io.NodeOutput (and anything with .args tuple)
    args = getattr(obj, "args", None)
    if isinstance(args, tuple) and len(args) > index:
        return args[index]

    result = getattr(obj, "result", None)
    if isinstance(result, tuple) and len(result) > index:
        return result[index]

    if isinstance(obj, dict):
        if "result" in obj:
            return obj["result"][index]
        return obj[index]

    try:
        return obj[index]
    except TypeError as exc:
        raise TypeError(
            f"Cannot index node return of type {type(obj)!r}; expected tuple/NodeOutput"
        ) from exc


def pil_to_comfy_tensor(im, torch):
    import numpy as np

    arr = np.asarray(im.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


def comfy_tensor_to_pil(img_bhwc):
    import numpy as np
    from PIL import Image

    x = img_bhwc[0].detach().float().cpu().numpy()
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def resolve_model_file(models_dir: Path, preferred: str, fallbacks: list[str] | None = None) -> str:
    candidates = [preferred] + (fallbacks or [])
    for name in candidates:
        if (models_dir / name).exists():
            return name
    stem = Path(preferred).stem.split(".")[0]
    for p in models_dir.rglob("*.safetensors"):
        if stem in p.name or preferred in p.name:
            return p.name
    return preferred
