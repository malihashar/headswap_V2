from __future__ import annotations

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


def bootstrap_comfy(init_custom_nodes: bool = False) -> dict[str, Any]:
    """Import ComfyUI node mappings into this process (Colab/RunPod style)."""
    path = comfyui_path()
    if not path.exists():
        raise FileNotFoundError(
            f"ComfyUI not found at {path}. Set COMFYUI_PATH or run scripts/setup_comfyui.sh"
        )

    # Drop conflicting modules
    for k in list(sys.modules.keys()):
        if k == "utils" or k.startswith("utils.") or k == "comfy" or k.startswith("comfy."):
            del sys.modules[k]

    sys.path = [p for p in sys.path if "custom_nodes" not in p]
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
    os.chdir(path)

    import asyncio
    import nodes

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        nodes.init_extra_nodes(init_custom_nodes=init_custom_nodes, init_api_nodes=False)
    )
    return nodes.NODE_CLASS_MAPPINGS


class NodeRuntime:
    def __init__(self, mappings: dict[str, Any] | None = None):
        self.mappings = mappings or bootstrap_comfy()
        self._cache: dict[str, Any] = {}
        self.models: dict[str, Any] = {}

    def get_node(self, name: str):
        if name not in self._cache:
            if name not in self.mappings:
                raise KeyError(f"ComfyUI node '{name}' not found. Update ComfyUI?")
            self._cache[name] = self.mappings[name]()
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self.mappings


def get_value_at_index(obj, index: int):
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]
    except TypeError:
        if hasattr(obj, "args"):
            return obj.args[index]
        raise


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
        # also search recursively one level of category folders already included
    # fuzzy: any file containing stem
    stem = Path(preferred).stem.split(".")[0]
    for p in models_dir.rglob("*.safetensors"):
        if stem in p.name or preferred in p.name:
            return p.name
    return preferred
