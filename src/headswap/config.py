from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DOTENV_LOADED = False


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_dotenv(dotenv_path: str | Path | None = None) -> bool:
    """Load KEY=VALUE pairs from project ``.env`` into os.environ (no overwrite).

    Returns True if a file was found and parsed. Missing file is a no-op.
    Does not log values — safe for API keys.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED and dotenv_path is None:
        return False
    path = Path(dotenv_path) if dotenv_path else project_root() / ".env"
    if not path.is_file():
        if dotenv_path is None:
            _DOTENV_LOADED = True
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        os.environ[key] = val
    if dotenv_path is None:
        _DOTENV_LOADED = True
    return True


def resolve_out_dir(cfg: dict[str, Any], override: str | Path | None = None) -> Path:
    if override:
        out = Path(override)
    else:
        out = project_root() / "results" / str(cfg.get("name", "run"))
    out.mkdir(parents=True, exist_ok=True)
    return out
