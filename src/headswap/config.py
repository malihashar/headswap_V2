from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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


def resolve_out_dir(cfg: dict[str, Any], override: str | Path | None = None) -> Path:
    if override:
        out = Path(override)
    else:
        out = project_root() / "results" / str(cfg.get("name", "run"))
    out.mkdir(parents=True, exist_ok=True)
    return out
