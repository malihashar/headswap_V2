#!/usr/bin/env python3
"""Patch REFace sources for Python 3.12+ (Colab default).

Upstream still has dead ``import imp`` (removed in 3.12) and a few other
imports that break headless Colab before inference starts.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


# Lines that are unused in the image-swap path but break imports on Py3.12 / Colab.
_DROP_LINE_PATTERNS = (
    re.compile(r"^\s*import\s+imp\s*$"),
    re.compile(r"^\s*from\s+turtle\s+import\s+.*$"),
)


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    changed = False
    for line in lines:
        if any(p.match(line.rstrip("\n")) for p in _DROP_LINE_PATTERNS):
            changed = True
            continue
        kept.append(line)
    if not changed:
        return False
    path.write_text("".join(kept), encoding="utf-8")
    return True


def patch_reface_tree(root: Path) -> list[str]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"REFace root not found: {root}")
    touched: list[str] = []
    for path in root.rglob("*.py"):
        # Skip vendored / huge trees if any appear later.
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        if patch_file(path):
            touched.append(str(path.relative_to(root)))
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reface-root",
        type=Path,
        default=Path(os.environ.get("REFACE_ROOT", "/content/REFace")),
    )
    args = parser.parse_args()
    touched = patch_reface_tree(args.reface_root)
    if touched:
        print("✓ patched REFace for Python 3.12:")
        for rel in touched:
            print(f"  - {rel}")
    else:
        print("✓ REFace already Python 3.12-safe (no patches needed)")


if __name__ == "__main__":
    main()
