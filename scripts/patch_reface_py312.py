#!/usr/bin/env python3
"""Patch REFace sources for modern Colab (Python 3.12 + current transformers/Lightning).

Upstream still has:
  - dead ``import imp`` (removed in 3.12)
  - unused ``from turtle import …`` (breaks headless)
  - CompVis safety-checker load that fails on current transformers
  - ``pytorch_lightning.utilities.distributed.rank_zero_only`` (moved in PL 2.x)
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

_SAFETY_LOAD_BLOCK = re.compile(
    r"# load safety model\n"
    r"safety_model_id = \"CompVis/stable-diffusion-safety-checker\"\n"
    r"safety_feature_extractor = AutoFeatureExtractor\.from_pretrained\(safety_model_id\)\n"
    r"safety_checker = StableDiffusionSafetyChecker\.from_pretrained\(safety_model_id\)\n",
    re.MULTILINE,
)

_SAFETY_LOAD_REPLACEMENT = """\
# Disabled for Colab: current transformers rejects CompVis safety-checker
# AutoFeatureExtractor (CLIP image processor, not an audio feature extractor).
# REFace's swap path already bypasses check_safety for outputs.
safety_model_id = None
safety_feature_extractor = None
safety_checker = None
"""

_CHECK_SAFETY_FN = re.compile(
    r"def check_safety\(x_image\):\n"
    r"(?:[ \t]+.*\n)+?"
    r"[ \t]+return x_checked_image, has_nsfw_concept\n",
    re.MULTILINE,
)

_CHECK_SAFETY_REPLACEMENT = """\
def check_safety(x_image):
    # No-op: safety checker disabled for Colab compatibility.
    n = int(getattr(x_image, "shape", [1])[0]) if hasattr(x_image, "shape") else 1
    return x_image, [False] * n
"""


_RANK_ZERO_IMPORT = re.compile(
    r"^from pytorch_lightning\.utilities\.distributed import rank_zero_only\s*$",
    re.MULTILINE,
)

_RANK_ZERO_REPLACEMENT = """\
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:  # pytorch-lightning >= 1.8 / 2.x
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
"""


def _patch_drop_lines(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    changed = False
    for line in lines:
        if any(p.match(line.rstrip("\n")) for p in _DROP_LINE_PATTERNS):
            changed = True
            continue
        kept.append(line)
    return "".join(kept), changed


def _patch_rank_zero_import(text: str) -> tuple[str, bool]:
    if "from pytorch_lightning.utilities.rank_zero import rank_zero_only" in text and \
       "except ImportError" in text:
        return text, False
    if not _RANK_ZERO_IMPORT.search(text):
        return text, False
    return _RANK_ZERO_IMPORT.sub(_RANK_ZERO_REPLACEMENT.rstrip("\n"), text), True


def _patch_safety_checker(text: str) -> tuple[str, bool]:
    changed = False
    if _SAFETY_LOAD_BLOCK.search(text):
        text = _SAFETY_LOAD_BLOCK.sub(_SAFETY_LOAD_REPLACEMENT, text, count=1)
        changed = True
    elif "safety_feature_extractor = None" not in text and "AutoFeatureExtractor.from_pretrained(safety_model_id)" in text:
        # Fallback: comment out load lines individually.
        text2 = []
        for line in text.splitlines(keepends=True):
            if "AutoFeatureExtractor.from_pretrained(safety_model_id)" in line:
                text2.append("safety_feature_extractor = None  # patched for Colab\n")
                changed = True
            elif "StableDiffusionSafetyChecker.from_pretrained(safety_model_id)" in line:
                text2.append("safety_checker = None  # patched for Colab\n")
                changed = True
            else:
                text2.append(line)
        text = "".join(text2)

    if _CHECK_SAFETY_FN.search(text) and "safety checker disabled for Colab" not in text:
        text = _CHECK_SAFETY_FN.sub(_CHECK_SAFETY_REPLACEMENT, text, count=1)
        changed = True
    return text, changed


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text
    text, _ = _patch_drop_lines(text)
    text, _ = _patch_rank_zero_import(text)
    if path.name == "inference_swap_selected.py" or "safety_model_id" in text:
        text, _ = _patch_safety_checker(text)
    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def patch_reface_tree(root: Path) -> list[str]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"REFace root not found: {root}")
    touched: list[str] = []
    for path in root.rglob("*.py"):
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
        print("✓ patched REFace for Colab compatibility:")
        for rel in touched:
            print(f"  - {rel}")
    else:
        print("✓ REFace already Colab-compatible (no patches needed)")


if __name__ == "__main__":
    main()
