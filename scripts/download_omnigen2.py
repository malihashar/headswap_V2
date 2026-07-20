#!/usr/bin/env python3
"""
Download OmniGen2 weights into /tmp/models (never into the repo).

Official sources (do not invent alternate filenames):
  Repo:    https://github.com/VectorSpaceLab/OmniGen2
  Weights: https://huggingface.co/OmniGen2/OmniGen2

This is a full Diffusers-style snapshot with custom remote code
(``OmniGen2Pipeline``). ComfyUI flat files (omnigen2_fp16.safetensors) are
a separate community packaging path — this downloader targets the official HF
snapshot used by ``inference.py``.

Examples:
  python scripts/download_omnigen2.py
  python scripts/download_omnigen2.py --verify-only
  HF_TOKEN=hf_xxx python scripts/download_omnigen2.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HEADSWAP_MODEL_STORE", "/tmp/models")
os.environ.setdefault("HEADSWAP_STAGING_DIR", "/tmp/_hf_dl_staging")

REPO_ID = "OmniGen2/OmniGen2"
LOCAL_DIRNAME = "OmniGen2"

# Marker files from model_index.json / HF siblings.
REQUIRED_RELATIVE = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "mllm/config.json",
    "mllm/model.safetensors.index.json",
    "mllm_processor/tokenizer_config.json",
    "processor/tokenizer_config.json",
)


def _reject_kaggle_working(label: str, path: str) -> None:
    if path.startswith("/kaggle/working"):
        raise SystemExit(
            f"ERROR: {label}={path} is under /kaggle/working (20GB loop). "
            "Use /tmp/models and /tmp/_hf_dl_staging instead."
        )


def _token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def snapshot_complete(root: Path) -> tuple[bool, list[str]]:
    missing = [rel for rel in REQUIRED_RELATIVE if not (root / rel).is_file()]
    return (not missing), missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--store-dir",
        default=os.environ.get("HEADSWAP_MODEL_STORE", "/tmp/models"),
        help="Persistent model store (default: /tmp/models)",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="Override snapshot directory (default: <store-dir>/OmniGen2)",
    )
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args(argv)

    store = Path(args.store_dir).expanduser().resolve()
    _reject_kaggle_working("--store-dir", str(store))
    local = (
        Path(args.local_dir).expanduser().resolve()
        if args.local_dir
        else store / LOCAL_DIRNAME
    )
    _reject_kaggle_working("--local-dir", str(local))

    print("=== download_omnigen2 ===")
    print(f"Repo:        {REPO_ID}")
    print(f"Revision:    {args.revision}")
    print(f"Model store: {store}")
    print(f"Local dir:   {local}")
    print("ComfyUI:     not required for official OmniGen2Pipeline path")
    print()

    ok, missing = snapshot_complete(local)
    if ok:
        print("Snapshot already complete — skipping download.")
        for rel in REQUIRED_RELATIVE:
            print(f"  OK  {local / rel}")
        print()
        print(f"Final model path: {local}")
        return 0

    if args.verify_only:
        print("VERIFY FAILED — missing:")
        for rel in missing:
            print(f"  MISSING  {local / rel}")
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "  pip install -U huggingface_hub\n"
            f"({exc})"
        ) from exc

    token = _token()
    print(f"Auth: {'HF_TOKEN present' if token else 'no HF_TOKEN (public repo)'}")

    staging = Path(os.environ.get("HEADSWAP_STAGING_DIR", "/tmp/_hf_dl_staging"))
    staging.mkdir(parents=True, exist_ok=True)
    cache_dir = staging / "hf_hub_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(staging / "hf_home"))

    local.mkdir(parents=True, exist_ok=True)
    print(f"Downloading snapshot → {local} (resumable)…")
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=str(local),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        cache_dir=str(cache_dir),
    )
    print(f"snapshot_download returned: {path}")

    ok, missing = snapshot_complete(local)
    if not ok:
        print("Download finished but required files still missing:", file=sys.stderr)
        for rel in missing:
            print(f"  MISSING  {local / rel}", file=sys.stderr)
        return 1

    print()
    print("Verified required files:")
    for rel in REQUIRED_RELATIVE:
        p = local / rel
        print(f"  OK  {p}  ({p.stat().st_size:,} bytes)")
    print()
    print(f"Final model path: {local}")
    print()
    print("Inference deps (official):")
    print("  git clone https://github.com/VectorSpaceLab/OmniGen2.git /tmp/OmniGen2")
    print("  pip install -r /tmp/OmniGen2/requirements.txt")
    print("  pip install -e /tmp/OmniGen2")
    print("  # optional: flash-attn for speed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
