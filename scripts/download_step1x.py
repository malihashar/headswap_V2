#!/usr/bin/env python3
"""
Download Step1X-Edit-v1p2 weights into /tmp/models (never into the repo).

Official sources (do not invent alternate filenames):
  Repo:    https://github.com/stepfun-ai/Step1X-Edit
  Weights: https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2
  Diffusers branch required for inference:
           https://github.com/Peyton-Chen/diffusers/tree/step1xedit_v1p2

This is a full Diffusers snapshot (~42 GiB on disk), not a ComfyUI flat-file set.
ComfyUI symlinks are therefore skipped (pipeline runs via Diffusers, not Comfy nodes).

Examples:
  python scripts/download_step1x.py
  python scripts/download_step1x.py --verify-only
  HF_TOKEN=hf_xxx python scripts/download_step1x.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Force store + staging onto /tmp before any download work.
os.environ.setdefault("HEADSWAP_MODEL_STORE", "/tmp/models")
os.environ.setdefault("HEADSWAP_STAGING_DIR", "/tmp/_hf_dl_staging")

REPO_ID = "stepfun-ai/Step1X-Edit-v1p2"
LOCAL_DIRNAME = "Step1X-Edit-v1p2"

# Marker files that must exist for a usable Diffusers snapshot (from model_index.json).
REQUIRED_RELATIVE = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
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
        help="Override snapshot directory (default: <store-dir>/Step1X-Edit-v1p2)",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check that required files already exist",
    )
    ap.add_argument(
        "--revision",
        default="main",
        help="HF revision / branch / commit (default: main)",
    )
    args = ap.parse_args(argv)

    store = Path(args.store_dir).expanduser().resolve()
    _reject_kaggle_working("--store-dir", str(store))
    local = (
        Path(args.local_dir).expanduser().resolve()
        if args.local_dir
        else store / LOCAL_DIRNAME
    )
    _reject_kaggle_working("--local-dir", str(local))

    print("=== download_step1x ===")
    print(f"Repo:        {REPO_ID}")
    print(f"Revision:    {args.revision}")
    print(f"Model store: {store}")
    print(f"Local dir:   {local}")
    print(f"Staging HF:  {os.environ.get('HF_HOME') or os.environ.get('HEADSWAP_STAGING_DIR')}")
    print("ComfyUI:     not required (Diffusers pipeline)")
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
    if token:
        print("Auth: HF_TOKEN present")
    else:
        print("Auth: no HF_TOKEN (public repo; token still helps with rate limits)")

    # Prefer /tmp for HF cache blobs on Kaggle.
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
        size = p.stat().st_size
        print(f"  OK  {p}  ({size:,} bytes)")
    print()
    print(f"Final model path: {local}")
    print()
    print("Inference deps (official):")
    print("  pip install 'transformers==4.55.0'")
    print("  git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git")
    print("  pip install -e ./diffusers")
    print("  # optional: pip install RegionE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
