#!/usr/bin/env python3
"""
Download FLUX.1 Kontext [dev] weights into /tmp/models (never into the repo).

Reuses scripts/download_models.py staging / HF-cache / Comfy symlink logic.
Skips Place It / Put It Here LoRAs (not part of the first integration).

Tokenizers for Flux/Kontext are embedded in the DualCLIP text-encoder
safetensors (clip_l + t5xxl); no separate tokenizer download is required.

Required assets (models.json set=kontext):
  - flux1-dev-kontext_fp8_scaled.safetensors  (~11.1 GiB)
  - clip_l.safetensors                        (~0.23 GiB)
  - t5xxl_fp8_e4m3fn_scaled.safetensors       (~4.8 GiB)
  - ae.safetensors                            (~0.31 GiB)

Examples:
  python scripts/download_kontext.py
  python scripts/download_kontext.py --verify-only
  python scripts/download_kontext.py --comfy /path/to/ComfyUI
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Force store + staging onto /tmp before importing download_models helpers.
os.environ.setdefault("HEADSWAP_MODEL_STORE", "/tmp/models")
os.environ.setdefault("HEADSWAP_STAGING_DIR", "/tmp/_hf_dl_staging")


def _reject_kaggle_working(label: str, path: str) -> None:
    if path.startswith("/kaggle/working"):
        raise SystemExit(
            f"ERROR: {label}={path} is under /kaggle/working (20GB loop). "
            "Use /tmp/models and /tmp/_hf_dl_staging instead."
        )


def main() -> int:
    # Import after env defaults so default_store_dir / default_staging_dir see them.
    import download_models as dm

    argv = list(sys.argv[1:])
    # Always target the kontext set unless the caller already chose one.
    if "--set" not in argv:
        argv = ["--set", "kontext", *argv]
    # Always pin store to /tmp/models unless explicitly overridden.
    if "--store-dir" not in argv:
        argv = ["--store-dir", "/tmp/models", *argv]
    if "--staging-dir" not in argv:
        argv = ["--staging-dir", "/tmp/_hf_dl_staging", *argv]

    # Guard against accidental /kaggle/working store even if flags override defaults.
    for i, arg in enumerate(argv):
        if arg == "--store-dir" and i + 1 < len(argv):
            _reject_kaggle_working("--store-dir", argv[i + 1])
        if arg == "--staging-dir" and i + 1 < len(argv):
            _reject_kaggle_working("--staging-dir", argv[i + 1])

    print("=== download_kontext ===")
    print(f"Model store: {os.environ.get('HEADSWAP_MODEL_STORE')}")
    print(f"Staging:     {os.environ.get('HEADSWAP_STAGING_DIR')}")
    print("Assets: Kontext FP8 UNET + clip_l + t5xxl_fp8 + ae VAE (~16.4 GiB)")
    print("Skipped: Place It / Put It Here LoRAs (intentional for v1)")
    print()

    sys.argv = [sys.argv[0], *argv]
    return dm.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
