#!/usr/bin/env python3
"""
Download Krea 2 Identity Edit stack into /tmp/models (never into the repo).

Required (set=krea2):
  - krea2_turbo_fp8_scaled.safetensors          (~12.2 GiB)
  - qwen3vl_4b_fp8_scaled.safetensors           (~4.9 GiB)
  - qwen_image_vae.safetensors                  (~0.24 GiB)  # shared with qwen set
  - krea2_identity_edit_v1_2_r64.safetensors    (~0.43 GiB)  # T4-friendly LoRA

Optional (--include-optional):
  - krea2_identity_edit_v1_2.safetensors        (~1.7 GiB) full LoRA

Also requires custom nodes (installed by setup_kaggle.sh --krea2):
  https://github.com/lbouaraba/comfyui-krea2edit

Examples:
  python scripts/download_krea2.py
  python scripts/download_krea2.py --include-optional
  python scripts/download_krea2.py --verify-only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("HEADSWAP_MODEL_STORE", "/tmp/models")
os.environ.setdefault("HEADSWAP_STAGING_DIR", "/tmp/_hf_dl_staging")


def _reject_kaggle_working(label: str, path: str) -> None:
    if path.startswith("/kaggle/working"):
        raise SystemExit(
            f"ERROR: {label}={path} is under /kaggle/working (20GB loop). "
            "Use /tmp/models and /tmp/_hf_dl_staging instead."
        )


def main() -> int:
    import download_models as dm

    argv = list(sys.argv[1:])
    if "--set" not in argv:
        argv = ["--set", "krea2", *argv]
    if "--store-dir" not in argv:
        argv = ["--store-dir", "/tmp/models", *argv]
    if "--staging-dir" not in argv:
        argv = ["--staging-dir", "/tmp/_hf_dl_staging", *argv]

    for i, arg in enumerate(argv):
        if arg == "--store-dir" and i + 1 < len(argv):
            _reject_kaggle_working("--store-dir", argv[i + 1])
        if arg == "--staging-dir" and i + 1 < len(argv):
            _reject_kaggle_working("--staging-dir", argv[i + 1])

    print("=== download_krea2 ===")
    print(f"Model store: {os.environ.get('HEADSWAP_MODEL_STORE')}")
    print(f"Staging:     {os.environ.get('HEADSWAP_STAGING_DIR')}")
    print(
        "Assets: Krea2 Turbo FP8 + Qwen3VL-4B FP8 + qwen_image_vae "
        "+ identity-edit LoRA r64 (~18 GiB)"
    )
    if "--include-optional" in argv:
        print("Optional: full krea2_identity_edit_v1_2 LoRA")
    else:
        print("Optional full LoRA skipped (pass --include-optional)")

    sys.argv = [sys.argv[0], *argv]
    return int(dm.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
