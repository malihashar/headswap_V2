#!/usr/bin/env bash
# Prepare a Kaggle GPU notebook for headswap_V2 (idempotent).
#
# Layout (verified against Kaggle's split filesystem):
#   /kaggle/working          → ~20GB loop device  → ComfyUI + notebook outputs only
#   /tmp (overlay root)      → ~1T free           → model store + HF staging
#
# Defaults:
#   COMFYUI_PATH          = /kaggle/working/ComfyUI
#   HEADSWAP_MODEL_STORE  = /tmp/models
#   HEADSWAP_STAGING_DIR  = /tmp/_hf_dl_staging
#
# Run from the repository root with Internet + GPU enabled.
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/kaggle/working/ComfyUI}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/tmp/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/tmp/_hf_dl_staging}"

echo "=== headswap_V2 Kaggle setup ==="
echo "Repository:  $REPO_ROOT"
echo "ComfyUI:     $COMFYUI_PATH"
echo "Model store: $HEADSWAP_MODEL_STORE"
echo "Staging:     $HEADSWAP_STAGING_DIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo
echo "Filesystem check (models must NOT land on the 20GB /kaggle/working loop):"
df -h /tmp || true
df -h /kaggle/working || true
echo

if [[ ! -d /kaggle/working ]]; then
  echo "ERROR: /kaggle/working not found — this script is for Kaggle notebooks." >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/scripts/setup_comfyui.sh" ]]; then
  echo "ERROR: Run this script from the repository root (scripts/setup_comfyui.sh not found)." >&2
  exit 1
fi

# Guard against accidental use of the 20GB loop for multi-GB weights.
case "$HEADSWAP_MODEL_STORE" in
  /kaggle/working/*)
    echo "ERROR: HEADSWAP_MODEL_STORE=$HEADSWAP_MODEL_STORE is under /kaggle/working." >&2
    echo "Use /tmp/models (overlay FS) instead." >&2
    exit 1
    ;;
esac
case "$HEADSWAP_STAGING_DIR" in
  /kaggle/working/*)
    echo "ERROR: HEADSWAP_STAGING_DIR=$HEADSWAP_STAGING_DIR is under /kaggle/working." >&2
    echo "Use /tmp/_hf_dl_staging (overlay FS) instead." >&2
    exit 1
    ;;
esac

echo "[1/5] ComfyUI → $COMFYUI_PATH"
bash "$REPO_ROOT/scripts/setup_comfyui.sh"
echo

echo "[2/5] Python dependencies (requirements.txt)"
echo "  Note: do not install hf_xet; setup forces HF_HUB_DISABLE_XET=1."
python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
echo "  Done."
echo

echo "[3/5] aria2 (resumable HTTP fallback)"
if command -v aria2c >/dev/null 2>&1; then
  echo "  aria2c already installed."
else
  echo "  Installing aria2 ..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2
  else
    echo "  WARN: apt-get not available; install aria2 manually for the HTTP fallback." >&2
  fi
fi
echo

mkdir -p "$HEADSWAP_MODEL_STORE" "$HEADSWAP_STAGING_DIR" "$COMFYUI_PATH/models"

DL_COMMON=(
  --comfy "$COMFYUI_PATH"
  --store-dir "$HEADSWAP_MODEL_STORE"
  --staging-dir "$HEADSWAP_STAGING_DIR"
  --backend auto
  --disable-xet
  --manifest "$REPO_ROOT/scripts/models.json"
)

echo "[4/5] Download Klein models (staging → verify → /tmp/models → symlink into ComfyUI)"
python3 "$REPO_ROOT/scripts/download_models.py" --set klein "${DL_COMMON[@]}"
echo

echo "[5/5] Download Qwen models (staging → verify → /tmp/models → symlink into ComfyUI)"
python3 "$REPO_ROOT/scripts/download_models.py" --set qwen "${DL_COMMON[@]}"
echo

echo "Setup complete."
echo "store_dir=$HEADSWAP_MODEL_STORE"
echo "staging_dir=$HEADSWAP_STAGING_DIR"
echo "COMFYUI_PATH=$COMFYUI_PATH"
echo
df -h /tmp || true
df -h /kaggle/working || true
echo
echo "Ready to run:"
echo "  export COMFYUI_PATH=$COMFYUI_PATH"
echo "  python scripts/run_compare.py --gpu --limit 12"
