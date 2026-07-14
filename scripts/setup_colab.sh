#!/usr/bin/env bash
# Prepare a fresh Google Colab runtime for headswap_V2 (idempotent).
# Run from the repository root after mounting Google Drive.
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/content/ComfyUI}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/content/drive/MyDrive/headswap_V2/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/content/_hf_dl_staging}"

echo "=== headswap_V2 Colab setup ==="
echo "Repository:  $REPO_ROOT"
echo "ComfyUI:     $COMFYUI_PATH"
echo "Model store: $HEADSWAP_MODEL_STORE"
echo "Staging:     $HEADSWAP_STAGING_DIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo

if [[ ! -f "$REPO_ROOT/scripts/setup_comfyui.sh" ]]; then
  echo "ERROR: Run this script from the repository root (scripts/setup_comfyui.sh not found)." >&2
  exit 1
fi

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "In a Colab cell, run:" >&2
  echo "  from google.colab import drive" >&2
  echo "  drive.mount('/content/drive')" >&2
  echo "Then re-run: bash scripts/setup_colab.sh" >&2
  exit 1
fi

echo "[1/5] ComfyUI"
if [[ -d "$COMFYUI_PATH" ]]; then
  echo "  Already installed at $COMFYUI_PATH — skipping clone."
else
  echo "  Installing via scripts/setup_comfyui.sh ..."
fi
bash "$REPO_ROOT/scripts/setup_comfyui.sh"
echo

echo "[2/5] Python dependencies (requirements.txt)"
echo "  Note: do not install hf_xet for Colab; setup forces HF_HUB_DISABLE_XET=1."
python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
echo "  Done."
echo

echo "[3/5] aria2 (resumable HTTP fallback)"
if command -v aria2c >/dev/null 2>&1; then
  echo "  aria2c already installed."
else
  echo "  Installing aria2 ..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2
fi
echo

mkdir -p "$HEADSWAP_MODEL_STORE" "$HEADSWAP_STAGING_DIR"

DL_COMMON=(
  --comfy "$COMFYUI_PATH"
  --store-dir "$HEADSWAP_MODEL_STORE"
  --staging-dir "$HEADSWAP_STAGING_DIR"
  --backend auto
  --disable-xet
  --manifest "$REPO_ROOT/scripts/models.json"
)

echo "[4/5] Download Klein models (staging → verify → Drive → symlink)"
python3 "$REPO_ROOT/scripts/download_models.py" --set klein "${DL_COMMON[@]}"
echo

echo "[5/5] Download Qwen models (staging → verify → Drive → symlink)"
python3 "$REPO_ROOT/scripts/download_models.py" --set qwen "${DL_COMMON[@]}"
echo

echo "Setup complete."
echo "Repository path: $REPO_ROOT"
echo "ComfyUI path:    $COMFYUI_PATH"
echo "Model store:     $HEADSWAP_MODEL_STORE"
echo "Ready to run:"
echo "  python scripts/run_compare.py --gpu --limit 12"
