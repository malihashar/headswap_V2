#!/usr/bin/env bash
# Prepare a fresh Google Colab runtime for headswap_V2 (idempotent).
# Run from the repository root.
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/content/ComfyUI}"

echo "=== headswap_V2 Colab setup ==="
echo "Repository: $REPO_ROOT"
echo "ComfyUI:    $COMFYUI_PATH"
echo

if [[ ! -f "$REPO_ROOT/scripts/setup_comfyui.sh" ]]; then
  echo "ERROR: Run this script from the repository root (scripts/setup_comfyui.sh not found)." >&2
  exit 1
fi

echo "[1/4] ComfyUI"
if [[ -d "$COMFYUI_PATH" ]]; then
  echo "  Already installed at $COMFYUI_PATH — skipping clone."
else
  echo "  Installing via scripts/setup_comfyui.sh ..."
fi
# setup_comfyui.sh is itself idempotent (skips clone if present).
bash "$REPO_ROOT/scripts/setup_comfyui.sh"
echo

echo "[2/4] Python dependencies (requirements.txt)"
python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
echo "  Done."
echo

echo "[3/4] Download Klein models"
python3 "$REPO_ROOT/scripts/download_models.py" --set klein --comfy "$COMFYUI_PATH"
echo

echo "[4/4] Download Qwen models"
python3 "$REPO_ROOT/scripts/download_models.py" --set qwen --comfy "$COMFYUI_PATH"
echo

echo "Setup complete."
echo "Repository path: $REPO_ROOT"
echo "ComfyUI path:    $COMFYUI_PATH"
echo "Ready to run:"
echo "  python scripts/run_compare.py --gpu --limit 12"
