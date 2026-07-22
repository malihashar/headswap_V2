#!/usr/bin/env bash
# Prepare a Google Colab runtime for headswap_V2 (idempotent).
#
# Layout (Colab):
#   /content/ComfyUI                              → ComfyUI (ephemeral disk, fast)
#   /content/drive/MyDrive/headswap_V2/models     → persistent model cache on Drive
#   /content/_hf_dl_staging                       → HF download staging (ephemeral)
#   /content/headswap_V2                          → this repository
#
# Model downloads are opt-in:
#   bash scripts/setup_colab.sh              # ComfyUI + deps only
#   bash scripts/setup_colab.sh --krea2      # + Krea2 Identity Edit nodes & weights
#   bash scripts/setup_colab.sh --klein      # + Klein weights
#   bash scripts/setup_colab.sh --kontext    # + Kontext weights
#   bash scripts/setup_colab.sh --qwen       # + Qwen weights
#
# Run from the repository root after mounting Google Drive (recommended for cache).
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/content/ComfyUI}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/content/drive/MyDrive/headswap_V2/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/content/_hf_dl_staging}"

DOWNLOAD_KONTEXT=0
DOWNLOAD_KLEIN=0
DOWNLOAD_QWEN=0
DOWNLOAD_KREA2=0
REQUIRE_DRIVE=1

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_colab.sh [options]

  (default)     Install ComfyUI + Python deps + aria2. Download NO models.
  --krea2       Install comfyui-krea2edit + download Krea 2 Identity Edit set.
  --kontext     Download FLUX.1 Kontext set.
  --klein       Download FLUX.2 Klein set.
  --qwen        Download Qwen Image Edit 2511 set.
  --no-drive    Allow ephemeral /content model store (not persisted).
  -h, --help    Show this help.

Models default to Google Drive so they survive runtime reconnects.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kontext) DOWNLOAD_KONTEXT=1 ;;
    --klein) DOWNLOAD_KLEIN=1 ;;
    --qwen) DOWNLOAD_QWEN=1 ;;
    --krea2) DOWNLOAD_KREA2=1 ;;
    --no-drive) REQUIRE_DRIVE=0 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

# If Drive is missing and --no-drive was not set, fall back to /content/models
# only when REQUIRE_DRIVE=0; otherwise error with clear instructions.
if [[ "$REQUIRE_DRIVE" -eq 1 && ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "In a Colab cell, run:" >&2
  echo "  from google.colab import drive" >&2
  echo "  drive.mount('/content/drive')" >&2
  echo "Or re-run with --no-drive to use ephemeral /content/models." >&2
  exit 1
fi

if [[ "$REQUIRE_DRIVE" -eq 0 && ! -d /content/drive/MyDrive ]]; then
  export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/content/models}"
fi

echo "=== headswap_V2 Colab setup ==="
echo "Repository:  $REPO_ROOT"
echo "ComfyUI:     $COMFYUI_PATH"
echo "Model store: $HEADSWAP_MODEL_STORE"
echo "Staging:     $HEADSWAP_STAGING_DIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "Downloads:   kontext=$DOWNLOAD_KONTEXT klein=$DOWNLOAD_KLEIN qwen=$DOWNLOAD_QWEN krea2=$DOWNLOAD_KREA2"
echo

if [[ ! -f "$REPO_ROOT/scripts/setup_comfyui.sh" ]]; then
  echo "ERROR: Run this script from the repository root (scripts/setup_comfyui.sh not found)." >&2
  exit 1
fi

echo "[1/3] ComfyUI → $COMFYUI_PATH"
bash "$REPO_ROOT/scripts/setup_comfyui.sh"
echo

echo "[2/3] Python dependencies (requirements.txt)"
echo "  Note: do not install hf_xet; setup forces HF_HUB_DISABLE_XET=1."
python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
echo "  Done."
echo

echo "[3/3] aria2 (resumable HTTP fallback)"
if command -v aria2c >/dev/null 2>&1; then
  echo "  aria2c already installed."
else
  echo "  Installing aria2 ..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2
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

STEP=3
if [[ "$DOWNLOAD_KONTEXT" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Kontext models → $HEADSWAP_MODEL_STORE"
  if [[ -f "$REPO_ROOT/scripts/download_kontext.py" ]]; then
    python3 "$REPO_ROOT/scripts/download_kontext.py" "${DL_COMMON[@]}"
  else
    python3 "$REPO_ROOT/scripts/download_models.py" --set kontext "${DL_COMMON[@]}"
  fi
  echo
fi

if [[ "$DOWNLOAD_KLEIN" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Klein models → $HEADSWAP_MODEL_STORE"
  python3 "$REPO_ROOT/scripts/download_models.py" --set klein "${DL_COMMON[@]}"
  echo
fi

if [[ "$DOWNLOAD_QWEN" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Qwen models → $HEADSWAP_MODEL_STORE"
  python3 "$REPO_ROOT/scripts/download_models.py" --set qwen "${DL_COMMON[@]}"
  echo
fi

if [[ "$DOWNLOAD_KREA2" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Install Krea2 edit custom nodes"
  bash "$REPO_ROOT/scripts/setup_krea2_nodes.sh"
  echo
  STEP=$((STEP + 1))
  echo "[$STEP] Download Krea2 Identity Edit models → $HEADSWAP_MODEL_STORE"
  if [[ -f "$REPO_ROOT/scripts/download_krea2.py" ]]; then
    python3 "$REPO_ROOT/scripts/download_krea2.py" "${DL_COMMON[@]}"
  else
    python3 "$REPO_ROOT/scripts/download_models.py" --set krea2 "${DL_COMMON[@]}"
  fi
  echo
fi

echo "Setup complete."
echo "COMFYUI_PATH=$COMFYUI_PATH"
echo "HEADSWAP_MODEL_STORE=$HEADSWAP_MODEL_STORE"
echo
if [[ "$DOWNLOAD_KREA2" -eq 1 ]]; then
  echo "Ready to run:"
  echo "  python scripts/run_pipeline.py --config configs/krea2_identity_edit.yaml --pair-id custom_001 --limit 1"
fi
