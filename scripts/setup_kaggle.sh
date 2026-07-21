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
# Model downloads are opt-in (default installs ComfyUI + Python deps only):
#   bash scripts/setup_kaggle.sh              # no model download
#   bash scripts/setup_kaggle.sh --kontext    # FLUX.1 Kontext weights → /tmp/models
#   bash scripts/setup_kaggle.sh --klein      # FLUX.2 Klein (+ BFS) weights → /tmp/models
#   bash scripts/setup_kaggle.sh --krea2      # Krea 2 Identity Edit stack → /tmp/models
#   bash scripts/setup_kaggle.sh --kontext --klein   # both sets
#
# Equivalent to --kontext:
#   bash scripts/setup_kaggle.sh
#   python scripts/download_kontext.py
#
# Run from the repository root with Internet + GPU enabled.
set -euo pipefail

REPO_ROOT="$(pwd)"
export COMFYUI_PATH="${COMFYUI_PATH:-/kaggle/working/ComfyUI}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/tmp/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/tmp/_hf_dl_staging}"

DOWNLOAD_KONTEXT=0
DOWNLOAD_KLEIN=0
DOWNLOAD_QWEN=0
DOWNLOAD_KREA2=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_kaggle.sh [options]

  (default)     Install ComfyUI + Python deps + aria2. Download NO models.
  --kontext     Download FLUX.1 Kontext set into /tmp/models (see configs/flux_kontext.yaml).
  --klein       Download FLUX.2 Klein set into /tmp/models (existing Klein baseline).
  --krea2       Download Krea 2 Identity Edit set + install comfyui-krea2edit nodes.
  --qwen        Download Qwen Image Edit 2511 set into /tmp/models (optional legacy).
  -h, --help    Show this help.

Model weights always go to HEADSWAP_MODEL_STORE (/tmp/models by default),
never under /kaggle/working.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kontext) DOWNLOAD_KONTEXT=1 ;;
    --klein) DOWNLOAD_KLEIN=1 ;;
    --qwen) DOWNLOAD_QWEN=1 ;;
    --krea2) DOWNLOAD_KREA2=1 ;;
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

echo "=== headswap_V2 Kaggle setup ==="
echo "Repository:  $REPO_ROOT"
echo "ComfyUI:     $COMFYUI_PATH"
echo "Model store: $HEADSWAP_MODEL_STORE"
echo "Staging:     $HEADSWAP_STAGING_DIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "Downloads:   kontext=$DOWNLOAD_KONTEXT klein=$DOWNLOAD_KLEIN qwen=$DOWNLOAD_QWEN krea2=$DOWNLOAD_KREA2"
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

STEP=3
if [[ "$DOWNLOAD_KONTEXT" -eq 1 ]]; then
  STEP=$((STEP + 1))
  echo "[$STEP] Download Kontext models → $HEADSWAP_MODEL_STORE"
  if [[ -f "$REPO_ROOT/scripts/download_kontext.py" ]]; then
    # Prefer the dedicated wrapper (pins /tmp store + kontext set).
    python3 "$REPO_ROOT/scripts/download_kontext.py" \
      --comfy "$COMFYUI_PATH" \
      --store-dir "$HEADSWAP_MODEL_STORE" \
      --staging-dir "$HEADSWAP_STAGING_DIR" \
      --backend auto \
      --disable-xet \
      --manifest "$REPO_ROOT/scripts/models.json"
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
    python3 "$REPO_ROOT/scripts/download_krea2.py" \
      --comfy "$COMFYUI_PATH" \
      --store-dir "$HEADSWAP_MODEL_STORE" \
      --staging-dir "$HEADSWAP_STAGING_DIR" \
      --backend auto \
      --disable-xet \
      --manifest "$REPO_ROOT/scripts/models.json"
  else
    python3 "$REPO_ROOT/scripts/download_models.py" --set krea2 "${DL_COMMON[@]}"
  fi
  echo
fi

echo "Setup complete."
echo "store_dir=$HEADSWAP_MODEL_STORE"
echo "staging_dir=$HEADSWAP_STAGING_DIR"
echo "COMFYUI_PATH=$COMFYUI_PATH"
echo
df -h /tmp || true
df -h /kaggle/working || true
echo

if [[ "$DOWNLOAD_KONTEXT" -eq 0 && "$DOWNLOAD_KLEIN" -eq 0 && "$DOWNLOAD_QWEN" -eq 0 && "$DOWNLOAD_KREA2" -eq 0 ]]; then
  echo "No model sets downloaded (default)."
  echo "Next (Kontext — recommended):"
  echo "  python scripts/download_kontext.py"
  echo "  # or: bash scripts/setup_kaggle.sh --kontext"
  echo "Optional:"
  echo "  bash scripts/setup_kaggle.sh --klein"
  echo "  bash scripts/setup_kaggle.sh --krea2"
  echo "  bash scripts/setup_kaggle.sh --qwen"
else
  echo "Ready to run:"
  echo "  export COMFYUI_PATH=$COMFYUI_PATH"
  if [[ "$DOWNLOAD_KONTEXT" -eq 1 ]]; then
    echo "  python scripts/run_pipeline.py --config configs/flux_kontext.yaml --limit 1"
  fi
  if [[ "$DOWNLOAD_KLEIN" -eq 1 ]]; then
    echo "  python scripts/run_pipeline.py --config configs/klein4b.yaml --limit 1"
  fi
  if [[ "$DOWNLOAD_KREA2" -eq 1 ]]; then
    echo "  python scripts/run_pipeline.py --config configs/krea2_identity_edit.yaml --pair-id custom_001 --limit 1"
  fi
fi
