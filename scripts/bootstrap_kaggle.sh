#!/usr/bin/env bash
# Single-command Kaggle bootstrap for headswap_V2 (idempotent).
#
# From the repo root (after clone or notebook cwd = repo):
#   !bash scripts/bootstrap_kaggle.sh
#
# Absolute fresh notebook (no local files yet) — one line:
#   !curl -fsSL https://raw.githubusercontent.com/malihashar/headswap_V2/main/scripts/bootstrap_kaggle.sh | bash
#
# Layout:
#   /kaggle/working          → ~20GB loop  → ComfyUI + notebook outputs
#   /tmp (overlay)           → ~1T free    → model store + HF staging
set -euo pipefail

REPO_URL="${HEADSWAP_REPO_URL:-https://github.com/malihashar/headswap_V2.git}"
REPO_DIR="${HEADSWAP_REPO_DIR:-/kaggle/working/headswap_V2}"
export COMFYUI_PATH="${COMFYUI_PATH:-/kaggle/working/ComfyUI}"
export HEADSWAP_MODEL_STORE="${HEADSWAP_MODEL_STORE:-/tmp/models}"
export HEADSWAP_STAGING_DIR="${HEADSWAP_STAGING_DIR:-/tmp/_hf_dl_staging}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

STAGE=0
stage() {
  STAGE=$((STAGE + 1))
  echo
  echo "========== [$STAGE] $* =========="
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Resolve / ensure repository, then always run from REPO_DIR.
# ---------------------------------------------------------------------------
ensure_repo() {
  stage "Clone / update repository"

  if [[ ! -d /kaggle/working ]]; then
    die "/kaggle/working not found — this script is for Kaggle notebooks."
  fi

  local script_src="${BASH_SOURCE[0]:-}"
  local from_repo=""

  # Invoked as scripts/bootstrap_kaggle.sh from an existing checkout.
  if [[ -n "$script_src" && -f "$script_src" && "$script_src" != /dev/fd/* ]]; then
    local here
    here="$(cd "$(dirname "$script_src")" && pwd)"
    if [[ -f "$here/setup_comfyui.sh" && -f "$here/../pyproject.toml" ]]; then
      from_repo="$(cd "$here/.." && pwd)"
    fi
  fi

  # Already sitting inside the repo.
  if [[ -z "$from_repo" && -f "$(pwd)/scripts/setup_comfyui.sh" && -f "$(pwd)/pyproject.toml" ]]; then
    from_repo="$(pwd)"
  fi

  if [[ -n "$from_repo" ]]; then
    REPO_DIR="$from_repo"
    echo "Using existing repo at $REPO_DIR"
    cd "$REPO_DIR"
    if [[ -d .git ]]; then
      echo "git pull --ff-only ..."
      git pull --ff-only || echo "  WARN: git pull failed (offline or dirty tree); continuing with local files."
    else
      echo "  No .git directory — skipping pull."
    fi
    return 0
  fi

  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Found $REPO_DIR — updating ..."
    cd "$REPO_DIR"
    git pull --ff-only || echo "  WARN: git pull failed; continuing with local files."
  elif [[ -f "$REPO_DIR/scripts/setup_comfyui.sh" ]]; then
    echo "Found $REPO_DIR (no .git) — using as-is."
    cd "$REPO_DIR"
  else
    echo "Cloning $REPO_URL → $REPO_DIR ..."
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
  fi

  # If we were started via curl|bash, re-exec the real script from the clone.
  if [[ ! -f "${script_src:-}" || "$script_src" == /dev/fd/* ]]; then
    echo "Re-executing $REPO_DIR/scripts/bootstrap_kaggle.sh ..."
    exec bash "$REPO_DIR/scripts/bootstrap_kaggle.sh"
  fi
}

# ---------------------------------------------------------------------------
# True when every required artifact for MODEL_SET is complete under store.
# MODEL_SET: kontext (default) | klein | qwen | all
# ---------------------------------------------------------------------------
models_complete() {
  local set_name="${1:-kontext}"
  python3 - "$HEADSWAP_MODEL_STORE" "$REPO_DIR/scripts/models.json" "$set_name" <<'PY'
import json, sys
from pathlib import Path

store = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
want_set = sys.argv[3]
missing = []
for name, entry in manifest.items():
    if not entry.get("required", True):
        continue
    set_name = str(entry.get("set", "all"))
    if want_set != "all" and set_name != want_set:
        continue
    path = store / entry["path"] / name
    want = int(entry["size"])
    try:
        got = path.stat().st_size if path.is_file() else -1
    except OSError:
        got = -1
    if got != want:
        missing.append(f"{name}: got={got} want={want} path={path}")
if missing:
    print("INCOMPLETE")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)
print("COMPLETE")
sys.exit(0)
PY
}

# ---------------------------------------------------------------------------
# Install deps only when needed (idempotent).
# ---------------------------------------------------------------------------
ensure_deps() {
  stage "Install Python / ComfyUI dependencies (skip if present)"

  mkdir -p "$HEADSWAP_MODEL_STORE" "$HEADSWAP_STAGING_DIR"

  case "$HEADSWAP_MODEL_STORE" in
    /kaggle/working/*) die "HEADSWAP_MODEL_STORE must not be under /kaggle/working (use /tmp/models)" ;;
  esac
  case "$HEADSWAP_STAGING_DIR" in
    /kaggle/working/*) die "HEADSWAP_STAGING_DIR must not be under /kaggle/working (use /tmp/_hf_dl_staging)" ;;
  esac

  echo "COMFYUI_PATH=$COMFYUI_PATH"
  echo "HEADSWAP_MODEL_STORE=$HEADSWAP_MODEL_STORE"
  echo "HEADSWAP_STAGING_DIR=$HEADSWAP_STAGING_DIR"
  echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
  echo
  df -h /tmp || true
  df -h /kaggle/working || true

  echo "→ setup_comfyui.sh"
  bash "$REPO_DIR/scripts/setup_comfyui.sh"

  if ! python3 -c "import headswap" 2>/dev/null; then
    echo "→ pip install -e . (headswap package missing)"
    python3 -m pip install -q -e "$REPO_DIR"
  else
    echo "→ headswap already importable — refreshing editable install lightly"
    python3 -m pip install -q -e "$REPO_DIR"
  fi

  if [[ -f "$REPO_DIR/requirements.txt" ]]; then
    echo "→ pip install -r requirements.txt"
    python3 -m pip install -q -r "$REPO_DIR/requirements.txt"
  fi

  python3 -m pip install -q -U "huggingface_hub>=0.24.0"

  if command -v aria2c >/dev/null 2>&1; then
    echo "→ aria2c already installed"
  else
    echo "→ installing aria2"
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2
    else
      echo "  WARN: apt-get unavailable; aria2 fallback disabled"
    fi
  fi

  echo "Deps OK."
}

# ---------------------------------------------------------------------------
# Models: default Kontext only (Klein/Qwen via BOOTSTRAP_MODEL_SET or flags).
# ---------------------------------------------------------------------------
ensure_models() {
  local model_set="${BOOTSTRAP_MODEL_SET:-kontext}"
  stage "Models under $HEADSWAP_MODEL_STORE (set=$model_set)"

  local dl_common=(
    --comfy "$COMFYUI_PATH"
    --store-dir "$HEADSWAP_MODEL_STORE"
    --staging-dir "$HEADSWAP_STAGING_DIR"
    --backend auto
    --disable-xet
    --manifest "$REPO_DIR/scripts/models.json"
  )

  if models_complete "$model_set"; then
    echo "Required $model_set models already present with correct sizes."
    echo "Recreating ComfyUI symlinks only (no re-download) ..."
  else
    echo "Missing or incomplete $model_set models — downloading into $HEADSWAP_MODEL_STORE ..."
    if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
      echo "  HINT: set HF_TOKEN if gated Hugging Face assets fail to download."
    fi
  fi

  if [[ "$model_set" == "kontext" && -f "$REPO_DIR/scripts/download_kontext.py" ]]; then
    python3 "$REPO_DIR/scripts/download_kontext.py" "${dl_common[@]}"
  else
    python3 "$REPO_DIR/scripts/download_models.py" --set "$model_set" "${dl_common[@]}"
  fi
}

# ---------------------------------------------------------------------------
# Custom eval or instructions.
# ---------------------------------------------------------------------------
run_or_prompt_custom() {
  stage "Custom eval (body.png / face.png)"

  local body="$REPO_DIR/data/custom/body.png"
  local face="$REPO_DIR/data/custom/face.png"

  if [[ -f "$body" && -f "$face" ]]; then
    echo "Found $body and $face"
    echo "→ prepare_eval_set.py --custom"
    python3 "$REPO_DIR/scripts/prepare_eval_set.py" --custom
    echo "→ run_compare.py --gpu --limit 1"
    python3 "$REPO_DIR/scripts/run_compare.py" --gpu --limit 1
    CUSTOM_RAN=1
  else
    CUSTOM_RAN=0
    echo "Custom images not found."
    echo
    echo "Upload these two files, then re-run this script:"
    echo "  $body   ← destination / body image"
    echo "  $face   ← source face / head image"
    echo
    echo "In a Kaggle cell you can also:"
    echo "  from pathlib import Path"
    echo "  Path('data/custom').mkdir(parents=True, exist_ok=True)"
    echo "  # then copy/upload body.png and face.png into data/custom/"
    echo "  !bash scripts/bootstrap_kaggle.sh"
  fi
}

# ---------------------------------------------------------------------------
# Final summary.
# ---------------------------------------------------------------------------
print_summary() {
  stage "Summary"

  echo "--- GPU ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    nvidia-smi -L || true
  else
    echo "(nvidia-smi not available)"
  fi
  python3 - <<'PY' || true
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"vram_gb={props.total_memory / 1024**3:.1f}")
PY

  echo
  echo "--- Paths ---"
  echo "REPO_DIR=$REPO_DIR"
  echo "COMFYUI_PATH=$COMFYUI_PATH"
  echo "HEADSWAP_MODEL_STORE=$HEADSWAP_MODEL_STORE"
  echo "HEADSWAP_STAGING_DIR=$HEADSWAP_STAGING_DIR"
  echo
  df -h /tmp || true
  df -h /kaggle/working || true

  echo
  echo "--- Model store (top level) ---"
  if [[ -d "$HEADSWAP_MODEL_STORE" ]]; then
    # Avoid SIGPIPE under pipefail when head closes early.
    set +o pipefail
    find "$HEADSWAP_MODEL_STORE" -maxdepth 2 -type f 2>/dev/null | head -40 || true
    set -o pipefail
  fi

  echo
  echo "--- Outputs ---"
  local result_paths metrics_paths
  result_paths="$(find "$REPO_DIR/results" -path '*/images/custom_001/result.png' 2>/dev/null | sort || true)"
  metrics_paths="$(find "$REPO_DIR/results" -name metrics.json 2>/dev/null | sort || true)"

  if [[ -n "$result_paths" ]]; then
    echo "Result image(s):"
    echo "$result_paths"
  else
    echo "Result image: (not generated yet — upload data/custom/body.png + face.png and re-run)"
  fi

  if [[ -n "$metrics_paths" ]]; then
    echo "Metrics:"
    echo "$metrics_paths"
  else
    echo "Metrics: (none yet)"
  fi

  # Preferred single-pair paths for the common case
  for pipe in flux_kontext klein4b qwen_improved qwen_baseline; do
    local r="$REPO_DIR/results/$pipe/images/custom_001/result.png"
    local m="$REPO_DIR/results/$pipe/metrics.json"
    if [[ -f "$r" ]]; then
      echo
      echo "Primary result:  $r"
      [[ -f "$m" ]] && echo "Primary metrics: $m"
      break
    fi
  done

  echo
  echo "Bootstrap complete."
}

# ---------------------------------------------------------------------------
main() {
  echo "=== headswap_V2 Kaggle bootstrap ==="
  # Default model set is Kontext. Override with:
  #   BOOTSTRAP_MODEL_SET=klein|qwen|all bash scripts/bootstrap_kaggle.sh
  #   bash scripts/bootstrap_kaggle.sh --klein
  #   bash scripts/bootstrap_kaggle.sh --kontext   (default)
  #   bash scripts/bootstrap_kaggle.sh --no-models
  local skip_models=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --kontext) BOOTSTRAP_MODEL_SET=kontext ;;
      --klein) BOOTSTRAP_MODEL_SET=klein ;;
      --qwen) BOOTSTRAP_MODEL_SET=qwen ;;
      --all-models) BOOTSTRAP_MODEL_SET=all ;;
      --no-models) skip_models=1 ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/bootstrap_kaggle.sh [--kontext|--klein|--qwen|--all-models|--no-models]
Default model download set: kontext → /tmp/models
EOF
        exit 0
        ;;
      *)
        echo "ERROR: unknown argument: $1" >&2
        exit 1
        ;;
    esac
    shift
  done
  export BOOTSTRAP_MODEL_SET="${BOOTSTRAP_MODEL_SET:-kontext}"

  CUSTOM_RAN=0
  ensure_repo
  ensure_deps
  if [[ "$skip_models" -eq 0 ]]; then
    ensure_models
  else
    echo "Skipping model download (--no-models)."
  fi
  run_or_prompt_custom
  print_summary
}

main "$@"
