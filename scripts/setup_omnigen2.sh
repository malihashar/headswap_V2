#!/usr/bin/env bash
# Install OmniGen2 code without downgrading Kaggle/Colab torch.
#
# The upstream repo has NO setup.py / pyproject.toml, so `pip install -e` fails.
# We clone to /tmp/OmniGen2 and put it on PYTHONPATH.
#
# Do NOT `pip install -r requirements.txt` wholesale — it pins torch==2.6.0 and
# will break an existing CUDA 12.8 / torch 2.10 stack.
set -euo pipefail

OMNIGEN2_DIR="${OMNIGEN2_DIR:-/tmp/OmniGen2}"
REPO_URL="${OMNIGEN2_REPO:-https://github.com/VectorSpaceLab/OmniGen2.git}"

echo "=== setup_omnigen2 ==="
echo "Target: $OMNIGEN2_DIR"

if [[ ! -d "$OMNIGEN2_DIR/.git" ]]; then
  git clone --depth 1 "$REPO_URL" "$OMNIGEN2_DIR"
else
  echo "Already cloned — fetching…"
  git -C "$OMNIGEN2_DIR" fetch --depth 1 origin || true
  git -C "$OMNIGEN2_DIR" reset --hard origin/main || true
fi

# Light deps only (skip torch/torchvision/transformers pins).
python3 - <<'PY'
import importlib.util
import subprocess
import sys

need = []
for pkg, mod in [
    ("einops", "einops"),
    ("omegaconf", "omegaconf"),
    ("python-dotenv", "dotenv"),
    ("timm", "timm"),
]:
    if importlib.util.find_spec(mod) is None:
        need.append(pkg)
if need:
    print("Installing missing light deps:", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
else:
    print("Light deps already present")
print("torch/transformers left untouched (use the env already on the machine)")
PY

# Smoke-import via PYTHONPATH
export PYTHONPATH="${OMNIGEN2_DIR}${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<PY
import sys
sys.path.insert(0, "${OMNIGEN2_DIR}")
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
print("OK: OmniGen2Pipeline importable from", "${OMNIGEN2_DIR}")
PY

echo
echo "Add to every notebook cell / shell before run_pipeline:"
echo "  export PYTHONPATH=${OMNIGEN2_DIR}:\$PYTHONPATH"
echo "Or rely on headswap.pipelines.omnigen2 auto-path injection."
echo
echo "Next:"
echo "  python scripts/download_omnigen2.py"
echo "  python scripts/run_pipeline.py --config configs/omnigen2.yaml --pair-id custom_001 --limit 1"
