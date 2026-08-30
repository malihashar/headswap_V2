#!/usr/bin/env bash
# Move InsightFace + rembg off the CPU.
#
# Measured: ~25s of a ~76s swap is CPU-side ONNX (InsightFace's 5 models plus
# rembg's 1GB bria-rmbg segmenter), running before any GPU work. No sampling
# parameter touches it, which is why it stayed invisible while steps and cfg
# were being tuned -- the profiler's clock started after that stage.
#
# Cause: insightface/rembg pull in the CPU-only `onnxruntime` wheel, so
# CUDAExecutionProvider is simply absent. The two packages CONFLICT when both
# are installed, so the CPU one must be removed first rather than installed
# alongside.
set -euo pipefail

echo "== before =="
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())" 2>/dev/null \
  || echo "  onnxruntime not importable"

echo "== removing CPU-only onnxruntime =="
pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true

echo "== installing onnxruntime-gpu =="
pip install -q --no-cache-dir onnxruntime-gpu

echo "== after =="
python3 - <<'PY'
import onnxruntime as ort
provs = ort.get_available_providers()
print(f"  onnxruntime {ort.__version__}")
print(f"  providers: {provs}")
if "CUDAExecutionProvider" in provs:
    print("  OK: CUDA provider present -- detection/segmentation will use the GPU")
else:
    print("  WARNING: no CUDAExecutionProvider. Detection stays on CPU and the")
    print("  ~25s pre-dispatch cost will NOT improve. Usually a CUDA/cuDNN")
    print("  version mismatch with the installed wheel.")
PY
