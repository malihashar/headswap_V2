# Winner

## Promote

`klein4b_mask_crop_stitch` — FLUX.2 [klein] 4B distilled with head/hair mask → crop edit → soft stitch.

## Fallback

`qwen_improved_mask_crop` — Qwen Image Edit 2511 + BFS V5 with the same locality pattern and official BFS prompt.

## Do not promote

`qwen_baseline` (full-frame Lightning edit) — keep only as legacy control until GPU cutover is validated.

## Evidence

- Research: locality + higher working resolution on the head crop is the main realism lever vs model-swap-alone.
- License: Klein 4B is Apache 2.0 (commercial-safe); Klein 9B is not.
- Offline harness (`scripts/run_compare.py`): smoke-tested on 24 synthetic pairs; masked pipelines preserve body PSNR; full-frame mock drifts.
- GPU confirmation still required: `python scripts/run_compare.py --gpu` on Colab/RunPod with real photos.

See [COMPARISON.md](COMPARISON.md) and [../README.md](../README.md).
