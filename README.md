# headswap_V2 — Magic Hour head swap prototype

Higher-quality open-source head swap for Magic Hour, focused on **determinism** and **local edits**.

## Winner (recommended promote)

**Primary:** `klein4b_mask_crop_stitch` — FLUX.2 [klein] 4B distilled + head/hair mask + crop → multi-ref edit → soft stitch  
**Fallback:** `qwen_improved_mask_crop` — same locality pattern on current Qwen 2511 + BFS V5  

The current production full-frame Qwen+Lightning path (`qwen_baseline`) stays as the control / legacy arm.

Re-run GPU eval before cutting over; mock ranking is for harness smoke only.

## What’s in this repo

| Path | Purpose |
| --- | --- |
| `configs/klein4b.yaml` | Klein 4B masked crop-stitch (main prototype) |
| `configs/qwen_baseline.yaml` | Faithful port of current Colab Cell 5 |
| `configs/qwen_improved.yaml` | Qwen + official BFS prompt + mask crop-stitch |
| `src/headswap/` | Pipelines, preprocess, metrics, eval harness |
| `workflows/` | Upstream ComfyUI Klein + BFS Klein graphs (reference) |
| `scripts/` | Setup, model download, eval, compare |
| `notebooks/colab_compare.ipynb` | GPU Colab runner |
| `results/COMPARISON.md` | Auto-written after `run_compare.py` |

## Architecture (Klein)

1. Detect face → expand to approximate **head+hair** mask  
2. Crop editable region; keep full body for stitch  
3. Face reference: tight head crop **with shoulders** (no blur-pad)  
4. FLUX.2 Klein 4B multi-ref edit (`ReferenceLatent` body_crop + face), 4 steps, CFG 1.0  
5. Soft-alpha stitch + mild LAB skin match  
6. Metrics: face detect, optional ArcFace identity, body PSNR outside mask, smoothness heuristic  

BFS Flux Klein LoRA is **off by default** (`bfs_lora_strength: 0.0`); A/B with `0.8–1.0`.

## Quick start (CPU mock / CI)

```bash
cd ~/repos/headswap_V2
python3 -m pip install -e ".[dev]"
python3 scripts/prepare_eval_set.py 24
python3 scripts/run_compare.py          # mock all pipelines + write results/COMPARISON.md
pytest -q
```

## GPU run (Colab / RunPod)

```bash
export COMFYUI_PATH=/content/ComfyUI   # or /workspace/ComfyUI
bash scripts/setup_comfyui.sh
python3 scripts/download_models.py --set all --comfy "$COMFYUI_PATH"
python3 scripts/prepare_eval_set.py 24
# Optional: replace data/eval/bodies + faces with real consented photos
python3 scripts/run_compare.py --gpu --limit 12
```

Single pipeline:

```bash
python3 -m headswap.cli --help
# or
PYTHONPATH=src python3 -c "from headswap.cli import main_run; main_run()" --config configs/klein4b.yaml
```

Use the CLI entry points after `pip install -e .`:

```bash
headswap-prepare-eval
headswap-run --config configs/klein4b.yaml --mock
headswap-compare
```

## Eval criteria

- **Success:** face detected + identity above threshold (when InsightFace available) + not pathologically smooth  
- **Body preserve PSNR** outside mask (masked pipelines should score high)  
- **Latency p95** vs ~30s budget (match current ~28s Colab)  

Composite score for promote decision:

`0.6 * success + 0.2 * body_psnr_norm + 0.2 * identity - 0.15 if over latency budget`

## Notes

- Synthetic eval images are for harness validation. Swap in real photos for product decisions.  
- Prefer **Klein 4B** (Apache 2.0). Do **not** ship Klein 9B commercially (non-commercial license).  
- OpenCV SSD mask is a portable fallback; swap in SAM3/BiRefNet for production masks when available.  
