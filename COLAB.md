# Running headswap_V2 on Google Colab

Internal **production demo** for Krea 2 Identity Edit (Magic Hour engineers).

## Open the notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. **File → Upload notebook** → `notebooks/krea2_identity_edit.ipynb`
3. **Runtime → Change runtime type → GPU** (prefer **A100**).
4. Edit §1 knobs if needed → **Runtime → Run all**.

## User knobs (§1)

| Knob | Default | Meaning |
| --- | --- | --- |
| `SEED` | `46` | Deterministic sampling seed |
| `PROMPT` | yaml default | Optional prompt override |
| `STEPS` | `8` | Denoising steps |
| `CFG` | `1.0` | Guidance scale |
| `OUTPUT_LONG_SIDE` | `1024` | Final body canvas long side (px) |
| `STITCH` | `True` | Mask → crop → edit → soft stitch |
| `DEBUG` | `False` | Verbose logs + `debug_*.png` |

## Flow

| Step | Action |
| --- | --- |
| GPU check | Fail early if no CUDA |
| Setup | Drive mount, clone, ComfyUI + nodes, model download/verify |
| Upload | Body + face (JPG/PNG/WEBP); face detection required |
| Preflight | GPU, weights, both faces |
| Run | Warm-held pipeline; clean progress; fixed seed |
| Results | Side-by-side preview + `HEADSWAP_RESULT.png` download |

## Paths

| Role | Path |
| --- | --- |
| Repo | `/content/headswap_V2` |
| ComfyUI | `/content/ComfyUI` |
| Model cache | `/content/drive/MyDrive/headswap_V2/models` |
| Stable output | `/content/headswap_outputs/HEADSWAP_RESULT.png` |

## Warm re-runs

After the first successful §6, re-run **§6 only** (or re-upload then §5–§7). Models stay loaded (`model_cache_hit=True`).

## First custom-node install

On a brand-new runtime, after the first `setup_krea2_nodes.sh`, **restart the session once**, then Run all from §1.

## HF login

Not required — Krea2 assets used here are public.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| No GPU | Runtime → GPU (A100 preferred) |
| Upload / WEBP errors | Re-pull repo; upload cell uses `io.BytesIO` via `colab_demo.save_upload` |
| No face detected | Use a clear front-facing photo |
| Slow on T4 | Expected; switch to A100 for demo latency |
| Need traceback | Set `DEBUG=True` in §1 |
