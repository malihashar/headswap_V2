# FLUX.2 Klein validation report

Date: 2026-07-13  
Project: `headswap_V2`  
Scope: validate FLUX.2 Klein 4B before trusting download URLs / architecture.

## 1. Model verification

| Question | Answer | Official source |
| --- | --- | --- |
| Is FLUX.2 Klein 4B publicly available? | **Yes.** Open weights on Hugging Face. | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence) |
| Apache 2.0 commercial use? | **Yes for 4B** (distilled + base). **Not for 9B** (FLUX Non-Commercial). | [HF card license:apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL model page](https://bfl.ai/models/flux-2-klein), [flux2 README](https://github.com/black-forest-labs/flux2) |
| Image editing supported? | **Yes.** Officially: T2I, single-reference I2I, multi-reference editing. | Same BFL / GitHub tables |
| Official ComfyUI support? | **Yes.** Built-in templates + docs. | [ComfyUI Klein guide](https://docs.comfy.org/tutorials/flux/flux-2-klein), [template library](https://comfy.org/templates/model/flux-2-klein/) |
| Official Comfy checkpoints? | **Yes.** Comfy docs link BFL FP8 UNETs + Comfy-Org text encoder / VAE packages. | Docs download links (see §2) |

### Required components (ComfyUI split-file path)

From Comfy docs storage layout + official template widgets:

1. **Diffusion / UNET** — `flux-2-klein-4b-fp8.safetensors` (distilled, 4 steps)  
2. **Text encoder** — `qwen_3_4b.safetensors` (CLIPLoader `type: flux2`)  
3. **VAE** — `flux2-vae.safetensors`  

Tokenizer: **not a separate Comfy download** for the native UNETLoader/CLIPLoader path; it is handled by Comfy when loading the Qwen3 encoder. (Diffusers users need the `tokenizer/` tree from the BFL repo.)

Optional:

4. **BFS Klein head-swap LoRA** — community MIT LoRA; not required for base multi-ref edit.

---

## 2. Download verification

### Method

1. Confirm file appears on Hugging Face model API (`siblings` / `tree`) with a size.  
2. Probe `resolve/main/...` and require HTTP **302/307** with **`X-Linked-Size`**.  
3. CDN final hop returned **403** from this sandbox to the signed Xet URL (environment restriction). That is **not** treated as “file missing”; existence is established by API size + HF resolve linked size. Fake repos return 404 (no linked size).

Re-run locally:

```bash
python scripts/download_models.py --set klein --verify-only
python scripts/download_models.py --set all --include-optional --verify-only
```

### Klein required (production defaults)

| File | Official repo | Resolve URL | API size | Resolve |
| --- | --- | --- | --- | --- |
| `flux-2-klein-4b-fp8.safetensors` | [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) | `.../resolve/main/flux-2-klein-4b-fp8.safetensors` | 4,070,624,520 | 302 + linked size |
| `qwen_3_4b.safetensors` | [Comfy-Org/flux2-klein-4B](https://huggingface.co/Comfy-Org/flux2-klein-4B) (307 → `Comfy-Org/vae-text-encorder-for-flux-klein-4b`) | docs link under `split_files/text_encoders/` | 8,044,982,048 | 307→302 + linked size |
| `flux2-vae.safetensors` | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) (docs link) | `.../split_files/vae/flux2-vae.safetensors` | 336,213,556 | 302 + linked size |

### Klein optional

| File | Repo | Notes |
| --- | --- | --- |
| `bfs_head_v1.1_optional_flux-klein_4b.safetensors` | [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) | OPTIONAL LoRA (MIT). Off by default (`bfs_lora_strength: 0`). |
| `flux-2-klein-4b.safetensors` | Comfy-Org/flux2-klein-4B | OPTIONAL bf16 distilled UNET (~7.75GB). Prefer FP8. |

### Qwen baseline set (also verified)

| File | Repo |
| --- | --- |
| `qwen_image_edit_2511_fp8mixed.safetensors` | Comfy-Org/Qwen-Image-Edit_ComfyUI |
| `qwen_image_vae.safetensors` | Comfy-Org/Qwen-Image_ComfyUI |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Comfy-Org/Qwen-Image_ComfyUI |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | lightx2v/Qwen-Image-Edit-2511-Lightning |
| `bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors` | Alissonerdx/BFS-Best-Face-Swap |

### Correction about previous script URLs

The earlier script pointed at `Comfy-Org/vae-text-encorder-for-flux-klein-4b`. That repo **does exist** and holds the same split files; official Comfy docs name **`Comfy-Org/flux2-klein-4B`**, which **307-redirects** to the `encorder` twin.  

What was wrong for production:

1. Docs’ preferred **UNET** is BFL **`flux-2-klein-4b-fp8`**, not only the bf16 Comfy package name.  
2. Docs’ preferred **VAE** URL is **`Comfy-Org/flux2-dev`**, not only the Klein twin package.  
3. BFS LoRA was bundled as if required; it is **optional**.

---

## 3. Architecture feasibility

Proposed: **Klein 4B + head/hair mask + crop → multi-ref edit → stitch (+ optional BFS LoRA)**

| Assumption | Feasible today? | Evidence |
| --- | --- | --- |
| Klein can do instruction multi-ref head/person swap | **Yes** | Official multi-ref editing; Comfy `ReferenceLatent` templates; BFS Klein workflow JSON on HF |
| ComfyUI can run it | **Yes** | Native templates (`image_flux2_klein_image_edit_4b_distilled.json` uses UNETLoader / CLIPLoader flux2 / VAELoader / ReferenceLatent / Flux2Scheduler / CFGGuider / SamplerCustomAdvanced) |
| Mask → crop → stitch | **Yes, with caveats** | Comfy has Inpaint Crop/Stitch and community RMBG/SAM nodes. Our portable OpenCV ellipse mask is an approximation; for production prefer SAM3 / BiRefNet / RMBG (BFS Klein workflow uses RMBG + ImageCropByMask) |
| Optional BFS LoRA | **Yes** | Published MIT weights; keep off unless A/B proves gain |

### Incorrect / risky assumptions (called out)

1. **Mask quality ≠ solved by Klein alone.** Without a good head+hair mask, full-frame denoise can still drift clothes/background.  
2. **BFS LoRA is not guaranteed better.** Treat as optional A/B (community mixed results).  
3. **Comfy must be recent enough** for Flux2 / Klein nodes and `CLIPLoader type=flux2`.  
4. **Do not use Klein 9B** in Magic Hour production (non-commercial license).

### Closest practical implementation (if stuck)

If FP8 or Klein templates are unavailable on an old Comfy build: fall back to **Qwen Image Edit 2511 + BFS V5 + mask crop/stitch** (`configs/qwen_improved.yaml`) — already implemented — until Comfy is updated. Do **not** invent alternate Klein checkpoints.

---

## 4. Project changes after validation

1. Rewrote [`scripts/download_models.py`](../scripts/download_models.py) to official docs/BFL URLs, with API + resolve probing, and `--include-optional` / `--verify-only`.  
2. Updated [`configs/klein4b.yaml`](../configs/klein4b.yaml) so primary UNET is `flux-2-klein-4b-fp8.safetensors` with documented sources; BFS remains strength `0.0`.  
3. Adjusted Klein pipeline loader fallback key to `unet_name_fallback`.  

Klein **can** be used as planned. Proceed with Comfy + verified downloads; harden masking for production.
