# full_frame head-swap pipeline: failure analysis

Investigation date: 2026-08-07. Scope: `src/headswap/pipelines/krea2.py`
(`Krea2IdentityEditPipeline`), `src/headswap/preprocess.py`,
`src/headswap/segmentation.py`, `configs/krea2_identity_edit.yaml`.

Method: 4 parallel read-only investigation agents (Pipeline/Routing,
Geometry/Alignment, Masking/Compositing, Quality/Resolution), each
independently reading the relevant code and reporting evidence with
file:line citations. Findings below are the synthesis, cross-checked
against the actual current source (all cited line numbers and yaml values
verified directly against the repo as of this writing).

**Environment constraint**: this investigation ran with no GPU/torch
available, so nothing here was confirmed by rendering an actual image. All
"confirmed" ratings mean *confirmed by reading the code* (the mechanism
provably exists and executes as described), not *confirmed by observing
the artifact on a real image*. Where a finding is architecturally certain
but its visual magnitude is unverified, that's called out explicitly.

## 1. Current pipeline (full_frame branch, as actually implemented)

```
body (original photo)
  -> body_full = resize_to_megapixels(body, target_mp=1.25, ...)      [krea2.py:3091-3106]
  -> selected_face, all_faces = select_face_box(body_full)            [krea2.py:3108-3113]  (PASS-1 detection)
  -> scene = resize_to_megapixels(body_full, ...)                     [_build_full_frame_inputs, krea2.py:1691-1702]
  -> person = resize_contain(face_crop, scene.size)                   [krea2.py:1714-1716]  (donor, full-bleed)
  -> freeze_mask = _build_full_frame_freeze_mask(body_full, selected_face, all_faces)   [krea2.py:1151-1190]  (PASS-1 mask)
  -> sample_meta = _sample_edit(scene, person, ...)                   [krea2.py:3433-ish]   (RAW Krea2 output, "edited")
  -> out = edited (resized to body_full.size if needed)                [krea2.py:3531-3533]
  -> [optional] clamp_edited_head_scale                                [krea2.py:3548, gated OFF -- see 3.2]
  -> [optional] procrustes_align_generated_to_body                     [krea2.py:3572-3597, gated OFF]
  -> out = _freeze_full_frame_outside_selected(body_full, out, freeze_mask, ...)  [krea2.py:1192-1266]  (PASS-1 COMPOSITE #1)
       - dilate(freeze_mask, 10) -> suppress_neighbor -> ensure_coverage
       - feathered_soft_composite(body_full, out, stitch_mask, extra_blur_px=30)
       - lab_histogram_match_face(out, body_full, stitch_mask, strength=0.15)
  -> out, refine_diag = _refine_full_frame_face(out, face_crop, body_full, ...)  [krea2.py:1268-1472]  (PASS 2)
       - selected2, faces2 = select_face_box(out)                     [krea2.py:1317-1322]  (PASS-2 detection, INDEPENDENT of pass 1)
       - flags = _tight_crop_flags(out, selected2, faces2)            [krea2.py:1327, reads BASE mask_* yaml keys]
       - built = _build_scene_person(out, face_crop, selected2, top_ext=flags[...], ...)  [krea2.py:1328-1341]  (PASS-2 mask, DIFFERENT shape)
       - refine_sample = _sample_edit(refine_scene, built["person"], ...)  [krea2.py:1378-1387]  (SECOND Krea2 sample)
       - [optional] procrustes_align_edited_crop_to_body_box + _blend_procrustes_face_only  [krea2.py:1391-1425, OFF by yaml:295]
       - refined = _stitch_edited(out, edited_for_stitch, built["mask"], ..., color_ref=out)  [krea2.py:1439-1450]  (PASS-1... no, PASS-2 COMPOSITE #2, onto already-composited `out`)
  -> out, pose_diag = _relock_head_direction(out, body_full, selected_face)  [krea2.py:3654-3658, OFF by yaml default]
  -> return PipelineResult(image=out, ...)                             [krea2.py:3801-3802 -- no resize back toward body's original resolution]
```

Everything downstream of the first `_sample_edit` call happens on a photo
permanently capped at `full_frame_target_mp` (default 1.25MP, `max_dim`
2048px) -- there is no upscale-back-to-source step anywhere in this path.

## 2. Confirmed problems (mechanism verified by reading the code)

### 2.1 The head/hair region is composited TWICE, by two independently-detected, independently-shaped masks

This is the single highest-confidence, most-convergent finding: **3 of 4
agents independently identified it**, and it directly explains the
persistent "ghost/leftover artifact around the old head and hair" that
survived several previous fix attempts (which targeted symptoms
downstream of this, not this root cause).

Pass 1 (`_freeze_full_frame_outside_selected`) and Pass 2
(`_refine_full_frame_face`'s own `_stitch_edited` call) each build and
apply their *own* mask to *roughly* the same region, but the two masks are
neither the same shape nor centered on the same detection:

| | Pass 1 (`_build_full_frame_freeze_mask`) | Pass 2 (`_build_scene_person` via `_tight_crop_flags`) |
|---|---|---|
| face box source | `selected_face`, detected on **`body_full`** (pristine) at krea2.py:3108-3113 | re-detected on **`out`** (already pass-1-generated) at krea2.py:1317-1322 |
| top_extend | 1.55 (yaml:223) | 1.55 (yaml:99) |
| side_extend | **0.42** (yaml:224) | **0.60** (yaml:100) |
| bot_extend | 0.40 (yaml:225) | 0.40 (yaml:101) |
| expand_px | 12 (yaml:226) | 18 (yaml:95) |
| blur_px | 24 (yaml:227) | 12 (yaml:96) |
| extra dilate | +10px (yaml:228) | none |
| composite feather | +30px extra blur (yaml:229) | +10px extra blur (yaml:102) |
| LAB match strength | 0.15, target=`body_full` (yaml:241) | 0.35, target=**`out`** i.e. pass-1's own output (yaml:103, krea2.py:1445) |
| coverage floor | `ensure_selected_face_mask_coverage` called (krea2.py:1178, 1222) | **not called** -- `_build_scene_person` has no equivalent |

Consequences, all confirmed by reading the code:

- **Different face box.** Pass 2 detects on pixels pass 1 already changed
  (new identity, possibly shifted eye/jaw landmarks), so its mask is
  centered on a potentially different point than pass 1's.
- **Different width (side_extend 0.42 vs 0.60).** The ring between the two
  extents gets *only* pass 2's edit, feathered at a different radius (10px)
  than the interior region, which carries *both* pass 1's edit (feathered
  up to ~64px: blur 24 + dilate 10 + feather 30) and pass 2's edit stacked
  on top.
- **Double, non-idempotent LAB color match**, targeting two different
  reference images (`body_full` then `out`) at two different strengths
  (0.15 then 0.35), in two different mask footprints.
- **No coverage guarantee.** Pass 1 has an explicit floor
  (`ensure_selected_face_mask_coverage`) making sure its mask fully covers
  the selected face; pass 2 has no equivalent, and also has no guarantee it
  *fully re-covers pass 1's own bleed zone* -- so a thin sliver of
  pass-1-only-altered pixels can be left exposed at the boundary.

**The ellipse mask backend is purely geometric, not hair-aware** (confirmed:
`head_mask_backend: ellipse` is the only backend actually active in this
deployment -- `sam2`/`birefnet` both fall back to `ellipse`, per
`segmentation.py:127-176`; `head_hair_mask_from_face`,
`preprocess.py:1166-1215`, computes the ellipse purely from the face bbox
+ extend fractions, no pixel/segmentation signal at all). For a
**short-haired subject** (the desert-photo test case), most of the
`top_extend=1.55` allowance above the face is open sky, not hair. Since
pass 1 and pass 2 build this "sky allowance" region from two different
face detections, their two sky-inclusive regions do not pixel-align --
this is the most likely specific mechanism for the artifact appearing
**above the head, in the sky**, rather than looking like true leftover
hair.

**Confirmed ruled out for the current default config**: the earlier
`_blend_procrustes_face_only` fix (a prior attempt at a related problem)
is verified OFF by `full_frame_face_refine_procrustes: false`
(yaml:295, overriding the code's `True` fallback at krea2.py:1397) — so it
cannot be contributing to the artifact visible in the most recent
screenshots. Whatever the user is still seeing predates/is independent of
that mechanism.

**Instrumentation gap (also confirmed):** `_stitch_edited`'s own debug
frames (`stitch_mask`, `composite_before_lab`, `composite_after_lab`) are
written into the `stitch_debug` dict during pass 2 (krea2.py:1448), but the
block that promotes `stitch_debug` into saved PNGs is gated by
`if do_stitch:` (krea2.py:3673), and `do_stitch` is hardcoded `False` for
`edit_mode == "full_frame"` (krea2.py:3143). **So pass 2's actual stitch
mask has never been saved to disk, even with `save_debug=True`** -- nobody
could have visually confirmed or denied this hypothesis from a real run
without a code change. Fixed as part of this round (see §6).

### 2.2 No scale anchor: the donor reference is placed independent of the target's actual head size, and every available safety clamp is disabled or gated off for exactly the case that matters most

`_build_full_frame_inputs`'s default person-prep
(`full_frame_match_crop_identity: true`, yaml:247, the production default)
does:

```python
person = resize_contain(face_crop.convert("RGB"), scene.size, fill=(0, 0, 0))
```

This full-bleeds the donor face crop to fill the *entire scene canvas*,
with **zero reference to `selected_face`'s actual height or position**.
The one branch that *does* measure target proportion
(`place_face_at_height_frac`, using `face_h_frac` from `selected_face.height`,
krea2.py:1710-1725) is dead code by default -- `match_crop_id=True` always
wins the `if` at krea2.py:1714.

Contrast with crop_stitch (`_build_scene_person`): the crop *window itself*
is sized directly from the detected face bbox (`crop_with_mask`,
krea2.py:688-690), so target head scale is implicitly encoded in the
canvas geometry. full_frame's canvas is purely a function of total photo
pixel count (`resize_to_megapixels(body_full, target_mp=1.25, ...)`) and
never reads `selected_face` at all. This is the architectural root cause:
**crop_stitch has an implicit scale constraint; full_frame does not.**

And there is currently no explicit safety net either:

- `full_frame_clamp_head_scale: false` by default (yaml:253, comment:
  *"Post-sample clamp on the full group frame can target the wrong face --
  OFF until inference-path identity parity is proven"*).
- Even if turned on, the gate at krea2.py:3548 is `if multi_person and
  do_clamp:` -- and `multi_person = len(all_faces) > 1` (krea2.py:3089).
  The single-person, full-body-route case (the primary way full_frame gets
  triggered today, via `_resolve_body_route`) has `multi_person = False`,
  so **the clamp would never fire for this case even if enabled.** This
  looks like a real gate bug, not an intentional design choice.
- `full_frame_procrustes_align: false` by default (yaml:257).
- The refine pass's own `_stitch_edited` call hardcodes `multi_person=False`
  (krea2.py:1446), which *also* disables `_stitch_edited`'s internal clamp
  gate (`stitch_multi = multi_person and not spp`).
- `_refine_full_frame_face`'s own docstring explicitly *assumes* "full_frame
  gives correct body/head proportions" and re-detects "the (already
  proportion-correct) head in `out`" -- an assumption directly contradicted
  by the above.

**Net: for the single-person full-body case (the main full_frame trigger),
there is currently no scale-correction mechanism anywhere in the pipeline.**
This is the most direct, best-evidenced explanation for "head
proportions/scale are noticeably wrong."

### 2.3 full_frame uses a different identity LoRA than crop_stitch -- a confound on the "full_frame face quality is worse" comparison

```
identity_lora_name:            krea2_identity_edit_v1_2_r64.safetensors   (yaml:21, crop_stitch)
full_frame_identity_lora_name: krea2_identity_edit_v1_2.safetensors       (yaml:197, full_frame -- no "r64")
```

Selected in `_load_models` (krea2.py:459-552) and `_sample_edit`
(krea2.py:1863-1866) whenever `multi_person_edit_mode` resolves to
`full_frame`. This was an empirical A/B pick (yaml:196 comment: *"A/B
winner d_ff_full_rb4: full v1.2 LoRA + ref_boost=4.0"*), not a
resolution-only ablation. **Any face-quality delta currently attributed to
"full_frame has lower effective resolution" is confounded with an
un-isolated LoRA-weights difference** -- these have never been tested
independently.

`full_frame_ref_boost` (4.0, yaml:198) is also higher than crop_stitch's
`ref_boost` (3.5, yaml:52). `ref_boost` is a cross-attention logit bias on
the donor reference (krea2.py:366-369 comment: *"ref_boost!=1 builds attn
logit bias -> masked SDPA path... required for identity fidelity"*); the
actual attention math lives in the external
`comfyui-krea2edit` node package, not in this repo, so its
behavior/artifacts at high values can't be traced further here -- it has
no in-repo-documented upper-bound safety validation.

### 2.4 Quantified resolution gap

`_build_full_frame_inputs` resizes the whole photo to `full_frame_target_mp`
(1.25MP, min 1.0/max 1.5, yaml:189-192). For a typical full-body photo
(aspect ~0.6-0.75) that's roughly 1291-1491px tall. With
`face_height_frac` 0.10-0.15 (typical full-body framing), the face renders
at only **~129-224px tall** in the diffusion scene.

crop_stitch's effective face render height is architecture-invariant:
`crop_long_side=768` (yaml:93) combined with the mask geometry
(`top_extend=1.55, bot_extend=0.40` -> crop height ≈ 2.95 × face height)
gives an effective face height of **≈768/2.95 ≈ 260px**, regardless of
source resolution.

Gap ≈ 1.2-2x for typical full-body shots, growing to 3x+ for smaller face
fractions (e.g. group shots, where full_frame is also the routed path).
This matches the pipeline's own diagnostic
(`_refine_full_frame_face`'s `resolution_gain_x`, krea2.py:1354, whose
docstring already cites "~3x" for `face_height_frac=0.12`).

Raising `full_frame_target_mp` only partially helps: face pixels scale as
√(target_mp), so even 1.25 -> 3.0MP only buys ~1.55x, while the dual-ref
attention cost is documented in-repo as super-linear on the target hardware
("O(n²) dominate on T4", krea2.py:370-373) -- an expensive, bounded lever
on its own.

**The donor identity reference ("person" image) is likely *not* the
resolution bottleneck.** In both paths, `face_crop` (a plain, un-resized
crop of the donor photo, `preprocess.py:782-833`) is placed via
`resize_contain` onto the scene canvas. full_frame's scene canvas
(~1.25MP) is *larger* than crop_stitch's (~0.54MP), so the *same* native
donor crop lands at equal-or-more pixels in full_frame, not fewer. The
constraint is the SCENE's spatial budget where the face must render, not
the donor reference's fidelity.

## 3. Likely problems (plausible mechanism, not independently confirmed by rendering)

- Pass 1's cumulative soft-bleed footprint (blur 24 + dilate 10 + feather 30)
  can plausibly extend past pass 2's narrower feathered boundary near the
  hair crown, leaving a low-alpha halo. Architecturally sound; visual
  magnitude unverified without a render.
- InsightFace landmark precision is plausibly degraded at the small
  face-pixel counts full_frame delivers (e.g. ~45-90px face height in
  low-resolution/group sources), especially combined with
  `full_frame_allow_upscale: false`. A real (non-mocked) trace on the
  crop_stitch/SPP path (`results/_head_scale_trace/night_group_offline/`)
  shows Krea2 itself amplifying scale error when conditioned on a weak
  face signal (S3_vs_S2_local 1.24x, S4_stitched 1.81x vs. true original) --
  a mechanism that plausibly also drives full_frame's proportion errors,
  and per §2.2, full_frame has *strictly less* implicit scale-anchoring
  than the path this trace measured.
- `full_frame_grounding_px=1024` (yaml:265) only partially compensates a
  second, compounding VLM-grounding resolution deficit on top of the
  diffusion-resolution one, since the scene being encoded is itself
  already downscaled.

## 4. Ruled out

- **Coordinate-space / resolution mismatch between the two masks.** Not the
  case: `out.resize(body_full.size, ...)` is enforced before the freeze
  composite (krea2.py:3532-3533) and again defensively inside
  `_freeze_full_frame_outside_selected` (krea2.py:1207-1210). Both masks
  are built on the *same* pixel grid. The mismatch is in mask
  *shape/parameters* and independently-*detected face box*, not resolution.
- **`_blend_procrustes_face_only` / `full_frame_face_refine_procrustes`**
  as a contributor to the *current* artifact -- confirmed disabled via
  yaml:295.
- **`clean_alpha_tails` being bypassed.** Both composite paths route
  through `feathered_soft_composite`, which unconditionally applies
  `clean_alpha_tails` (`preprocess.py:1873-1874`, floor/ceil defaults never
  overridden to 0/255 at either call site). The alpha-tail-ghost failure
  mode that fix targeted is not the active mechanism here -- the residual
  issue is the mismatch *between* two independently-cleaned masks, not a
  leaked tail within either one.
- **Stale face detection carried across resolutions.** full_frame
  explicitly re-runs `select_face_box` on the newly-built full_frame-scale
  `body_ff` (krea2.py:3108-3113) before building scene/mask -- the earlier
  1024px-capped detection is only used for upstream routing decisions, not
  full_frame's own geometry.
- **Refine-pass LoRA reload as an artifact source.**
  `full_frame_face_refine_reload_lora: false` by default (yaml:280) -- the
  refine pass reuses the exact same model bundle as pass 1, so this isn't a
  mid-run model-state-change risk.

## 5. Conflicting findings between agents

None of substance. All 4 agents converged on the double-composite/mask-
mismatch mechanism as the top artifact hypothesis and on the missing
scale-anchor as the top proportions hypothesis, using independent reads of
the same code. The only nuance: the routing agent initially phrased part
of the mismatch as a "different resolution/crop" risk; the geometry agent's
independent read confirmed resolution/coordinate space is *not* actually
divergent (both operate on `body_full`-sized pixels) -- the mismatch is
purely in mask shape and face-box source. This is reflected in §4 above.

## 6. Recommended experiments, in priority order

| # | Problem | Hypothesis | Test | Expected result | Actual result | Conclusion |
|---|---|---|---|---|---|---|
| 1 | Ghost/artifact around head | Two independently-detected, independently-shaped masks (pass 1 freeze vs pass 2 refine) create a non-coincident boundary in the same region | Make pass 2 reuse pass 1's exact face box and mask shape (top/side/bot/expand/blur) instead of re-detecting + using crop_stitch's defaults; fix pass 2's LAB `color_ref` from `out` to pristine `body_full` | Single coincident mask boundary; artifact/ghost/discoloration halo reduced or eliminated | **Implemented this round** (see below); GPU visual confirmation still needed | Pending user verification |
| 2 | Head proportions/scale wrong | No scale anchor in full_frame's donor placement, and the one scale-clamp gate excludes the single-person case that triggers full_frame most often | Fix the `if multi_person and do_clamp` gate to also cover the single-person full-body-route case | Generated head shrinks+recenters onto the target's real detected face when oversized; scale/position ratio closer to 1.0 | **Implemented this round** (2026-08-07 follow-up): gate fixed, `full_frame_clamp_head_scale` flipped to `true`. Synthetic end-to-end test (`results/_head_scale_clamp_experiment/`, not committed): a 1.89x-oversized, off-center generated head clamped to ratio 1.00 and recentered on the target's face, then correctly freeze-composited (everything outside the mask restored from the pristine original). | GPU visual confirmation on the real desert photo still needed. `full_frame_match_crop_identity=False` A/B (the *generation-time* half of this problem -- donor reference conditioning still carries no target-scale signal) remains a follow-up if the post-hoc clamp alone isn't sufficient. |
| 3 | Face quality gap vs crop_stitch | Confounded by LoRA choice (v1_2 vs v1_2_r64) and ref_boost (4.0 vs 3.5), not purely resolution | A/B `full_frame_identity_lora_name` forced to the same `r64` LoRA crop_stitch uses; isolate LoRA contribution from resolution contribution | Quantify how much of the quality gap is LoRA vs resolution | Not yet run | Recommended after #1/#2 are validated |
| 4 | Missing visual evidence | `do_stitch` gate blocks full_frame's refine-pass debug frames from ever being saved | Extend the debug-save gate to `do_stitch or edit_mode == "full_frame"` | Next real GPU run with `save_debug=true` produces `debug_stitch_mask.png` etc. for the refine pass | **Implemented this round** | Enables experiments 1-3 to actually be visually verified next time |

## 7. Recommended architecture

**The full_frame + local-refine *concept* is sound and should be kept**:
full-frame generation for correct body geometry, plus a second, higher-
resolution pass for face quality, is the right shape for this problem, and
none of the evidence suggests otherwise. The concept is not what's broken.

**The current *implementation* of the second pass is the wrong shape**: it
runs *after* the first pass has already composited onto the final canvas,
re-detects independently, and builds an independently-shaped mask -- i.e.
it behaves as a second, uncoordinated stitch rather than a refinement of
the first. The fix implemented this round (reuse pass 1's face box +
mask geometry, fix the LAB reference) is the smallest change that makes
the two passes share one boundary instead of two. A more thorough
long-term fix (out of scope for "smallest change") would restructure so
the resolution-boost refinement happens on the *raw, uncomposited* Krea2
sample *before* the single freeze-composite ever runs -- collapsing this
to one detection, one mask, one composite, one LAB match -- which several
agents flagged as strictly safer but is a larger structural change than
this round's brief calls for.

**Root cause #2 (no scale anchor) is architecturally simple to fix** (wire
`selected_face`'s real height into the donor placement, and fix the
`multi_person` gate bug) and should be the next experiment, since "head
proportions/scale are noticeably wrong" is one of the most direct user-
reported symptoms and is not addressed by this round's fix.

## 8. What was actually implemented this round

Per explicit instruction, only experiment #1's smallest change plus the
debug-instrumentation fix (#4, needed to verify #1 and future experiments
visually) were implemented:

1. `_refine_full_frame_face` now accepts and reuses pass 1's
   `selected_face`/`all_faces` (falling back to re-detection only if not
   provided) instead of always re-detecting on the already-generated `out`.
2. `_refine_full_frame_face`'s mask is now built with the *same*
   `full_frame_mask_top_extend` / `side_extend` / `bot_extend` /
   `expand_px` / `blur_px` values pass 1's freeze mask uses, instead of
   crop_stitch's own (different) `mask_*` defaults.
3. `_refine_full_frame_face`'s `_stitch_edited` call now color-matches
   against pristine `body_full` instead of `out` (pass 1's already-shifted
   output), so the two LAB corrections target the same ground truth instead
   of compounding toward each other.
4. The debug-save gate (`if do_stitch:`) now also fires for
   `edit_mode == "full_frame"`, so a future `save_debug=true` GPU run
   actually captures pass 2's stitch mask and before/after-LAB composite
   frames -- these were silently discarded before.

**Not implemented this round** (deferred per explicit "smallest change for
the first experiment" instruction): the LoRA-confound isolation (§6
experiment 3), and the larger refine-before-composite restructuring
mentioned in §7.

**Validation status**: unit/logic-level verification only (no GPU
available in this environment -- see the constraint noted at the top of
this document). A real GPU run against the desert-photo test case is
required to visually confirm this fix actually resolves the artifact.

## 9. Follow-up round (2026-08-07): experiment #2, the scale/position clamp

With the ghost fixed (confirmed by the user on a real render), the next
reported symptom was exactly experiment #2 from §6: "head alignment and
sizing are still incorrect." Implemented the smallest fix for it:

- `_maybe_clamp_full_frame_head_scale` (extracted from inline code for
  testability, mirroring the existing `_maybe_procrustes_edited_crop`
  pattern): the `if multi_person and do_clamp:` gate is now just
  `if do_clamp:`. `full_frame_clamp_head_scale` flipped to `true` in
  `configs/krea2_identity_edit.yaml`.
- Mechanism (pre-existing, unmodified): `clamp_edited_head_scale`
  (`preprocess.py:157`) detects the face on both the pristine `body_full`
  (the geometric anchor -- exactly what the user asked to use) and the raw
  full_frame sample `out`. If the generated face is taller than the
  target's real face by more than `max_edited_head_height_ratio` (1.08), it
  shrinks the whole `out` image about the generated face's own center and
  translates it so that center lands exactly on the target's real face
  center -- one bounded (shrink-only, 0.55-1.0x), rotation/shear-free
  similarity transform fixing scale AND position (both axes) together. This
  runs *before* the freeze composite, so only the shrunk+recentered head
  region survives into the final image; everything outside the mask is
  still restored from the pristine original regardless.
- Why this was previously off: the yaml comment cited `detect_best_face`
  possibly grabbing the wrong face "on the full group frame" -- a
  multi-person (>1 face) risk. The gate this round only newly activates the
  clamp for the `multi_person=False` case, where that risk doesn't apply
  (there's only one face to detect). Behavior for actual multi-person
  full_frame runs is unchanged.

**Validated**: 4 new unit tests (`tests/test_full_frame_head_scale_clamp.py`)
plus a synthetic end-to-end experiment (mocked face detection, real
`_maybe_clamp_full_frame_head_scale` + `_build_full_frame_freeze_mask` +
`_freeze_full_frame_outside_selected` code, saved to
`results/_head_scale_clamp_experiment/`, not committed): a 1.89x-oversized,
off-center generated head was clamped to ratio 1.00 and correctly
recentered on the target's face position. No GPU available to confirm on
the actual photo.

**Not addressed**: the *generation-time* half of the scale problem --
`_build_full_frame_inputs`'s donor placement (`resize_contain`) still
carries no target-scale signal into the raw Krea2 sample itself (§2.2,
§6 experiment 2's second half). This clamp is a post-hoc correction, not a
fix to the underlying conditioning; if the model drifts scale
*consistently* rather than occasionally, the clamp's `0.55` minimum-shrink
floor could still leave a visibly-too-large head in extreme cases. Revisit
`full_frame_match_crop_identity=False` (activating the currently-dead
`place_face_at_height_frac` branch) if the clamp alone proves insufficient.
