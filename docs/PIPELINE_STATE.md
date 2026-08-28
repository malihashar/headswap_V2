# headswap_V2 — Pipeline State Log

Append-only changelog of validated pipeline checkpoints. Every future change gets a new
entry here BEFORE being treated as current. To roll back, find the checkpoint name/date
below and restore that exact config block — do not rely on memory of what "should" be set.

Status legend: ✅ validated on real Colab GPU output · 🟡 proposed/pushed, not yet visually
confirmed · ❌ tested and rejected (kept for reference, do not silently retry)

---

## CHECKPOINT-01 — Baseline production (pre-multi-person work)
**Status:** ✅ (long-standing, single-person)
- Single-person: `crop_stitch`, unchanged
- No lighting route, no multi-person special-casing

---

## CHECKPOINT-02 — Full-frame A/B selected recipe
**Status:** 🟡 (some reports marked prep/mock — needs GPU reconfirmation if not already done)
- Multi-person, dark lighting → `full_frame`, full v1.2 LoRA, `ref_boost=4.0`, `grounding_px=768`
- Multi-person, normal lighting → `crop_stitch`, r64 LoRA, `ref_boost=3.5`, `grounding_px=768`

---

## CHECKPOINT-03 — Lighting route threshold, final value
**Status:** ✅
- `dark_lighting_threshold = 110.0`
- `enable_lighting_route = true`
- Known trap: stale notebook cell can silently hold `70.0` in session memory even after
  file/git shows `110.0` — always verify via printed runtime log, not the file.

---

## CHECKPOINT-04 — Full-frame target selection fix
**Status:** ✅ — DO NOT REVERT
- Full-frame mode now uses the same shared target-selector as `crop_stitch`
  (`body_face_policy` / `body_face_index` / `TARGET_HEAD`)
- Previously hardcoded to always pick rightmost face — confirmed fixed via log:
  `policy=index index=1 position=center`

---

## CHECKPOINT-05 — preserve_headwear fix
**Status:** ✅ — DO NOT REVERT
- `preserve_headwear: false`
- Root cause of the smooth-dome "hat" artifact — styled hair/hair accessories were being
  classified as headwear and preserved instead of replaced. NOT a compositing bug — several
  rounds were wrongly spent debugging masks/blending before this was found.

---

## CHECKPOINT-06 — ref_boost ceiling established
**Status:** ✅ (as a boundary/lesson, not a single "best" value)
- `ref_boost = 4.0` — most defensible stable global value
- `ref_boost = 7.0` global — ❌ rejected, overcooks face (synthetic/uncanny look)
- Region-masked boost (core face high / periphery low) tested — periphery boost increase
  made identity worse, not better — ❌ do not push periphery ref_boost_a up
- `union_bbox` crop widening — ❌ rejected, dilutes identity conditioning same as full_frame

---

## CHECKPOINT-07 — Dual-mask compositing architecture
**Status:** 🟡 GHOST ROOT-CAUSED AND FIXED (2026-08-18), residual re-verification pending.
The `pair2` floating-headwear ghost (and `pair1`'s border seam, same class of bug, not yet
separately re-tested post-fix) traced to two real, stacked bugs, both fixed and confirmed via
real Colab runs with runtime-verified evidence (mask dumps, a raw pixel diff, and a diagnostic
log), not code-inspection alone:
1. `composite_isolated_head_layer`'s (preprocess.py) blur/erosion kernels were fixed-size
   regardless of donor-content size, letting alpha bleed into black canvas (commits `1ab8ede`,
   `8f06345`). Verified: this function's own mask/layer dumps became clean and tight.
2. The real bug: `_head_matte_mask` (segmentation.py) intersects a geometric ellipse with a
   person-segmentation matte, but the matte returns **zero** foreground for rigid headwear
   (confirmed via a diagnostic log: `alpha_max_in_crown=0` before the fix) — segmentation
   backends are trained on human bodies, not accessories. The intersection chopped the hat out
   of the mask entirely regardless of ellipse size, leaving it as real, un-regenerated content
   inside the mask's feather band — composited toward transparent, reading as a ghost of the
   original headwear. Fixed (commit `15226c1`) by bypassing the matte requirement for the crown
   band directly above the face when that band has real object texture (Laplacian variance —
   worn headwear vs. smooth sky), not a blanket ellipse-only fallback (which would revive the
   older short-hair-ghost-oval bug this intersection exists to prevent).
   Confirmed on Colab: the actual generation crop grew (`crop_native` [80,112]→[80,144]),
   the mask now covers the hat, and the final composite shows a solid, properly regenerated
   hat instead of the floating disconnected fragment. A small soft edge remains right at the
   crown tip (normal feathering) — much smaller than the original artifact, not yet confirmed
   as fully eliminated across multiple seeds/cases.
- **Generation mask**: tight, head+hair only. Controls what Krea2 regenerates. Never widen
  this to fix a color/blend problem.
- **Harmonization mask**: separate, wider, skin-adaptive (mediapipe limb detection + HSV
  skin classification, intersected with body matte). Extends into neck/shoulders/arms only
  where actually visible. Must fully CONTAIN the generation mask.
- Generation mask interior: generated pixels fully replace original, erode+max technique
  for opaque core + single thin feathered edge.
- Harmonization-only ring: LAB statistical color transfer (Reinhard-style) toward donor
  tone sampled from the generated head — tone/color only, never content replacement.
- **One smooth outer feather only** — at harmonization mask's outer edge. A second
  independent feather between the two masks is what caused historical ghosting.

---

## CHECKPOINT-08 — Geometry / head-scale correction
**Status:** ✅ (full-frame similarity transform) — replaces abandoned local-box clamp
- Detect landmarks/bbox on original target face vs. generated head
- Compute similarity transform (scale + translate, +rotate if justified)
- Apply to the ENTIRE generated frame, not a local box
- Build compositing mask AFTER the transform (mask reflects true post-transform position)
- Composite once: `final = mask * transformed + (1-mask) * original`
- ❌ Local-box clamp (any variant) — causes double-head ghosting, clipped hair,
  rectangular seams. Confirmed bad across multiple independent attempts. Do not revive.

---

## CHECKPOINT-09 — extend_skin_harmonization (body-skin LAB tone match post-composite)
**Status:** ✅ (confirmed isolated per code audit; not yet independently GPU-reconfirmed under this log's validation discipline)
- Adds a post-composite skin-tone harmonization pass: LAB statistical color transfer
  toward donor tone sampled from the generated head, applied to a body-skin mask.
- Body-skin mask built from mediapipe limb landmarks ∩ HSV skin classification ∩
  person/body matte — same construction style as CHECKPOINT-07's harmonization mask.
- Hard-excludes the head region (`skin_mask` zeroed at `head_bottom + 10px`), with a
  byte-identity safety assertion on the head region post-composite confirming this mask
  never overlaps or touches the CHECKPOINT-07 generation-mask region.
- Has its own independent edge feather (`feather_px=20`), but operates on a
  non-overlapping region by construction — not the "second feather between the two
  masks" ghost-bug CHECKPOINT-07 warns against, since there is no shared boundary.
- Gated by `extend_skin_harmonization: true` (`configs/krea2_identity_edit.yaml:594`).

---

## Known-bad list (cross-session, do not silently retry)
- Local-box head-scale clamp — any variant
- Widening generation mask to fix a blend/seam issue
- Independently alpha-blending two overlapping masks against the original
- Fixed-radius / flat-average neck-tone fill
- `ref_boost` global > ~5 without region-masking
- `union_bbox` / generation-crop widening for "more donor context"

## Validation discipline (applies to every future change)
1. One variable changed at a time.
2. Real Colab GPU run required for any visual-quality claim — no code-inspection-only claims.
3. Trace runtime-printed values, not just config files (lighting threshold + ref_boost_mask
   both had silent stale-value bugs that looked fine in the file).
4. Compare against the most recent ✅ checkpoint above, not vibes.
5. New checkpoint gets appended here with ✅/🟡/❌ before being treated as "current."

---

## Open items (not yet checkpointed)
- `grounding_px=768` re-verification: reverted from 1024 back to the shared 768 default
  (CHECKPOINT-02, 2026-08-18) since the 1024 test wasn't validated. The removed config
  comment claimed 1024 was "the primary driver of full_frame's face-quality gap" — that
  claim is plausible but plausibly measured under pre-`preserve_headwear`-fix conditions,
  same confound as the `ref_boost` question below. Re-verify with a clean A/B once the
  `ref_boost`/LoRA round (below) is settled — one A/B round at a time.
- `ref_boost` (7.0 vs 4.0) and `full_frame_identity_lora_name` (r64 vs non-r64) for the
  full_frame dark route: config comments cite a 2026-08-11 GPU A/B favoring 7.0/r64, which
  conflicts with CHECKPOINT-06's "7.0 rejected as overcooked." A clean re-run under current
  code (`preserve_headwear=false` fixed) is in progress — see
  `results/_ab_ref_boost_lora_checkpoint02/`. Not yet resolved.
- Full_frame head-scale clamp default: `full_frame_clamp_head_scale=true` is currently an
  undescribed whole-frame-warp+border-patch variant, not the checkpoint-08 mask-after-
  transform design (`crop_stitch_full_frame_head_clamp`, currently disabled). Needs a
  head-to-head Colab comparison before flipping the default — not yet run.
- Neck-as-generated-content (extend generation mask/crop through neck, straight-head
  prompt) — prompt sent, result not yet reported/checkpointed.
- Crop-tightness vs. identity test on athlete/well-known-face case — interrupted, test
  case never saved to `data/testcases/`.
- Whether Krea2 can reach acceptable identity fidelity on highly-recognizable faces at all,
  vs. requiring a dedicated deterministic swap model (InSwapper/SimSwap-class) as the core
  identity stage with Krea2 demoted to harmonization-only.

---

## CHECKPOINT-10 — T4: the approved simple_full_body recipe
**Status:** ✅ adopted as product default (commit `5e60e78`, 2026-08-27)

Selected from an 8-pair x 5-arm sweep as the only arm worth keeping. Restore
ALL FIVE items below to revert to T4 — the four values alone are not enough.

### Sampling (configs/krea2_identity_edit.yaml)
- `ref_boost: 5.5` — between CHECKPOINT-06's rejected 7.0 (overcooked,
  synthetic skin) and its defensible 4.0; tested better than both here
- `denoise: 0.85` — meaningful only because sampling seeds from the SOURCE
  latent (see below); at 1.0 the img2img branch is unreachable
- `cfg: 1.8` — at 1.0 there is NO classifier-free guidance, so the prompt
  carries zero weight and every prompt edit is inert
- `seed: 46`

### Latent seeding
`_sample_edit` must reach the `elif denoise < 0.999 and scene_lat is not None`
branch BEFORE the `EmptySD3LatentImage` branch. Starting from empty noise
regenerates the whole frame: a black robe returned as a bare torso, a lace
dress as a top plus skirt, a chain appeared on an athlete wearing none.
Confirm with `[krea2 img2img] starting from the SOURCE latent at denoise=0.85`.

### Prompt (part of the recipe, not incidental)
The `run_simple_full_body` default prompt must be byte-identical to the A/B
text — 564 chars, logged as `[krea2 prompt] source=built-in default chars=564`.

**Measured:** adding an expression-preservation block plus an absolute clothing
prohibition, with sampling and seed unchanged, moved the generated face from
**38.7% -> 45.0%** of the frame (visibly tighter, worse) while leaving the
expression unchanged. On this route the prompt is a framing control as well as
a content control.

### Rule for future prompt work
A/B every prompt edit on the same seed and compare the `face is N% of frame`
value from `[krea2 body_restore] SKIPPED - bust shot (face is N% of frame)`
before landing it. Prefer replacing prompt text over adding to it, and test
new prompts via the `simple_full_body_prompt` config override so the default
recipe is never disturbed by an experiment.

### Known-unfixed at this checkpoint
- A robed subject can still come back shirtless (garment removed outright)
- Some subjects get no skin-tone change at all
- Expression still follows the donor portrait, not the body photo

---

## CHECKPOINT-11 — Expression is NOT reachable from the prompt
**Status:** ❌ closed as a dead end (two attempts, both measured)

Goal was: keep the BODY photo's expression, take only identity from the donor.

| attempt | prompt delta | face fraction | expression changed? |
|---|---|---|---|
| meta-instruction ("identity from image 2, expression from image 1") | +~400 chars | 38.7% -> **45.0%** | no |
| measured hint ("The person is not smiling, with the mouth closed") | +1 sentence | 40.3% -> **42.0%** | no |

Same seed and sampling in both cases. Two conclusions, both measured rather
than argued:

1. **Any prompt addition moves framing on this route.** Even one sentence cost
   1.7 points of face fraction and visibly degraded head position, size and
   realism. Prompt length is a framing control here, not just content.
2. **Expression is not prompt-governed at all.** The measurement was correct
   (`smile_ratio=0.808 open_ratio=0.25` -> "not smiling, with the mouth
   closed", which matched the body photo), and the output smiled regardless.
   The donor's expression arrives as IMAGE conditioning at `ref_boost=5.5`,
   which no prompt text touches.

`_measure_expression_hint` is kept behind `expression_prompt_hint` (default
**false**) because the measurement is sound and reusable — only the lever was
wrong.

### Where to look instead
Not the prompt. Expression enters with the donor pixels, so the candidates are
the image-conditioning side: the `face_refine` pass (re-renders the face from
the donor crop and is the most likely single source), `ref_boost` / `ref_boost_a`,
and the donor crop geometry itself (`crop_face_reference`). Any attempt must be
A/B'd on one seed with the `face is N% of frame` value compared, which is what
caught both failures above.

---

## CHECKPOINT-12 — face_refine: innocent on expression, wasteful on bust shots
**Status:** ✅ gated by face size

A/B on one bust-shot pair (face 42.0% of frame), same seed and sampling:

| arm | time | result |
|---|---|---|
| refine ON (T4 default) | 76s | smiling (donor expression) |
| refine OFF | 53s | **visually indistinguishable** |

Two findings.

**Expression: eliminated.** face_refine was the leading suspect from
CHECKPOINT-11 — it re-renders the face from the donor crop, the largest single
dose of donor pixels. Turning it off changed the expression not at all. The
donor's expression therefore enters through the main pass's image
conditioning, not the refine. Next lever is `ref_boost` / `ref_boost_a`.

**Speed: ~30% for nothing, on bust shots.** face_refine exists to recover
identity detail when the face is SMALL — it re-renders a head crop at full
resolution. At 42% of frame the main pass already rendered the face at high
resolution, so there is nothing to recover. It was not gated on face size at
all; it ran unconditionally.

**REJECTED as a default (2026-08-27).** "Visually indistinguishable" was
judged from a small side-by-side grid; at full size the refined result was
clearly preferred and the skip was rejected. refine ON is part of T4
(CHECKPOINT-10) and stays on. The 30% saving is real but not free.

The gate remains as opt-in: `simple_full_body_refine_max_face_frac` defaults
to **1.01** (unreachable). Set it to 0.25 to skip on bust shots, matching
`simple_full_body_restore_max_face_frac`, which already encodes "is this a
bust shot?".

Lesson: a small comparison grid is not enough to judge "no visible
difference". Full-size review is required before any quality/speed trade is
taken by default. Skipping also removes a composite — and so a boundary — from
exactly the frames where the face is large enough for misalignment to show,
which is the class of the Iron Man helmet/face offset.

**Detector hazard found while building the gate:** `detect_best_face` falls
back to a CENTRE BOX when it finds nothing — on a flat grey image it returns
0.417, above the threshold. Gating on it would have silently skipped the
refine on every frame where detection failed, while looking like a real
measurement. The gate uses `detect_faces` instead, which reports real
detections only, and treats None as "unknown" rather than as either answer.

---

## CHECKPOINT-13 — Expression: ref_boost eliminated too
**Status:** ❌ third dead end; one lever left

`ref_boost` sweep on the athlete pair, T4 otherwise unchanged, same seed:

| arm | ref_boost | expression |
|---|---|---|
| R55_T4 | 5.5 | donor smile |
| R40 | 4.0 | donor smile |
| R30 | 3.0 | donor smile |
| R40_a08 | 4.0 + ref_boost_a 0.8 | donor smile |

A 45% cut in donor conditioning strength produced no expression change, and
identity held up throughout. Combined with the earlier results, every channel
I proposed has now been tested and eliminated:

| channel | attempts | result |
|---|---|---|
| prompt text | 2 (meta-instruction, measured hint) | no change; both moved framing |
| face_refine pass | 1 (ON vs OFF) | no change |
| ref_boost / ref_boost_a | 4 (5.5 / 4.0 / 3.0 / +a0.8) | no change |

### What that leaves
`identity_lora_strength` (currently 1.0) is the only untested lever.
`krea2_identity_edit_v1_2_r64` is trained to transplant the head from image 2
-- and a head includes its expression. If the LoRA is what carries it, that is
not a bug to tune out; it is what the LoRA does.

### If the LoRA test also fails
Expression is intrinsic to this model + LoRA and cannot be separated from
identity by configuration. The honest options then are: accept the donor
expression, choose donor photos whose expression suits the target scene, or
change the identity model. Further sampling-parameter sweeps would be waste --
three channels, seven arms, zero movement is enough evidence to stop.
