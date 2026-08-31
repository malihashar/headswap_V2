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

---

## CHECKPOINT-14 — Pre-editing the DONOR's expression: closed as a dead end
**Status:** ❌ closed on CLEAN data — four GPU rounds, zero expression
movement, plus two structural defects. Round 4 was verified unconfounded
(`knobs: using code defaults`, no override warnings), so both the tuning
axis and the structural argument are now supported.
**Default:** `pre_edit_donor_expression: false` — T4's own render is unaffected.

CHECKPOINTs 11–13 all edited the TARGET side or the conditioning strength.
This was the first attempt at the OTHER side: edit the DONOR photo's
expression to match the target's BEFORE T4 ever sees it, via a Krea2
self-edit (`scene == person ==` the donor photo, no mask), then hand the
edited donor to T4's existing two-step pass unchanged.

Rationale at the time: CHECKPOINT-13 established that the donor's expression
transfers reliably and unstoppably. That was read as a feature — fix the
donor's expression and it should carry through.

| round | lora | ref_boost | cfg | denoise | result |
|---|---|---|---|---|---|
| 1 | on | 2.0 | 1.8 | 0.35 | donor came back with **no visible change at all** |
| 2 | on | 0.5 | 3.0 | 0.35 | **identity drifted** (head shape/angle); expression unchanged |
| 3 | **off** | 2.0 | 4.0 | 0.35 | identity drifted **worse**; expression still unchanged |
| 4 | **off** | 2.0 | 4.0 | **0.45** | ✅ clean/unconfounded. Smile **got WIDER**; identity drifted again |

Round 4 is the decisive one: it is the only round verified free of the
stale-form confound below. The prediction going in was that the measured
label's correct half ("not smiling" — the target genuinely is not) would at
least partially land and reduce the donor's smile. **It did the opposite.**
Raising denoise from 0.35 to 0.45 gave the sampler more room and it spent
all of it on identity, none on expression.

Across four rounds the sweep covers lora on/off, cfg 1.8→4.0,
denoise 0.35→0.45 and ref_boost 0.5→2.0, with zero expression movement in
any arm. There is no remaining confound to attribute that to.

Round 3 was the decisive one. The hypothesis was that
`krea2_identity_edit_v1_2_r64` — trained to make image A's head look like
image B's — collapses to "reproduce this exact head" when A and B are the
same photo, and that this attractor was overpowering the prompt. Confirmed
loaded-out via `[krea2 sampling] ... lora=[]`. **Removing it did not unlock
the expression; it only cost identity.** The hypothesis was wrong.

### Why no knob on this route can work
`ref_boost` anchors fidelity to the "person" reference — and that reference
CONTAINS the donor's smile. It therefore preserves identity and expression
*together* and cannot separate them: raising it stops the identity drift but
re-pins the expression, lowering it frees both. The same is true of img2img
seeding and of low denoise. Every purely SPATIAL preservation mechanism on
this route has this property, because identity and expression live in the
same pixels.

### Two defects found while measuring, both independent of the tuning

**1. `_measure_expression_hint`'s openness measurement is broken by
construction.** It computes

```python
mouth_y = 0.5 * (lm[3][1] + lm[4][1])          # landmarks5: mouth CORNERS
open_ratio = abs(mouth_y - nose_y) / face.height
```

`lm[3]`/`lm[4]` are the mouth **corners**, which barely move vertically when
a jaw drops — the lower lip and chin do. landmarks5 has no lower-lip or jaw
point, so `open_ratio` **cannot detect an open mouth at all**; it measures a
nose-to-mouth face proportion. Measured on the athlete pair: the target's
mouth is visibly OPEN and it returned `open_ratio=0.25` (< 0.42 threshold)
→ `"not smiling, with the mouth closed"`. CHECKPOINT-11 recorded this
measurement as "sound"; that is correct for smile width and **wrong for
openness**. Fixing it needs `landmark_2d_106` (already loaded by
insightface, used inline in `preprocess.py` for a head bbox but with no
reusable accessor) and a verified inner-lip index map. NOT fixed here —
a wrong index map would be worse than a documented known-bad measurement.

**2. The text channel is too narrow to carry an expression.** Even with a
correct measurement and a working editor, this design compresses the
target's expression into two scalars → one of four canned phrases. The
athlete's actual expression is *mouth open, head tilted up, mid-exertion*;
no sentence in that fixed set represents it. This is a ceiling of the
architecture, not of the tuning.

### Where to look instead
Not any text-conditioned path. Expression should be driven from the target
image's own dense keypoints, which is what face-reenactment models
(LivePortrait class) do. Recommended placement is a POST-step, not a
pre-step: T4's output already has the right identity, pose, head scale,
lighting and framing — only the expression is wrong — so use T4's output as
the source and the ORIGINAL target photo as the driving frame, in
relative-motion mode. It runs last, so nothing can revert it, and T4 stays
untouched. Known risk: it re-introduces a crop/paste boundary, which is the
exact failure class CHECKPOINT-07 and `simple_full_body_raw_model` exist to
avoid. Not yet tested.

### Trap worth recording (cost three runs)
Colab `#@param` form values live in the browser TAB, not on disk, so
`git reset --hard` in the setup cell does NOT update them. All three rounds
above ran at `denoise=0.35` — a stale form value — while newly-added params
picked up fresh code defaults, producing mixed configs that matched no
intended arm. `notebooks/pre_edit_expression_test.ipynb` now defaults
`USE_CODE_DEFAULTS=True` so code defaults win unless deliberately overridden;
the tab must still be RELOADED for notebook changes to appear.

---

## CHECKPOINT-15 — Expression IS controllable: upstream's single-image edit graph
**Status:** ✅ first working expression change in this investigation.
🟡 identity retention is a known, measured gap — not yet solved.
**Selected for now:** `single_edit_denoise = 0.65` (chosen visually by Ali;
see the caveat about the metric below).

### What unblocked it
CHECKPOINT-11..14 spent five sessions and nine levers failing to move
expression. All of them ran the **two-image head-swap graph**. Comparing this
repo against the **Krea 2 production Colab** (the reference notebook for the
`comfyui-krea2edit` nodes) showed its `edit_workflow()` calls
`Krea2EditModelPatch` with **one** image:

    model, source_latent, vae, source_image, fit_mode

No `source_image_b`, no `source_latent_b`, **no `ref_boost`**.

"Change this person's expression" is a single-image instruction edit, so the
canonical graph for it had never been run. Every prior null result was a
head-transplant graph contorted into imitating an editor. Implemented as
`Krea2IdentityEditPipeline.edit_single_image()`.

Upstream edit-mode values, adopted verbatim rather than blended with ours:
`er_sde`/`simple` (not `euler`), 4 steps (not 8), cfg 1.0 (not 1.8), and
denoise 1.0 where preservation comes from `Krea2EditModelPatch` rather than
from img2img latent seeding — a different MECHANISM, not a different number.
No `ModelSamplingFlux` shift; upstream applies none.

**The LoRA is not the suppressor.** The identity LoRA was loaded at
strength 1.0 in the run that finally changed the expression, and was fully
OFF in two earlier runs that did not. Graph shape was the variable
throughout. Do not spend another round on LoRA strength for this.

### Resolution bug found on the way (cost one full GPU round)
Upstream's `upload_and_prep` upscales every edit input to >=1024 on the long
side. The first port brought the workflow but not the prep, and this repo's
`resize_max_keep_ar` is `min(1.0, ...)` — a CAP that never upscales — so a
small donor reached the sampler at **192x176**, a thumbnail that cannot hold
a likeness. `_fit_body_dim` documents the identical trap for the body image.
Fixed via `single_edit_min_long_side` (default 1024) and logged.

### Denoise sweep (athlete pair, seed 46, `scripts/sweep_expression_denoise.py`)

| denoise | ArcFace identity vs donor |
|---|---|
| 1.0 | 0.471 |
| 0.8 | **0.402** |
| 0.65 | 0.427 |
| 0.5 | **0.574** |

**Not monotonic**, and the spread (0.40-0.57) is narrow enough that the
ordering may not be meaningful at n=1 pair / 1 seed. `0.5` scored highest but
`0.65` was preferred visually. Treat the number as a tripwire for gross
identity loss, not as a ranking to optimise — that is why 0.65 was selected
over the metric winner.

End to end on the chosen arm:
`id_donor_vs_final = 0.408`, `id_edited_vs_final = 0.697`.
Read that as: **step 1 loses the identity; T4 then transfers the edited face
faithfully.** Any further identity work belongs in the expression edit, not
in the swap.

### Known-unfixed
Identity retention through step 1 is the open problem. Upstream's own README
disclaims it: *"structure-preserving i2i — can't guarantee 1:1 content
preservation, confirmed not solved by us or the wider community."* So a
ceiling here is expected. If it proves unacceptable, LivePortrait
(`src/headswap/expression_transfer.py`, Cell 5) preserves identity
structurally rather than by tuning and has still never had a valid run.

### Revert recipe
T4 is untouched by all of this — `pre_edit_donor_expression` defaults to
`false` and nothing in `run_simple_full_body` changed. To drop the expression
work entirely, simply do not call `edit_single_image()`; T4 behaves exactly
as CHECKPOINT-10 describes.

---

## Colab workflow — READ THIS BEFORE DEBUGGING ANY COLAB RESULT
**Roughly ten GPU runs in this investigation executed STALE CODE while their
logs showed a fresh commit.** Two independent causes stack:

1. Colab caches notebook `#@param` form values in the **browser tab**.
   `git reset --hard` in a setup cell updates the repo on disk but **cannot**
   touch them. A stale tab silently re-sends old numbers, and newly-added
   params pick up fresh code defaults — producing mixed configs that match no
   intended arm. Symptoms seen: `denoise` pinned at 0.35 for four rounds, then
   at 1.0 for three more, while `ref_boost`/`cfg` moved.
2. A notebook **saved to Drive stops re-fetching from GitHub entirely**, so
   even reloading the page does not update it. Confirmed when a traceback
   failed inside `if RUN_THE_SWAP:` — a variable that did not exist in the
   pushed cell.

**Therefore: put logic in `scripts/`, not in notebook cells.** A script is
pulled by `git pull` like any other file, so what runs is what is in the repo.
The Colab cell should be one line that never changes:

    !cd /content/headswap_V2 && git pull -q && python scripts/<name>.py

`scripts/sweep_expression_denoise.py` is the worked example. Verify any
surprising result by checking the values the run actually PRINTED before
concluding anything about the model — several conclusions in
CHECKPOINT-14 had to be softened afterwards for exactly this reason.

---

## CHECKPOINT-16 — Headwear and garment control: what is and is not reachable
**Status:** ❌ both closed as unreachable by the routes attempted.
**Defaults:** unchanged. T4's approved 564-char prompt is byte-identical.

### Headwear removal: 4 prompt wordings, none worked
| attempt | wording | result |
|---|---|---|
| 1 | base "anything worn on the head" | cap survived |
| 2 | appended CRITICAL remove-and-replace clause (943 ch) | cap survived |
| 3 | as 2, with "keep the clothing" scoped below the neck (958 ch) | cap AND a durag survived |
| 4 | fact-style sentence moved INTO sentence one (665 ch) | cap survived |

The reason is structural, not lexical. `denoise=0.85` seeds from the SOURCE
latent, which already contains a large, high-contrast cap, and img2img at
that setting exists precisely to preserve such structure. Describing what to
draw does not remove what is already there. A fifth wording is not the
answer.

`erase_headwear()` (LaMa inpainting, already in the repo) DOES remove it --
the mask found a cap at 10.1% coverage first try -- but it was rejected on
output quality. A negative-prompt arm was added and never confirmed firing.

### Garment preservation: adding words always broke something else
| attempt | result |
|---|---|
| do-not-expose prohibition | torso covered, SLEEVES removed instead |
| drop the body-part enumeration | sleeves back |
| name "robe, shirt or top" | a tennis POLO rendered as a fluffy bathrobe |

The mechanism is consistent and worth stating plainly: **on this route the
model draws whatever the prompt names.** The skin clause enumerates "the
neck, arms, hands and legs that are already bare", a covered subject has
none, so the model creates some; forbidding one named part just moves the
exposure to the next one; and naming garment types renders those garments.

### Why "drop the clause instead of adding words" also failed
Removing the sentence for covered subjects is the right shape -- a shorter
prompt cannot introduce a garment that was not there. But it needs a
reliable "is this subject covered?" signal, and three attempts at one all
inverted on the two real test images:

| variant | robe (covered) | tennis (bare arms) |
|---|---|---|
| full rectangle below chin | 53.0% | 5.5% |
| 3 face-widths column | 18.0% | 6.9% |

Both confounded. Desert sand sits almost exactly on skin hue; the robed
subject's bare praying hands sit dead-centre below the face; and the tennis
player's face is 30% of frame, so a face-relative column spans mostly dark
background and dilutes his genuinely bare arms. Framing scale and pose
dominate the measurement.

A correct signal needs the person's SILHOUETTE, i.e. segmentation. That was
explicitly ruled out for this work, so the approach is closed rather than
mistuned. `skip_skin_clause_when_covered` is default OFF; leaving it on
would let a wrong verdict silently rewrite the prompt.

### Honest options from here
1. Accept both (a robed subject may come back shirtless; hats survive).
2. Allow segmentation for the covered/bare DECISION only -- no output mask.
3. Allow `erase_headwear()` (LaMa) for hats and tune its quality.
4. Curate inputs: no hats, no fully-covered subjects.

### Process note that cost the most time
Roughly a dozen renders were spent on stale code or broken instruments
rather than on the models: Colab caches cell source in the browser tab, and
Cell 3 neither pulls nor re-imports, so it re-runs whatever is in the
kernel. One "fix" returned byte-identical numbers (53.0%/5.5%) to the run
before it, which is only possible if the new code never loaded. `run_chain`
now compares HEAD at load time against HEAD at run time and prints a banner
when they differ. Verify a number MOVED before interpreting it.

---

## CHECKPOINT-17 — Sampling containment on T4: WORKED mechanically, REVERTED on looks

Wired `noise_mask` (EXPERIMENT E1) into T4's main `_sample_edit` — ellipse head
region, `denoise=1.0` inside it, everything outside re-pinned to source every
step. Ran on GPU on the robe pair. **Reverted in the next commit. Do not
re-attempt.**

### It is not a Turbo no-op — that risk is now closed

The prediction was that the distilled Turbo LoRA might make `noise_mask`
inert. It does not:

```
[krea2 containment] head region = 5.4% of frame (backend=ellipse)
[krea2 containment] scene_latent+noise_mask coverage=0.054 mask=[672,1024]
[krea2 containment] mean|diff| inside=35.40  outside=0.91  (0-255)
```

`inside=35.4` — the masked region genuinely regenerated. `outside=0.91` on a
0-255 scale is essentially pixel-identical: the pin holds, and the black robe
came through intact, which is the clothing result that `ref_boost_a` sweeps and
three prompt rewrites never achieved.

**So the mechanism works and the arithmetic was right.** That is not the
problem.

### Why it was reverted anyway

The hard pin boundary is a composite boundary by another name. `denoise=1.0`
inside and a per-step re-pin outside means the two regions are rendered under
completely different conditions and meet at the mask edge with nothing
reconciling them. On this render the seam lands across the neck and collar and
is plainly visible at full size.

This is the **same failure class masks were banned for** — CHECKPOINT-07's
ghost, the "glowing oval" in `headwear_erase.py`, the clamp that was pulled.
The earlier claim in this file that *"there is no composite and no boundary, so
the ghost class cannot occur"* was **wrong**: it reasoned about compositing in
pixel space and missed that a noise_mask creates the same discontinuity in
latent space. Correct that reasoning wherever it is repeated.

The instruction stands and now has a measurement behind it: **no output masks,
no sampling masks, no clamps on this route.** Solve it inside Krea2's own
conditioning or not at all.

### What this leaves

Carrier 3 — `Krea2EditGroundedEncode`, which receives `image=scene_t` at
`grounding_px=768` (`krea2.py:2679-2687`) — is now the only untouched
conditioning path, and it is the one that tells the VLM what it is looking at.
It is internal to Krea2, needs no mask and adds no boundary. Never swept.

Note the run above also shows two contradictory skin lines
(`raw_model ... all DISABLED` immediately followed by `LAB wash ON`); worth a
look independently of headwear.
