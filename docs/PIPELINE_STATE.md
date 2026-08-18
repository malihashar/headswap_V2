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
**Status:** ✅ — core architectural rule, do not violate
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
