"""Post-render diff correction: restore ORIGINAL garment pixels wherever the
render turned fabric into skin, and nowhere else.

Three prompt-text fixes for "a covered subject can come back with clothing
stripped near the torso" were tried and rejected -- see
tests/test_skin_clause_drop.py and tests/test_garment_protection.py for that
history. This is a structurally different mechanism: not a prompt change, a
post-render diff between what was fabric in the ORIGINAL photo and what
reads as skin in the RENDER. Because the diff can only ever include pixels
that were classified as clothing in the source, it structurally cannot touch
already-bare skin (hands, neck, face) -- so it cannot reproduce the
white-hands regression that killed the previous two attempts.

Grep-on-source-text pattern, same as test_torso_clothed_skin_clause.py: no
isolated-exec, no GPU. These tests pin STRUCTURE (gating, fail-closed
behaviour, which primitives are used, ordering of operations), not pixel
output -- that needs the GPU validation matrix documented in the function's
own call-site comment in krea2.py.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
SKIN_HARM = (ROOT / "src" / "headswap" / "skin_harmonize.py").read_text()


def _function_body(text: str, def_line: str) -> str:
    i = text.find(def_line)
    assert i > 0, f"could not find {def_line!r}"
    j = text.find("\ndef ", i + 10)
    return text[i: j if j > 0 else len(text)]


def test_default_is_off():
    """Ship opt-in until GPU-validated on both the target case and the
    regression-prone bare-armed case -- same discipline as
    skip_skin_clause_when_covered and protect_garments, both of which
    shipped on and had to be reverted."""
    i = KREA2.find('"simple_full_body_restore_stripped_garment"')
    assert i > 0, "could not find the cfg key for the new flag"
    window = KREA2[i:i + 60]
    assert "False" in window


def test_not_gated_by_raw_model():
    """The correction's boundary sits on a real garment silhouette from the
    source photo, not an arbitrary geometric/probability line -- the case
    _raw_model's own reasoning does not cover. _raw_model=True is the
    production default and where this bug was measured, so nesting the fix
    inside `not _raw_model` would make it inert in production."""
    i = KREA2.find('restore_stripped_garment(')
    assert i > 0
    # Walk back to the nearest enclosing "if" line.
    if_start = KREA2.rfind("\n        if ", 0, i)
    assert if_start > 0
    if_line_end = KREA2.find("\n", if_start + 1)
    if_line = KREA2[if_start:if_line_end]
    assert "_raw_model" not in if_line, (
        "the restore-stripped-garment call must not be gated on _raw_model "
        "-- that would make it inert on the production default"
    )


def test_runs_before_skin_harmonize():
    """An erroneously-exposed patch must be restored BEFORE
    extend_skin_harmonization runs, or that call samples the stripped
    pixels as 'current body skin' and skews its donor-tone transfer."""
    i_restore = KREA2.find("restore_stripped_garment(")
    i_harmonize = KREA2.find("extend_skin_harmonization(")
    assert 0 < i_restore < i_harmonize


def test_fails_closed_when_masks_unavailable():
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    # Every skip path must return the ORIGINAL `rendered` image unchanged,
    # never a mutated one.
    skip_returns = body.count("return rendered, info")
    assert skip_returns >= 4, (
        "expected multiple fail-safe early returns (clothes segmenter "
        "unavailable, skin segmenter unavailable, below pixel floor, below "
        "area floor) -- found fewer than expected"
    )


def test_uses_real_classifiers_not_colour():
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    assert "semantic_clothes_mask" in body
    assert "_semantic_skin_mask" in body


def test_head_exclusion_is_geometric_and_semantic():
    """_semantic_category_mask has a documented failure mode where it
    silently returns an all-zero array (not None) on some frames -- a
    semantic-only exclusion would then evaluate to 'no exclusion at all'
    and paste original pixels under the just-swapped neck. Both guards must
    be present."""
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    assert "semantic_head_mask" in body
    assert "head_excl" in body


def test_head_zero_survives_the_blur():
    """extend_skin_harmonization hit this exact 'hard-zero, then blur'
    reorder bug once already (measured max_diff 19 and 11 in the head
    region before the fix). The geometric zero must be applied both before
    AND after GaussianBlur, not just before."""
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    i_blur = body.find("GaussianBlur")
    assert i_blur > 0
    before = body[:i_blur]
    after = body[i_blur:]
    assert "stripped[:head_excl] = 0.0" in before
    assert "weight[:head_excl] = 0.0" in after


def test_speckle_guard_present():
    """cv2.INTER_LINEAR resizing inside the mask functions produces a thin
    fractional halo along EVERY garment edge on every render, not just a
    buggy one -- unguarded, this would fix the target case while adding a
    hairline fringe on cases that were already correct. A total-pixel-count
    floor alone is not enough (a long thin halo can sum past it); a
    connected-component AREA filter is required too."""
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    assert "MORPH_OPEN" in body
    assert "connectedComponentsWithStats" in body
    assert "min_component_frac" in body


def test_pasted_patch_is_tone_matched_to_the_render():
    """Measured on GPU: pasting RAW original pixels produced a visible
    rectangular patch on the shoulders, because the diffusion pass shifts
    global exposure/white-balance even at high denoise-preserve -- the
    original photo and the render are two different colour grades, and a
    boundary between them is visible exactly where they differ. Unlike
    extend_skin_harmonization's rejected skin-wash (which shifts GENERATED
    skin toward a donor face and 'cannot produce skin rendered under the
    scene's light'), this samples its transfer target from garment pixels
    ADJACENT to the patch in the SAME render -- literally how this render
    already lit this same fabric a few centimetres away -- so it is not the
    same critique. Must fall back to unmatched original pixels when too
    little reference garment survives to measure a transform from."""
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    assert "_reinhard_transfer" in body
    assert "_robust_lab_stats" in body
    assert "paste_source" in body
    assert "ref_mask" in body


def test_log_line_names_skip_reason():
    body = _function_body(SKIN_HARM, "def restore_stripped_garment(")
    assert body.count('info["reason"]') >= 3


def test_meta_dict_exposes_the_diagnostic():
    assert '"restore_stripped_garment": restore_diag' in KREA2
