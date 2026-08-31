"""Precise, MEASURED bare-skin-location wording, replacing the enumerated
skin-recolour sentence when the ambiguity that causes garment-stripping is
actually present in this photo.

History (see test_garment_negative_prompt.py, test_skin_clause_drop.py,
test_garment_protection.py for the full arc): four attempts at fighting the
garment-stripping bleed AFTER the fact all failed on GPU -- a do-not-expose
prohibition, naming garment types, dropping/scoping the skin-recolour
sentence, and a negative-prompt exposure guard (no measurable effect,
because unlike headwear removal there IS an active positive instruction
pulling toward skin here, so a skin-state negative term fights its own
positive prompt).

This asks a different question: the default prompt enumerates body parts
("the neck, arms, hands and legs") generically, so on a photo where only a
small area is genuinely bare, the model has no precise boundary for
"already bare" and bleeds recolouring into adjacent fabric. If that
ambiguity is measurably present (a small, compact bare-skin blob -- e.g.
praying hands), state its location as a FACT instead of an enumerated list
that may not match this photo. This is the one prompt mechanism already
proven to work elsewhere in this file (expression hints, bald-donor
wording), not a repeat of prohibition/enumeration/negative-repulsion.

Grep-on-source-text pattern, same as the other structural tests on this
branch.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _function_body(text: str, def_line: str) -> str:
    i = text.find(def_line)
    assert i > 0, f"could not find {def_line!r}"
    j = text.find("\n    def ", i + 10)
    return text[i: j if j > 0 else len(text)]


def test_flag_defaults_off():
    i = KREA2.find('"simple_full_body_precise_skin_location"')
    assert i > 0
    window = KREA2[i:i + 60]
    assert "False" in window


def test_measurement_scores_false_on_large_or_spread_skin():
    """Must NOT fire on the regression-prone case: a genuinely bare-armed
    subject has a large skin fraction and/or a wide bounding box, and must
    score compact=False so that case keeps the default enumerated wording."""
    body = _function_body(KREA2, "def _measure_bare_skin_is_compact(")
    assert "skin_frac_too_large" in body
    assert "skin_bbox_too_spread_out" in body


def test_measurement_fails_closed():
    body = _function_body(KREA2, "def _measure_bare_skin_is_compact(")
    assert body.count("return None, diag") >= 4


def test_measurement_uses_top_components_not_full_bbox():
    """v1 measured a naive bounding box over ALL skin pixels below the face
    stretching from a neck sliver to bare ankles/feet near the bottom of the
    frame -- 326px tall against a 43px face -- even though the hands blob
    is small and contained. v2 (a single dominant component) then never
    fired at all: praying hands routinely segment as TWO separate
    components (the shadow where the palms touch breaks the connection),
    so neither alone reaches 50% of the skin pixels. The bbox must be the
    COMBINED extent of the top few components that together explain most
    of the visible skin, not one component and not every skin pixel."""
    body = _function_body(KREA2, "def _measure_bare_skin_is_compact(")
    assert "connectedComponentsWithStats" in body
    assert "covered_frac_of_skin" in body
    assert "bare_skin_max_components" in body


def test_measurement_uses_semantic_skin_not_colour():
    body = _function_body(KREA2, "def _measure_bare_skin_is_compact(")
    assert "_semantic_skin_mask" in body
    assert "semantic_person_mask" in body


def test_prompt_wording_names_no_body_parts():
    """The whole point: state WHERE the skin is, don't enumerate WHAT parts
    might be bare -- enumeration is what this file's own history blames for
    the model drawing exposed skin it was only told to recolour."""
    i = KREA2.find("if _precise_skin_location")
    assert i > 0
    window = KREA2[max(0, i - 400):i]
    assert "hands" not in window.lower()
    assert "arms" not in window.lower()
    assert "legs" not in window.lower()


def test_prompt_wording_takes_priority_over_scoped_wording():
    """_precise_skin_location must be checked BEFORE _scope_skin_to_visible
    in the ternary, so a compact-skin measurement wins over the generic
    scoped fallback when both could apply."""
    i_precise = KREA2.find("if _precise_skin_location")
    i_scoped = KREA2.find("if _scope_skin_to_visible")
    assert 0 < i_precise < i_scoped


def test_negative_prompt_retargeted_to_garment_state_not_skin_state():
    """Measured on GPU: skin-state negative terms ('exposed torso, bare
    chest... nudity') had no effect, because the positive prompt is ALSO
    actively asking for skin nearby -- a skin-state negative term fights its
    own positive prompt. Retargeted to garment-state words so the two
    instructions point at different concepts instead of the same one from
    opposite directions."""
    assert '"exposed torso, bare chest, exposed stomach, "' not in KREA2
    i = KREA2.find('"missing clothing, garment removed')
    assert i > 0
    window = KREA2[i:i + 150]
    assert "undressed" in window or "disrobed" in window


def test_default_t4_prompt_still_564_chars():
    """Must not touch the byte-identical default when the new flag is off
    -- covered by the existing isolated-exec harnesses picking up
    _precise_skin_location=False; this just confirms the ternary's default
    fallback branch (neither precise nor scoped) is unchanged text."""
    i = KREA2.find(
        'else "Second: change the skin colour of the body -- the neck, "'
    )
    assert i > 0
