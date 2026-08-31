"""Garment-exposure guard via the NEGATIVE prompt, not a positive rewrite.

History: three POSITIVE-prompt fixes for "a covered subject's clothing gets
stripped near the hands/torso" already failed on GPU -- a do-not-expose
prohibition, naming garment types, and dropping/scoping the skin-recolour
sentence (see test_skin_clause_drop.py, test_garment_protection.py). "On
this route the model draws whatever the positive prompt names."

This uses the SAME different mechanism that already worked for headwear
removal on this route: classifier-free guidance repulsion via the negative
prompt at cfg=1.8, not an instruction the model has to obey -- so it adds
nothing to the positive prompt and cannot repeat those three failure modes.
Gated on the SAME already-measured "is this subject covered" signal used by
skip_skin_clause_when_covered (_measure_visible_skin / _measure_torso_
clothed, additive OR), so an already-bare-armed subject that legitimately
needs skin exposed never receives these tokens and cannot regress.

Grep-on-source-text pattern, same as the other structural tests on this
branch: no isolated-exec, no GPU.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_flag_defaults_off():
    i = KREA2.find('"simple_full_body_garment_negative_prompt"')
    assert i > 0
    window = KREA2[i:i + 60]
    assert "False" in window


def test_gated_on_the_existing_covered_measurement_not_new_detection():
    """Must reuse _measure_visible_skin and _measure_torso_clothed --
    already built, already GPU-validated -- not introduce a third detector."""
    i = KREA2.find('"simple_full_body_garment_negative_prompt"')
    assert i > 0
    window = KREA2[i:i + 1500]
    assert "self._measure_visible_skin(body_full)" in window
    assert "self._measure_torso_clothed(body_full)" in window
    assert "_ngp_skin_covered or _ngp_torso_covered" in window, (
        "must be an additive OR of both signals, same as "
        "skip_skin_clause_when_covered's own decision"
    )


def test_does_not_touch_the_positive_prompt():
    """The whole point: this must be reachable without altering `prompt` at
    all. The garment-negative-prompt block must sit in the negative-prompt
    section, entirely separate from where `prompt = ...` is assembled."""
    i_prompt_build = KREA2.find("prompt = _prompt_override or (")
    i_negative_block = KREA2.find('"simple_full_body_garment_negative_prompt"')
    assert 0 < i_prompt_build < i_negative_block, (
        "the negative-prompt guard must be built AFTER (and separate from) "
        "prompt assembly, never folded into the positive prompt text"
    )


def test_headwear_and_garment_negative_clauses_coexist():
    """Restructured from a single overwrite-if-empty assignment to an
    appended list, so remove_headwear and the garment guard can both fire on
    the same render without one clobbering the other."""
    i = KREA2.find("_neg_clauses: list[str] = []")
    assert i > 0
    window = KREA2[i:i + 3000]
    assert '_neg_clauses.append(' in window
    assert window.count('_neg_clauses.append(') >= 2
    assert '", ".join(_neg_clauses)' in window


def test_garment_clause_wording_is_concrete_not_vague():
    i = KREA2.find('"exposed torso, bare chest')
    assert i > 0
    window = KREA2[i:i + 200]
    assert "nudity" in window


def test_logs_which_signal_fired_or_why_not():
    i = KREA2.find('"simple_full_body_garment_negative_prompt"')
    assert i > 0
    window = KREA2[i:i + 2200]
    assert "[krea2 garment_negative]" in window
    assert "skin_covered=" in window
    assert "torso_covered=" in window
