"""Bare skin rendered in a garment-like tone must not be frozen at the original.

Observed as a hard vertical edge down one leg: pale/original on one side,
correctly toned on the other. A soft under-correction fades out; a hard edge
is a mask boundary. A bare leg rendered in a pale cream tone reads as a
stocking or legging to the class segmenter and is labelled CLOTHES, which
drops it from the skin gate entirely. Because that veto happens BEFORE
weighting, no change to transfer weight or strength can reach it.

The rescue overrides the veto where the pixel scores highly on skin-likeness
against this person's OWN face in this photo's light, and rembg agrees the
pixel belongs to the person.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import headswap.skin_harmonize as sh

HARM = (ROOT / "src" / "headswap" / "skin_harmonize.py").read_text()


def test_rescue_threshold_is_high_enough_to_reject_a_garment():
    """A blue jersey scored 1.00 on colour alone in an earlier regression."""
    assert sh._SKIN_RESCUE_FROM_CLOTHES >= 0.7, (
        f"threshold {sh._SKIN_RESCUE_FROM_CLOTHES} is too permissive; a "
        "skin-coloured garment would be recoloured"
    )


def test_rescue_requires_the_rembg_matte_too():
    """Colour evidence alone is not sufficient -- it must also be the person."""
    assert "(_pm_norm > 0.5)" in HARM, (
        "the rescue does not require rembg agreement, so background pixels "
        "with skin-like colour could be recoloured"
    )


def test_rescue_only_adds_coverage():
    assert "_sem = np.maximum(_sem, _rescue)" in HARM, (
        "the rescue must be a union; anything else could REMOVE coverage the "
        "class segmenter got right"
    )


def test_rescue_is_disableable():
    assert "HEADSWAP_SKIN_RESCUE_THR" in HARM, (
        "no escape hatch: if a skin-coloured garment is recoloured there must "
        "be a way to raise the threshold without a code change"
    )


def test_rescue_runs_before_the_gate_is_applied():
    i_rescue = HARM.find("_sem = np.maximum(_sem, _rescue)")
    i_apply = HARM.find("weight = np.maximum(weight * _sem_c, _sem_c * _matte)")
    assert 0 < i_rescue < i_apply, (
        "the rescue must widen the gate BEFORE the gate is multiplied into "
        "the weight, or it has no effect"
    )
