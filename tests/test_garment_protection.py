"""A robed subject must keep the robe.

CHECKPOINT-10 lists "a robed subject can still come back shirtless (garment
removed outright)" as known-unfixed, and this file's own history names the
cause: "naming body parts as 'bare skin' once made the model strip a robe".

The skin-recolour clause asks for the neck, arms, hands and legs "that are
already bare". On a covered subject the model has nothing to recolour, so it
EXPOSES some. "Do not turn any clothed area into skin" was the counterweight
and was measured insufficient -- a desert robe came back bare-chested.

Gated rather than unconditional: CHECKPOINT-10 fixes T4's approved text at
564 chars as part of the recipe, and CHECKPOINT-11 measured that prompt
LENGTH alone moves face fraction (38.7 -> 45.0, and 40.3 -> 42.0 for a
single added sentence). So the default must stay byte-identical and callers
opt in.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
CHAIN = (ROOT / "src" / "headswap" / "chain.py").read_text()


def _assemble(headwear: bool, garments: bool) -> str:
    i = KREA2.find("prompt = _prompt_override or (")
    j = KREA2.find("\n        # Explicit headwear-removal clause", i)
    assert 0 < i < j
    ns = {
        "_prompt_override": "",
        "_expr_inline": "",
        "self": type("C", (), {"cfg": {
            "simple_full_body_remove_headwear": headwear,
            "simple_full_body_protect_garments": garments,
        }})(),
    }
    exec(KREA2[i:j], {}, ns)  # noqa: S102
    return ns["prompt"]


def test_t4_default_stays_byte_identical_at_564_chars():
    """The approved recipe must be untouched when nothing is opted into."""
    assert len(_assemble(False, False)) == 564


def test_default_keeps_the_original_weak_prohibition():
    assert _assemble(False, False).endswith(
        "Do not turn any clothed area into skin."
    )


def test_opt_in_forbids_exposing_new_skin():
    p = _assemble(False, True)
    assert "do not expose any new skin" in p
    assert "do not remove, shorten or open any garment" in p
    assert "Do not turn any clothed area into skin." not in p, (
        "the two prohibitions must not stack: length alone moves framing"
    )


def test_opt_in_names_the_garment_types_that_failed():
    """A desert robe was the measured failure; naming it beats a generic
    'clothing', which the model already had and ignored."""
    assert "robe, shirt or top" in _assemble(False, True)


def test_nothing_enables_garment_protection_by_default():
    """The wording stays available behind a flag so the arms are
    reproducible, but NOTHING turns it on.

    Measured on GPU: naming garment types rendered them. A tennis polo came
    back as a fluffy bathrobe because the clause said "robe, shirt or top".
    That is the same mechanism as the body-part enumeration it was written to
    replace -- on this route the prompt cannot name a thing without the model
    drawing it -- so it is a wrong approach, not a wrong wording.
    """
    assert 'self.cfg.get("simple_full_body_protect_garments", False)' in KREA2
    assert '"protect_garments": False' in CHAIN, (
        "the chain must not opt in: it altered clothing on inputs that were "
        "previously correct"
    )


def test_opt_in_stops_enumerating_body_parts():
    """The enumeration is the mechanism, not a missing prohibition.

    "the neck, arms, hands and legs that are already bare" names parts, and
    the model exposes whichever named part is covered so it has something to
    recolour. Measured twice: a robed subject came back bare-chested; adding
    a do-not-expose prohibition fixed the torso and the model moved down the
    list, returning the same robe with the SLEEVES removed.
    """
    p = _assemble(False, True)
    assert "the neck, arms, hands and legs that are already bare" not in p
    assert "wherever skin is already visible in the first image" in p


def test_default_keeps_the_enumeration():
    """T4's approved 564-char text is unchanged; only opt-in callers differ."""
    assert "the neck, arms, hands and legs that are already bare" in _assemble(
        False, False
    )
