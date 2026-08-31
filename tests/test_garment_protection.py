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


def test_chain_opts_in_but_t4_default_does_not():
    assert 'self.cfg.get("simple_full_body_protect_garments", False)' in KREA2
    assert '"protect_garments": True' in CHAIN
    assert 'cfg["simple_full_body_protect_garments"] = True' in CHAIN
