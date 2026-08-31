"""On a covered subject, DROP the skin sentence instead of adding words.

The skin-recolour sentence asks to recolour the neck, arms, hands and legs
"that are already bare". A robed subject has none, so the model exposes some
to have something to recolour -- CHECKPOINT-10 lists the shirtless robe as
known-unfixed and this file's history names the cause: "naming body parts as
'bare skin' once made the model strip a robe".

Three attempts to fix that by ADDING words each broke something else:
  1. do-not-expose prohibition   torso covered, SLEEVES removed instead
  2. drop the body-part list     sleeves back
  3. "robe, shirt or top"        a tennis POLO rendered as a fluffy bathrobe

On this route the model draws whatever the prompt names. So the fix removes
words rather than adding them, and because nothing new is named, a subject
that was already correct cannot change -- which is the property all three
earlier attempts lacked.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()
CHAIN = (ROOT / "src" / "headswap" / "chain.py").read_text()


def _assemble(drop: bool, headwear: bool = False) -> str:
    i = KREA2.find("prompt = _prompt_override or (")
    # End at the negative-prompt block, not at the prompt LOG. Slicing to
    # the log swallows that block, whose different indent makes exec() raise
    # IndentationError -- which surfaces as every assertion here failing at
    # once rather than as anything about prompts.
    j = KREA2.find("\n        # Steer AWAY from headwear", i)
    assert 0 < i < j
    ns = {
        "_prompt_override": "",
        "_expr_inline": "",
        "_drop_skin_clause": drop,
        "self": type("C", (), {"cfg": {
            "simple_full_body_remove_headwear": headwear,
            "simple_full_body_protect_garments": False,
        }})(),
    }
    exec(KREA2[i:j], {}, ns)  # noqa: S102
    return ns["prompt"]


def test_t4_default_is_still_564_chars():
    """Nothing about this may touch the approved recipe."""
    assert len(_assemble(False)) == 564


def test_covered_subject_loses_the_whole_skin_sentence():
    p = _assemble(True)
    assert "skin colour" not in p
    assert "already bare" not in p


def test_covered_subject_adds_no_new_words():
    """The property the three failed attempts lacked: a shorter prompt cannot
    introduce a garment or body part that was not there before."""
    assert len(_assemble(True)) < len(_assemble(False))


def test_dropping_also_fixes_the_dangling_two_changes_contract():
    """"Two changes ... First ... Second" only parses as instructions if the
    second one exists."""
    assert _assemble(False).startswith("Two changes. First:")
    assert _assemble(True).startswith("One change:")
    assert "Second:" not in _assemble(True)


def test_measurement_compares_against_the_subjects_own_face():
    """A fixed skin-tone range would fail on dark or very pale subjects; the
    face is the correct per-image reference, and a*b* only so shadow does not
    read as clothing."""
    i = KREA2.find("def _measure_visible_skin(")
    assert i > 0
    body = KREA2[i:KREA2.find('\n    def ', i + 10)]
    assert "np.median(lab[fy0:fy1, fx0:fx1]" in body
    assert "body[:, 1:] - ref[1:]" in body


def test_measurement_never_breaks_a_render():
    i = KREA2.find("def _measure_visible_skin(")
    assert "must never break a render" in KREA2[i:KREA2.find('\n    def ', i + 10)]


def test_chain_enables_it_and_leaves_add_words_off():
    assert '"skip_skin_clause_when_covered": True' in CHAIN
    assert '"protect_garments": False' in CHAIN, (
        "the add-words approach must stay off: it altered clothing on inputs "
        "that were previously correct"
    )
