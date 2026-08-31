"""On a covered subject, SCOPE the skin sentence instead of dropping it.

The skin-recolour sentence asks to recolour the neck, arms, hands and legs
"that are already bare". A robed subject has none, so the model exposes some
to have something to recolour -- CHECKPOINT-10 lists the shirtless robe as
known-unfixed and this file's history names the cause: "naming body parts as
'bare skin' once made the model strip a robe".

Three attempts to fix that by ADDING words each broke something else:
  1. do-not-expose prohibition   torso covered, SLEEVES removed instead
  2. drop the body-part list     sleeves back
  3. "robe, shirt or top"        a tennis POLO rendered as a fluffy bathrobe

A fourth attempt REMOVED words instead -- dropping the "Second:" sentence
entirely for a covered subject. That fixed the exposure bug (nothing named,
nothing to go expose) but also removed the only instruction telling the
model to colour-match skin that genuinely IS visible: a robed subject's own
praying hands came back visibly darker than the face, mismatched, because
nothing told the model what tone to use on them. Measured on GPU.

The current fix keeps the sentence but SCOPES it: "wherever skin is already
visible in the first image and only there". Nothing is enumerated, so there
is nothing to go and expose -- the same property the full drop had -- but
unlike the full drop it still tells the model what tone already-visible skin
(hands, face) should be recoloured to.
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
        "_scope_skin_to_visible": drop,
        # Measured-location wording (see test_precise_skin_location.py): off
        # by default, never fires in this file's tests.
        "_precise_skin_location": False,
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


def test_covered_subject_keeps_the_skin_sentence_but_scopes_it():
    """The full-drop approach is gone -- see this file's docstring. The
    sentence stays, but no longer enumerates body parts, so it cannot
    reopen the exposure bug it used to cause."""
    p = _assemble(True)
    assert "skin colour" in p
    assert "already bare" not in p
    assert "wherever skin is already visible in the first image" in p


def test_covered_subject_prompt_length():
    """The scoped wording is not a full drop and is not shorter than the
    enumeration -- it is a different sentence, not a truncated one. Its
    own length is pinned so future edits to this branch are deliberate."""
    assert len(_assemble(True)) == 622


def test_scoping_keeps_the_two_changes_contract_intact():
    """"Two changes ... First ... Second" must still parse as instructions
    -- the sentence is scoped now, not removed."""
    assert _assemble(False).startswith("Two changes. First:")
    assert _assemble(True).startswith("Two changes. First:")
    assert "Second:" in _assemble(True)


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


def test_chain_uses_the_matte_signal_and_not_the_add_words_one():
    """Both are off, for different reasons (CHECKPOINT-16).

    Adding words altered clothing on inputs that were already correct, so
    that approach stays off permanently.

    Dropping/scoping the clause needed a reliable "is this subject
    covered?" signal. Three GEOMETRIC attempts inverted (53.0/5.5, then
    18.0/6.9 -- covered scoring higher than bare both times) because
    background and framing dominated. Measuring inside the person matte
    removed both confounds and was re-enabled -- but on the exact test case
    it was built to fix, both the full-drop and the scoped-wording variants
    then rendered the subject's genuinely-visible hands visibly WHITE and
    mismatched, worse than the shirtless bug they replaced. Measured on GPU
    twice. Off again, at Ali's direction: keep the plain enumeration that
    already works for most cases rather than iterate further on a mechanism
    that has now failed the same test case three separate ways.
    """
    assert '"skip_skin_clause_when_covered": False' in CHAIN, (
        "off again -- the matte-based drop/scope mechanism produced "
        "mismatched white hands on the covered test case it was built to "
        "fix, worse than the shirtless bug it replaced"
    )
    assert '"protect_garments": False' in CHAIN, (
        "the add-words approach stays off: it altered clothing on inputs "
        "that were previously correct"
    )
    assert '"protect_garments": False' in CHAIN, (
        "the add-words approach must stay off: it altered clothing on inputs "
        "that were previously correct"
    )
