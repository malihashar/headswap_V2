"""Headwear removal is a prompt clause INSIDE sentence one, not an appendix.

History, all measured on GPU:
  1. base "anything worn on the head"      -> cap survived
  2. CRITICAL clause appended last (943)   -> cap survived
  3. same + "keep clothing" scoped to
     below the neck (958)                  -> cap AND durag survived
  4. LaMa/Telea inpaint of the target      -> cap removed, but Telea smears
                                              and the artifacts survived into
                                              the final image. Rejected.

Attempts 2 and 3 appended the clause DEAD LAST, after the prohibitions.
CHECKPOINT-10 records that exact failure mode for a different instruction:
the skin-colour clause sat third, after two clauses of prohibitions, and
"the model simply did not act on a buried clause" -- moving it up is what
made it work. So the clause now sits inside the head sentence, before
"Second:", and is phrased as a fact rather than a CRITICAL order.

It is also ~300 chars shorter than the appended version, which matters
independently: CHECKPOINT-10/11 measured that prompt LENGTH alone moves
face fraction on this route.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def _assemble(remove_headwear: bool, donor_bald: bool = False) -> str:
    """Build the prompt exactly as run_simple_full_body does."""
    i = KREA2.find("prompt = _prompt_override or (")
    j = KREA2.find("\n        # Print the prompt that will actually be used", i)
    assert 0 < i < j
    ns = {
        "_prompt_override": "",
        "_expr_inline": "",
        # T4's default path: the skin sentence is kept.
        # It is dropped only for a subject with no visible skin.
        "_drop_skin_clause": False,
        # Bald-donor wording (see test_donor_baldness_headwear_wording.py):
        # False reproduces the original, already-working sentence exactly;
        # True selects the bare-head branch for a donor with no hair.
        "_donor_bald": donor_bald,
        "self": type("C", (), {
            "cfg": {"simple_full_body_remove_headwear": remove_headwear}
        })(),
    }
    exec(KREA2[i:j], {}, ns)  # noqa: S102
    return ns["prompt"]


def test_default_prompt_is_byte_identical_to_t4():
    """CHECKPOINT-10 records the approved text as 564 chars and treats it as
    part of the recipe, not incidental."""
    assert len(_assemble(False)) == 564


def test_clause_lands_inside_sentence_one():
    p = _assemble(True)
    i_clause = p.find("hat, cap or head covering is gone")
    i_second = p.find("Second:")
    assert 0 < i_clause < i_second, "the clause must not be buried after the prohibitions"


def test_clause_is_a_fact_not_a_shouted_order():
    """"CRITICAL: ... REMOVE it completely" was tried twice and ignored.
    Measured facts are the pattern with precedent here."""
    p = _assemble(True)
    assert "CRITICAL" not in p
    assert "is gone" in p and "is there instead" in p


def test_clause_is_shorter_than_the_appended_version():
    """The appended block took the prompt to 958 chars. Length alone moves
    face fraction on this route."""
    assert len(_assemble(True)) < 800


def test_absent_when_disabled():
    assert "head covering" not in _assemble(False)


def test_override_still_wins():
    """A caller-supplied prompt must not get the clause spliced into it."""
    i = KREA2.find("prompt = _prompt_override or (")
    assert i > 0, "the override must short-circuit the whole assembled prompt"


def test_bald_donor_gets_bare_head_wording_not_hair():
    p = _assemble(True, donor_bald=True)
    assert "there is no hair on it, it is bald" in p
    assert "the second person's hair is there instead" not in p


def test_non_bald_donor_wording_is_unaffected_by_the_bald_branch():
    """The False path must produce EXACTLY what it produced before the
    bald-donor branch was added -- same length as the original 4-arm sweep
    measured (CHECKPOINT-16)."""
    p_default = _assemble(True, donor_bald=False)
    assert "the second person's hair is there instead" in p_default
    assert "it is bald" not in p_default
