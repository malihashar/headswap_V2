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


def _assemble(remove_headwear: bool) -> str:
    """Build the prompt exactly as run_simple_full_body does."""
    i = KREA2.find("prompt = _prompt_override or (")
    j = KREA2.find("\n        # Print the prompt that will actually be used", i)
    assert 0 < i < j
    ns = {
        "_prompt_override": "",
        "_expr_inline": "",
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
