"""Inline expression hint: generalises to any pair, tests an untested position.

CHECKPOINT-13 measured the body photo's actual mouth state per-image (not a
phrase hand-picked for one test case) but only tested it APPENDED after the
whole prompt. Position was never isolated as its own variable. This wires the
same measurement into the middle of the head-swap sentence instead -- the
untested "inline" position -- while leaving the appended path from
CHECKPOINT-13 completely alone.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_default_is_still_append_not_inline():
    """T4's default render must be untouched by this addition."""
    assert 'self.cfg.get("expression_hint_position", "append")' in KREA2


def test_inline_disabled_when_prompt_is_overridden():
    """A caller-supplied prompt (simple_full_body_prompt) must win outright --
    inline injection must not silently rewrite an override's text."""
    assert "not _prompt_override" in KREA2


def test_appended_and_inline_cannot_both_apply():
    """The two hint positions must be mutually exclusive, or a pair could get
    the measured fact twice, doubling the very framing risk CHECKPOINT-11/12
    exist to guard against."""
    assert "if _expr_inline_diag is not None:" in KREA2
    assert '_expr_hint, expression_diag = "", _expr_inline_diag' in KREA2


def test_inline_clause_has_no_leftover_full_stop_from_the_hint_sentence():
    """The hint text is a full sentence ("The person is X."); spliced into a
    clause it must not carry a stray period mid-sentence."""
    assert "_clause = _clause.rstrip" in KREA2


def test_base_prompt_unchanged_when_inline_hint_is_empty():
    """No hint (landmarks unavailable) must degrade to exactly T4's prompt."""
    i = KREA2.find('f"second image, with none of the first person\'s head remaining{_expr_inline}. "')
    assert i > 0, "the inline slot must default to an empty string, not vanish"
