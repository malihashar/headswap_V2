"""Expression is MEASURED from the body and stated as fact, not instructed.

An earlier attempt appended a meta-instruction telling the model to source
identity from image 2 while holding the expression to image 1.
It failed twice over: the donor's smile came through unchanged, and the added
text moved the generated face from 38.7% to 45.0% of the frame at identical
sampling and seed (docs/PIPELINE_STATE.md CHECKPOINT-10).

That phrasing asks the model which INPUT to obey. It cannot be checked against
anything the model is producing, and it competes with the donor portrait
arriving as image conditioning at ref_boost=5.5, which the prompt does not
govern at all.

A measured hint is a different kind of claim: "The person is not smiling, with
the mouth closed" describes the content of the picture being made, like
"wearing a blue shirt". That is why _measure_head_direction_hint works, and it
is the only expression lever in this repo with a working precedent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


class _Pipe(Krea2IdentityEditPipeline):
    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.cache_dir = ROOT / "results" / "_cache"


def test_hint_is_appended_after_the_prompt_is_built():
    """The T4 base text must stay byte-identical; the hint is appended."""
    i_base = KREA2.find('"Two changes. First:')
    i_hint = KREA2.find("_expr_hint, expression_diag = self._measure_expression_hint")
    assert i_base > 0 and i_hint > i_base, (
        "the hint must be appended after the prompt is assembled, or the T4 "
        "recipe's prompt is no longer byte-identical"
    )
    assert 'prompt = f"{prompt} {_expr_hint}"' in KREA2


def test_hint_states_a_fact_not_an_instruction():
    """No 'take only from image N' phrasing -- that is what already failed."""
    assert '"The person is {label}."' in KREA2 or 'f"The person is {label}."' in KREA2
    for banned in ("stays exactly as it is in the first image",
                   "ONLY the identity from the second image"):
        assert banned not in KREA2, f"reverted meta-instruction is back: {banned}"


def test_hint_can_be_disabled():
    assert 'self.cfg.get("expression_prompt_hint", True)' in KREA2


def test_disabled_returns_no_hint():
    pipe = _Pipe({"expression_prompt_hint": False})
    text, diag = pipe._measure_expression_hint(None, None)
    assert text == ""
    assert diag["reason"] == "disabled"


def test_missing_face_returns_no_hint():
    pipe = _Pipe({})
    text, diag = pipe._measure_expression_hint(None, None)
    assert text == ""
    assert diag["applied"] is False


def test_failure_never_raises():
    """A hint must never break a render -- it is diagnostics, not pixels."""
    class _Bad:
        def convert(self, *_a, **_k):
            raise RuntimeError("boom")

    class _Box:
        height = 100
    pipe = _Pipe({})
    text, diag = pipe._measure_expression_hint(_Bad(), _Box())
    assert text == ""
    assert "reason" in diag


def test_hint_does_not_read_selected_face_before_it_is_assigned():
    """Regression: the hint call crashed every render with UnboundLocalError.

    The prompt is assembled near the top of run_simple_full_body, but
    `selected_face` is not assigned until well below it. Because that name IS
    assigned somewhere in the function, Python treats it as local for the
    entire scope, so reading it early raises UnboundLocalError immediately --
    it is not a None check that can save you. The same trap is already
    documented in this file for the np/cv2 imports.

    The fix is that _measure_expression_hint detects its own face box, so the
    call site passes nothing and the ordering cannot matter.
    """
    src = KREA2
    i_call = src.find("self._measure_expression_hint(body_full)")
    assert i_call > 0, (
        "the hint call must pass only body_full; passing selected_face "
        "reintroduces the UnboundLocalError crash"
    )
    # The call must not name selected_face at all.
    call_line_end = src.find("\n", i_call)
    assert "selected_face" not in src[i_call:call_line_end]
    # And the assignment must genuinely still be below the call. Search from
    # the call onward: the same text also appears earlier as the hint method's
    # own default argument, which is not the binding that caused the crash.
    i_assign = src.find("selected_face: FaceBox | None = None", i_call)
    assert i_assign > i_call, (
        "selected_face is now assigned before the hint call; this test's "
        "premise changed and it should be re-examined rather than deleted"
    )


def test_hint_detects_its_own_face():
    assert "selected_face = detect_best_face(rgb, self.cache_dir)" in KREA2
