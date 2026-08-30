"""The clothing-preservation clause must stay at T4's approved wording.

It briefly read "Keep the clothing below the neck ...", on the theory that
"keep the clothing" was contradicting the headwear removal clause, since a
cap is clothing. The cap survived that change anyway. So the qualifier fixed
nothing while weakening a preservation instruction on a route whose
known-unfixed list (CHECKPOINT-10) already includes "a robed subject can
still come back shirtless" -- and a robed subject then came back shirtless.

Cause was never isolated; that failure predates the change. The point is
that an edit which bought nothing does not get to remain a suspect, and
CHECKPOINT-10 requires prompt edits be A/B'd on face fraction before
landing, which this one never was.

Headwear is now handled by ERASING it from the target before the swap, which
takes it out of the source latent rather than arguing with the prompt.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_clothing_clause_is_t4s_approved_wording():
    assert ('"Keep the clothing, pose, body shape, background and lighting "'
            in KREA2)


def test_the_below_the_neck_variant_is_gone():
    """Reverted, not merely disabled -- it is a suspect in a garment
    regression and bought nothing.

    Matches the string LITERAL, not prose: the comment explaining the revert
    necessarily quotes the phrase, and a plain substring check flags that as
    the code still being present. This repo already documents that exact
    false positive elsewhere.
    """
    # Discriminate on text only the CODE had. The comment quotes the phrase
    # as '"Keep the clothing below the neck ..."', so both a substring check
    # and a regex anchored on the opening quote match it -- I got this wrong
    # twice. The real literal continued ", pose, body shape, "; the comment
    # elides that with an ellipsis.
    assert '"Keep the clothing below the neck, pose' not in KREA2, (
        "the qualified variant is still in the prompt"
    )


def test_headwear_is_handled_before_the_swap_instead():
    chain = (ROOT / "src" / "headswap" / "chain.py").read_text()
    assert "erase_headwear_first" in chain
    assert 'fallback="telea"' in chain
