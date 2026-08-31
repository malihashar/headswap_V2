"""Bald-donor headwear wording must be MEASURED per-donor, never a default.

The remove-headwear clause claims "the second person's hair is there
instead" -- a claim about pixels that do not exist when the donor is bald.
On a bald donor the model has nothing coherent to draw where the target's
cap was removed, and the cap can survive as the path of least resistance.

The fix must not become a blanket "everyone is bald" instruction: a hair-
having donor (e.g. Kevin Hart in the tests this was found against) must keep
the exact wording that was already working, unchanged. Regression risk was
explicit: "sometimes it loosens and makes everybody bald."
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KREA2 = (ROOT / "src" / "headswap" / "pipelines" / "krea2.py").read_text()


def test_baldness_measurement_fails_closed_to_not_bald():
    """Any error/unmeasurable case must return False, not True.

    Failing OPEN (defaulting to bald) would risk exactly the regression
    named above on any donor the segmenter can't read.
    """
    i = KREA2.find("def _measure_donor_baldness")
    body = KREA2[i:i + 2000]
    assert "return False, diag" in body, (
        "the except branch must fail closed to False (not bald)"
    )
    # Every explicit early-return before the real measurement must also be
    # False, not True.
    unreachable = body[:body.find("hair_frac < thresh")]
    assert "return True" not in unreachable, (
        "an early-return before the real measurement returned True -- that "
        "is a fail-OPEN path, the opposite of what this function promises"
    )


def test_default_wording_is_untouched_when_not_bald():
    """The original, already-working sentence's WORDS must be unchanged.

    Adjacent Python string literals concatenate at runtime regardless of
    where the source wraps them across lines, so this checks the phrase is
    present rather than pinning an exact line-break position.
    """
    assert "hat, cap or head covering is gone" in KREA2
    assert "the second person's hair is there instead" in KREA2


def test_bald_wording_is_a_distinct_branch_not_a_default():
    assert "if _donor_bald else" in KREA2, (
        "bald wording must be a conditional branch, not unconditional text"
    )
    assert '_donor_bald = False' in KREA2, (
        "_donor_bald must default False before any measurement runs"
    )


def test_measurement_only_runs_when_headwear_removal_is_requested():
    """Wasted work (and a wrong log line) if it ran unconditionally."""
    i = KREA2.find("_donor_bald, _bald_diag = self._measure_donor_baldness")
    assert i > 0
    guard = KREA2[max(0, i - 300):i]
    assert 'simple_full_body_remove_headwear' in guard


def test_chain_pipeline_flags_are_explicitly_set_both_ways():
    """The sticky-cfg bug: flags were only ever set True, never False.

    A cached cfg dict plus 'if flag: cfg[...] = True' with no else meant a
    flag enabled once in a Colab kernel stayed on for every later call in
    that session regardless of what was passed -- which is how
    protect_garments reached a sweep that never asked for it.
    """
    chain_py = (ROOT / "src" / "headswap" / "chain.py").read_text()
    for key in (
        "simple_full_body_remove_headwear",
        "skip_skin_clause_when_covered",
        "simple_full_body_protect_garments",
    ):
        i = chain_py.find(f'cfg["{key}"] = True')
        assert i > 0, f"{key} True-branch missing"
        tail = chain_py[i:i + 200]
        assert f'cfg["{key}"] = False' in tail or "else:" in chain_py[i:i + 60], (
            f"{key} has no explicit False branch nearby -- it can only ever "
            "be turned on, never off, across cached calls"
        )


def test_load_models_default_matches_documented_default():
    """protect_garments's own parameter default must match DEFAULTS.

    DEFAULTS documents in detail why it is False (the bathrobe finding).
    The function's own default previously said True, so any caller not
    passing the argument explicitly got the opposite of the documented,
    intended default.
    """
    chain_py = (ROOT / "src" / "headswap" / "chain.py").read_text()
    assert 'DEFAULTS["protect_garments"] = False' not in chain_py  # comment form differs
    assert '"protect_garments": False,' in chain_py
    i = chain_py.find("def load_models(")
    sig = chain_py[i:i + 600]
    assert "protect_garments: bool = False" in sig, (
        "load_models' own default for protect_garments does not match "
        "DEFAULTS['protect_garments'] (False) -- a caller relying on the "
        "function default instead of DEFAULTS would silently get the "
        "bathrobe-causing wording"
    )


def test_pipeline_cache_is_flag_aware():
    """Rebuilding must happen when requested flags differ from the cache.

    Without this, `_STATE['pipe'] is not None` returned the FIRST call's
    pipeline forever and silently ignored every argument on every later
    call in the same kernel.
    """
    chain_py = (ROOT / "src" / "headswap" / "chain.py").read_text()
    assert "_flags_key" in chain_py
    assert '_STATE.get("flags_key") == _flags_key' in chain_py
