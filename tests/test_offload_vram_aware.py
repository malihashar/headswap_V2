"""The pre-sample offload must be skipped when VRAM is ample.

It was unconditional -- correct on a 16GB T4, where the UNet cannot sit
alongside CLIP/VAE. On a 40GB A100 with ~34GB free and a ~13GB UNet it
evicts for headroom that already exists and pays a full reload on the next
sample: measured at 6.3s per swap over four evict/reload cycles.

Gated on MEASURED free VRAM, not on a device name, so a small card keeps the
old behaviour without any caller knowing which GPU it is on. "Unknown" must
count as "not ample" -- guessing wrong here is an OOM, not a slowdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "src" / "headswap" / "comfy" / "full_load.py").read_text()


def test_offload_is_gated_on_free_vram():
    assert "_cuda_free_mb()" in SRC
    assert "offload_skipped_ample_vram" in SRC
    assert 'reason="before_sample"' in SRC, "the offload path must still exist"


def test_unknown_vram_is_treated_as_not_ample():
    """None means CUDA is unavailable or the query failed. That must fall
    through to the safe offload, never to the skip."""
    assert "_free_mb is not None and _free_mb >= _skip_free_mb" in SRC


def test_threshold_is_well_above_the_working_set():
    """~13GB UNet + ~4GB CLIP. The default must leave a wide margin, because
    being wrong about skipping is an OOM rather than a slow render."""
    import re
    m = re.search(r'HEADSWAP_OFFLOAD_SKIP_FREE_MB", (\d+)', SRC)
    assert m, "threshold must be env-overridable"
    assert int(m.group(1)) >= 20000, "margin too tight over a ~17GB working set"


def test_cuda_free_mb_returns_none_without_cuda():
    """Runs on CPU CI, so this exercises the real fallback path."""
    from headswap.comfy.full_load import _cuda_free_mb

    val = _cuda_free_mb()
    assert val is None or isinstance(val, float)


def test_every_after_sample_offload_site_is_gated():
    """There are TWO after_sample call sites: one in the `not enabled`
    branch and one in the live force_full_load path. The first fix patched
    only the dead one, so churn stayed at 4.9s and the skip line never
    printed. Both must be gated, or the fix is invisible rather than absent.
    """
    import re

    # Count the GATE itself, not a log string -- the log phrase also appears
    # in a comment, which made the first version of this test count three
    # gates for two call sites.
    n_calls = SRC.count('reason="after_sample"')
    n_gates = len(re.findall(
        r'if _ample:\s*\n\s*print\(\s*\n?\s*"\[full_load\] post-sample', SRC))
    assert n_calls >= 1
    assert n_gates == n_calls, (
        f"{n_calls} after_sample call sites but {n_gates} gated -- an "
        "ungated site silently keeps the churn"
    )
