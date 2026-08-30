"""The LaMa dependency check must live in repo code, not a notebook cell.

Only setup_colab.sh installs simple-lama, and Cell 1 skips that script
whenever ComfyUI is already present -- so a reconnected runtime silently
lacks it, erase_headwear() falls back, and the hat survives the swap while
the log says the mask found it (coverage=10.098%).

Putting the install in the notebook cell did NOT fix it: Colab caches cell
source in the browser tab, so a stale tab kept running the old cell while
the repo code was already current. Repo code is what `git pull` reaches,
which is why this check has to sit inside warmup().
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "src" / "headswap" / "chain.py").read_text()


def test_bootstrap_is_in_repo_code_not_the_notebook():
    from headswap.chain import ensure_simple_lama  # noqa: F401

    assert "def ensure_simple_lama" in SRC


def test_warmup_calls_it():
    i = SRC.find("def warmup(")
    j = SRC.find("def run_chain(", i)
    assert 0 < i < j
    assert "ensure_simple_lama()" in SRC[i:j]


def test_pin_repair_accompanies_the_install():
    """simple-lama pulls pillow 9.5 / numpy 1.26, which break rembg and
    restore_background silently. Installing without repairing trades a
    visible hat for two invisible regressions."""
    assert "pillow==11.3.0" in SRC
    assert "numpy==2.4.6" in SRC


def test_failure_is_reported_not_silent():
    assert "headwear erase will skip and the hat will remain" in SRC
