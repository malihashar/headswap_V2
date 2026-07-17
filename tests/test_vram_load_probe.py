from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.vram_load_probe import VramLoadProbe


def test_ingest_partial_log_line():
    p = VramLoadProbe()
    p._ingest_log_line(
        "loaded partially; 4128.00 MB usable, 3901.25 MB loaded, 15200.50 MB offloaded, "
        "512.00 MB buffer reserved, lowvram patches: 42"
    )
    assert p.report.sampling_load_mode == "partial"
    ev = p.report.events[0]
    assert ev.kind == "partial"
    assert ev.mb_loaded == 3901.25
    assert ev.mb_offloaded == 15200.50
    assert ev.mb_usable == 4128.0
    assert ev.lowvram_patches == 42


def test_ingest_complete_log_line():
    p = VramLoadProbe()
    p._ingest_log_line("loaded completely; 19483.95 MB loaded, full load: True")
    assert p.report.sampling_load_mode == "complete"
    ev = p.report.events[0]
    assert ev.kind == "complete"
    assert ev.mb_loaded == 19483.95
    assert ev.full_load_flag is True


def test_finalize_prefers_partial():
    p = VramLoadProbe()
    p._ingest_log_line("loaded completely; 100.00 MB loaded, full load: True")
    p._ingest_log_line(
        "loaded partially; 50.00 MB usable, 40.00 MB loaded, 900.00 MB offloaded, "
        "10.00 MB buffer reserved, lowvram patches: 1"
    )
    rep = p.finalize()
    assert rep.sampling_load_mode == "partial"
    assert rep.summary["mb_offloaded"] == 900.0
