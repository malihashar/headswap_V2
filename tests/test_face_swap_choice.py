"""Unit tests for Colab multi-face selector parsing (no GPU / PIL needed)."""
from __future__ import annotations

from pathlib import Path


def _load_parse():
    """Load parse_face_swap_choice without importing the full colab_demo module."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "colab_demo.py").read_text()
    start = text.find("def parse_face_swap_choice")
    end = text.find("\ndef show_face_swap_selector")
    assert start >= 0 and end > start
    ns: dict = {}
    exec(  # noqa: S102 — test helper isolates pure function
        "from __future__ import annotations\nfrom typing import Any\n"
        + text[start:end],
        ns,
    )
    return ns["parse_face_swap_choice"]


def test_swap_all():
    parse = _load_parse()
    out = parse("Swap All Faces")
    assert out["face_swap_mode"] == "all"
    assert out["body_face_index"] == 0


def test_swap_face_n():
    parse = _load_parse()
    out = parse("Swap Face 3")
    assert out["face_swap_mode"] == "single"
    assert out["body_face_policy"] == "index"
    assert out["body_face_index"] == 2


def test_default_single():
    parse = _load_parse()
    out = parse(None)
    assert out["face_swap_mode"] == "single"
    assert out["body_face_index"] == 0
