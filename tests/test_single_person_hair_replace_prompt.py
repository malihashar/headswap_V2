"""single_person_parity's early-return path silently skipped the
hair-force prompt reinforcement.

single_person_parity defaults true, so _single_person_parity() is true for
EVERY render in the default production config -- single or multi-person.
That made the "Replace the hair completely..." reinforcement dead code in
production (never appended to any real render), which is
multi_hair_replace_prompt's entire purpose.

GPU-confirmed against two known-good reference renders of the same
target/donor pair (2026-08-07, predating this session): the one WITHOUT
this reinforcement kept the target's own long hair with only the face
swapped; the one WITH it correctly replaced the hair with the donor's
actual hairstyle -- i.e. a real head swap vs. a face swap, this project's
stated goal.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.pipelines.krea2 import Krea2IdentityEditPipeline

HAIR_CLAUSE = "Replace the hair completely with the hairstyle from the second"


def _pipe(cfg: dict | None = None) -> Krea2IdentityEditPipeline:
    class _Pipe(Krea2IdentityEditPipeline):
        def __init__(self, cfg):
            self.cfg = dict(cfg or {})
            self.cache_dir = ROOT / "results" / "_cache"

    return _Pipe(cfg)


def test_single_person_parity_gets_hair_replace_clause_by_default():
    pipe = _pipe({"prompt": "base prompt.", "single_person_parity": True})
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False, direction_hint="")
    assert HAIR_CLAUSE in out


def test_single_person_parity_respects_flag_disabled():
    pipe = _pipe(
        {
            "prompt": "base prompt.",
            "single_person_parity": True,
            "multi_hair_replace_prompt": False,
        }
    )
    out = pipe._prompt_for_edit(use_tight=False, multi_person=False, direction_hint="")
    assert HAIR_CLAUSE not in out


def test_non_spp_path_still_gets_hair_replace_clause_for_tight_crop():
    pipe = _pipe({"prompt": "base prompt.", "single_person_parity": False})
    out = pipe._prompt_for_edit(use_tight=True, multi_person=False, direction_hint="")
    assert HAIR_CLAUSE in out
