from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from headswap.profiling.path_proof import (
    PATH_PROOF_TOKEN,
    dump_identity,
    dump_vaedecode_impl,
    enter,
    install_decode_chain_enters,
)


def test_enter_emits_to_stdout_and_stderr(capsys, tmp_path, monkeypatch):
    log = tmp_path / "path_proof.log"
    monkeypatch.setenv("HEADSWAP_PATH_PROOF_LOG", str(log))
    enter("Z", "unit_test")
    captured = capsys.readouterr()
    assert "[path_proof] ENTER Z unit_test" in captured.out
    assert "[path_proof] ENTER Z unit_test" in captured.err
    assert log.read_text().strip().endswith("ENTER Z unit_test")


def test_dump_identity_includes_token(capsys, tmp_path, monkeypatch):
    log = tmp_path / "path_proof.log"
    monkeypatch.setenv("HEADSWAP_PATH_PROOF_LOG", str(log))
    dump_identity("unit")
    out = capsys.readouterr().out
    assert PATH_PROOF_TOKEN in out
    assert "path_proof.__file__=" in out
    assert "headswap.pipelines.qwen" in out or "NOT_IN_sys.modules" in out


def test_dump_vaedecode_impl_with_fake_runtime(capsys, tmp_path, monkeypatch):
    log = tmp_path / "path_proof.log"
    monkeypatch.setenv("HEADSWAP_PATH_PROOF_LOG", str(log))

    class FakeVAEDecode:
        FUNCTION = "decode"

        def decode(self, vae, samples):
            return (samples,)

    rt = MagicMock()
    rt.mappings = {"VAEDecode": FakeVAEDecode}
    dump_vaedecode_impl(rt)
    out = capsys.readouterr().out
    assert "ENTER D1" in out
    assert "VAEDecode class=" in out
    assert "getsourcefile=" in out


def test_install_decode_chain_enters_calls_F(capsys, tmp_path, monkeypatch):
    import types

    log = tmp_path / "path_proof.log"
    monkeypatch.setenv("HEADSWAP_PATH_PROOF_LOG", str(log))

    class FakeVAEDecode:
        FUNCTION = "decode"

        def decode(self, vae, samples):
            return ("ok",)

    fake_nodes = types.ModuleType("nodes")
    fake_nodes.VAEDecode = FakeVAEDecode

    class FakeVAE:
        def decode(self, samples_in, vae_options=None):
            return samples_in

    fake_comfy = types.ModuleType("comfy")
    fake_sd = types.ModuleType("comfy.sd")
    fake_sd.VAE = FakeVAE
    fake_mm = types.ModuleType("comfy.model_management")

    def fake_lmg(*a, **k):
        return None

    fake_mm.load_models_gpu = fake_lmg

    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", fake_sd)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_mm)

    rt = MagicMock()
    rt.mappings = {"VAEDecode": FakeVAEDecode}

    with install_decode_chain_enters(runtime=rt):
        FakeVAEDecode().decode(vae=object(), samples={"samples": 1})

    out = capsys.readouterr().out
    assert "ENTER E0" in out
    assert "ENTER F" in out
    assert "ENTER E1" in out
