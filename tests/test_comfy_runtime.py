"""Unit tests for ComfyUI V1/V3 node invocation helpers (no ComfyUI install required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headswap.comfy.runtime import NodeRuntime, get_value_at_index, invoke_node


class _V1ZeroOut:
    """Mirrors nodes.ConditioningZeroOut (FUNCTION-based V1 API)."""

    FUNCTION = "zero_out"

    def zero_out(self, conditioning):
        return (f"zeroed:{conditioning}",)


class _V3NodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args if self.args else None


class _V3ReferenceLatent:
    """Mirrors io.ComfyNode (FUNCTION → EXECUTE_NORMALIZED)."""

    @classmethod
    def VALIDATE_CLASS(cls):
        return None

    @classmethod
    def EXECUTE_NORMALIZED(cls, **kwargs):
        return _V3NodeOutput(f"ref:{kwargs['conditioning']}:{kwargs.get('latent')}")

    # classproperty simulation: execution.py reads cls.FUNCTION as this string
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def execute(cls, conditioning, latent=None):
        raise AssertionError("call path should use EXECUTE_NORMALIZED, not raw execute")


def test_invoke_v1_uses_function_method_not_execute():
    out = invoke_node(_V1ZeroOut, conditioning="pos")
    assert out == ("zeroed:pos",)
    assert get_value_at_index(out, 0) == "zeroed:pos"


def test_invoke_v1_has_no_execute_attribute():
    assert not hasattr(_V1ZeroOut, "execute")
    with pytest.raises(AttributeError):
        # Belongs to the old broken call style that triggered in Colab:
        # AttributeError: type object 'ConditioningZeroOut' has no attribute 'execute'
        _V1ZeroOut.execute(conditioning="pos")  # type: ignore[attr-defined]


def test_invoke_v3_uses_execute_normalized():
    out = invoke_node(_V3ReferenceLatent, conditioning="c", latent="L")
    assert get_value_at_index(out, 0) == "ref:c:L"


def test_runtime_call_dispatches_both():
    rt = NodeRuntime(
        mappings={
            "ConditioningZeroOut": _V1ZeroOut,
            "ReferenceLatent": _V3ReferenceLatent,
        }
    )
    assert get_value_at_index(rt.call("ConditioningZeroOut", conditioning="x"), 0) == "zeroed:x"
    assert get_value_at_index(rt.call("ReferenceLatent", conditioning="a", latent="b"), 0) == "ref:a:b"


def test_get_value_at_index_node_output_and_tuple():
    assert get_value_at_index(("a", "b"), 1) == "b"
    assert get_value_at_index(_V3NodeOutput("x", "y"), 0) == "x"
    assert get_value_at_index({"result": ("r0", "r1")}, 1) == "r1"
    assert get_value_at_index(SimpleNamespace(args=("p",), result=("p",)), 0) == "p"
