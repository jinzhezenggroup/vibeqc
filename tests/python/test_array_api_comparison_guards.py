"""Symbolic equality must not choose Python branches during capture."""

import operator
import typing

import pytest
from vibeqc_compiler.array_api import VibeArray, input_array, trace
from vibeqc_compiler.tensor import Index, IndexSpace, TensorSpec


def _spec() -> TensorSpec:
    return TensorSpec((Index("i", IndexSpace("ao", "ao", 2)),), role="input")


@pytest.mark.parametrize("comparison", [operator.eq, operator.ne])
@pytest.mark.parametrize(
    "case", ["different", "same", "rewrapped", "scalar", "reflected", "none"]
)
def test_unsupported_symbolic_comparisons_raise(
    comparison: typing.Callable[[object, object], object], case: str
) -> None:
    x = input_array("x", _spec())
    y = input_array("y", _spec())
    operands = {
        "different": (x, y),
        "same": (x, x),
        "rewrapped": (x, VibeArray(x.node)),
        "scalar": (x, 0),
        "reflected": (0, x),
        "none": (x, None),
    }
    with pytest.raises(TypeError, match="symbolic VibeArray comparisons"):
        comparison(*operands[case])


@pytest.mark.parametrize("comparison", [operator.eq, operator.ne])
@pytest.mark.parametrize("scalar", [False, True])
def test_trace_rejects_comparison_driven_control_flow(
    comparison: typing.Callable[[object, object], object], scalar: bool
) -> None:
    def expression(x: VibeArray, y: VibeArray) -> VibeArray:
        return x if comparison(x, 0 if scalar else y) else y

    with pytest.raises(TypeError, match="symbolic VibeArray comparisons"):
        trace(expression, {"x": _spec(), "y": _spec()})


def test_explicit_node_identity_remains_available() -> None:
    x = input_array("x", _spec())
    wrapped = VibeArray(x.node)
    assert x.node is wrapped.node
    assert x.shape == (2,)
    assert x.dtype == "float64"
