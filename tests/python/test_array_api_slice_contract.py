"""Subscript syntax must preserve static integer slice admission."""

import pytest
from vibeqc_compiler.array_api import trace
from vibeqc_compiler.tensor import Index, IndexSpace, TensorSpec


@pytest.mark.parametrize("step", [True, False, 1.0, 2.0])
def test_symbolic_subscript_rejects_non_integer_steps(step: object) -> None:
    spec = TensorSpec((Index("p", IndexSpace("ao", "ao", 4)),), role="input")
    with pytest.raises(ValueError, match="unit step"):
        trace(lambda x: x[slice(None, None, step)], {"x": spec})
