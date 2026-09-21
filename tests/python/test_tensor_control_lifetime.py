"""Discrete input controls cannot bypass an execution owner's closed state."""

from types import SimpleNamespace

import pytest
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda


@pytest.mark.parametrize("dtype", ("int64", "float32", "float64"))
def test_closed_owner_rejects_before_inspecting_control_data(dtype: str) -> None:
    prepared = object.__new__(PreparedCuda)
    prepared._mask = None
    node = SimpleNamespace(spec=SimpleNamespace(dtype=dtype))
    with pytest.raises(RuntimeError, match="validation scratch is closed"):
        prepared._validate(None, node)
