"""Source inventories must preserve the caller's declared reduction order."""

import typing
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.second_order import (
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
)


def _executor(sources: typing.Any) -> StationarySecondOrderExecutor:
    return StationarySecondOrderExecutor(
        SimpleNamespace(identity="plan-v1", source_names=sources),
        natoms=1,
        perturbation=StationaryPerturbationProvider("rhs-v1", lambda value: value),
        response=StationaryResponseDriver("response-v1", lambda value: value),
        contributors=(
            StationaryHVPContributor("a", "a-v1", lambda context: context.direction),
            StationaryHVPContributor("b", "b-v1", lambda context: context.direction),
        ),
    )


@pytest.mark.parametrize("sources", ["ab", {"a", "b"}, frozenset(("a", "b"))])
def test_executor_rejects_text_and_unordered_source_inventories(
    sources: typing.Any,
) -> None:
    with pytest.raises(TypeError, match="ordered source_names"):
        _executor(sources)


@pytest.mark.parametrize("sources", [("b", "a"), ["b", "a"]])
def test_executor_preserves_explicit_source_order(sources: typing.Any) -> None:
    result = _executor(sources).apply([[1.0, -2.0, 3.0]])
    assert tuple(result.components) == ("b", "a")
    np.testing.assert_array_equal(result.value, [[2.0, -4.0, 6.0]])
