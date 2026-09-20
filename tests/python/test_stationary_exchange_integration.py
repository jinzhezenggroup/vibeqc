"""Range and global exchange retain distinct stationary provider identities."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import execute


@pytest.mark.parametrize("spin,blocks", [("unpolarized", 1), ("polarized", 2)])
@pytest.mark.parametrize("range_index", [1, 2])
def test_single_range_primitive_is_not_a_global_hybrid(
    spin: str, blocks: int, range_index: int
) -> None:
    method = resolve_method("CAM-B3LYP", spin=spin)
    primitive = method.primitives[range_index]
    method = replace(
        method,
        identifier="single-range-test",
        primitives=(method.primitives[0], primitive),
    )
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    assert plan.exchange is None
    assert "exact_exchange" not in plan.source_names
    source = plan.range_exchange_sources[0].name
    left = np.arange(1, 1 + 3 * blocks, dtype=np.float64).reshape(blocks, 3)
    right = left[:, ::-1].copy()
    block = plan.integral_block(source, terms=3)
    result = execute(
        block.weights, {"density_left": left, "density_right": right}
    ).outputs["weights"]
    factor = -float(primitive.coefficient) * (0.25 if blocks == 1 else 0.5)
    expected = np.array(
        [
            factor * sum(float(left[s, t]) * float(right[s, t]) for s in range(blocks))
            for t in range(3)
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15)
    with pytest.raises(ValueError, match="integral-gradient"):
        plan.integral_block("exact_exchange", terms=3)
