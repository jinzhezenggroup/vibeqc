"""Native bounded fixed-grid VV10/rVV10 execution gates for #491D."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc.nonlocal_runtime import NonlocalFixedGridPlan
from vibeqc_compiler.dft import (
    nonlocal_energy_reference,
    nonlocal_explicit_geometry_derivatives_reference,
    nonlocal_feature_derivatives_reference,
)
from vibeqc_compiler.method import original_nonlocal_correlation


@pytest.fixture
def fixed_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.2, -0.1],
            [-0.3, 0.8, 0.5],
            [1.1, -0.4, 0.9],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.3, 0.5, 0.4, 0.2], dtype=np.float64)
    density = np.array([0.42, 0.31, 0.18, 0.27], dtype=np.float64)
    gradient = np.array(
        [
            [0.05, -0.02, 0.01],
            [-0.03, 0.04, 0.02],
            [0.01, 0.02, -0.04],
            [-0.02, -0.01, 0.03],
        ],
        dtype=np.float64,
    )
    return coordinates, weights, density, gradient


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_native_pair_plan_matches_independent_reference_for_all_outputs(
    fixed_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], variant: str
) -> None:
    coordinates, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation(variant)
    coefficient = Fraction(7, 10)
    expected_energy = float(coefficient) * nonlocal_energy_reference(
        coordinates, weights, density, gradient, spec, tile_size=2
    )
    expected_vrho, expected_vsigma = nonlocal_feature_derivatives_reference(
        coordinates, weights, density, gradient, spec, tile_size=2
    )
    expected_point, expected_weight = nonlocal_explicit_geometry_derivatives_reference(
        coordinates, weights, density, gradient, spec, tile_size=2
    )

    with NonlocalFixedGridPlan(
        spec,
        len(density),
        coefficient=coefficient,
        tile_points=2,
        maximum_bytes=4096,
    ) as plan:
        result = plan.execute(
            coordinates, weights, density, gradient, features=True, geometry=True
        )
        diagnostic = plan.diagnostic()

    assert result.backend == diagnostic.backend == "cpu"
    assert result.energy == pytest.approx(expected_energy, rel=2e-14, abs=1e-16)
    np.testing.assert_allclose(
        result.vrho, float(coefficient) * expected_vrho, rtol=2e-14, atol=2e-16
    )
    np.testing.assert_allclose(
        result.vsigma, float(coefficient) * expected_vsigma, rtol=3e-14, atol=2e-16
    )
    np.testing.assert_allclose(
        result.point_derivative,
        float(coefficient) * expected_point,
        rtol=3e-14,
        atol=2e-16,
    )
    np.testing.assert_allclose(
        result.weight_derivative,
        float(coefficient) * expected_weight,
        rtol=2e-14,
        atol=2e-16,
    )
    assert diagnostic.workspace_bytes == 6 * len(density) * 8
    assert diagnostic.maximum_bytes == 4096
    assert diagnostic.pair_evaluations == len(density) ** 2
    assert diagnostic.point_count == len(density)
    assert diagnostic.tile_points == 2


def test_native_pair_plan_schedule_does_not_change_science(
    fixed_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coordinates, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("vv10")
    values = []
    for tile in (1, 2, 3, 16):
        with NonlocalFixedGridPlan(spec, len(density), tile_points=tile) as plan:
            result = plan.execute(
                coordinates, weights, density, gradient, features=True, geometry=True
            )
            values.append(result)
            assert plan.diagnostic().tile_points == min(tile, len(density))
    for candidate in values[1:]:
        assert candidate.energy == pytest.approx(values[0].energy, rel=2e-15, abs=1e-17)
        for name in ("vrho", "vsigma", "point_derivative", "weight_derivative"):
            np.testing.assert_allclose(
                getattr(candidate, name),
                getattr(values[0], name),
                rtol=2e-15,
                atol=2e-17,
            )


def test_native_pair_plan_rejects_budget_before_scientific_execution() -> None:
    spec = original_nonlocal_correlation("rvv10")
    with pytest.raises(RuntimeError, match="workspace exceeds maximum_bytes"):
        NonlocalFixedGridPlan(spec, 100, maximum_bytes=6 * 100 * 8 - 1)


def test_native_pair_plan_keeps_cuda_fail_closed() -> None:
    spec = original_nonlocal_correlation("vv10")
    with pytest.raises(NotImplementedError, match="CUDA lowerer"):
        NonlocalFixedGridPlan(spec, 4, device="cuda")
