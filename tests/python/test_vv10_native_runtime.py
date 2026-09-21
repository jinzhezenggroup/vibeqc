"""Native bounded fixed-grid VV10/rVV10 execution gates for #491D."""

import ctypes
from fractions import Fraction

import numpy as np
import pytest
from vibeqc import _native
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
    assert diagnostic.host_workspace_bytes == 12 * len(density) * 8
    assert diagnostic.device_workspace_bytes == 0
    assert diagnostic.workspace_bytes == diagnostic.host_workspace_bytes
    assert diagnostic.maximum_bytes == 4096
    assert diagnostic.pair_evaluations == len(density) ** 2
    assert diagnostic.tiles == 2
    assert diagnostic.point_count == len(density)
    assert diagnostic.tile_points == 2
    assert result.identity
    assert result.plan_identity == plan.identity


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
    assert len({candidate.plan_identity for candidate in values}) == len(values)


def test_native_pair_plan_rejects_budget_before_scientific_execution() -> None:
    spec = original_nonlocal_correlation("rvv10")
    with pytest.raises(RuntimeError, match="workspace exceeds maximum_bytes"):
        NonlocalFixedGridPlan(spec, 100, maximum_bytes=12 * 100 * 8 - 1)


def test_native_pair_plan_cuda_is_native_or_fails_closed() -> None:
    spec = original_nonlocal_correlation("vv10")
    try:
        plan = NonlocalFixedGridPlan(spec, 4, device="cuda")
    except (NotImplementedError, RuntimeError):
        return
    with plan:
        assert plan.diagnostic().backend == "cuda"


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_native_cuda_matches_cpu_on_real_device_when_available(
    fixed_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], variant: str
) -> None:
    coordinates, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation(variant)
    coefficient = Fraction(7, 10)
    with NonlocalFixedGridPlan(
        spec, len(density), coefficient=coefficient, tile_points=3
    ) as cpu_plan:
        cpu = cpu_plan.execute(
            coordinates, weights, density, gradient, features=True, geometry=True
        )
    try:
        cuda_plan = NonlocalFixedGridPlan(
            spec,
            len(density),
            coefficient=coefficient,
            tile_points=2,
            device="cuda",
            device_id=0,
        )
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"CUDA nonlocal provider unavailable: {error}")
    with cuda_plan:
        cuda = cuda_plan.execute(
            coordinates, weights, density, gradient, features=True, geometry=True
        )
        diagnostic = cuda_plan.diagnostic()

    assert cuda.backend == diagnostic.backend == "cuda"
    assert cuda.energy == pytest.approx(cpu.energy, rel=3e-13, abs=3e-16)
    np.testing.assert_allclose(cuda.vrho, cpu.vrho, rtol=5e-13, atol=5e-15)
    np.testing.assert_allclose(cuda.vsigma, cpu.vsigma, rtol=5e-13, atol=5e-15)
    np.testing.assert_allclose(
        cuda.point_derivative, cpu.point_derivative, rtol=5e-13, atol=5e-15
    )
    np.testing.assert_allclose(
        cuda.weight_derivative, cpu.weight_derivative, rtol=5e-13, atol=5e-15
    )
    assert diagnostic.host_workspace_bytes == 7 * len(density) * 8
    assert diagnostic.device_workspace_bytes == (21 * len(density) + 1) * 8
    assert diagnostic.workspace_bytes == (
        diagnostic.host_workspace_bytes + diagnostic.device_workspace_bytes
    )
    assert diagnostic.pair_evaluations == len(density) ** 2
    assert diagnostic.tiles == 2


def test_native_failure_does_not_partially_publish_caller_buffers(
    fixed_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coordinates, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("vv10")
    bad_density = density.copy()
    bad_density[1] = 0.0
    sentinel = 9182.5
    with NonlocalFixedGridPlan(spec, len(density), maximum_bytes=4096) as plan:
        library = plan._library

        def pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
            return value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        vrho = np.full(len(density), sentinel, dtype=np.float64)
        vsigma = np.full(len(density), sentinel, dtype=np.float64)
        point = np.full((len(density), 3), sentinel, dtype=np.float64)
        weight = np.full(len(density), sentinel, dtype=np.float64)
        inputs = _native.NonlocalInputDescriptor(
            ctypes.sizeof(_native.NonlocalInputDescriptor),
            _native.ABI_VERSION,
            pointer(coordinates),
            coordinates.size,
            pointer(weights),
            weights.size,
            pointer(bad_density),
            bad_density.size,
            pointer(gradient),
            gradient.size,
        )
        output = _native.NonlocalResultDescriptor(
            ctypes.sizeof(_native.NonlocalResultDescriptor),
            _native.ABI_VERSION,
            sentinel,
            pointer(vrho),
            vrho.size,
            pointer(vsigma),
            vsigma.size,
            pointer(point),
            point.size,
            pointer(weight),
            weight.size,
            -17,
        )
        status = library.vibeqc_nonlocal_plan_execute(
            plan._plan, ctypes.byref(inputs), ctypes.byref(output)
        )

    assert status != _native.STATUS_SUCCESS
    assert output.energy == sentinel
    assert output.executed_backend == -17
    for value in (vrho, vsigma, point, weight):
        np.testing.assert_array_equal(value, sentinel)
