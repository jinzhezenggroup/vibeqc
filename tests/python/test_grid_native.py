"""Independent Decimal/FD gates for the bounded native Becke adjoint."""

import ctypes as ct
import typing
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from test_grid_response import CENTERS, DC, POINTS, decimal_partition
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.dft.grid import partition_weights
from vibeqc_compiler.xc.grid_native import NativeGridContraction
from vibeqc_compiler.xc.grid_response import partition_response

OWNERS = np.array([0, 1, 2], dtype=np.int64)
SEEDS = np.array([0.3, -0.2, 0.7])


def native(tmp_path: typing.Any, **kwargs: typing.Any) -> typing.Any:
    return NativeGridContraction(
        compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path, **kwargs
    )


@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_independent_decimal_fd_translation_permutation_and_tiles(
    tmp_path: typing.Any, iterations: typing.Any
) -> None:
    executor = native(tmp_path, iterations=iterations)
    gradient = executor.contract(POINTS, CENTERS, OWNERS, SEEDS)
    derivative = np.sum(gradient * DC)
    with localcontext() as context:
        context.prec = 60
        h = Decimal("1e-16")

        def moved(
            values: typing.Any, motion: typing.Any, sign: typing.Any
        ) -> typing.Any:
            return [
                [Decimal(str(x)) + sign * h * Decimal(str(dx)) for x, dx in zip(row, d)]
                for row, d in zip(values, motion)
            ]

        plus, minus = [
            decimal_partition(
                moved(POINTS, DC[OWNERS], s), moved(CENTERS, DC, s), iterations
            )
            for s in (1, -1)
        ]
        expected = sum(
            Decimal(str(q)) * (p[o] - m[o]) / (2 * h)
            for q, p, m, o in zip(SEEDS, plus, minus, OWNERS)
        )
        assert abs(derivative - float(expected)) < 2e-14
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        values = [
            SEEDS
            @ partition_weights(
                POINTS + s * h * DC[OWNERS], CENTERS + s * h * DC, iterations=iterations
            )[np.arange(3), OWNERS]
            for s in (1, -1)
        ]
        errors.append(abs((values[0] - values[1]) / (2 * h) - derivative))
    assert errors[-1] < 3e-10 and errors[-1] < errors[0] / 100
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-15)
    shift = np.array([0.3, -0.7, 0.2])
    np.testing.assert_allclose(
        executor.contract(POINTS + shift, CENTERS + shift, OWNERS, SEEDS),
        gradient,
        atol=2e-15,
    )
    order = [2, 0, 1]
    np.testing.assert_allclose(
        executor.contract(POINTS, CENTERS[order], np.argsort(order)[OWNERS], SEEDS),
        gradient[order],
        atol=2e-15,
    )
    pieces = [
        executor.contract(POINTS[a:b], CENTERS, OWNERS[a:b], SEEDS[a:b])
        for a, b in ((0, 2), (2, 3), (3, 3))
    ]
    np.testing.assert_allclose(sum(pieces), gradient, atol=2e-15)
    np.testing.assert_array_equal(pieces[-1], 0)
    reference = partition_response(
        POINTS,
        CENTERS,
        point_motion=DC[OWNERS],
        center_motion=DC,
        iterations=iterations,
    )
    assert derivative == pytest.approx(
        SEEDS @ reference.directional[np.arange(3), OWNERS], abs=2e-14
    )
    np.testing.assert_array_equal(
        executor.contract(POINTS, CENTERS[:1], np.zeros(3, dtype=np.int64), SEEDS), 0
    )


@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_saturated_exact_zero_and_single_zero_factor(
    tmp_path: typing.Any, iterations: typing.Any
) -> None:
    executor = native(tmp_path, iterations=iterations)
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    owners = np.array([0, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        executor.contract(points, centers, owners, np.ones(2)), 0
    )
    # The middle product has exactly one zero; endpoint clipping must not
    # evaluate log(0) or silently contaminate its normalized adjoint.
    np.testing.assert_array_equal(
        executor.contract(points[:1], centers, np.array([1]), np.ones(1)), 0
    )


def test_native_bounds_invalid_inputs_and_transactional_late_failure(
    tmp_path: typing.Any,
) -> None:
    executor = native(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        native(tmp_path, max_bytes=1).contract(POINTS, CENTERS, OWNERS, SEEDS)
    with pytest.raises(ValueError, match="work budget"):
        native(tmp_path, max_pair_visits=1).contract(POINTS, CENTERS, OWNERS, SEEDS)
    with pytest.raises(TypeError, match="CPU compiler"):
        NativeGridContraction(compiler=object(), cache=tmp_path)
    for owners, seeds in (
        (OWNERS.astype(np.int32), SEEDS),
        (OWNERS, SEEDS.astype(np.float32)),
        (OWNERS, SEEDS[:2]),
    ):
        with pytest.raises(ValueError):
            executor.contract(POINTS, CENTERS, owners, seeds)
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    output = np.full((3, 3), 42.0)
    points = POINTS.copy()
    points[-1] = CENTERS[0]  # Fail after two completed point contractions.
    for npnt, na, budget in (
        (3, 3, executor.max_bytes),
        (2**64 - 1, 3, executor.max_bytes),
        (3, 2**64 - 1, executor.max_bytes),
        (3, 3, 0),
    ):
        assert (
            executor.call(
                ptr(points),
                npnt,
                ptr(CENTERS),
                na,
                OWNERS.ctypes.data_as(ct.POINTER(ct.c_int64)),
                ptr(SEEDS),
                ptr(output),
                output.size,
                budget,
                executor.max_pair_visits,
                1e-12,
            )
            != 0
        )
        np.testing.assert_array_equal(output, 42.0)
    for centers, points, owners, seeds, tolerance in (
        (CENTERS, POINTS, np.array([-1, 1, 2]), SEEDS, 1e-12),
        (CENTERS, POINTS, OWNERS, np.array([1.0, np.nan, 2.0]), 1e-12),
        (CENTERS, POINTS, OWNERS, SEEDS, 10.0),
        (CENTERS, POINTS, OWNERS, SEEDS, np.nan),
        (np.zeros_like(CENTERS), POINTS, OWNERS, SEEDS, 1e-12),
    ):
        with pytest.raises(ValueError):
            executor.contract(
                points, centers, owners, seeds, coincident_tolerance=tolerance
            )
    assert np.isfinite(executor.contract(POINTS, CENTERS, OWNERS, SEEDS)).all()


def test_one_rounded_zero_factor_keeps_nonzero_product_derivative(
    tmp_path: typing.Any,
) -> None:
    """A single zero factor can have a nonzero tangent before saturation.

    The final pair value rounds to zero at iteration one, but its derivative
    survives. A large finite seed makes an incorrect zero-product shortcut
    visible. Decimal differentiates the independent direct-product objective.
    """
    points = np.array([[-1.0, 2e-5, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    owners = np.array([1], dtype=np.int64)
    seed = np.array([1e12])
    motion = DC[:2]
    assert partition_weights(points, centers, iterations=1)[0, 1] == 0
    gradient = native(tmp_path, iterations=1).contract(points, centers, owners, seed)
    with localcontext() as context:
        context.prec = 70
        h = Decimal("1e-20")

        def moved(
            values: typing.Any, directions: typing.Any, sign: typing.Any
        ) -> typing.Any:
            return [
                [Decimal(str(x)) + sign * h * Decimal(str(dx)) for x, dx in zip(row, d)]
                for row, d in zip(values, directions)
            ]

        plus, minus = [
            decimal_partition(
                moved(points, motion[owners], s), moved(centers, motion, s), 1
            )
            for s in (1, -1)
        ]
        expected = float(Decimal("1e12") * (plus[0][1] - minus[0][1]) / (2 * h))
    assert abs(expected) > 1e-3
    assert abs(np.sum(gradient * motion) - expected) < 2e-9
