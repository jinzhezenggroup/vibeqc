"""Real native LDA/PBE RKS nuclear response for the DFT Hessian path."""

import typing

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.dft import NativeAO

from tools.vibeqc_hessian import directional_rks_response
from tools.vibeqc_response import GMRESOptions, NativeRKSResponse

H2 = [("H", (0.0, 0.0, -0.72)), ("H", (0.08, -0.03, 0.71))]
GRID = GridSpec(radial_points=10, angular_polar=4, angular_azimuth=8)


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


def _moved(atoms: typing.Any, direction: np.ndarray, scale: float) -> list:
    return [
        (symbol, np.asarray(position) + scale * delta)
        for (symbol, position), delta in zip(atoms, direction, strict=True)
    ]


@pytest.fixture(params=("lda-rks", "pbe-rks"), scope="module")
def case(request: typing.Any) -> typing.Iterator[typing.Any]:
    with (
        _calculator(request.param).prepare_batch([H2]) as batch,
        NativeAO(H2) as basis,
    ):
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as operator:
            direction = np.array(
                [[0.17, -0.09, 0.31], [-0.13, 0.07, -0.26]], dtype=np.float64
            )
            direction /= np.linalg.norm(direction)
            result = directional_rks_response(
                operator,
                direction,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )
            yield request.param, operator, direction, result


def test_real_rks_geometry_direction_solves_shared_cpks(case: typing.Any) -> None:
    _, operator, direction, result = case
    assert result.response.solve_result.converged
    assert result.response.solve_result.residual_norm < 1e-9
    np.testing.assert_array_equal(result.direction, direction)
    np.testing.assert_allclose(
        result.frozen_fock_derivative,
        result.integral_frozen_fock_derivative + result.xc_frozen_fock_derivative,
        atol=2e-14,
        rtol=0,
    )
    assert np.max(np.abs(result.xc_frozen_fock_derivative)) > 1e-8
    assert result.diagnostics["nuclear_response_solves"] == 1
    assert result.diagnostics["response_operator"] == "shared-native-rks-cpks"
    for value in (
        result.direction,
        result.integral_frozen_fock_derivative,
        result.xc_frozen_fock_derivative,
        result.frozen_fock_derivative,
        result.overlap_derivative,
        result.response.density_derivative,
        result.response.energy_weighted_density_derivative,
    ):
        assert not value.flags.writeable


def test_rks_nuclear_response_matches_reconverged_density_and_weighted_density(
    case: typing.Any,
) -> None:
    method, _, direction, result = case
    expected = (
        result.response.density_derivative,
        result.response.energy_weighted_density_derivative,
    )
    errors = []
    for step in (2e-3, 7e-4, 2e-4):
        displaced = []
        for sign in (1, -1):
            atoms = _moved(H2, direction, sign * step)
            with (
                _calculator(method).prepare_batch([atoms]) as batch,
                NativeAO(atoms) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    displaced.append(
                        (
                            np.array(current.state.density[0]),
                            np.array(current.state.weighted_density[0]),
                        )
                    )
        numeric = tuple(
            (plus - minus) / (2 * step)
            for plus, minus in zip(displaced[0], displaced[1], strict=True)
        )
        errors.append(
            [
                float(np.max(np.abs(actual - target)))
                for actual, target in zip(numeric, expected, strict=True)
            ]
        )
    errors = np.asarray(errors)
    assert np.all(errors[-1] < 4e-5), errors
    assert np.all(errors[-1] < np.maximum(0.3 * errors[0], 8e-7)), errors


def test_global_translation_has_zero_rks_nuclear_rhs(case: typing.Any) -> None:
    _, operator, _, _ = case
    direction = np.tile([0.13, -0.21, 0.31], (2, 1))
    result = directional_rks_response(
        operator,
        direction,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(result.overlap_derivative, 0, atol=3e-11, rtol=0)
    np.testing.assert_allclose(result.frozen_fock_derivative, 0, atol=3e-9, rtol=0)
    np.testing.assert_allclose(result.response.density_derivative, 0, atol=3e-8, rtol=0)
    np.testing.assert_allclose(
        result.response.energy_weighted_density_derivative, 0, atol=3e-8, rtol=0
    )
