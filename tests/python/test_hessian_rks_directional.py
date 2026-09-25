"""Real native LDA/PBE RKS nuclear response for the DFT Hessian path."""

import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import _native_ao_atoms
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.grid_response import partition_response

from tools.vibeqc_hessian import (
    directional_rks_response,
    native_rks_xc_hvp_components,
)
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


def _native_xc_gradient(operator: NativeRKSResponse) -> dict[str, np.ndarray]:
    """Re-evaluate only the native semilocal XC first-gradient sources."""
    operator.validate_current()
    state = operator.state
    source = state._source
    basis = operator.xc_kernel.basis
    spec = operator.xc_kernel.spec
    grid = state.grid
    points = np.asarray(grid.points)
    owners = np.asarray(grid.owners, dtype=np.int64)
    contraction = ExternalPointContraction(spec, "geometry")
    jets = basis.evaluate(points, contraction.contract.ao_order)
    features = contraction.features(jets, state.density[0])
    zero_gradient = np.zeros((2, len(points), 3))
    values = source.evaluate_xc_points(
        spec,
        features["rho"],
        features.get("gradient", zero_gradient),
    )
    partials = contraction.geometry_from_cartesian_coefficients(
        jets,
        state.density[0],
        grid.weights,
        values["energy"],
        values["rho"],
        values["gradient"] if "sigma" in spec.ingredients else None,
        ao_atoms=_native_ao_atoms(basis),
        natom=basis.natom,
    )
    result = {
        "xc_ao": np.array(partials.centers),
        "xc_grid": np.zeros((basis.natom, 3)),
        "xc_weight": np.zeros((basis.natom, 3)),
    }
    np.add.at(result["xc_grid"], owners, partials.points)

    centers = np.asarray([atom.position for atom in basis.atoms], dtype=np.float64)
    atomic_weights = np.asarray(source.atomic_weights)
    grid_spec = source.grid_spec
    assert grid_spec is not None
    selected = (np.arange(len(points)), owners)
    for atom in range(basis.natom):
        for axis in range(3):
            motion = np.zeros((basis.natom, 3))
            motion[atom, axis] = 1.0
            response = partition_response(
                points,
                centers,
                point_motion=motion[owners],
                center_motion=motion,
                iterations=grid_spec.partition_iterations,
                coincident_tolerance=grid_spec.coincident_tolerance,
            )
            result["xc_weight"][atom, axis] = np.dot(
                partials.weights,
                atomic_weights * response.directional[selected],
            )
    return result


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
    _, _operator, direction, result = case
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


def test_native_rks_xc_hvp_matches_reconverged_xc_gradient(
    case: typing.Any,
) -> None:
    method, operator, direction, directional = case
    actual = native_rks_xc_hvp_components(operator, directional)
    assert actual.diagnostics["additional_response_solves"] == 0
    assert actual.diagnostics["source_names"] == ("xc_ao", "xc_grid", "xc_weight")
    np.testing.assert_allclose(
        actual.total,
        actual.xc_ao + actual.xc_grid + actual.xc_weight,
        atol=0,
        rtol=0,
    )
    assert np.max(np.abs(actual.total)) > 1e-7
    for value in (actual.xc_ao, actual.xc_grid, actual.xc_weight, actual.total):
        assert not value.flags.writeable

    errors = []
    for step in (1.5e-3, 5e-4, 1.7e-4):
        displaced = []
        for sign in (1, -1):
            atoms = _moved(H2, direction, sign * step)
            with (
                _calculator(method).prepare_batch([atoms]) as batch,
                NativeAO(atoms) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    displaced.append(_native_xc_gradient(current))
        numeric = {
            name: (displaced[0][name] - displaced[1][name]) / (2 * step)
            for name in ("xc_ao", "xc_grid", "xc_weight")
        }
        numeric["total"] = sum(numeric.values())
        errors.append(
            [
                float(np.max(np.abs(numeric["xc_ao"] - actual.xc_ao))),
                float(np.max(np.abs(numeric["xc_grid"] - actual.xc_grid))),
                float(np.max(np.abs(numeric["xc_weight"] - actual.xc_weight))),
                float(np.max(np.abs(numeric["total"] - actual.total))),
            ]
        )
    errors = np.asarray(errors)
    assert np.all(errors[-1] < 3e-4), errors
    assert np.all(errors[-1] < np.maximum(0.45 * errors[0], 2e-5)), errors

    with pytest.raises(ValueError, match="does not belong"):
        native_rks_xc_hvp_components(
            operator,
            replace(directional, identity="not-the-current-response"),
        )


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
