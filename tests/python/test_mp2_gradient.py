"""Analytic MP2 energy-adjoint contracts for the complete-gradient chain."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator
from vibeqc_compiler.tensor import execute

from tools.vibeqc_mp2.equations import energy_program
from tools.vibeqc_mp2.gradient import (
    canonical_energy_adjoint,
    canonical_lagrangian_weights,
    canonical_orbital_rhs,
    dense_molecular_gradient_oracle,
    solve_canonical_orbital_response,
    tile_energy_adjoint,
)
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    GMRESOptions,
    ResponseSolveError,
)


def test_dense_derivative_oracle_rejects_output_budget_before_allocation(monkeypatch):
    meta, _ = load_fixture("h2")
    arguments = source_arguments(meta)
    with NativeSource(**arguments) as source:
        monkeypatch.setattr(
            np,
            "empty",
            lambda *args, **kwargs: pytest.fail("output allocated before budget check"),
        )
        with pytest.raises(ValueError, match="output exceeds"):
            source.integral_derivatives(output_budget_bytes=1)


def _energy(feeds):
    values = execute(energy_program(feeds["g"].shape), feeds).outputs
    return float(values["opposite_spin"] + values["same_spin"])


def test_tile_energy_adjoint_matches_closed_form_and_dot_identity():
    rng = np.random.default_rng(193)
    shape = (2, 3, 2, 4)
    feeds = {
        "g": rng.normal(scale=0.2, size=shape),
        "x": rng.normal(scale=0.2, size=shape),
        "ei": np.array([-0.9, -0.7]),
        "ej": np.array([-1.0, -0.8, -0.6]),
        "ea": np.array([0.2, 0.4]),
        "eb": np.array([0.1, 0.3, 0.5, 0.7]),
    }
    adjoint = tile_energy_adjoint(feeds)
    denominator = (
        feeds["ei"][:, None, None, None]
        + feeds["ej"][None, :, None, None]
        - feeds["ea"][None, None, :, None]
        - feeds["eb"][None, None, None, :]
    )
    numerator = 2 * feeds["g"] ** 2 - feeds["g"] * feeds["x"]
    np.testing.assert_allclose(
        adjoint.direct, (4 * feeds["g"] - feeds["x"]) / denominator
    )
    np.testing.assert_allclose(adjoint.exchange, -feeds["g"] / denominator)
    bar_denominator = -numerator / denominator**2
    np.testing.assert_allclose(adjoint.occupied_i, bar_denominator.sum(axis=(1, 2, 3)))
    np.testing.assert_allclose(adjoint.occupied_j, bar_denominator.sum(axis=(0, 2, 3)))
    np.testing.assert_allclose(adjoint.virtual_a, -bar_denominator.sum(axis=(0, 1, 3)))
    np.testing.assert_allclose(adjoint.virtual_b, -bar_denominator.sum(axis=(0, 1, 2)))

    directions = {name: rng.normal(size=value.shape) for name, value in feeds.items()}
    reverse_dot = sum(
        np.vdot(getattr(adjoint, attribute), directions[name])
        for name, attribute in (
            ("g", "direct"),
            ("x", "exchange"),
            ("ei", "occupied_i"),
            ("ej", "occupied_j"),
            ("ea", "virtual_a"),
            ("eb", "virtual_b"),
        )
    )
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        plus = {name: value + step * directions[name] for name, value in feeds.items()}
        minus = {name: value - step * directions[name] for name, value in feeds.items()}
        finite_difference = (_energy(plus) - _energy(minus)) / (2 * step)
        errors.append(abs(finite_difference - reverse_dot))
    assert errors[-1] < 1e-8
    assert errors[-1] < errors[0]


def test_energy_program_default_remains_nondifferentiable():
    shape = (1, 1, 2, 2)
    primal = energy_program(shape)
    differentiable = energy_program(shape, differentiable=True)
    assert primal.logical_hash != differentiable.logical_hash
    inputs = [node for node in primal.nodes if node.op == "input"]
    assert inputs and all(not node.spec.differentiable for node in inputs)


def test_canonical_adjoint_accumulates_exchange_and_repeated_energy_feeds():
    rng = np.random.default_rng(194)
    no, nv = 2, 3
    g = rng.normal(scale=0.1, size=(no, no, nv, nv))
    eps = np.array([-0.9, -0.6, 0.1, 0.3, 0.8])
    adjoint = canonical_energy_adjoint(
        g,
        eps,
        no,
        reference_identity="synthetic-canonical-194",
        hamiltonian_id="conventional-unscreened",
    )
    dg = rng.normal(size=g.shape)
    de = rng.normal(size=eps.shape)
    reverse_dot = np.vdot(adjoint.integrals_iajb, dg) + np.vdot(
        adjoint.orbital_energies, de
    )

    def total(values, energies):
        return _energy(
            {
                "g": values,
                "x": values.swapaxes(2, 3),
                "ei": energies[:no],
                "ej": energies[:no],
                "ea": energies[no:],
                "eb": energies[no:],
            }
        )

    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        finite_difference = (
            total(g + step * dg, eps + step * de)
            - total(g - step * dg, eps - step * de)
        ) / (2 * step)
        errors.append(abs(finite_difference - reverse_dot))
    assert errors[-1] < 1e-8
    assert errors[-1] < errors[0]


def _symmetric_eri(rng, size):
    eri = rng.normal(scale=0.08, size=(size,) * 4)
    eri = 0.5 * (eri + eri.swapaxes(0, 1))
    eri = 0.5 * (eri + eri.swapaxes(2, 3))
    return 0.5 * (eri + eri.transpose(2, 3, 0, 1))


def _transform_eri(eri, rotation):
    return np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        eri,
        rotation,
        rotation,
        rotation,
        rotation,
        optimize=True,
    )


def _fock(hcore, eri, occupied):
    result = hcore.copy()
    for i in range(occupied):
        result += 2 * eri[:, :, i, i] - eri[:, i, i, :]
    return result


def test_canonical_orbital_rhs_matches_rebuilt_fock_rotation():
    rng = np.random.default_rng(195)
    occupied, size = 2, 5
    eri = _symmetric_eri(rng, size)
    target_energies = np.array([-1.0, -0.7, 0.2, 0.5, 0.9])
    hcore = np.diag(target_energies) - _fock(np.zeros((size, size)), eri, occupied)
    np.testing.assert_allclose(
        _fock(hcore, eri, occupied), np.diag(target_energies), atol=1e-15
    )
    g = eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3)
    adjoint = canonical_energy_adjoint(
        g,
        target_energies,
        occupied,
        reference_identity="synthetic-canonical-195",
        hamiltonian_id="conventional-unscreened",
    )
    orbital = canonical_orbital_rhs(hcore, eri, adjoint, occupied)
    direction = rng.normal(size=(occupied, size - occupied))
    reverse_dot = np.vdot(orbital.energy_gradient, direction)
    generator = np.zeros((size, size))
    generator[:occupied, occupied:] = direction
    generator[occupied:, :occupied] = -direction.T

    def rotated_energy(step):
        identity = np.eye(size)
        rotation = np.linalg.solve(
            identity - 0.5 * step * generator,
            identity + 0.5 * step * generator,
        )
        h = rotation.T @ hcore @ rotation
        transformed = _transform_eri(eri, rotation)
        energies = np.diag(_fock(h, transformed, occupied))
        values = transformed[:occupied, occupied:, :occupied, occupied:].transpose(
            0, 2, 1, 3
        )
        return _energy(
            {
                "g": values,
                "x": values.swapaxes(2, 3),
                "ei": energies[:occupied],
                "ej": energies[:occupied],
                "ea": energies[occupied:],
                "eb": energies[occupied:],
            }
        )

    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        finite_difference = (rotated_energy(step) - rotated_energy(-step)) / (2 * step)
        errors.append(abs(finite_difference - reverse_dot))
    assert errors[-1] < 1e-8
    assert errors[-1] < errors[0]


def test_orbital_rhs_rejects_nonfinite_hamiltonian_data():
    energies = np.array([-0.8, 0.3])
    eri = np.zeros((2, 2, 2, 2))
    adjoint = canonical_energy_adjoint(
        np.ones((1, 1, 1, 1)),
        energies,
        1,
        reference_identity="synthetic-nonfinite",
        hamiltonian_id="conventional-unscreened",
    )
    hcore = np.diag(energies)
    hcore[0, 1] = np.nan
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        canonical_orbital_rhs(hcore, eri, adjoint, 1)


def _explicit_rhf_matrix(energies, eri, occupied):
    rotations = [
        (i, a) for i in range(occupied) for a in range(occupied, len(energies))
    ]
    matrix = np.empty((len(rotations), len(rotations)))
    for row, (i, a) in enumerate(rotations):
        for column, (j, b) in enumerate(rotations):
            matrix[row, column] = (
                (energies[a] - energies[i]) * (i == j and a == b)
                + 4 * eri[a, i, b, j]
                - eri[a, b, i, j]
                - eri[a, j, i, b]
            )
    return matrix


def test_mp2_orbital_response_reuses_shared_solver_and_explicit_matrix():
    meta, arrays = load_fixture("water")
    reference = fixture_snapshot(meta, arrays)
    occupied = reference.nocc
    eri = arrays["conventional_mo"]
    g = eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3)
    adjoint = canonical_energy_adjoint(
        g,
        reference.orbital_energies,
        occupied,
        reference_identity=reference.identity,
        hamiltonian_id=reference.hamiltonian_id,
    )
    hcore_mo = (
        reference.coefficients.T @ arrays["conventional_h"] @ reference.coefficients
    )
    orbital = canonical_orbital_rhs(hcore_mo, eri, adjoint, occupied)
    backend = DenseAOResponseBackend(arrays["ao"])
    options = GMRESOptions(rtol=1e-12, atol=1e-13, restart=20, max_iterations=100)
    result = solve_canonical_orbital_response(
        reference, backend, orbital, options=options
    )
    explicit = _explicit_rhf_matrix(reference.orbital_energies, eri, occupied)
    expected = np.linalg.solve(explicit, orbital.response_rhs.reshape(-1))
    np.testing.assert_allclose(result.solution, expected, atol=2e-9, rtol=2e-9)
    assert result.converged and result.residual_norm < 1e-10
    weights = canonical_lagrangian_weights(
        hcore_mo,
        eri,
        adjoint,
        result,
        occupied,
    )
    assert weights.stationarity_residual < 1e-9
    np.testing.assert_allclose(weights.overlap, weights.overlap.T, atol=1e-14)
    assert weights.reference_identity == reference.identity
    assert weights.hamiltonian_id == reference.hamiltonian_id
    assert weights.operator_identity
    rng = np.random.default_rng(196)
    metric_direction = rng.normal(size=(len(reference.orbital_energies),) * 2)
    metric_direction = 0.5 * (metric_direction + metric_direction.T)

    def weighted_hamiltonian(step):
        connection = np.eye(len(metric_direction)) - 0.5 * step * metric_direction
        transformed_h = connection.T @ hcore_mo @ connection
        transformed_eri = _transform_eri(eri, connection)
        return np.vdot(weights.one_electron, transformed_h) + np.vdot(
            weights.two_electron, transformed_eri
        )

    metric_fd = (weighted_hamiltonian(1e-5) - weighted_hamiltonian(-1e-5)) / 2e-5
    np.testing.assert_allclose(
        metric_fd, np.vdot(weights.overlap, metric_direction), atol=2e-8, rtol=2e-8
    )
    with pytest.raises(ResponseSolveError, match="workspace_limit"):
        solve_canonical_orbital_response(
            reference,
            backend,
            orbital,
            options=GMRESOptions(max_workspace_bytes=1),
        )
    stale = replace(reference, generation_id="another-generation")
    with pytest.raises(ValueError, match="different reference"):
        solve_canonical_orbital_response(stale, backend, orbital, options=options)
    with pytest.raises(ValueError, match="Z-vector belongs to a different reference"):
        canonical_lagrangian_weights(
            hcore_mo,
            eri,
            adjoint,
            replace(result, reference_identity=stale.identity),
            occupied,
        )


@pytest.mark.parametrize("name", ["h2", "water"])
def test_dense_complete_gradient_matches_fully_resolved_finite_differences(name):
    meta, arrays = load_fixture(name)
    arguments = source_arguments(meta)
    reference = fixture_snapshot(meta, arrays)
    occupied = reference.nocc
    eri = arrays["conventional_mo"]
    g = eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3)
    adjoint = canonical_energy_adjoint(
        g,
        reference.orbital_energies,
        occupied,
        reference_identity=reference.identity,
        hamiltonian_id=reference.hamiltonian_id,
    )
    hcore_mo = (
        reference.coefficients.T @ arrays["conventional_h"] @ reference.coefficients
    )
    orbital = canonical_orbital_rhs(hcore_mo, eri, adjoint, occupied)
    backend = DenseAOResponseBackend(arrays["ao"])
    response = solve_canonical_orbital_response(
        reference,
        backend,
        orbital,
        options=GMRESOptions(rtol=1e-12, atol=1e-13, max_iterations=100),
    )
    weights = canonical_lagrangian_weights(hcore_mo, eri, adjoint, response, occupied)
    with NativeSource(**arguments) as source:
        analytic = dense_molecular_gradient_oracle(reference, source, weights)

    calculator = Calculator(
        method="mp2",
        basis=arguments["basis"],
        basis_representation=arguments["representation"],
    )
    base_atoms = arguments["atoms"]
    finite = []
    for step in (3e-3, 1e-3, 3e-4):
        gradient = np.empty_like(analytic)
        for atom in range(len(base_atoms)):
            for axis in range(3):
                displaced = []
                for index, value in enumerate(base_atoms):
                    position = list(value.position)
                    if index == atom:
                        position[axis] += step
                    displaced.append((value.atomic_number, position))
                plus = calculator.singlepoint(
                    displaced, charge=arguments["charge"]
                ).energy
                displaced[atom][1][axis] -= 2 * step
                minus = calculator.singlepoint(
                    displaced, charge=arguments["charge"]
                ).energy
                gradient[atom, axis] = (plus - minus) / (2 * step)
        finite.append(gradient)
    errors = [float(np.max(np.abs(value - analytic))) for value in finite]
    assert errors[-1] < 1e-6
    assert min(errors) < 1e-7
    assert errors[-1] < errors[0]
