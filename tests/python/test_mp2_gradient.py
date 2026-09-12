"""Analytic MP2 energy-adjoint contracts for the complete-gradient chain."""

import ctypes as ct
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc import Calculator, _native
from vibeqc_compiler.tensor import execute

from tools.vibeqc_mp2.complete_gradient import complete_gradient_validation
from tools.vibeqc_mp2.equations import energy_program
from tools.vibeqc_mp2.gradient import (
    ao_lagrangian_weights,
    canonical_energy_adjoint,
    canonical_lagrangian_weights,
    canonical_orbital_rhs,
    dense_molecular_gradient_oracle,
    dense_ri_lagrangian_weights_oracle,
    dense_ri_molecular_gradient_oracle,
    fused_cuda_conventional_molecular_gradient,
    fused_cuda_ri_molecular_gradient,
    solve_canonical_orbital_response,
    tile_energy_adjoint,
)
from tools.vibeqc_posthf.df import MetricFactor
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
        with pytest.raises(ValueError, match="output exceeds"):
            source.df_integral_derivatives(output_budget_bytes=1)


def test_inverse_sqrt_metric_response_is_included_in_ri_gradient():
    from tools.vibeqc_mp2.gradient import _inverse_sqrt_metric_response

    metric = np.array([[2.0, 0.2], [0.2, 1.1]])
    bar = np.array([[0.3, -0.4], [0.2, 0.7]])
    direction = np.array([[0.1, 0.3], [0.3, -0.2]])
    reverse = np.vdot(_inverse_sqrt_metric_response(metric, bar, 1e-10), direction)

    def scalar(step):
        values, vectors = np.linalg.eigh(metric + step * direction)
        root = vectors @ np.diag(values**-0.5) @ vectors.T
        return np.vdot(bar, root)

    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        finite = (scalar(step) - scalar(-step)) / (2 * step)
        errors.append(abs(finite - reverse))
    assert errors[-1] < 1e-9 and errors[-1] < errors[0]


def test_nuclear_repulsion_gradient_matches_independent_oracle_block():
    from tools.vibeqc_mp2.gradient import _nuclear_repulsion_gradient

    meta, _ = load_fixture("water")
    arguments = source_arguments(meta)
    with NativeSource(**arguments) as source:
        expected = source.integral_derivatives()["nuclear"].reshape(-1, 3)
    np.testing.assert_allclose(
        _nuclear_repulsion_gradient(arguments["atoms"]), expected, atol=1e-13, rtol=0
    )


def test_gradient_validation_helpers_route_explicit_device():
    from tools.vibeqc_validation.df_gradient import execute_df_gradient
    from tools.vibeqc_validation.one_electron_gradient import execute_gradient

    captured = []

    class Function:
        def __init__(self, implementation=lambda *_: 0):
            self.implementation = implementation

        def __call__(self, *arguments):
            return self.implementation(*arguments)

    def capture(descriptor, _):
        value = ct.cast(descriptor, ct.POINTER(_native.ContextDescriptor)).contents
        captured.append(value.device_id)
        raise RuntimeError("captured device")

    library = SimpleNamespace(
        vibeqc_system_one_electron_gradient_cuda=Function(),
        vibeqc_system_df_gradient_cuda=Function(),
        vibeqc_context_create=Function(capture),
    )
    calculator = SimpleNamespace(_library=library)
    atoms = [("H", (0.0, 0.0, 0.0))]
    with pytest.raises(RuntimeError, match="captured device"):
        execute_gradient(calculator, atoms, np.zeros((3, 1, 1)), device_id=7)
    with pytest.raises(RuntimeError, match="captured device"):
        execute_df_gradient(
            calculator,
            calculator,
            atoms,
            np.zeros((1, 1, 1)),
            np.zeros((1, 1)),
            device_id=9,
        )
    assert captured == [7, 9]


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)
def test_weighted_eri_cuda_spherical_pullback_matches_dense_oracle(monkeypatch):
    meta, _ = load_fixture("f_heh")
    arguments = source_arguments(meta)
    assert arguments["representation"] == "spherical"
    with NativeSource(**arguments) as source:
        rng = np.random.default_rng(144193)
        weights = rng.normal(scale=0.02, size=(source.nbf,) * 4)
        derivatives = source.integral_derivatives()["eri"]
        expected = np.einsum("xpqrs,pqrs->x", derivatives, weights).reshape(-1, 3)
        monkeypatch.setattr(
            source,
            "integral_derivatives",
            lambda **_: pytest.fail("weighted bridge called dense derivatives"),
        )
        actual = source.weighted_eri_gradient_cuda(weights)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)
@pytest.mark.parametrize("density_fitted", [False, True])
def test_complete_gradient_validation_facade_matches_public_finite_difference(
    density_fitted, monkeypatch
):
    meta, _ = load_fixture("h2")
    arguments = source_arguments(meta)
    with NativeSource(**arguments) as source:
        orbital = Calculator(
            basis=arguments["basis"],
            basis_representation=arguments["representation"],
            device="cuda",
        )
        auxiliary = (
            Calculator(
                basis=arguments["auxiliary_basis"],
                basis_representation=arguments["representation"],
                device="cuda",
            )
            if density_fitted
            else None
        )
        monkeypatch.setattr(
            source,
            "integral_derivatives",
            lambda **_: pytest.fail("facade called dense conventional derivatives"),
        )
        monkeypatch.setattr(
            source,
            "df_integral_derivatives",
            lambda **_: pytest.fail("facade called dense DF derivatives"),
        )
        result = complete_gradient_validation(
            source,
            orbital,
            auxiliary,
            density_fitted=density_fitted,
        )
    label = "df" if density_fitted else "conventional"
    record = meta["records"][label]
    assert (
        abs(result.total_energy - record["hf_energy"] - record["correlation_energy"])
        < 1e-9
    )
    calculator = Calculator(
        method="mp2",
        basis=arguments["basis"],
        auxiliary_basis=arguments["auxiliary_basis"] if density_fitted else None,
        basis_representation=arguments["representation"],
        density_fitting="cuda" if density_fitted else "none",
        device="cuda",
    )
    step = 3e-4
    finite = np.empty_like(result.gradient)
    for atom in range(len(arguments["atoms"])):
        for axis in range(3):
            displaced = [
                (value.atomic_number, list(value.position))
                for value in arguments["atoms"]
            ]
            displaced[atom][1][axis] += step
            plus = calculator.singlepoint(displaced).energy
            displaced[atom][1][axis] -= 2 * step
            minus = calculator.singlepoint(displaced).energy
            finite[atom, axis] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(result.gradient, finite, atol=1e-6, rtol=1e-6)
    assert result.response_residual < 1e-9
    assert result.stationarity_residual < 1e-8
    assert result.diagnostics["global_derivative_tensors"] is False


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
def test_dense_complete_gradient_matches_fully_resolved_finite_differences(
    name, monkeypatch
):
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
    if name == "h2":
        with pytest.raises(MemoryError, match="output budget"):
            ao_lagrangian_weights(
                reference,
                weights,
                output_budget_bytes=reference.nmo**4 * 8,
            )
    with NativeSource(**arguments) as source:
        analytic = dense_molecular_gradient_oracle(reference, source, weights)
        if name == "h2":
            fitted_reference = replace(
                reference, hamiltonian_id="density-fitting:stale-test"
            )
            fitted_weights = replace(
                weights,
                reference_identity=fitted_reference.identity,
                hamiltonian_id=fitted_reference.hamiltonian_id,
            )
            cpu_calculator = Calculator(
                basis=arguments["basis"],
                basis_representation=arguments["representation"],
            )
            with pytest.raises(ValueError, match="unscreened exact Hamiltonian"):
                fused_cuda_conventional_molecular_gradient(
                    fitted_reference,
                    source,
                    fitted_weights,
                    cpu_calculator,
                )
        if os.environ.get("VIBEQC_MP2_CUDA_TEST") == "1":
            calculator_cuda = Calculator(
                basis=arguments["basis"],
                basis_representation=arguments["representation"],
                device="cuda",
            )
            monkeypatch.setattr(
                source,
                "integral_derivatives",
                lambda **_: pytest.fail("fused path called dense derivatives"),
            )
            fused, diagnostics = fused_cuda_conventional_molecular_gradient(
                reference, source, weights, calculator_cuda
            )
            np.testing.assert_allclose(fused, analytic, atol=2e-9, rtol=2e-9)
            assert diagnostics["global_derivative_tensors"] is False
            assert (
                diagnostics["weight_output_bytes"]
                <= diagnostics["weight_output_budget_bytes"]
            )

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


@pytest.mark.parametrize("name", ["h2", "water"])
def test_dense_complete_ri_gradient_matches_fully_resolved_finite_differences(
    name, monkeypatch
):
    meta, arrays = load_fixture(name)
    arguments = source_arguments(meta)
    with NativeSource(**arguments) as source:
        metric = MetricFactor.from_source(source)
        reference = fixture_snapshot(meta, arrays, label="df", metric=metric)
        occupied = reference.nocc
        eri = arrays["df_mo"]
        g = eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3)
        adjoint = canonical_energy_adjoint(
            g,
            reference.orbital_energies,
            occupied,
            reference_identity=reference.identity,
            hamiltonian_id=reference.hamiltonian_id,
        )
        hcore_mo = reference.coefficients.T @ arrays["df_h"] @ reference.coefficients
        orbital = canonical_orbital_rhs(hcore_mo, eri, adjoint, occupied)
        response = solve_canonical_orbital_response(
            reference,
            DenseAOResponseBackend(arrays["df_ao"]),
            orbital,
            options=GMRESOptions(rtol=1e-12, atol=1e-13, max_iterations=100),
        )
        weights = canonical_lagrangian_weights(
            hcore_mo, eri, adjoint, response, occupied
        )
        if name == "h2":
            incomplete_output_budget = (
                reference.nmo**2 * source.naux + source.naux**2
            ) * 8
            with pytest.raises(MemoryError, match="output budget"):
                dense_ri_lagrangian_weights_oracle(
                    reference,
                    source,
                    metric,
                    weights,
                    output_budget_bytes=incomplete_output_budget,
                )
        analytic = dense_ri_molecular_gradient_oracle(
            reference, source, metric, weights
        )
        if os.environ.get("VIBEQC_MP2_CUDA_TEST") == "1":
            orbital_calculator = Calculator(
                basis=arguments["basis"],
                basis_representation=arguments["representation"],
                device="cuda",
            )
            auxiliary_calculator = Calculator(
                basis=arguments["auxiliary_basis"],
                basis_representation=arguments["representation"],
                device="cuda",
            )
            monkeypatch.setattr(
                source,
                "integral_derivatives",
                lambda **_: pytest.fail(
                    "fused path called conventional dense derivatives"
                ),
            )
            monkeypatch.setattr(
                source,
                "df_integral_derivatives",
                lambda **_: pytest.fail("fused path called dense DF derivatives"),
            )
            fused, diagnostics = fused_cuda_ri_molecular_gradient(
                reference,
                source,
                metric,
                weights,
                orbital_calculator,
                auxiliary_calculator,
            )
            np.testing.assert_allclose(fused, analytic, atol=2e-9, rtol=2e-9)
            assert diagnostics["global_derivative_tensors"] is False
            assert diagnostics["excluded_from_bridge_budget"]
            assert (
                diagnostics["weight_output_bytes"]
                <= diagnostics["weight_output_budget_bytes"]
            )
        elif name == "h2":
            cpu_calculator = Calculator(
                basis=arguments["basis"],
                basis_representation=arguments["representation"],
            )
            with pytest.raises(ValueError, match="calculators differ"):
                fused_cuda_ri_molecular_gradient(
                    reference,
                    source,
                    metric,
                    weights,
                    cpu_calculator,
                    cpu_calculator,
                )
            for stale in (
                {"charge": 1, "multiplicity": 2},
                {"charge": 0, "multiplicity": 3},
            ):
                with (
                    NativeSource(**{**arguments, **stale}) as stale_source,
                    pytest.raises(ValueError, match="electron state mismatch"),
                ):
                    fused_cuda_ri_molecular_gradient(
                        reference,
                        stale_source,
                        metric,
                        weights,
                        cpu_calculator,
                        cpu_calculator,
                    )
    if name == "h2":
        first = arguments["basis"][0]
        changed_primitive = replace(
            first.primitives[0], exponent=first.primitives[0].exponent * 1.01
        )
        changed_basis = (
            replace(first, primitives=(changed_primitive, *first.primitives[1:])),
            *arguments["basis"][1:],
        )
        with NativeSource(**{**arguments, "basis": changed_basis}) as changed_source:
            changed_metric = MetricFactor.from_source(changed_source)
            changed_reference = replace(
                reference, hamiltonian_id=changed_metric.hamiltonian_id
            )
            changed_weights = replace(
                weights,
                reference_identity=changed_reference.identity,
                hamiltonian_id=changed_reference.hamiltonian_id,
            )
            with pytest.raises(ValueError, match="identity mismatch"):
                dense_ri_molecular_gradient_oracle(
                    changed_reference,
                    changed_source,
                    changed_metric,
                    changed_weights,
                )

    calculator = Calculator(
        method="mp2",
        basis=arguments["basis"],
        auxiliary_basis=arguments["auxiliary_basis"],
        basis_representation=arguments["representation"],
        density_fitting="cpu",
    )
    finite = []
    for step in (1e-3, 3e-4):
        gradient = np.empty_like(analytic)
        for atom in range(len(arguments["atoms"])):
            for axis in range(3):
                displaced = []
                for index, value in enumerate(arguments["atoms"]):
                    position = list(value.position)
                    if index == atom:
                        position[axis] += step
                    displaced.append((value.atomic_number, position))
                plus = calculator.singlepoint(displaced).energy
                displaced[atom][1][axis] -= 2 * step
                minus = calculator.singlepoint(displaced).energy
                gradient[atom, axis] = (plus - minus) / (2 * step)
        finite.append(gradient)
    errors = [float(np.max(np.abs(value - analytic))) for value in finite]
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]
