"""Directional nuclear inputs/response, checked independently of Hessian assembly."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    NativeRHFState,
    directional,
    directional_rhf_response,
    first_order,
    perturbation,
)
from tools.vibeqc_posthf.export import conventional_fock
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import GMRESOptions, ResponseSolveError
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs


@pytest.fixture(scope="module", params=("h2", "water"))
def case(request):
    with NativeSource(**fixture_inputs(request.param)) as source:
        state = NativeRHFState.from_source(source)
        v = np.random.default_rng(180).normal(size=(state.nat, 3))
        v /= np.linalg.norm(v)
        result = directional_rhf_response(state, v)
        yield request.param, state, v, result


def forbidden(*args, **kwargs):
    raise AssertionError("directional path used an all-coordinate/dense/oracle input")


def test_directional_sources_match_independent_native_integral_derivatives(case):
    _, s, v, result = case
    # Independent native derivative implementation exists only on the assertion side.
    integrals = s.source.integral_derivatives()
    eri1 = np.einsum("a,apqrs->pqrs", v.ravel(), integrals["eri"])
    h1 = np.einsum("a,apq->pq", v.ravel(), integrals["hcore"])
    expected = h1 + np.einsum("pqrs,rs->pq", eri1, s.P0)
    expected -= 0.5 * np.einsum("prqs,rs->pq", eri1, s.P0)
    np.testing.assert_allclose(
        result.frozen_fock_derivative, expected, atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        result.overlap_derivative,
        np.einsum("a,apq->pq", v.ravel(), integrals["overlap"]),
        atol=2e-11,
        rtol=2e-11,
    )


def test_does_not_materialize_all_coordinate_inputs_or_rerun_scf(case, monkeypatch):
    _, s, v, expected = case
    monkeypatch.setattr(NativeRHFState, "first_order_inputs", property(forbidden))
    monkeypatch.setattr(first_order, "generated_first_order", forbidden)
    monkeypatch.setattr(s.source, "integral_derivatives", forbidden)
    monkeypatch.setattr(s.source, "rhf_density", forbidden)
    original_zeros = first_order.np.zeros

    def checked_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape == (s.nat, 3, s.nbf, s.nbf):
            forbidden()
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(first_order.np, "zeros", checked_zeros)
    original_solve = perturbation.solve
    solves = []

    def counted_solve(*args, **kwargs):
        solves.append(1)
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(perturbation, "solve", counted_solve)
    actual = directional_rhf_response(s, v)
    np.testing.assert_allclose(
        actual.response.density_derivative,
        expected.response.density_derivative,
        atol=2e-11,
        rtol=2e-11,
    )
    assert actual.diagnostics["first_order_matrix_bytes"] == 2 * s.nbf**2 * 8
    assert actual.diagnostics["nuclear_response_solves"] == 1
    assert len(solves) == 1


def test_response_metric_and_immutable_result_contract(case):
    _, s, v, result = case
    r = result.response
    c = s.C[:, : s.nocc]
    c1 = r.coefficient_derivative
    np.testing.assert_allclose(
        c.T @ s.S0 @ c1 + c1.T @ s.S0 @ c + c.T @ result.overlap_derivative @ c,
        0,
        atol=2e-10,
        rtol=0,
    )
    assert (
        abs(
            np.einsum("pq,pq", r.density_derivative, s.S0)
            + np.einsum("pq,pq", s.P0, result.overlap_derivative)
        )
        < 2e-10
    )
    for value in (r.density_derivative, r.energy_weighted_density_derivative):
        np.testing.assert_allclose(value, value.T, atol=2e-10, rtol=0)
    for value in (
        result.direction,
        result.frozen_fock_derivative,
        result.overlap_derivative,
        r.rhs,
        r.coefficient_derivative,
        r.occupied_energy_derivative,
        r.density_derivative,
        r.energy_weighted_density_derivative,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    np.testing.assert_array_equal(result.direction, v)
    diag = result.diagnostics
    diag["rhs_count"] = 500
    assert result.diagnostics["rhs_count"] == 1
    assert not result.diagnostics["molecular_hvp"]
    assert result.response.solve_result.converged
    assert result.response.solve_result.residual_norm < 1e-9


def test_three_step_finite_difference_of_native_D_W_and_frozen_F_S(case):
    _, s, v, result = case
    targets = (
        result.frozen_fock_derivative,
        result.overlap_derivative,
        result.response.density_derivative,
        result.response.energy_weighted_density_derivative,
    )
    errors = []
    for step in (3e-3, 1e-3, 3e-4):
        displaced = []
        for sign in (1, -1):
            atoms = [
                (a.atomic_number, xyz)
                for a, xyz in zip(
                    s.source.atoms, s.coords + sign * step * v, strict=True
                )
            ]
            with NativeSource(
                atoms, basis=s.source.shells, charge=s.source.charge
            ) as source:
                moved = NativeRHFState.from_source(source, tolerance=1e-13)
                frozen = conventional_fock(source, s.P0)
                w = (
                    moved.C[:, : moved.nocc] * (2 * moved.eps[: moved.nocc])
                ) @ moved.C[:, : moved.nocc].T
                displaced.append((frozen, moved.S0, moved.P0, w))
        errors.append(
            [
                float(np.max(np.abs((a - b) / (2 * step) - target)))
                for a, b, target in zip(*displaced, targets, strict=True)
            ]
        )
    errors = np.asarray(errors)
    assert np.all(errors[-1] < 2e-5), errors
    assert np.all(errors[-1] < np.maximum(0.2 * errors[0], 2e-7)), errors


def test_translation_zero_and_direction_scaling(case):
    _, s, v, result = case
    zero = directional_rhf_response(s, np.zeros_like(v))
    assert zero.response.solve_result.iterations == 0
    np.testing.assert_array_equal(zero.response.density_derivative, np.zeros_like(s.P0))
    translation = directional_rhf_response(s, np.tile([0.13, -0.21, 0.31], (s.nat, 1)))
    np.testing.assert_allclose(
        translation.frozen_fock_derivative, 0, atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        translation.response.energy_weighted_density_derivative, 0, atol=2e-9, rtol=0
    )
    scaled = directional_rhf_response(s, -0.7 * v)
    assert scaled.identity != result.identity
    np.testing.assert_allclose(
        scaled.response.density_derivative,
        -0.7 * result.response.density_derivative,
        atol=2e-9,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        scaled.response.energy_weighted_density_derivative,
        -0.7 * result.response.energy_weighted_density_derivative,
        atol=2e-9,
        rtol=2e-9,
    )


@pytest.mark.parametrize("case", ["water"], indirect=True)
def test_metric_omission_changes_independently_checked_response(case, monkeypatch):
    _, s, v, expected = case
    monkeypatch.setattr(
        perturbation, "metric_density_response_mo", lambda s1, **_: np.zeros_like(s1)
    )
    wrong = directional_rhf_response(s, v)
    assert (
        np.max(
            np.abs(
                wrong.response.density_derivative - expected.response.density_derivative
            )
        )
        > 1e-4
    )
    assert (
        np.max(
            np.abs(
                wrong.response.energy_weighted_density_derivative
                - expected.response.energy_weighted_density_derivative
            )
        )
        > 1e-4
    )


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 2.0, 3.0],
        np.zeros((1, 3)),
        np.zeros((2, 3), dtype=complex),
        np.ones((2, 3), dtype=bool),
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
        np.full((2, 3), "x", dtype=object),
    ],
)
def test_invalid_direction_rejected_before_provider_work(bad, monkeypatch):
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        monkeypatch.setattr(directional, "generated_directional_first_order", forbidden)
        with pytest.raises(ValueError, match="direction"):
            directional_rhf_response(state, bad)


def test_invalid_backend_options_and_closed_state_fail_before_sources(monkeypatch):
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        monkeypatch.setattr(directional, "generated_directional_first_order", forbidden)
        with pytest.raises(ValueError, match="jk_backend"):
            directional_rhf_response(state, np.zeros((2, 3)), jk_backend="auto")
        with pytest.raises(ValueError, match="response_execution"):
            directional_rhf_response(
                state, np.zeros((2, 3)), response_execution="device"
            )
        with pytest.raises(ValueError, match="requires jk_backend"):
            directional_rhf_response(
                state, np.zeros((2, 3)), response_execution="cuda-resident"
            )
        with pytest.raises(ValueError, match="response_device_budget"):
            directional_rhf_response(
                state, np.zeros((2, 3)), response_device_budget_bytes=0
            )
        with pytest.raises(TypeError, match="solver_options"):
            directional_rhf_response(state, np.zeros((2, 3)), solver_options={})
        with pytest.raises(ValueError, match="geometry"):
            replace(state, reference=replace(state.reference, geometry_hash="other"))
    with pytest.raises(RuntimeError, match="closed"):
        directional_rhf_response(state, np.zeros((2, 3)))


def test_insufficient_solver_workspace_does_not_publish_or_poison_result(case):
    _, s, v, expected = case
    with pytest.raises(ResponseSolveError):
        directional_rhf_response(
            s, v, solver_options=GMRESOptions(max_workspace_bytes=1)
        )
    actual = directional_rhf_response(s, v)
    np.testing.assert_allclose(
        actual.response.density_derivative,
        expected.response.density_derivative,
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize("case", ["h2"], indirect=True)
def test_late_first_component_failure_replays_cleanly(case, monkeypatch):
    _, s, v, expected = case
    original = first_order.FirstDerivativeEvaluator.contract
    count = 0

    def fail_late(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise FloatingPointError("injected late derivative failure")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(first_order.FirstDerivativeEvaluator, "contract", fail_late)
        with pytest.raises(FloatingPointError, match="late derivative"):
            directional_rhf_response(s, v)
    actual = directional_rhf_response(s, v)
    np.testing.assert_array_equal(
        actual.frozen_fock_derivative, expected.frozen_fock_derivative
    )


@pytest.mark.parametrize("case", ["water"], indirect=True)
def test_nonconverged_response_fails_without_partial_publication(case):
    _, state, v, expected = case
    with pytest.raises(ResponseSolveError):
        directional_rhf_response(
            state,
            v,
            solver_options=GMRESOptions(restart=1, max_iterations=1, rtol=1e-15),
        )
    replay = directional_rhf_response(state, v)
    np.testing.assert_allclose(
        replay.response.energy_weighted_density_derivative,
        expected.response.energy_weighted_density_derivative,
        atol=2e-9,
        rtol=2e-9,
    )


@pytest.mark.parametrize("case", ["h2"], indirect=True)
def test_one_perturbation_rejects_bad_matrices_before_jk(case, monkeypatch):
    from tools.vibeqc_response import NativeJKBackend, RHFResponseOperator

    _, state, _, result = case
    backend = NativeJKBackend(state.source)
    operator = RHFResponseOperator(
        RHFResponseOperator.build_problem(state.reference, backend), backend
    )
    monkeypatch.setattr(backend, "coulomb_exchange", forbidden)
    for bad in (
        np.eye(2, dtype=complex),
        np.zeros((1, 1)),
        np.full((2, 2), np.inf),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
    ):
        with pytest.raises(ValueError, match="matrix|symmetric"):
            perturbation.solve_rhf_nuclear_perturbation(
                operator, bad, result.overlap_derivative
            )
