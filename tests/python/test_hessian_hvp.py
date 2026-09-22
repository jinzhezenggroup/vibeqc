"""Complete conventional RHF HVP assembly and directional acceptance gates."""

import typing

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    NativeRHFState,
    analytic_hessian,
    cphf_relaxation,
    rhf_hvp,
)
from tools.vibeqc_hessian.directional import directional_rhf_response
from tools.vibeqc_hessian.first_order import generated_rhf_relaxation_contraction
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs


@pytest.fixture(scope="module")
def h2_case() -> typing.Any:
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        rng = np.random.default_rng(1803)
        v = rng.normal(size=(state.nat, 3))
        v /= np.linalg.norm(v)
        u = rng.normal(size=(state.nat, 3))
        u /= np.linalg.norm(u)
        response = directional_rhf_response(state, v)
        relaxation = generated_rhf_relaxation_contraction(
            state,
            response.response.density_derivative,
            response.response.energy_weighted_density_derivative,
        )
        yield state, v, u, relaxation


def test_directional_relaxation_equals_dense_cphf_contraction(
    h2_case: typing.Any,
) -> None:
    state, v, _, actual = h2_case
    dense = cphf_relaxation(state)
    expected = np.einsum("abxy,by->ax", dense, v)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-10)


def test_complete_hvp_matches_native_dense_assembly_by_component(
    h2_case: typing.Any,
) -> None:
    state, v, _, _ = h2_case
    actual = rhf_hvp(state, v)
    dense = analytic_hessian(state)
    for name in ("nuclear", "core", "pulay", "two_electron", "relaxation"):
        expected = np.einsum("abxy,by->ax", dense[name], v)
        np.testing.assert_allclose(
            actual.components[name], expected, atol=8e-10, rtol=3e-10
        )
    np.testing.assert_allclose(
        actual.value,
        np.einsum("abxy,by->ax", dense["total"], v),
        atol=1e-9,
        rtol=4e-10,
    )
    diag = actual.diagnostics
    assert diag["molecular_hvp"]
    assert not diag["full_molecular_hessian_allocated"]
    assert not diag["all_coordinate_first_integrals_allocated"]
    assert diag["response_operator_actions"] >= 0
    assert diag["solver_workspace_bytes"] > 0
    assert diag["published_hvp_bytes"] == actual.value.nbytes
    assert diag["transfers"]["directional_first"]["device_transfers"] == 0
    assert diag["transfers"]["response_jk"]["device_transfers"] == 0
    timings = diag["timings_seconds"]
    assert timings["complete_hvp"] > 0
    assert timings["directional_response"] >= timings["response_operator"]
    assert timings["second_integral_hvp"] > 0
    for value in (actual.direction, actual.value, *actual.components.values()):
        assert not value.flags.writeable


def test_hvp_bilinear_symmetry_without_posthoc_symmetrization(
    h2_case: typing.Any,
) -> None:
    state, v, u, _ = h2_case
    hv = rhf_hvp(state, v).value
    hu = rhf_hvp(state, u).value
    left = float(np.einsum("ax,ax->", u, hv))
    right = float(np.einsum("ax,ax->", v, hu))
    assert abs(left - right) < 2e-9


def test_hvp_matches_three_step_reconverged_gradient_difference(
    h2_case: typing.Any,
) -> None:
    from vibeqc import Calculator

    state, v, _, _ = h2_case
    expected = rhf_hvp(state, v).value
    calculator = Calculator(
        method="rhf",
        basis=state.source.shells,
        device="cpu",
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        max_iterations=400,
    )
    errors = []
    for step in (3e-3, 1e-3, 3e-4):
        gradients = []
        for sign in (1, -1):
            xyz = state.coords + sign * step * v
            result = calculator.singlepoint(
                [
                    (atom.atomic_number, position)
                    for atom, position in zip(state.source.atoms, xyz, strict=True)
                ],
                charge=state.source.charge,
                multiplicity=1,
            )
            assert result.forces is not None
            gradients.append(-np.asarray(result.forces))
        numeric = (gradients[0] - gradients[1]) / (2 * step)
        errors.append(float(np.max(np.abs(numeric - expected))))
    assert errors[-1] < 2e-5, errors
    assert errors[-1] < max(0.2 * errors[0], 1e-7), errors


def test_relaxation_is_required_for_complete_hvp(h2_case: typing.Any) -> None:
    state, v, _, _ = h2_case
    result = rhf_hvp(state, v)
    frozen = result.nuclear + result.core + result.pulay + result.two_electron
    assert np.max(np.abs(result.value - frozen)) > 1e-4
    np.testing.assert_allclose(
        result.value - frozen, result.relaxation, atol=2e-12, rtol=0
    )


def test_hvp_path_does_not_materialize_dense_hessian_or_coordinate_sources(
    h2_case: typing.Any, monkeypatch: typing.Any
) -> None:
    from tools.vibeqc_hessian import analytic, first_order

    state, v, _, _ = h2_case

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("matrix-free HVP used a dense/all-coordinate path")

    monkeypatch.setattr(analytic, "_scatter", forbidden)
    monkeypatch.setattr(analytic, "analytic_hessian", forbidden)
    monkeypatch.setattr(first_order, "generated_first_order", forbidden)
    monkeypatch.setattr(NativeRHFState, "first_order_inputs", property(forbidden))
    result = rhf_hvp(state, v)
    assert np.isfinite(result.value).all()
