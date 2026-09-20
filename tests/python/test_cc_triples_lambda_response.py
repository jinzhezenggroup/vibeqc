"""Corrected standard-(T) Lambda and fixed-orbital response tests for #154 B."""

import typing
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest
from test_cc_lambda_response import _STRICT, _direction, _state

from tools.vibeqc_cc import PreparedCCSD
from tools.vibeqc_cc import solver as solver_module
from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_cc.triples_lambda_response import (
    BoundCCSDTResponse,
    solve_corrected_lambda,
)
from tools.vibeqc_response.implicit import ImplicitSolveError
from tools.vibeqc_response.problem import ResponseCompatibilityError


@lru_cache(maxsize=2)
def _corrected_state() -> typing.Any:
    snapshot, provider, cc, bound, baseline, _, _ = _state("h2o")
    corrected = solve_corrected_lambda(bound, baseline, vir_chunk_size=1)
    response = BoundCCSDTResponse(bound, baseline, corrected, vir_chunk_size=1)
    return snapshot, provider, cc, bound, baseline, corrected, response


def _resolved_total_correlation(
    snapshot: typing.Any, provider: typing.Any, cc: typing.Any, changes: typing.Any
) -> float:
    """Re-solve CCSD at perturbed fixed-orbital inputs, then recompute standard (T)."""

    prepared = PreparedCCSD(snapshot, provider, _STRICT, t1=cc.t1, t2=cc.t2)
    for name, change in changes.items():
        prepared.feeds[name] = prepared.feeds[name] + change
    with patch.object(solver_module, "PreparedCCSD", return_value=prepared):
        result = solver_module.solve(snapshot, provider, options=_STRICT)
    assert result.converged, result.reason
    assert (
        max(
            result.history[-1]["independent_r1_max"],
            result.history[-1]["independent_r2_max"],
        )
        <= 1e-12
    )
    o = snapshot.nocc
    feeds = prepared.feeds
    et = triples_energy(
        o,
        snapshot.nmo - o,
        feeds["ovvv"],
        feeds["ovoo"],
        feeds["ovov"],
        feeds["fov"],
        result.t1,
        result.t2,
        snapshot.orbital_energies[:o],
        snapshot.orbital_energies[o:],
    )
    return result.correlation_energy + et


def test_corrected_lambda_replays_total_stationarity_and_is_nontrivial() -> None:
    _, _, _, bound, baseline, corrected, response = _corrected_state()
    assert corrected.lambda_residual_norm <= bound.options.lambda_tolerance
    assert corrected.independent_lambda_residual_norm <= bound.options.lambda_tolerance
    assert corrected.independent_lambda_residual_max <= bound.options.lambda_tolerance
    assert (
        np.linalg.norm(corrected.delta_lambda1)
        + np.linalg.norm(corrected.delta_lambda2)
        > 1e-7
    )
    np.testing.assert_allclose(
        corrected.lambda1,
        baseline.lambda1 + corrected.delta_lambda1,
        atol=1e-13,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        corrected.lambda2,
        baseline.lambda2 + corrected.delta_lambda2,
        atol=1e-13,
        rtol=1e-12,
    )
    assert corrected.provenance["jacobian"].startswith("RCCSD residual Jacobian")
    assert corrected.provenance["orbital_response"] == "excluded"
    assert response.corrected_lambda_identity


@pytest.mark.parametrize("parameter", ("foo", "fov", "ovov", "oooo"))
def test_combined_weights_match_reconverged_ccsd_t_fixed_orbital_differences(
    parameter: typing.Any,
) -> None:
    snapshot, provider, cc, _, _, _, response = _corrected_state()
    weight = response.weight(parameter, reference_identity=snapshot.identity)
    direction = _direction(
        weight.spec, seed=15420 + response.parameters.index(parameter)
    )
    expected = weight.contract(direction)
    for step in (1e-4, 3e-5):
        lower = _resolved_total_correlation(
            snapshot, provider, cc, {parameter: -step * direction}
        )
        upper = _resolved_total_correlation(
            snapshot, provider, cc, {parameter: step * direction}
        )
        np.testing.assert_allclose(
            (upper - lower) / (2 * step),
            expected,
            atol=2e-8,
            rtol=3e-6,
        )
    assert weight.independent_delta_max_abs_error <= 1e-12


def test_negative_decomposition_detects_pure_ccsd_lambda_and_missing_direct_triples() -> (
    None
):
    snapshot, _, _, _, _, _, response = _corrected_state()

    delta_only = response.weight("foo", reference_identity=snapshot.identity)
    assert np.linalg.norm(delta_only.direct_triples) == 0
    assert np.linalg.norm(delta_only.delta_lambda) > 1e-7
    direction = delta_only.delta_lambda / np.linalg.norm(delta_only.delta_lambda)
    assert (
        abs(
            delta_only.contract(direction)
            - float(np.sum(delta_only.baseline_ccsd * direction))
        )
        > 1e-7
    )

    direct = response.weight("fov", reference_identity=snapshot.identity)
    assert np.linalg.norm(direct.direct_triples) > 1e-7
    direction = direct.direct_triples / np.linalg.norm(direct.direct_triples)
    without_direct = direct.baseline_ccsd + direct.delta_lambda
    assert (
        abs(direct.contract(direction) - float(np.sum(without_direct * direction)))
        > 1e-7
    )


def test_orbital_energy_denominator_sources_match_fixed_amplitude_finite_differences() -> (
    None
):
    snapshot, _, _, bound, _, _, response = _corrected_state()
    weights = response.orbital_energy_weights(reference_identity=snapshot.identity)
    o = snapshot.nocc
    arrays = (
        bound.feeds["ovvv"],
        bound.feeds["ovoo"],
        bound.feeds["ovov"],
        bound.feeds["fov"],
        bound.feeds["t1"],
        bound.feeds["t2"],
    )
    eps_o = np.array(snapshot.orbital_energies[:o], copy=True)
    eps_v = np.array(snapshot.orbital_energies[o:], copy=True)
    for index, name in enumerate(("eps_o", "eps_v")):
        direction = np.random.default_rng(15450 + index).normal(
            size=weights[name].shape
        )
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(weights[name] * direction))
        for step in (2e-5, 2e-6):
            energies = []
            for sign in (-1.0, 1.0):
                eo, ev = np.array(eps_o, copy=True), np.array(eps_v, copy=True)
                (eo if name == "eps_o" else ev)[:] += sign * step * direction
                energies.append(
                    triples_energy(
                        o,
                        snapshot.nmo - o,
                        *arrays,
                        eo,
                        ev,
                    )
                )
            np.testing.assert_allclose(
                (energies[1] - energies[0]) / (2 * step),
                analytic,
                atol=2e-9,
                rtol=2e-6,
            )
    np.testing.assert_allclose(
        np.sum(weights["eps_o"]) + np.sum(weights["eps_v"]),
        0.0,
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ("total", "delta", "state", "schedule"))
def test_forged_corrected_lambda_cannot_publish_response(mode: typing.Any) -> None:
    _, _, _, bound, baseline, corrected, _ = _corrected_state()
    if mode == "total":
        values = np.array(corrected.lambda1)
        values[0, 0] += 1e-3
        corrected = replace(corrected, lambda1=values)
    elif mode == "delta":
        values = np.array(corrected.delta_lambda2)
        values.flat[0] += 1e-3
        corrected = replace(corrected, delta_lambda2=values)
    elif mode == "state":
        corrected = replace(corrected, cc_state_identity="other-cc-state")
    else:
        corrected = replace(
            corrected,
            provenance={**corrected.provenance, "vir_chunk_size": 2},
        )
    with pytest.raises((ResponseCompatibilityError, ImplicitSolveError, ValueError)):
        BoundCCSDTResponse(bound, baseline, corrected, vir_chunk_size=1)


def test_all_parameter_blocks_have_checked_decomposition_and_state_identity() -> None:
    snapshot, _, _, bound, _, _, response = _corrected_state()
    for parameter in response.parameters:
        weight = response.weight(parameter, reference_identity=snapshot.identity)
        np.testing.assert_allclose(
            weight.values,
            weight.baseline_ccsd + weight.direct_triples + weight.delta_lambda,
            atol=1e-14,
            rtol=1e-13,
        )
        assert np.isfinite(weight.values).all()
        assert weight.reference_identity == snapshot.identity
        assert weight.cc_state_identity == bound.cc_state_identity
        assert weight.corrected_lambda_identity == response.corrected_lambda_identity


def test_parameter_weight_rejects_invalid_directions() -> None:
    snapshot, _, _, _, _, _, response = _corrected_state()
    weight = response.weight("foo", reference_identity=snapshot.identity)
    direction = np.eye(snapshot.nocc, dtype=np.float64)
    for bad in (
        direction.astype(np.float32),
        direction.reshape(-1),
        direction * np.nan,
    ):
        with pytest.raises(ValueError):
            weight.contract(bad)
    direction[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetry"):
        weight.contract(direction)


def test_outputs_are_immutable_and_identity_bound() -> None:
    snapshot, _, _, bound, _, corrected, response = _corrected_state()
    for value in (
        corrected.lambda1,
        corrected.lambda2,
        corrected.delta_lambda1,
        corrected.delta_lambda2,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    weight = response.weight("ovov", reference_identity=snapshot.identity)
    for value in (
        weight.values,
        weight.baseline_ccsd,
        weight.direct_triples,
        weight.delta_lambda,
    ):
        assert not value.flags.writeable
    with pytest.raises(ResponseCompatibilityError):
        response.weight("ovov", reference_identity="stale-reference")
    assert corrected.cc_state_identity == bound.cc_state_identity


@pytest.mark.parametrize("where", ("field", "provenance", "both"))
def test_corrected_source_identity_is_recomputed(where: str) -> None:
    _, _, _, bound, baseline, corrected, _ = _corrected_state()
    changes = {}
    if where in ("field", "both"):
        changes["triples_source_identity"] = "forged-source"
    if where in ("provenance", "both"):
        changes["provenance"] = {
            **corrected.provenance,
            "triples_source_identity": "forged-source",
        }
    with pytest.raises(ResponseCompatibilityError, match="source identity"):
        BoundCCSDTResponse(
            bound, baseline, replace(corrected, **changes), vir_chunk_size=1
        )


@pytest.mark.parametrize("field", ("corrected", "vir_chunk_size", "baseline"))
def test_bound_corrected_response_cannot_be_replaced_after_validation(
    field: str,
) -> None:
    from dataclasses import FrozenInstanceError

    _, _, _, bound, baseline, corrected, _ = _corrected_state()
    response = BoundCCSDTResponse(bound, baseline, corrected, vir_chunk_size=1)
    value = 2 if field == "vir_chunk_size" else None
    with pytest.raises(FrozenInstanceError):
        setattr(response, field, value)
