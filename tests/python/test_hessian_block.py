"""Bounded multi-RHS and full conventional RHF Hessian assembly gates."""

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    NativeRHFState,
    analytic_hessian,
    rhf_hessian,
    rhf_hvp_many,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs


@pytest.fixture(scope="module")
def h2_case():
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        dense = analytic_hessian(state)["total"]
        rng = np.random.default_rng(1804)
        directions = rng.normal(size=(3, state.nat, 3))
        directions /= np.linalg.norm(directions.reshape(3, -1), axis=1)[:, None, None]
        yield state, dense, directions


def test_hvp_many_recycled_matches_dense_columns_and_reports_shared_solve(h2_case):
    state, dense, directions = h2_case
    result = rhf_hvp_many(state, directions, strategy="recycled")
    expected = np.stack(
        [np.einsum("abxy,by->ax", dense, vector) for vector in directions]
    )
    np.testing.assert_allclose(result.values, expected, atol=1e-9, rtol=4e-10)
    diag = result.diagnostics
    assert diag["nrhs"] == len(directions)
    assert diag["strategy"] == "recycled"
    assert diag["complete_numeric_peak_bound_bytes"] <= diag["total_budget_bytes"]
    assert diag["response_peak_workspace_bytes"] > 0
    assert result.response_batch.solve_result.converged
    assert len(result.response_batch.responses) == len(directions)
    for value in (result.directions, result.values, *result.components.values()):
        assert not value.flags.writeable


def test_hvp_many_uses_solve_many_not_single_solver(h2_case, monkeypatch):
    from tools.vibeqc_hessian import perturbation

    state, dense, directions = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("block Hessian response used the scalar solve path")

    monkeypatch.setattr(perturbation, "solve", forbidden)
    result = rhf_hvp_many(state, directions[:2], strategy="blocked")
    expected = np.stack(
        [np.einsum("abxy,by->ax", dense, vector) for vector in directions[:2]]
    )
    np.testing.assert_allclose(result.values, expected, atol=1e-9, rtol=4e-10)
    assert result.response_batch.solve_result.strategy == "blocked"


def test_full_hessian_blocked_matches_independent_dense_reference(h2_case):
    state, dense, _ = h2_case
    result = rhf_hessian(state, block_size=2, strategy="recycled")
    expected = dense.transpose(0, 2, 1, 3).reshape(6, 6)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-9, rtol=4e-10)
    diag = result.diagnostics
    assert diag["block_size"] == 2
    assert diag["block_count"] == 3
    assert diag["raw_symmetry_error"] < 2e-9
    assert not diag["posthoc_symmetrization"]
    assert not result.matrix.flags.writeable


def test_block_budget_rejects_before_first_integral_work(h2_case, monkeypatch):
    from tools.vibeqc_hessian import block

    state, _, directions = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("insufficient block budget reached provider work")

    monkeypatch.setattr(block, "generated_directional_first_order", forbidden)
    with pytest.raises(ValueError, match="persistent numeric storage"):
        rhf_hvp_many(state, directions[:1], total_budget_bytes=1)


def test_full_output_budget_rejects_without_partial_hessian(h2_case, monkeypatch):
    from tools.vibeqc_hessian import block

    state, _, _ = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized full Hessian request started a block")

    monkeypatch.setattr(block, "rhf_hvp_many", forbidden)
    output_bytes = (3 * state.nat) ** 2 * 8
    with pytest.raises(ValueError, match="full Hessian output"):
        rhf_hessian(state, total_budget_bytes=output_bytes)
