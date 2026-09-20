"""The #180 block consumer uses the same resident #179 multi-RHS solver."""

import os
import typing

import numpy as np
import pytest
from test_hessian_block import h2_case as _h2_case

from tools.vibeqc_hessian import rhf_hvp_many
from tools.vibeqc_validation.hessian_fixtures import oracle_analytic_hessian

# Native dense assembly checks parity; it shares the response solver under test.
h2_case = _h2_case

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


@pytest.fixture(scope="module")
def h2_external_hessian(h2_case: typing.Any) -> typing.Any:
    """Build the external oracle with native Hessian/response seams forbidden."""
    pytest.importorskip("pyscf")
    from tools.vibeqc_hessian import analytic, perturbation
    from tools.vibeqc_response import krylov

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("external Hessian oracle called native Hessian/response code")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(analytic, "analytic_hessian", forbidden)
        for module in (perturbation, krylov):
            patch.setattr(module, "solve", forbidden)
            patch.setattr(module, "solve_many", forbidden)
        return oracle_analytic_hessian(h2_case[0].source)


@pytest.mark.parametrize("strategy", ("sequential", "blocked", "recycled"))
def test_complete_hvp_block_uses_resident_shared_solve(
    h2_case: typing.Any,
    h2_external_hessian: typing.Any,
    strategy: str,
    monkeypatch: typing.Any,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from tools.vibeqc_hessian import perturbation

    state, dense, directions = h2_case
    original = perturbation.solve_many
    calls = []

    def solve_many(
        operator: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        assert operator._krylov_engine.resident
        assert kwargs["collect_basis"] is False
        calls.append(kwargs["strategy"])

        def forbidden(*args: typing.Any) -> None:
            pytest.fail("nuclear multi-RHS solve used host AO/MO actions")

        with monkeypatch.context() as patch:
            patch.setattr(operator, "apply", forbidden)
            return original(operator, *args, **kwargs)

    monkeypatch.setattr(perturbation, "solve_many", solve_many)
    actual = rhf_hvp_many(
        state,
        directions,
        strategy=strategy,
        jk_backend="cuda",
        response_execution="cuda-resident",
    )
    expected = np.stack([np.einsum("abxy,by->ax", dense, v) for v in directions])
    np.testing.assert_allclose(actual.values, expected, atol=1e-9, rtol=4e-10)
    external = np.stack(
        [np.einsum("abxy,by->ax", h2_external_hessian, v) for v in directions]
    )
    np.testing.assert_allclose(actual.values, external, atol=1e-9, rtol=0)
    assert calls == [strategy]
    diag = actual.diagnostics
    assert diag["response_execution"] == "cuda-resident"
    assert (
        diag["resident_response"]["operator_actions"]
        == diag["response_operator_actions"]
    )
    assert (
        diag["retained_response_device_bytes"] <= diag["response_device_budget_bytes"]
    )
    assert diag["complete_numeric_peak_bound_bytes"] <= diag["total_budget_bytes"]
    assert diag["response_phase_numeric_bound_bytes"] >= (
        diag["retained_response_device_bytes"] + diag["response_peak_workspace_bytes"]
    )


def test_resident_consumer_budget_rejection_precedes_first_sources(
    h2_case: typing.Any, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from tools.vibeqc_hessian import block

    state, _, directions = h2_case

    def forbidden(*args: typing.Any) -> None:
        pytest.fail("impossible response budget reached first-integral work")

    monkeypatch.setattr(block, "generated_directional_first_order", forbidden)
    with pytest.raises((ValueError, MemoryError)):
        rhf_hvp_many(
            state,
            directions,
            jk_backend="cuda",
            response_execution="cuda-resident",
            response_device_budget_bytes=1,
        )
    state.validate()
