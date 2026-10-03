"""RHF density-fitting source weights generated through StationaryProblem (#358)."""

import subprocess
import sys
import typing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method import DensityFittingRHFResponsePlan
from generativeqc_compiler.method.df_hf_response_contract import CONTRACT_IDENTITY
from generativeqc_compiler.method.df_hf_response_cuda import (
    df_rhf_charge_gemm_kind,
    emit_df_hf_response_contract,
    emit_df_hf_response_cuda,
)
from generativeqc_compiler.tensor import execute


def _fixture() -> typing.Any:
    rng = np.random.default_rng(358)
    n, a = 2, 3
    raw = rng.normal(size=(a, a))
    metric = raw @ raw.T + np.eye(a)
    three_center = rng.normal(size=(a, n, n))
    three_center = (three_center + three_center.transpose(0, 2, 1)) / 2
    density = rng.normal(size=(n, n))
    density = (density + density.T) / 2
    weighted = rng.normal(size=(n, n))
    weighted = (weighted + weighted.T) / 2
    one_electron = rng.normal(size=(n, n))
    overlap = rng.normal(size=(n, n))
    fitted = np.linalg.solve(metric, three_center.reshape(a, -1)).reshape(a, n, n)
    feeds = {
        "density": density,
        "weighted_density": weighted,
        "one_electron": one_electron,
        "overlap": overlap,
        "three_center": three_center,
        "metric": metric,
        "fitted": fitted,
    }
    return rng, DensityFittingRHFResponsePlan(n, a), feeds


def _weights(owner: typing.Any, feeds: typing.Any) -> typing.Any:
    plan = owner.compile()
    rhs = execute(plan.rhs, feeds).outputs["fitted"]
    metric = feeds["metric"]
    multiplier = np.linalg.solve(metric.T, rhs.reshape(owner.naux, -1)).reshape(
        owner.naux, owner.nbf, owner.nbf
    )
    result = execute(
        plan.partials,
        {**feeds, plan.multiplier_inputs["fitted"]: multiplier},
    ).outputs
    assert np.linalg.norm(result[plan.stationarity_outputs["fitted"]]) < 2e-13
    return plan, {name: result[output] for name, output in plan.weight_outputs.items()}


def _resolved_energy(owner: typing.Any, feeds: typing.Any) -> typing.Any:
    density = feeds["density"]
    weighted = feeds["weighted_density"]
    hamiltonian = feeds["one_electron"]
    overlap = feeds["overlap"]
    three_center = feeds["three_center"]
    metric = feeds["metric"]
    fitted = np.linalg.solve(metric, three_center.reshape(owner.naux, -1)).reshape(
        owner.naux, owner.nbf, owner.nbf
    )
    raw_charge = np.einsum("ij,pij->p", density, three_center)
    fitted_charge = np.einsum("ij,pij->p", density, fitted)
    return (
        np.einsum("ij,ij->", density, hamiltonian)
        - np.einsum("ij,ij->", weighted, overlap)
        + float(owner.coulomb_coefficient) * 0.5 * np.dot(raw_charge, fitted_charge)
        - float(owner.exchange_coefficient)
        * np.einsum("pij,ki,pkl,lj->", three_center, density, fitted, density)
    )


def test_common_stationary_plan_matches_pre_migration_rhf_df_algebra() -> None:
    _, owner, feeds = _fixture()
    assert owner.coulomb_coefficient == Fraction(1, 1)
    assert owner.exchange_coefficient == Fraction(1, 4)
    plan, weights = _weights(owner, feeds)
    density = feeds["density"]
    fitted = feeds["fitted"]
    raw_charge = np.einsum("ij,pij->p", density, feeds["three_center"])
    potential = np.linalg.solve(feeds["metric"], raw_charge)
    exchange_response = np.einsum("ki,pkl,lj->pij", density, fitted, density)

    legacy_three_center = (
        density[None, :, :] * potential[:, None, None]
        - 2 * float(owner.exchange_coefficient) * exchange_response
    )
    legacy_metric = -0.5 * float(owner.coulomb_coefficient) * np.outer(
        potential, potential
    ) + float(owner.exchange_coefficient) * np.einsum(
        "pij,qij->pq", fitted, exchange_response
    )

    np.testing.assert_allclose(weights["one_electron"], density, atol=2e-14)
    np.testing.assert_allclose(
        weights["overlap"], -feeds["weighted_density"], atol=2e-14
    )
    np.testing.assert_allclose(
        weights["three_center"], legacy_three_center, atol=3e-14, rtol=3e-14
    )
    np.testing.assert_allclose(weights["metric"], legacy_metric, atol=3e-14, rtol=3e-14)

    graph = plan.problem.dependency_graph
    assert graph["implicit_region"]["states"] == ["fitted"]
    assert graph["providers"]["three_center"] == ["geometry"]
    assert graph["providers"]["metric"] == ["geometry"]
    assert set(plan.weight_outputs) == {
        "one_electron",
        "overlap",
        "three_center",
        "metric",
    }


def test_generated_source_weights_match_resolved_finite_differences() -> None:
    rng, owner, feeds = _fixture()
    _, weights = _weights(owner, feeds)
    directions = {
        "one_electron": rng.normal(size=(owner.nbf, owner.nbf)),
        "overlap": rng.normal(size=(owner.nbf, owner.nbf)),
        "three_center": rng.normal(size=(owner.naux, owner.nbf, owner.nbf)),
        "metric": rng.normal(size=(owner.naux, owner.naux)),
    }
    directions["three_center"] = (
        directions["three_center"] + directions["three_center"].transpose(0, 2, 1)
    ) / 2
    directions["metric"] = (directions["metric"] + directions["metric"].T) / 2

    for name, direction in directions.items():
        analytic = np.vdot(weights[name], direction)
        errors = []
        for step in (2e-4, 7e-5, 2e-5):
            plus = {**feeds, name: feeds[name] + step * direction}
            minus = {**feeds, name: feeds[name] - step * direction}
            finite = (
                _resolved_energy(owner, plus) - _resolved_energy(owner, minus)
            ) / (2 * step)
            errors.append(abs(finite - analytic))
        # Linear h/S directions can hit FP64 roundoff before the smallest
        # step, so the gate is absolute after the central-difference sweep.
        assert min(errors) < 2e-8
        assert max(errors) < 2e-7


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("density_scale", [0.5, 2.0])
def test_rooted_final_projection_recovers_fitted_rhf_potential(
    symmetric: bool, density_scale: float
) -> None:
    """Rooted trace recovers X(D:B) across occupied gauge and B asymmetry."""
    rng = np.random.default_rng(1690 + 10 * int(symmetric) + int(2 * density_scale))
    n, r, a = 9, 4, 7
    density_coefficients = np.linalg.qr(rng.normal(size=(n, r)))[0]
    occupied_rotation = np.linalg.qr(rng.normal(size=(r, r)))[0]
    final_coefficients = density_coefficients @ occupied_rotation
    fitted = rng.normal(size=(a, n, n))
    if symmetric:
        fitted = 0.5 * (fitted + fitted.transpose(0, 2, 1))
    metric_rotation = np.linalg.qr(rng.normal(size=(a, a)))[0]
    metric_eigenvalues = np.geomspace(0.7, 3.1, a)
    inverse_root = (metric_rotation / np.sqrt(metric_eigenvalues)) @ metric_rotation.T
    density = density_scale * density_coefficients @ density_coefficients.T
    direct_charge = np.einsum("mn,qmn->q", density, fitted)
    direct_potential = inverse_root @ direct_charge
    projected = np.einsum(
        "mi,qmn,nj->qij", final_coefficients, fitted, final_coefficients
    )
    rooted = np.einsum("pq,qij->pij", inverse_root, projected)
    reused_potential = density_scale * np.trace(rooted, axis1=1, axis2=2)
    np.testing.assert_allclose(
        reused_potential, direct_potential, atol=4e-13, rtol=4e-13
    )


def test_final_projection_potential_reuse_avoids_retained_fitted_charge_pass() -> None:
    """The qualified route roots S once and never rereads B for Coulomb charge."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_response_weights.cu").read_text()
    start = source.index("if (reuse_final_fitted_projection) {")
    stop = source.index(
        "} else if (fitted_occupied || buffers.read_occupied_panels)", start
    )
    reuse = source[start:stop]
    assert "df_occupied_finish_projection" in reuse
    assert "df_occupied_apply_metric_root" in reuse
    assert "df_rhf_potential_from_rooted_projection" in reuse
    assert "response_final_fitted_potential_reused" in reuse
    assert "response_occupied_charge_inverse_gemms_avoided" in reuse
    assert "launch_unpack_df_values" not in reuse
    assert "gather_final_fitted_projection" in reuse
    assert "df_rhf_charge_contract" not in reuse
    assert source.count("generated::df_occupied_finish_projection(") == 1
    exchange = source[source.index("exchange_response_occupied_weight_gemm") :]
    retained_start = exchange.index("if (retained_root) {")
    retained_stop = exchange.index("} else if (fitted_occupied)", retained_start)
    retained = exchange[retained_start:retained_stop]
    assert "if (reuse_final_fitted_projection)" in retained
    assert "U=X*S was formed" in retained


def test_metric_custom_rule_is_explicit_fixed_rank_pseudoinverse() -> None:
    owner = DensityFittingRHFResponsePlan(2, 3)
    rule = owner.metric_rule(0.1)
    assert rule.function == "pseudoinverse"
    state = rule.prepare(np.diag([0.02, 1.0, 3.0]))
    assert state.rank == 2
    tangent = np.array([[0.0, 0.3, -0.1], [0.3, 0.2, 0.4], [-0.1, 0.4, -0.2]])
    assert np.linalg.norm(state.jvp(tangent)[:1, 1:]) > 0


def test_production_native_lowering_is_bound_to_stationary_plan() -> None:
    owner = DensityFittingRHFResponsePlan(1, 1)
    plan = owner.compile(max_elements=256)
    contract = emit_df_hf_response_contract()
    cuda = emit_df_hf_response_cuda()
    assert CONTRACT_IDENTITY in plan.problem.model_identity
    assert CONTRACT_IDENTITY in contract and CONTRACT_IDENTITY in cuda
    assert "df_rhf_exchange_coefficient =\n    0.25;" in contract
    assert df_rhf_charge_gemm_kind() == "direct-NT"
    assert "charge-contraction: tij,pij->tp" in cuda
    assert "tensorir-charge-lowering: direct-NT" in cuda
    assert "df_rhf_charge_contract" in cuda
    assert "df_rhf_potential_from_rooted_projection" in cuda
    assert "rooted[q * stride + i * diagonal_stride]" in cuda
    assert "symmetric_pairs ? 1 : rank + 1" in cuda
    assert "potentials[q] = density_scale * value" in cuda
    assert "cublasDgemm" in cuda
    for kernel in (
        "coulomb_weights_kernel",
        "coulomb_metric_kernel",
        "exchange_weights_kernel",
        "exchange_metric_kernel",
        "fitted_exchange_weights_kernel",
    ):
        assert kernel in cuda


def test_production_lowering_codegen_import_is_dependency_light(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    cuda = tmp_path / "generated_df_hf_response.cuh"
    contract = tmp_path / "generated_df_hf_response_contract.hpp"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools" / "generate_df_hf_response.py"),
            "--cuda-output",
            str(cuda),
            "--contract-output",
            str(contract),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tensorir-charge-lowering: direct-NT" in cuda.read_text()
    assert "df_rhf_exchange_coefficient" in contract.read_text()
