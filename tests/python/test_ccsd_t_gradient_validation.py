"""Independent PySCF/finite-difference RCCSD(T) gradient acceptance for #155."""

from __future__ import annotations

import numpy as np
import pytest

from tools.generate_cc_triples_references import GROUND_TRUTH, _triples_feeds
from tools.validate_ccsd_t_gradient import (
    FD_STEPS,
    analytic_oracle,
    finite_difference,
)
from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_cc.triples_response import full_triples_vjp


@pytest.mark.parametrize("name", ("h2o", "nh3"))
def test_pinned_pyscf_ccsdt_oracle_has_nontrivial_triples_force(name: str) -> None:
    record = analytic_oracle(name)
    analytic = record["analytic"]
    gradient = np.asarray(analytic["gradient"])
    ccsd_gradient = np.asarray(analytic["ccsd_gradient"])
    triples_gradient = np.asarray(analytic["triples_gradient"])

    assert record["pyscf"] == "2.14.0"
    assert analytic["scf_converged"]
    assert analytic["ccsd_converged"]
    assert analytic["ccsd_lambda_converged"]
    assert analytic["ccsdt_lambda_converged"]
    np.testing.assert_allclose(
        analytic["triples_energy"],
        GROUND_TRUTH[name][2],
        atol=1e-9,
        rtol=0,
    )
    np.testing.assert_allclose(
        gradient - ccsd_gradient,
        triples_gradient,
        atol=0,
        rtol=0,
    )
    assert np.max(np.abs(triples_gradient)) > 1e-8
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-8, rtol=0)
    assert analytic["scf_commutator_max"] < 1e-8
    assert analytic["ccsd_amplitude_update_max"] < 1e-8
    assert analytic["canonical_fock_offdiag_max"] < 1e-9
    assert set(analytic["source_sha256"]) == {
        "pyscf.cc.ccsd_t",
        "pyscf.cc.ccsd_t_lambda",
        "pyscf.grad.ccsd",
        "pyscf.grad.ccsd_t",
    }


def test_complete_ccsdt_energy_matches_three_step_reconverged_finite_difference() -> (
    None
):
    fd = finite_difference("h2o")
    assert tuple(point["step_bohr"] for point in fd["points"]) == FD_STEPS
    assert fd["fresh_displaced_geometries"] == 2 * len(FD_STEPS)
    assert min(fd["errors"][1:]) < 1e-6, fd["errors"]
    assert fd["errors"][-1] < 2e-6, fd["errors"]


def test_delta_lambda_omission_is_detectable_on_physical_water() -> None:
    analytic = analytic_oracle("h2o")["analytic"]
    correct = np.asarray(analytic["gradient"])
    omitted = np.asarray(analytic["without_delta_lambda_gradient"])
    delta = np.asarray(analytic["delta_lambda_gradient"])
    np.testing.assert_allclose(correct - omitted, delta, atol=2e-12, rtol=2e-12)
    assert np.max(np.abs(delta)) > 1e-10


@pytest.mark.parametrize("name", ("h2o", "nh3"))
def test_denominator_response_is_nonzero_on_physical_ccsdt_cases(name: str) -> None:
    feeds = _triples_feeds(name)
    nocc, nvir, arrays = feeds[0], feeds[1], feeds[2:]
    response = full_triples_vjp(
        nocc,
        nvir,
        *arrays,
        inputs=("eps_o", "eps_v"),
    )
    assert np.linalg.norm(response["eps_o"]) > 1e-10
    assert np.linalg.norm(response["eps_v"]) > 1e-10

    rng = np.random.default_rng(15520 + ("h2o", "nh3").index(name))
    d_occ = rng.normal(size=response["eps_o"].shape)
    d_vir = rng.normal(size=response["eps_v"].shape)
    scale = max(1.0, float(np.linalg.norm(d_occ)), float(np.linalg.norm(d_vir)))
    d_occ /= scale
    d_vir /= scale
    analytic = float(
        np.vdot(response["eps_o"], d_occ) + np.vdot(response["eps_v"], d_vir)
    )
    assert abs(analytic) > 1e-10

    arrays = [np.array(value, copy=True) for value in arrays]
    for step in (2e-5, 2e-6):
        samples = []
        for sign in (-1.0, 1.0):
            changed = [np.array(value, copy=True) for value in arrays]
            changed[-2] += sign * step * d_occ
            changed[-1] += sign * step * d_vir
            samples.append(triples_energy(nocc, nvir, *changed))
        finite = (samples[1] - samples[0]) / (2.0 * step)
        np.testing.assert_allclose(analytic, finite, atol=2e-7, rtol=2e-6)


def test_ccsdt_oracle_is_not_the_runtime_force_endpoint() -> None:
    record = analytic_oracle("h2o")
    assert record["restrictions"] == {
        "reference": "canonical RHF",
        "frozen_core": 0,
        "density_fitting": False,
        "open_shell": False,
    }
    # The validator is intentionally independent evidence. It must not claim
    # that VibeQC's public RCCSD(T) force capability is already available.
    assert "vibeqc" not in record["analytic"]["source_sha256"]
