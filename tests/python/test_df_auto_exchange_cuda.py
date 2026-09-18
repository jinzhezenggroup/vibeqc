"""Automatic SCF/response work selection preserves independent molecular forces."""

import json
import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    "case_name",
    [
        "water-tetramer-def2-svp-spherical",
        "water-octamer-s4-def2-svp-spherical",
        "water-hexadecamer-2s4-def2-svp-spherical",
        "water-32mer-4s4-def2-svp-spherical",
    ],
)
def test_automatic_cold_and_warm_force_matches_independent_oracle(
    monkeypatch, tmp_path, case_name
):
    """96/192/384/768 AO are validation points, with executed provenance required."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()[case_name]
    mol = gto.M(atom=case.atoms, unit="Bohr", basis="def2-svp", verbose=0)
    oracle = scf.RHF(mol).density_fit(auxbasis="def2-svp")
    # Match the independent SCF acceptance used by the existing force suite;
    # absolute energy steps below the 768-AO total-energy ULP are not meaningful.
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
    oracle.kernel()
    assert oracle.converged
    expected_force = -oracle.nuc_grad_method().kernel()
    np.savez(tmp_path / "oracle.npz", energy=oracle.e_tot, forces=expected_force)
    for control in (
        "EXCHANGE",
        "SEED_EXCHANGE",
        "FINAL_EXCHANGE",
        "RESPONSE_SPACE",
        "RESPONSE_STORAGE",
        "FINAL_PROJECTION",
    ):
        monkeypatch.setenv("VIBEQC_DF_" + control, "auto")
    calc = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    results = []
    with calc.prepare_batch([case.atoms]) as batch:
        for phase in ("cold", "warm"):
            trace = tmp_path / f"{phase}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            actual = batch.execute(strict=True).items[0]
            results.append(
                {
                    "phase": phase,
                    "energy_error": abs(actual.energy - oracle.e_tot),
                    "force_error": float(
                        np.max(np.abs(actual.forces - expected_force))
                    ),
                    "iterations": actual.iterations,
                }
            )
            (tmp_path / "errors.json").write_text(json.dumps(results, indent=2) + "\n")
            assert results[-1]["energy_error"] < 1e-9
            assert results[-1]["force_error"] < 1e-8
            records = read_trace(trace)
            assert any(r["operation"] == "occupied_scf_provenance" for r in records)
            if phase == "warm":
                (response,) = [r for r in records if r["operation"] == "force_response"]
                assert response["counters"]["response_occupied_projection_products"] > 0
                assert not response["counters"].get("response_ao_matrix_products", 0)
