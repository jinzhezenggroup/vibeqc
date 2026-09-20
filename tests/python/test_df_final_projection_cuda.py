"""Exact-state final occupied-K projection reuse, with independent force gates."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_final_projection_replay_and_geometry(
    method: typing.Any, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """Only singleton RHF may lend U; warm and geometry replays bind fresh states."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    for name, value in {
        "VIBEQC_DF_EXCHANGE": "occupied",
        "VIBEQC_DF_FINAL_EXCHANGE": "occupied",
        "VIBEQC_DF_RESIDENT_EXCHANGE": "full",
        "VIBEQC_DF_RESPONSE_STORAGE": "jk-scratch",
        "VIBEQC_DF_RESPONSE_SPACE": "occupied",
        "VIBEQC_DF_FINAL_PROJECTION": "reuse",
    }.items():
        monkeypatch.setenv(name, value)
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch([case.atoms]) as batch:
        batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        for moved in (False, True):
            coordinates = np.array([r for _, r in case.atoms])
            coordinates[1, 0] += 0.001 if moved else 0
            atoms = [(z, r) for (z, _), r in zip(case.atoms, coordinates)]
            mol = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", verbose=0)
            oracle = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
                auxbasis="def2-svp"
            )
            oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
            oracle.kernel()
            assert oracle.converged
            expected = -oracle.nuc_grad_method().kernel()
            answers = []
            for policy in ("off", "reuse", "reuse"):
                monkeypatch.setenv("VIBEQC_DF_FINAL_PROJECTION", policy)
                trace = tmp_path / f"{moved}-{len(answers)}.jsonl"
                monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
                actual = batch.execute(
                    [coordinates] if moved else None, strict=True
                ).items[0]
                assert actual.energy == pytest.approx(oracle.e_tot, abs=1e-9, rel=0)
                np.testing.assert_allclose(actual.forces, expected, atol=1e-8, rtol=0)
                answers.append(np.asarray(actual.forces))
                rows = [
                    r for r in read_trace(trace) if r["operation"] == "force_response"
                ]
                assert len(rows) == 1
                reused = rows[0]["counters"].get("response_final_projection_reused", 0)
                if method == "uhf" or policy == "off":
                    assert reused == 0
                elif not moved:
                    assert reused == 1
            np.testing.assert_allclose(answers[0], answers[1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(answers[1], answers[2], atol=1e-10, rtol=0)
