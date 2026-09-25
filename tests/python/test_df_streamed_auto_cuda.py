"""Constrained-memory SCF uses exact streamed factors without response borrowing."""

import json
import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("forces,budget_mib", [(False, 16), (True, 24)])
@pytest.mark.parametrize(
    ("shared_policy", "expect_shared"),
    [(None, True), ("auto", True), ("1", True), ("0", False)],
)
@pytest.mark.parametrize("final_exchange", [None, "auto", "occupied"])
def test_streamed_auto_cold_warm_and_changed_geometry(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    forces: bool,
    budget_mib: int,
    shared_policy: str | None,
    expect_shared: bool,
    final_exchange: str | None,
) -> None:
    """Independent libcint/PySCF gates cover all solves and geometry rebuilding."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = [
        (symbol, tuple(np.array(xyz) + (0, 0, 0.025) if i == 1 else xyz))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis="def2-svp")
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        references.append(
            (mf.e_tot, -mf.nuc_grad_method().kernel() if forces else None)
        )
    for control in (
        "EXCHANGE",
        "SEED_EXCHANGE",
        "RESPONSE_SPACE",
        "RESPONSE_STORAGE",
        "FINAL_PROJECTION",
    ):
        monkeypatch.setenv("VIBEQC_DF_" + control, "auto")
    if final_exchange is None:
        monkeypatch.delenv("VIBEQC_DF_FINAL_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_FINAL_EXCHANGE", final_exchange)
    if shared_policy is None:
        monkeypatch.delenv("VIBEQC_DF_JK_SHARED_SOURCE", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_JK_SHARED_SOURCE", shared_policy)
    calculator = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget_mib << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    properties = ("energy", "forces") if forces else ("energy",)
    rows = []
    with calculator.prepare_batch([atoms]) as batch:
        for phase, geometry, reference in (
            ("cold", None, references[0]),
            ("warm", None, references[0]),
            ("changed", [np.array([xyz for _, xyz in moved])], references[1]),
        ):
            trace = tmp_path / f"{phase}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(geometry, properties=properties, strict=True).items[
                0
            ]
            energy_error = abs(result.energy - reference[0])
            force_error = (
                float(np.max(np.abs(result.forces - reference[1]))) if forces else None
            )
            rows.append(
                {
                    "phase": phase,
                    "energy_error": energy_error,
                    "force_error": force_error,
                    "iterations": result.iterations,
                }
            )
            (tmp_path / "errors.json").write_text(json.dumps(rows, indent=2) + "\n")
            assert energy_error < 1e-9
            if forces:
                assert force_error < 1e-8
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            assert any(
                record["counters"].get("streamed_occupied_source_first")
                for record in records
            )
            joint = [
                record for record in records if record["operation"] == "ri_jk_shared"
            ]
            if expect_shared:
                assert joint
                for record in joint:
                    assert record["counters"]["shared_coulomb_charge_rows"] == 96
                    assert record["counters"]["shared_raw_source_values"] == (
                        96 * 96 * record["naux"]
                    )
            else:
                assert not joint
            retained_final = [
                record
                for record in records
                if record["operation"] == "final_state_retained_jk"
            ]
            assert retained_final
            assert all(record["counters"].get("accepted") for record in retained_final)
            if forces:
                # Temporary eigenbasis projections never become a resident
                # symmetric-C force-response lease, even after warm SCF.
                response = [r for r in records if r["operation"] == "force_response"]
                assert response
                for record in response:
                    counts = record["counters"]
                    assert not counts.get("response_final_projection_reused", 0)
                    if counts.get("response_occupied_projection_products", 0):
                        # Existing streamed response may use independently
                        # validated canonical C, with its own charged U. That
                        # is different from borrowing private value projections.
                        assert counts["response_owned_occupied_projection_bytes"] > 0
