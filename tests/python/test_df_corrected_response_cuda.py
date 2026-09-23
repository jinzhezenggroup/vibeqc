"""A corrected final density cannot reuse stale SCF factors for streamed forces."""

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


@pytest.mark.parametrize("response_space", ["auto", "occupied"])
@pytest.mark.parametrize("case_name", ["water-tetramer", "ammonia-trimer"])
def test_corrected_streamed_response_preserves_force_oracle(
    monkeypatch: typing.Any, tmp_path: typing.Any, response_space: str, case_name: str
) -> None:
    """Gate corrected factor replay on streamed water and non-water workloads."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    if case_name == "water-tetramer":
        atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    else:
        ammonia = benchmark_cases()["ammonia-def2-svp-spherical"].atoms
        atoms = tuple(
            (symbol, tuple(np.asarray(position) + offset))
            for offset in ((0, 0, 0), (8, 0, 0), (0, 8, 0))
            for symbol, position in ammonia
        )
    moved = [
        (
            symbol,
            tuple(np.asarray(position) + (0, 0, 0.025) if index == 1 else position),
        )
        for index, (symbol, position) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        molecule = gto.M(
            atom=geometry, unit="Bohr", basis="def2-svp", cart=False, verbose=0
        )
        reference = scf.RHF(molecule).density_fit(auxbasis="def2-svp")
        reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = (
            1e-12,
            1e-10,
            100,
        )
        reference.kernel()
        assert reference.converged
        references.append((reference.e_tot, -reference.nuc_grad_method().kernel()))
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", response_space)
    monkeypatch.setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", "1")
    monkeypatch.setenv("VIBEQC_DF_FINAL_EXCHANGE", "auto")
    calculator = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=(
            24 if case_name == "water-tetramer" else 16
        )
        << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calculator.prepare_batch([atoms]) as batch:
        for phase, coordinates, reference in (
            ("cold", None, references[0]),
            ("warm", None, references[0]),
            ("changed", [np.asarray([xyz for _, xyz in moved])], references[1]),
        ):
            trace = tmp_path / f"{phase}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(
                coordinates, properties=("energy", "forces"), strict=True
            ).items[0]
            assert abs(result.energy - reference[0]) < 1e-9
            assert float(np.max(np.abs(result.forces - reference[1]))) < 1e-8
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            assert any(
                record["operation"] == "ri_j" and record["streamed"]
                for record in records
            )
            assert (
                sum(
                    record["operation"] == "final_state_physical_fock"
                    for record in records
                )
                >= 2
            )
            corrected = [
                record
                for record in records
                if record["operation"] == "corrected_response_factor"
            ]
            if response_space == "occupied":
                assert corrected
                assert all(record["counters"].get("accepted") for record in corrected)
            else:
                assert not corrected
            force = [
                record for record in records if record["operation"] == "force_response"
            ]
            assert force
            assert all(
                not record["counters"].get("response_final_projection_reused", 0)
                for record in force
            )
            if response_space == "occupied":
                assert all(
                    record["counters"].get("response_occupied_projection_products", 0)
                    for record in force
                )
                assert all(
                    not record["counters"].get("response_ao_matrix_products", 0)
                    for record in force
                )
            else:
                assert all(
                    not record["counters"].get(
                        "response_occupied_projection_products", 0
                    )
                    for record in force
                )
                assert all(
                    record["counters"].get("response_ao_matrix_products", 0)
                    for record in force
                )
