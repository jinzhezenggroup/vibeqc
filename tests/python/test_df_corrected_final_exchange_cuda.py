"""Corrected streamed RHF final K reconstructs only qualified factors."""

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


@pytest.mark.parametrize(
    ("final_exchange", "expect_corrected"),
    [("auto", True), ("occupied", True), ("dense", False)],
)
def test_corrected_streamed_final_exchange_has_strict_provenance(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    final_exchange: str,
    expect_corrected: bool,
) -> None:
    """Forced correction uses algebraic occupied K only when qualification allows it."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = [
        (symbol, tuple(np.asarray(xyz) + (0, 0, 0.025) if i == 1 else xyz))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis="def2-svp")
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        references.append(mf.e_tot)

    for control in (
        "EXCHANGE",
        "SEED_EXCHANGE",
        "RESPONSE_SPACE",
        "RESPONSE_STORAGE",
        "FINAL_PROJECTION",
    ):
        monkeypatch.setenv("VIBEQC_DF_" + control, "auto")
    monkeypatch.setenv("VIBEQC_DF_FINAL_EXCHANGE", final_exchange)
    monkeypatch.setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", "1")

    calculator = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=16 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calculator.prepare_batch([atoms]) as batch:
        for phase, geometry, reference in (
            ("cold", None, references[0]),
            ("warm", None, references[0]),
            ("changed", [np.asarray([xyz for _, xyz in moved])], references[1]),
        ):
            trace = tmp_path / f"{phase}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(geometry, properties=("energy",), strict=True).items[
                0
            ]
            assert abs(result.energy - reference) < 1e-9

            records = [json.loads(line) for line in trace.read_text().splitlines()]
            assert (
                sum(r["operation"] == "final_state_physical_fock" for r in records) >= 2
            )
            corrected = [
                r for r in records if r["operation"] == "final_state_corrected_jk"
            ]
            if expect_corrected:
                assert corrected
                assert all(r["counters"].get("accepted") for r in corrected)
                assert all(
                    1 <= r["counters"].get("correction_generations", 0) <= 16
                    for r in corrected
                )
            else:
                assert not corrected
