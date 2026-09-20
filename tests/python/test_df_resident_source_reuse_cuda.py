"""Resident forward DF values must survive full/partial response panel selection."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(
    scope="module",
    params=["water-tetramer-def2-svp-spherical", "oh-def2-svp-spherical-uhf"],
)
def reference(request: Any) -> Any:
    from pyscf import gto, scf

    case = benchmark_cases()[request.param]
    coordinates = np.array([r for _, r in case.atoms])
    moved = coordinates.copy()
    moved[1, 0] += 0.001
    records = []
    for xyz in (coordinates, moved):
        atoms = [(z, r) for (z, _), r in zip(case.atoms, xyz, strict=True)]
        mol = gto.M(
            atom=atoms,
            basis=case.pyscf_basis,
            unit="Bohr",
            cart=False,
            charge=case.charge,
            spin=case.multiplicity - 1,
            verbose=0,
        )
        mf = (scf.UHF if case.method == "uhf" else scf.RHF)(mol).density_fit(
            auxbasis=case.pyscf_basis
        )
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        records.append((mf.e_tot, -mf.nuc_grad_method().kernel()))
    return case, moved, records


@pytest.mark.parametrize("response_budget", [None, 4 << 20])
def test_resident_source_reuses_values_for_full_and_partial_force_panels(
    reference: Any,
    response_budget: int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Independent forces and work counters protect geometry/property replay."""
    assert os.environ.get("SLURM_JOB_ID")
    case, moved, records = reference
    for key in list(os.environ):
        if key.startswith("VIBEQC_DF_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
    if response_budget is not None:
        monkeypatch.setenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES", str(response_budget))
    calc = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    with calc.prepare_batch(
        [case.atoms],
        charges=[case.charge],
        multiplicities=[case.multiplicity],
        warm_start=True,
    ) as batch:
        batch.execute(strict=True, properties=("energy", "forces"))
        batch.set_warm_start_updates(False)
        batch.execute(strict=True, properties=("energy", "forces"))
        for step, changed in enumerate((False, True, False)):
            path = tmp_path / f"response-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(path))
            coordinates = moved if changed else np.array([r for _, r in case.atoms])
            item = batch.execute(
                [coordinates], strict=True, properties=("energy", "forces")
            ).items[0]
            energy, forces = records[int(changed)]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            np.testing.assert_allclose(item.forces, forces, atol=1e-8, rtol=0)
            (response,) = [
                r for r in read_trace(path) if r["operation"] == "force_response"
            ]
            counters = response["counters"]
            assert response["source_backed"]
            assert counters.get("raw_tile_productions", 0) == 0
            assert counters["response_borrowed_whitened_bytes"] > 0
            assert counters["response_fitted_panel_calls"] >= 1
            assert counters["response_metric_blas_gemms"] >= 1
            if response_budget is not None:
                assert counters["response_scratch_bytes"] <= response_budget
            if case.expected_ao_count == 96:
                assert counters["three_center_shell_panels"] >= 1
                assert (counters["response_auxiliary_blocks"] == 1) == (
                    response_budget is None
                )
            monkeypatch.delenv("VIBEQC_DF_TRACE")
            if response_budget is None:
                # Each replay's coordinates are explicit; omitting them uses
                # the prepared geometry, not the prior call's displaced frame.
                energy_only = batch.execute(
                    [coordinates], strict=True, properties=("energy",)
                ).items[0]
                assert energy_only.energy == pytest.approx(energy, abs=1e-9, rel=0)
                assert energy_only.forces is None
