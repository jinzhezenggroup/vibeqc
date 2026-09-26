"""Qualified warm frames save SCF work without skipping physical force checks."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("auxiliary", ["def2-svp", "cc-pvdz-jkfit"])
def test_one_step_warm_force_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frozen: bool, auxiliary: str
) -> None:
    """Frozen/latest entries survive repeats; geometry and disabled reuse miss."""
    assert os.environ.get("SLURM_JOB_ID")
    from pyscf import gto, scf
    from vibeqc import Calculator

    from benchmarks._cases import benchmark_cases
    from benchmarks.df_component_ledger import read_trace

    atoms = benchmark_cases()["water-tetramer-def2-svp-spherical"].atoms
    moved = [
        (z, tuple(np.asarray(x) + ((0, 0, 0.001) if i == 1 else (0, 0, 0))))
        for i, (z, x) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=auxiliary)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        gradient = mf.nuc_grad_method()
        gradient.auxbasis_response = True
        references.append((mf.e_tot, -gradient.kernel()))
    for key in list(os.environ):
        if key.startswith("VIBEQC_DF_"):
            monkeypatch.delenv(key)
    for key, value in {
        "VALUE_STORAGE": "packed-single",
        "EXCHANGE": "occupied",
        "FINAL_EXCHANGE": "auto",
        "RESPONSE_SPACE": "occupied",
        "OCCUPIED_RESPONSE_SOURCE": "fitted",
        "SOURCE_DERIVATIVE_SCHEDULE": "qualify",
        "RESPONSE_ALGEBRA": "blas",
        "RESPONSE_BUDGET_BYTES": str(64 << 20),
    }.items():
        monkeypatch.setenv("VIBEQC_DF_" + key, value)
    record = (
        auxiliary
        if auxiliary == "def2-svp"
        else Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    calc = Calculator(
        device="cuda",
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        auxiliary_basis=record,
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch([atoms], warm_start=True) as owner:

        def execute(label: str, geometry: int = 0, *, reused: bool = False) -> None:
            trace = tmp_path / (label + ".jsonl")
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            item = owner.execute(
                [np.asarray([x for _, x in moved])] if geometry else None, strict=True
            ).items[0]
            energy, forces = references[geometry]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            np.testing.assert_allclose(item.forces, forces, atol=1e-8, rtol=0)
            rows = read_trace(trace)
            provenance = [
                r for r in rows if r["operation"] == "occupied_scf_provenance"
            ]
            assert len(provenance) == 1
            assert provenance[0]["counters"]["warm_seed_iterations"] == int(reused)
            if reused:
                assert item.iterations == 1
                assert not any(r["operation"] == "density_exchange_seed" for r in rows)
                # One actual SCF K plus one separately validated final K.
                assert (
                    sum(
                        r["operation"] == "ri_k_occupied" and r["execution"] == "stream"
                        for r in rows
                    )
                    == 2
                )
            else:
                assert item.iterations > 1

        execute("cold")
        owner.set_warm_start_updates(not frozen)
        for repeat in range(3):
            execute(f"warm-{repeat}", reused=True)
        owner.set_warm_start_updates(True)
        execute("moved", geometry=1)
        owner.set_warm_start_updates(not frozen)
        for repeat in range(3):
            execute(f"moved-warm-{repeat}", geometry=1, reused=True)
        monkeypatch.setenv("VIBEQC_DF_WARM_REUSE", "0")
        execute("disabled", geometry=1)
        monkeypatch.delenv("VIBEQC_DF_WARM_REUSE")
        # Disabling reuse revokes the old record; one complete ordinary solve
        # republishes a qualified frame, rather than resurrecting stale state.
        owner.set_warm_start_updates(True)
        execute("requalify", geometry=1)
        owner.set_warm_start_updates(not frozen)
        execute("requalified-warm", geometry=1, reused=True)
