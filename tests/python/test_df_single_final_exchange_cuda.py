"""Single-B final K and fitted response preserve independent molecular forces."""

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks._retained_basis import load_retained_basis

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("auxiliary", ["def2-svp", "cc-pvdz-jkfit"])
@pytest.mark.parametrize(
    "final_exchange,metric,corrected,projection",
    [
        ("auto", "auto", False, "auto"),
        ("auto", "auto", False, "off"),
        ("occupied", "spectral", False, "auto"),
        ("auto", "auto", True, "auto"),
        ("dense", "spectral", True, "auto"),
    ],
)
def test_single_final_k_and_response_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auxiliary: str,
    final_exchange: str,
    metric: str,
    corrected: bool,
    projection: str,
) -> None:
    """Final K accepts exact/corrected factors without lending an absent raw owner."""
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator
    from pyscf import gto, scf

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
        references.append((mf.e_tot, -mf.nuc_grad_method().kernel()))

    for key in list(os.environ):
        if key.startswith("GENERATIVEQC_DF_"):
            monkeypatch.delenv(key)
    for key, value in {
        "VALUE_STORAGE": "packed-single",
        "EXCHANGE": "occupied",
        "FINAL_EXCHANGE": final_exchange,
        "FINAL_PROJECTION": projection,
        "RESPONSE_SPACE": "occupied",
        "OCCUPIED_RESPONSE_SOURCE": "fitted",
        "OCCUPIED_METRIC": metric,
        "SOURCE_DERIVATIVE_SCHEDULE": "qualify",
        "RESPONSE_ALGEBRA": "blas",
        "RESPONSE_BUDGET_BYTES": str(64 << 20),
    }.items():
        monkeypatch.setenv("GENERATIVEQC_DF_" + key, value)
    if corrected:
        monkeypatch.setenv("GENERATIVEQC_DF_FORCE_FINAL_REBUILD", "1")
    record = (
        auxiliary
        if auxiliary == "def2-svp"
        else load_retained_basis(
            Path(__file__).resolve().parents[2]
            / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
        )
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
    observed_reuse = False
    with calc.prepare_batch([atoms], warm_start=True) as owner:
        for phase, geometry in enumerate((None, None, moved, moved)):
            trace = tmp_path / f"phase-{phase}.jsonl"
            monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
            coords = (
                None if geometry is None else [np.asarray([x for _, x in geometry])]
            )
            item = owner.execute(coords, strict=True).items[0]
            energy, force = references[int(phase >= 2)]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            np.testing.assert_allclose(item.forces, force, atol=1e-8, rtol=0)
            rows = read_trace(trace)
            accepted = [
                r
                for r in rows
                if r["operation"]
                in ("final_state_retained_jk", "final_state_corrected_jk")
                and r["counters"].get("accepted")
            ]
            if final_exchange == "dense":
                assert not accepted
            else:
                assert accepted
                if corrected:
                    assert any(
                        r["operation"] == "final_state_corrected_jk" for r in accepted
                    )
            response = [r for r in rows if r["operation"] == "force_response"]
            assert len(response) == 1
            counters = response[0]["counters"]
            assert counters.get("response_final_projection_reused", 0) == 0
            # Even an unforced solve can need a strict final correction. Only
            # the last accepted producer can publish the current exact lease;
            # an earlier retained build may have been superseded by correction.
            final_fitted = bool(
                accepted and accepted[-1]["counters"].get("final_projection_fitted", 0)
            )
            reused = metric != "spectral" and projection != "off" and final_fitted
            observed_reuse |= reused
            assert counters.get("response_final_fitted_projection_reused", 0) == int(
                reused
            )
            assert counters.get("response_final_fitted_potential_reused", 0) == int(
                reused
            )
            assert counters.get("response_retained_fitted_projection_passes", 0) == int(
                not reused
            )
            assert counters.get("response_occupied_charge_inverse_gemms", 0) == (
                0 if reused else 2
            )
            assert counters.get(
                "response_occupied_charge_inverse_gemms_avoided", 0
            ) == (2 if reused else 0)
            if reused:
                rank = mol.nelectron // 2
                auxiliary_count = mf.with_df.auxmol.nao_nr()
                occupied_pairs = rank * (rank + 1) // 2
                assert counters["response_symmetric_occupied_pairs"] == 1
                assert (
                    counters["response_occupied_root_elements"]
                    == auxiliary_count * occupied_pairs
                )
                assert (
                    counters["response_occupied_inverse_flops"]
                    == 2 * auxiliary_count**2 * occupied_pairs
                )
                assert (
                    counters["response_occupied_metric_flops"]
                    == auxiliary_count * (auxiliary_count + 1) * occupied_pairs
                )
                assert (
                    counters["response_symmetric_expansion_copy_bytes"]
                    == auxiliary_count * rank**2 * 8
                )
                assert counters.get("response_retained_fitted_charge_panels", 0) == 0
                assert (
                    counters["response_retained_fitted_charge_source_elements_avoided"]
                    > 0
                )
                assert (
                    counters["response_retained_fitted_charge_unpack_elements_avoided"]
                    > 0
                )
            else:
                assert counters.get("response_symmetric_occupied_pairs", 0) == 0
                assert counters.get("response_symmetric_expansion_copy_bytes", 0) == 0
                assert counters["response_retained_fitted_charge_panels"] > 0
            assert counters.get("response_fitted_occupied_metric_root_gemms") == (
                2 if metric == "spectral" else 1
            )
            assert counters.get("response_retained_metric_root", 0) == (
                metric != "spectral"
            )
            assert counters.get("response_scratch_bytes", 0) <= 64 << 20

    if (
        not corrected
        and metric != "spectral"
        and projection != "off"
        and final_exchange != "dense"
    ):
        assert observed_reuse, "fixture did not exercise projection reuse"
