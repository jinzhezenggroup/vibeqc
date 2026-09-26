"""Practical unequal auxiliary bases expose ill-conditioned metric response."""

import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import (
    load_comparison_basis,
    native_build_metadata,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def _practical_model(case_name: typing.Any) -> typing.Any:
    """Load the retained explicit bases; historical names supply geometry only."""
    case = benchmark_cases()[case_name]
    identity = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity"
    )
    orbital, cpu_orbital = load_comparison_basis(
        identity / "cc-pvdz.json", case, role="orbital", compute_forces=True
    )
    auxiliary, cpu_auxiliary = load_comparison_basis(
        identity / "cc-pvdz-jkfit.json", case, role="auxiliary", compute_forces=True
    )
    return case, orbital, auxiliary, cpu_orbital, cpu_auxiliary


@pytest.fixture(
    scope="module",
    params=["oh-def2-svp-spherical-uhf", "water-tetramer-def2-svp-spherical"],
)
def practical_reference(request: typing.Any) -> typing.Any:
    """Solve original/moved cc-pVDZ/JKFIT models independently with libcint."""
    from pyscf import gto, scf

    case, orbital, auxiliary, cpu_orbital, cpu_auxiliary = _practical_model(
        request.param
    )
    moved = [
        (symbol, np.array(position, dtype=float)) for symbol, position in case.atoms
    ]
    moved[1][1][0] += 0.001
    references = []
    for geometry in (case.atoms, moved):
        mol = gto.M(
            atom=geometry,
            basis=cpu_orbital,
            unit="Bohr",
            spin=case.multiplicity - 1,
            charge=case.charge,
            verbose=0,
        )
        oracle = (scf.UHF if case.method == "uhf" else scf.RHF)(mol).density_fit(
            auxbasis=cpu_auxiliary
        )
        oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-12, 200
        oracle.direct_scf_tol = 1e-14
        oracle.kernel()
        assert oracle.converged
        gradient = oracle.nuc_grad_method()
        gradient.auxbasis_response = True
        references.append((oracle.e_tot, -gradient.kernel()))
    return case, orbital, auxiliary, moved, references


@pytest.mark.parametrize(
    "values,storage,space,algebra,budget",
    [
        ("dense", "auto", "auto", "blas", 0),
        ("dense", "panel", "dense", "scalar", 0),
        ("dense", "panel", "dense", "blas", 0),
        ("dense", "jk-scratch", "occupied", "blas", 0),
        ("packed", "auto", "occupied", "blas", 0),
        ("packed", "panel", "dense", "blas", 0),
        ("dense", "panel", "dense", "blas", 128 << 20),
    ],
)
def test_practical_full_force_cold_warm_and_changed_geometry(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    practical_reference: typing.Any,
    values: typing.Any,
    storage: typing.Any,
    space: typing.Any,
    algebra: typing.Any,
    budget: typing.Any,
    *,
    density_tolerance: typing.Any = 1e-12,
) -> None:
    """Check complete forces, including Coulomb/exchange and metric cancellation.

    Accuracy is tested at every endpoint, including the first cold solve. All
    completed arrays are persisted before assertions so later phases cannot
    hide an earlier inaccurate result. These are untimed regression tests.

    The default 1e-12 request isolates response arithmetic. Frozen v12 failed
    the original 1e-10 request near 4e-11; the separate original-request cases
    below exercise the physical-state repair without reclassifying that old
    failure. Both requests retain the same 3e-11 energy/force gates.
    """
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require finite Slurm"
    case, orbital, auxiliary, moved, references = practical_reference
    for name, value in {
        "VIBEQC_DF_VALUE_STORAGE": values,
        "VIBEQC_DF_RESPONSE_STORAGE": storage,
        "VIBEQC_DF_RESPONSE_SPACE": space,
        "VIBEQC_DF_RESPONSE_ALGEBRA": algebra,
        "VIBEQC_DF_EXCHANGE": "auto" if space == "auto" else "occupied",
        "VIBEQC_DF_FORCE_SCREEN_ABS": "off",
        "VIBEQC_DF_REFERENCE_FINAL_VALIDATION": "0",
        "VIBEQC_DF_WEIGHTED_EXECUTION": "shell",
        "VIBEQC_DF_DERIVATIVE_PAIRS": "packed" if values == "packed" else "full",
    }.items():
        monkeypatch.setenv(name, value)
    calculator = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=density_tolerance,
        screening_tolerance=1e-14,
    )
    (tmp_path / "identity.json").write_text(
        json.dumps(
            {
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "native_build": native_build_metadata(calculator),
                "case": case.name if hasattr(case, "name") else repr(case.atoms),
                "values": values,
                "storage": storage,
                "space": space,
                "algebra": algebra,
                "budget": budget,
                "density_tolerance": density_tolerance,
            },
            indent=2,
        )
        + "\n"
    )
    monkeypatch.setenv("VIBEQC_DF_TRACE", str(tmp_path / "prepare.jsonl"))
    rows = []
    with calculator.prepare_batch(
        [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
    ) as batch:
        for phase, changed in (
            ("cold", False),
            ("warm", False),
            ("changed", True),
            ("changed_warm", True),
        ):
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(tmp_path / f"{phase}.jsonl"))
            item = batch.execute(
                [np.array([r for _, r in moved])] if changed else None, strict=False
            ).items[0]
            if not item.succeeded:
                rows.append(
                    {
                        "phase": phase,
                        "status": item.status,
                        "status_message": item.status_message,
                    }
                )
                (tmp_path / "results.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                pytest.fail(f"{phase}: {item.status_message}")
            energy, force = references[int(changed)]
            rows.append(
                {
                    "phase": phase,
                    "energy": item.energy,
                    "forces": np.asarray(item.forces).tolist(),
                    "energy_error": abs(item.energy - energy),
                    "force_error": float(np.max(np.abs(item.forces - force))),
                    "iterations": item.iterations,
                }
            )
            (tmp_path / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        assert row["energy_error"] <= 3e-11, row
        assert row["force_error"] <= 3e-11, row


@pytest.mark.parametrize(
    "values,storage,space,algebra",
    [
        ("dense", "panel", "dense", "scalar"),
        ("dense", "panel", "dense", "blas"),
        ("dense", "jk-scratch", "occupied", "blas"),
        ("packed", "auto", "occupied", "blas"),
        ("packed", "panel", "dense", "blas"),
    ],
)
def test_practical_original_density_request(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    practical_reference: typing.Any,
    values: typing.Any,
    storage: typing.Any,
    space: typing.Any,
    algebra: typing.Any,
) -> None:
    """Keep the original RHF/UHF request and every failed route's strict gate.

    Cold, warm and both changed-geometry endpoints are independently compared
    with the same CPU oracle, rather than replacing a failed cold result with
    a better-converged warm sample.
    """
    test_practical_full_force_cold_warm_and_changed_geometry(
        monkeypatch,
        tmp_path,
        practical_reference,
        values,
        storage,
        space,
        algebra,
        0,
        density_tolerance=1e-10,
    )


@pytest.mark.parametrize(
    "practical_reference", ["oh-def2-svp-spherical-uhf"], indirect=True
)
def test_practical_packed_oh_at_128_mib(
    monkeypatch: typing.Any, tmp_path: typing.Any, practical_reference: typing.Any
) -> None:
    """Keep the admitted small packed case separate from water's rejection."""
    test_practical_full_force_cold_warm_and_changed_geometry(
        monkeypatch,
        tmp_path,
        practical_reference,
        "packed",
        "panel",
        "dense",
        "blas",
        128 << 20,
    )


@pytest.mark.parametrize(
    "values,budget", [("dense", 80 << 20), ("packed", 128 << 20), ("packed", 160 << 20)]
)
def test_practical_water_insufficient_budget(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    values: typing.Any,
    budget: typing.Any,
) -> None:
    """Reject allowances below the value plan's independently checked lower bound.

    The force adapter assigns half its allowance to J/K. The native CPU planner
    requires at least 43,241,659 dense or 85,079,227 packed bytes for this shape,
    before source/DIIS reservations. This checks the explicit resource status,
    not a numerical endpoint. Earlier qualification runs remain failed records.
    """
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests require finite Slurm"
    case, orbital, auxiliary, _, _ = _practical_model(
        "water-tetramer-def2-svp-spherical"
    )
    for name, value in {
        "VIBEQC_DF_VALUE_STORAGE": values,
        "VIBEQC_DF_RESPONSE_STORAGE": "panel",
        "VIBEQC_DF_RESPONSE_SPACE": "dense",
        "VIBEQC_DF_RESPONSE_ALGEBRA": "blas",
        "VIBEQC_DF_EXCHANGE": "occupied",
        "VIBEQC_DF_REFERENCE_FINAL_VALIDATION": "0",
        "VIBEQC_DF_TRACE": str(tmp_path / "rejection.jsonl"),
    }.items():
        monkeypatch.setenv(name, value)
    calculator = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-12,
        screening_tolerance=1e-14,
    )
    identity = {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "native_build": native_build_metadata(calculator),
        "values": values,
        "budget": budget,
        "expected_status": _native.STATUS_OUT_OF_MEMORY,
    }
    (tmp_path / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    with calculator.prepare_batch(
        [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
    ) as batch:
        item = batch.execute(strict=False).items[0]
        # Separate filename prevents evidence auditors from treating a resource
        # contract check as a completed scientific result or an erased failure.
        (tmp_path / "resource-result.json").write_text(
            json.dumps(
                {"status": item.status, "status_message": item.status_message}, indent=2
            )
            + "\n"
        )
        assert item.status == _native.STATUS_OUT_OF_MEMORY, item.status_message
        # The diagnostic API reports unavailable capability when no value
        # plan was admitted, rather than returning an empty diagnostic tuple.
        with pytest.raises(NotImplementedError, match="VIBEQC error 3"):
            batch.last_density_fitting_metric_diagnostics()


@pytest.mark.parametrize("buckets", ("off", "packet"))
def test_practical_auxiliary_f_rys_is_executed(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    practical_reference: typing.Any,
    buckets: typing.Any,
) -> None:
    """Observe the generated f classes, not just agreement of two fallbacks."""
    from benchmarks.df_component_ledger import aggregate, read_trace

    assert os.environ.get("SLURM_JOB_ID")
    case, orbital, auxiliary, _, references = practical_reference
    for key, value in {
        "SHELL_POLICY": "candidate",
        "WEIGHTED_EXECUTION": "shell",
        "DERIVATIVE_PAIRS": "symmetric",
        "SHELL_SCHEDULE": "compact",
        "PRIMITIVE_BUCKETS": buckets,
        "FORCE_SCREEN_ABS": "off",
    }.items():
        monkeypatch.setenv("VIBEQC_DF_" + key, value)
    calculator = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    trace = tmp_path / "auxiliary-f.jsonl"
    with calculator.prepare_batch(
        [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
    ) as batch:
        batch.execute(strict=True)
        monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
        result = batch.execute(strict=True).items[0]
    energy, force = references[0]
    assert abs(result.energy - energy) <= 3e-11
    np.testing.assert_allclose(result.forces, force, rtol=0, atol=3e-11)
    groups = [
        g
        for g in aggregate(read_trace(trace))["groups"]
        if g["operation"] == "force_response"
    ]
    assert len(groups) == 1
    counts = groups[0]["counter_sums"]
    for cls in ("003", "103", "113", "203", "213"):
        assert counts[f"shell_{cls}_rys_selected"] == 1, (cls, counts)
    # d-d-f requires five roots, which is not admitted by the strict DF quadrature.
    assert counts["shell_223_rys_selected"] == 0
