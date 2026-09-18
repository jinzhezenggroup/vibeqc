"""Real automatic derivative admission retains full independent HF forces."""

import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator
from vibeqc.profiles import probe_device

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    "case_name,practical,batch_size,shell,packet",
    [
        ("water-def2-svp-spherical", False, 1, False, False),
        ("water-tetramer-def2-svp-spherical", False, 1, True, False),
        ("water-tetramer-def2-svp-spherical", False, 4, True, False),
        ("water-octamer-s4-def2-svp-spherical", False, 1, True, True),
        ("water-octamer-s4-def2-svp-spherical", False, 4, True, True),
        ("water-tetramer-def2-svp-spherical", True, 1, True, False),
        ("water-octamer-s4-def2-svp-spherical", True, 1, True, True),
        ("oh-def2-svp-spherical-uhf", True, 1, False, False),
    ],
)
def test_automatic_derivative_route_cold_warm_and_moved(
    case_name, practical, batch_size, shell, packet, monkeypatch, tmp_path
):
    """No forced shell/candidate control can stand in for the public auto path."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()[case_name]
    orbital, cpu_orbital = case.vibeqc_basis, case.pyscf_basis
    auxiliary, cpu_auxiliary = orbital, cpu_orbital
    if practical:
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
    for name in (
        "WEIGHTED_EXECUTION",
        "DERIVATIVE_PAIRS",
        "SHELL_SCHEDULE",
        "PRIMITIVE_BUCKETS",
        "RESPONSE_UPLOAD_PROBE",
        "RESPONSE_SCATTER_PROBE",
        "SERIAL_RESPONSE_DOT",
    ):
        monkeypatch.delenv("VIBEQC_DF_" + name, raising=False)
    monkeypatch.setenv("VIBEQC_DF_SHELL_POLICY", "auto")
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in case.atoms]
    moved[1][1][0] += 0.001
    references = []
    for geometry in (case.atoms, moved):
        mol = gto.M(
            atom=geometry,
            unit="Bohr",
            basis=cpu_orbital,
            cart=case.basis_representation == "cartesian",
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
        derivative = oracle.nuc_grad_method()
        derivative.auxbasis_response = True
        references.append((oracle.e_tot, -derivative.kernel()))
    calc = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation=case.basis_representation,
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    device = probe_device(calc._library, calc._device_id)["device"]
    qualified = (device["major"], device["minor"]) == (12, 0)
    with calc.prepare_batch(
        [case.atoms] * batch_size,
        charges=[case.charge] * batch_size,
        multiplicities=[case.multiplicity] * batch_size,
    ) as owner:
        for index, changed in enumerate((False, False, True, True)):
            trace = tmp_path / f"phase-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            positions = (
                [np.asarray([xyz for _, xyz in moved])] * batch_size
                if changed
                else None
            )
            result = owner.execute(positions, strict=True)
            energy, forces = references[int(changed)]
            for item in result.items:
                assert abs(item.energy - energy) <= (3e-11 if practical else 1e-9)
                np.testing.assert_allclose(
                    item.forces, forces, atol=3e-11 if practical else 1e-8, rtol=0
                )
            responses = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            assert len(responses) == batch_size
            for response in responses:
                counters = response["counters"]
                assert bool(counters.get("three_center_shell_panels", 0)) == (
                    qualified and shell
                )
                assert bool(
                    counters.get("three_center_signature_packet_launches", 0)
                ) == (qualified and packet)
