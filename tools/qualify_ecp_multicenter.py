"""Record bounded mixed LANL2DZ Na / Stuttgart RLC K ECP evidence using a test-only PySCF oracle."""

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import typing
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import scf
from vibeqc import Calculator, ResourceBudget
from vibeqc.ecp import ecp_integrals
from vibeqc.profiles import file_hash


def maximum(value: typing.Any) -> typing.Any:
    return float(np.max(np.abs(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "tests/python/test_ecp_multicenter.py"
    sys.path.insert(0, str(fixture_path.parent))
    spec = importlib.util.spec_from_file_location("multicenter_oracle", fixture_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    record = {
        "device": args.device,
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "fixture_sha256": file_hash(fixture_path),
        "shared_helper_sha256": file_hash(fixture_path.with_name("test_ecp_heavy.py")),
        "driver_sha256": file_hash(Path(__file__)),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "timing_scope": "one synchronous fresh Calculator singlepoint, no warmup; no performance claim",
        "grids": [[160, 32, 64], [224, 44, 88]],
        "cases": [],
    }
    if args.device == "cuda":
        record["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    for reverse in (False, True):
        for spin in (0, 1):
            atoms, basis, mol = oracle.multicenter_fixture(spin=spin, reverse=reverse)
            options = {"charge": spin, "multiplicity": spin + 1}
            raw = ecp_integrals(atoms, basis, device=args.device, **options)
            fine = ecp_integrals(
                atoms,
                basis,
                device=args.device,
                radial_points=224,
                polar_points=44,
                **options,
            )
            expected = oracle.reference_components(mol)
            target = (scf.UHF(mol) if spin else scf.RHF(mol)).run(
                conv_tol=1e-13, conv_tol_grad=1e-10
            )
            assert target.converged
            calculator = oracle.calculator(basis, args.device, spin)
            start = perf_counter()
            result = calculator.singlepoint(atoms, **options)
            elapsed = (perf_counter() - start) * 1000
            assert result.converged
            assert result.executed_backend == (
                "cuda" if args.device == "cuda" else "cpu_reference"
            )
            expected_forces = -target.nuc_grad_method().kernel()
            errors = {
                "local_matrix": maximum(raw.local - expected[0]),
                "nonlocal_matrix": maximum(raw.nonlocal_ - expected[1]),
                "energy": abs(result.energy - target.e_tot),
                "force": maximum(result.forces - expected_forces),
                "net_force": maximum(result.forces.sum(axis=0)),
            }
            assert max(errors["local_matrix"], errors["nonlocal_matrix"]) < 2e-9
            assert errors["energy"] < 2e-8
            assert errors["force"] < 2e-6
            assert errors["net_force"] < 2e-7
            plan = calculator.estimate_resources(
                [atoms], charges=[spin], multiplicities=[spin + 1]
            ).require_feasible()
            bounded = Calculator(
                basis=basis,
                device=args.device,
                method="uhf" if spin else "rhf",
                resource_budget=ResourceBudget(
                    host_bytes=plan.peak_bytes["host"],
                    device_bytes=plan.peak_bytes.get("device"),
                ),
            )
            with bounded.prepare_batch(
                [atoms], charges=[spin], multiplicities=[spin + 1]
            ) as batch:
                replay = batch.execute(strict=True).items[0]
                assert replay.converged
                np.testing.assert_allclose(
                    replay.energy, result.energy, atol=2e-10, rtol=0
                )
                ledger = batch.resource_diagnostics["observation"].get("device_ledger")
                if args.device == "cuda":
                    assert ledger["rejected_allocations"] == 0
                    assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]
            case = {
                "order": "K-Na" if reverse else "Na-K",
                "method": "uhf" if spin else "rhf",
                "atoms_bohr": atoms,
                "charge": spin,
                "multiplicity": spin + 1,
                "core_electrons": [18, 10] if reverse else [10, 18],
                "nelectron": mol.nelectron,
                "nao": mol.nao_nr(),
                "basis_identity": basis.identity,
                "parameter_sha256": basis.provenance.checksum,
                "errors": errors,
                "quadrature_difference": raw.quadrature_difference(fine),
                "energy": result.energy,
                "reference_energy": target.e_tot,
                "forces": result.forces.tolist(),
                "reference_forces": expected_forces.tolist(),
                "singlepoint_ms": elapsed,
                "resource_peak_bytes": plan.peak_bytes,
                "device_ledger": ledger,
            }
            if args.device == "cuda":
                cpu = ecp_integrals(atoms, basis, **options)
                case["cpu_cuda_max_abs"] = {
                    field: maximum(getattr(raw, field) - getattr(cpu, field))
                    for field in (
                        "local",
                        "nonlocal_",
                        "local_derivative",
                        "nonlocal_derivative",
                    )
                }
            record["cases"].append(case)
            args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
