"""Matched ECP correctness and synchronous API timing; test-only PySCF oracle.

Component timings zero the other coefficients, retaining the full quadrature
engine. They are endpoint measurements, not isolated kernel times or speedups.
"""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import scf
from vibeqc import Calculator
from vibeqc.ecp import ecp_integrals
from vibeqc.profiles import file_hash


def timed(call, repeats):
    result = call()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append(1000 * (perf_counter() - start))
    return result, {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ecp_oracle", root / "tests/python/test_ecp.py"
    )
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    atoms, basis, mol = oracle.fixture()
    record = {
        "device": args.device,
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "machine": platform.machine(),
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "grid": {"radial": 160, "polar": 32, "azimuth": 64},
        "refined_grid": {"radial": 224, "polar": 44, "azimuth": 88},
        "atoms_bohr": atoms,
        "basis_identity": basis.identity,
        "timing_scope": "warm synchronous API including context, metadata, allocations and transfers",
        "component_timing": "other coefficients zeroed; full local/projector engine retained",
    }
    record["cpu"] = subprocess.check_output(["lscpu"], text=True)
    if args.device == "cuda":
        record["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    actual = ecp_integrals(atoms, basis, device=args.device)
    refined = ecp_integrals(
        atoms, basis, device=args.device, radial_points=224, polar_points=44
    )
    record["quadrature_difference"] = actual.quadrature_difference(refined)
    record["libcint_matrix_max_abs"] = float(
        np.max(np.abs(actual.matrix - oracle.reference(mol)))
    )
    record["cpu_gpu_max_abs"] = {}
    cpu = ecp_integrals(atoms, basis)
    for field in ("local", "nonlocal_", "local_derivative", "nonlocal_derivative"):
        record["cpu_gpu_max_abs"][field] = float(
            np.max(np.abs(getattr(actual, field) - getattr(cpu, field)))
        )
    record["components"] = {}
    for part in ("local", "nonlocal"):
        potentials = json.loads(basis.by_element[11].ecp_data)
        local = max(p["angular_momentum"][0] for p in potentials)
        for p in potentials:
            if (p["angular_momentum"][0] == local) != (part == "local"):
                p["coefficients"] = [["0"] * len(p["gaussian_exponents"])]
        selected = replace(
            basis,
            elements=tuple(
                replace(e, ecp_data=json.dumps(potentials))
                if e.atomic_number == 11
                else e
                for e in basis.elements
            ),
        )
        for derivatives in (False, True):
            _, timing = timed(
                lambda selected=selected, derivatives=derivatives: ecp_integrals(
                    atoms, selected, device=args.device, derivatives=derivatives
                ),
                args.repeats,
            )
            record["components"][f"{part}_{'gradient' if derivatives else 'value'}"] = (
                timing
            )
    record["hf"] = []
    for spin in (0, 1):
        atoms, basis, mol = oracle.fixture(spin=spin)
        target = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
        calculator = Calculator(
            basis=basis, device=args.device, method="uhf" if spin else "rhf"
        )
        result, timing = timed(
            lambda spin=spin, calculator=calculator, atoms=atoms: (
                calculator.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
            ),
            args.repeats,
        )
        plan = calculator.estimate_resources(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        )
        record["hf"].append(
            {
                "method": "uhf" if spin else "rhf",
                "timing": timing,
                "energy_abs_error": abs(result.energy - target.e_tot),
                "force_max_abs_error": float(
                    np.max(np.abs(result.forces + target.nuc_grad_method().kernel()))
                ),
                "resource_peak_bytes": plan.peak_bytes,
            }
        )
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
