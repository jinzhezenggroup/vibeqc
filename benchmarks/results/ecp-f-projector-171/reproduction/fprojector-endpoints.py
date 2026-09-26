"""Record complete f-projector/f-orbital HF qualification, with independent errors."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import scf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
sys.path[:0] = [str(args.source / "python"), str(args.source / "tests/python")]
from test_ecp import fixture, reference
from vibeqc import Calculator
from vibeqc.ecp import ecp_integrals

record = {
    "device": args.device,
    "python": platform.python_version(),
    "pyscf": pyscf.__version__,
    "library_sha256": hashlib.sha256(
        Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
    ).hexdigest(),
    "timing_scope": "synchronous complete singlepoint; CPU one cold sample, CUDA one warmup and three samples",
    "claim": "bounded synthetic f-projector qualification, not broad heavy-element support or speedup",
    "threads": {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    },
    "cases": [],
}
cases = [("cartesian", 0, False), ("spherical", 1, False)]
if args.device == "cuda":
    cases += [("cartesian", 1, False), ("spherical", 0, False), ("cartesian", 0, True)]
try:
    for representation, spin, both in cases:
        atoms, basis, mol = fixture(
            representation=representation,
            spin=spin,
            f_shell=True,
            f_on_h=both,
            f_projector=True,
        )
        target = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
        assert target.converged
        force = -target.nuc_grad_method().kernel()
        raw = ecp_integrals(atoms, basis, device=args.device)
        matrix_error = float(np.max(np.abs(raw.matrix - reference(mol))))
        calc = Calculator(
            basis=basis, method="uhf" if spin else "rhf", device=args.device
        )

        def call(calc=calc, atoms=atoms, spin=spin):
            result = calc.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
            assert result.converged
            assert result.executed_backend == (
                "cpu_reference" if args.device == "cpu" else "cuda"
            )
            return result

        if args.device == "cuda":
            call()
        samples, energy_errors, force_errors, iterations = [], [], [], []
        for _ in range(1 if args.device == "cpu" else 3):
            begin = perf_counter()
            result = call()
            samples.append(1000 * (perf_counter() - begin))
            energy_errors.append(abs(result.energy - target.e_tot))
            force_errors.append(float(np.max(np.abs(result.forces - force))))
            iterations.append(result.iterations)
        plan = calc.estimate_resources(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        )
        record["cases"].append(
            {
                "representation": representation,
                "spin": spin,
                "f_on_h": both,
                "ao_count": mol.nao_nr(),
                "cartesian_ao_count": mol.nao_cart(),
                "shell_count": mol.nbas,
                "primitive_count": sum(mol.bas_nprim(i) for i in range(mol.nbas)),
                "projector_channels": [0, 1, 3],
                "harmonic_slots": 16,
                "basis_identity": basis.identity,
                "atoms_bohr": atoms,
                "raw_grid": [160, 32, 64],
                "hf_grids": [[160, 32, 64], [224, 44, 88]],
                "matrix_max_abs_error": matrix_error,
                "energy_abs_errors": energy_errors,
                "force_max_abs_errors": force_errors,
                "timing_ms": {"median": statistics.median(samples), "samples": samples},
                "iterations": iterations,
                "resource_status": plan.status,
                "planned_peak_bytes": dict(plan.peak_bytes),
            }
        )
        assert matrix_error < 2e-9
        assert max(energy_errors) < 2e-8
        assert max(force_errors) < 2e-6
finally:
    args.output.write_text(json.dumps(record, indent=2) + "\n")
