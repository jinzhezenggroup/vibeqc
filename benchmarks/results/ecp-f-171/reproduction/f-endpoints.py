"""Retained f-orbital qualification; invoke with candidate source and exact library."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import scf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.device == "cpu" else 3
    if args.repeats < 1:
        parser.error("repeats must be positive")
    sys.path[:0] = [str(args.source / "python"), str(args.source / "tests/python")]
    from test_ecp import fixture, reference
    from vibeqc import Atom, Calculator, ResourceBudget
    from vibeqc.ecp import ecp_integrals, resolve_ecp

    record = {
        "device": args.device,
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "library_sha256": hashlib.sha256(
            Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "timing_scope": "warm synchronous Calculator.singlepoint, including fresh system/SCF and complete energy/forces; one warmup per case",
        "claim": "bounded f capability qualification, no f speedup comparison with unsupported baseline",
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "cases": [],
    }
    if args.device == "cuda":
        record["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()

    cases = [(r, s, False) for r in ("cartesian", "spherical") for s in (0, 1)]
    # The independent CPU reference is deliberately expensive at larger f
    # sizes. Its two-f-center prepared replay is an optional qualification;
    # retain the larger Cartesian timing/resource workload on CUDA only.
    if args.device == "cuda":
        cases.append(("cartesian", 0, True))
    try:
        for representation, spin, both_centers in cases:
            atoms, basis, mol = fixture(
                representation=representation,
                spin=spin,
                f_shell=True,
                f_on_h=both_centers,
            )
            raw = ecp_integrals(atoms, basis, device=args.device)
            target = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
            assert target.converged
            target_force = -target.nuc_grad_method().kernel()
            method = "uhf" if spin else "rhf"
            calculator = Calculator(basis=basis, method=method, device=args.device)
            call = partial(
                calculator.singlepoint, atoms, charge=spin, multiplicity=spin + 1
            )
            warm = call()
            assert warm.converged
            samples, energies, forces, iterations = [], [], [], []
            for _ in range(args.repeats):
                start = perf_counter()
                result = call()
                samples.append(1000 * (perf_counter() - start))
                assert result.converged
                expected_backend = "cpu_reference" if args.device == "cpu" else "cuda"
                assert result.executed_backend == expected_backend, (
                    result.executed_backend
                )
                energies.append(abs(result.energy - target.e_tot))
                forces.append(float(np.max(np.abs(result.forces - target_force))))
                iterations.append(result.iterations)
            plan = calculator.estimate_resources(
                [atoms], charges=[spin], multiplicities=[spin + 1]
            )
            observed = diagnostics = None
            if plan.status == "unsupported":
                assert args.device == "cuda" and mol.nao_nr() > 16
                assert "<=16 public AOs" in plan.diagnostic
                try:
                    plan.require_feasible()
                except NotImplementedError:
                    pass
                else:
                    raise AssertionError("unsupported inventory was silently accepted")
            else:
                plan.require_feasible()
                bounded = Calculator(
                    basis=basis,
                    method=method,
                    device=args.device,
                    resource_budget=ResourceBudget(
                        host_bytes=plan.peak_bytes["host"],
                        device_bytes=plan.peak_bytes.get("device"),
                    ),
                )
                with bounded.prepare_batch(
                    [atoms], charges=[spin], multiplicities=[spin + 1]
                ) as batch:
                    observed = batch.execute(strict=True).items[0]
                    assert observed.converged
                    diagnostics = batch.resource_diagnostics
                    if args.device == "cuda":
                        ledger = diagnostics["observation"]["device_ledger"]
                        assert ledger["rejected_allocations"] == 0
                        assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]
            _, terms = resolve_ecp(basis, tuple(Atom.from_value(a) for a in atoms))
            row = {
                "representation": representation,
                "executed_backend": result.executed_backend,
                "method": method,
                "f_centers": ["Na", "H"] if both_centers else ["Na"],
                "atoms_bohr": atoms,
                "ao_count": mol.nao_nr(),
                "shell_count": mol.nbas,
                "basis_identity": basis.identity,
                "shells": [
                    {
                        "atom": int(mol.bas_atom(i)),
                        "angular": int(mol.bas_angular(i)),
                        "primitives": int(mol.bas_nprim(i)),
                        "contractions": int(mol.bas_nctr(i)),
                    }
                    for i in range(mol.nbas)
                ],
                "ecp_terms": len(terms),
                "ecp_centers": 1,
                "raw_grid": [160, 32, 64],
                "hf_grids": [[160, 32, 64], [224, 44, 88]],
                "ao_pair_count": mol.nao_nr() ** 2,
                "matrix_max_abs_error": float(
                    np.max(np.abs(raw.matrix - reference(mol)))
                ),
                "timing_ms": {"median": statistics.median(samples), "samples": samples},
                "iterations": iterations,
                "energy_abs_errors": energies,
                "force_max_abs_errors": forces,
                "resource_status": plan.status,
                "resource_diagnostic": plan.diagnostic,
                "resource_peak_bytes": plan.peak_bytes
                if observed is not None
                else None,
                "resource_diagnostics": diagnostics,
            }
            record["cases"].append(row)
            assert row["matrix_max_abs_error"] < 2e-9
            assert max(energies) < 2e-8 and max(forces) < 2e-6
            if observed is not None:
                np.testing.assert_allclose(
                    observed.energy, target.e_tot, atol=2e-8, rtol=0
                )
                np.testing.assert_allclose(
                    observed.forces, target_force, atol=2e-6, rtol=0
                )
            print(representation, method, row["ao_count"], row["timing_ms"], flush=True)
    finally:
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
