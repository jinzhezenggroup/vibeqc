"""Complete ECP HF schedule endpoints; retain and validate every timing sample."""

import argparse
import json
import os
import platform
import statistics
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/python"))
from test_ecp import fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-raw", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.trace_raw:
        atoms, basis, _ = fixture()
        ecp_integrals(atoms, basis, device="cuda", radial_points=161, polar_points=32)
        return
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyscf": pyscf.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "driver_sha256": file_hash(Path(__file__)),
        "timing_scope": "synchronous complete energy+forces; every sample checked; first call is not process-cold",
        "repeats": args.repeats,
        "cases": [],
    }
    for label, options, spin in (
        ("9ao-rhf", {}, 0),
        ("9ao-uhf", {}, 1),
        ("16ao-rhf", {"f_on_h": True}, 0),
        (
            "29ao-rhf-fallback",
            {"f_shell": True, "f_on_h": True, "representation": "cartesian"},
            0,
        ),
    ):
        atoms, basis, mol = fixture(spin=spin, **options)
        xyz = mol.atom_coords()
        moved_xyz = xyz + np.array([[0.013, -0.017, 0.011], [-0.019, 0.023, 0.029]])
        moved_atoms = [
            (s, tuple(p)) for (s, _), p in zip(atoms, moved_xyz, strict=True)
        ]
        refs = []
        for coords in (xyz, moved_xyz):
            refmol = mol.copy().set_geom_(coords, unit="Bohr")
            ref = (scf.UHF(refmol) if spin else scf.RHF(refmol)).run(conv_tol=1e-12)
            assert ref.converged
            refs.append((ref.e_tot, -ref.nuc_grad_method().kernel()))
        calc = Calculator(basis=basis, device="cuda", method="uhf" if spin else "rhf")
        case = {
            "label": label,
            "nao": mol.nao_nr(),
            "atoms_bohr": atoms,
            "moved_atoms_bohr": moved_atoms,
            "basis_identity": basis.identity,
            "samples": [],
            "timings": {},
        }

        def checked(
            result: typing.Any,
            which: typing.Any,
            stage: typing.Any,
            elapsed: typing.Any,
            refs: typing.Any = refs,
            case: typing.Any = case,
        ) -> None:
            assert result.converged and result.executed_backend == "cuda"
            energy_error = abs(result.energy - refs[which][0])
            force_error = float(np.max(np.abs(result.forces - refs[which][1])))
            assert energy_error < 2e-8 and force_error < 2e-6
            case["samples"].append(
                {
                    "stage": stage,
                    "ms": elapsed,
                    "energy": result.energy,
                    "energy_error": energy_error,
                    "force_error": force_error,
                    "iterations": result.iterations,
                }
            )

        start = perf_counter()
        result = calc.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
        checked(result, 0, "first_call", 1000 * (perf_counter() - start))
        # Measure reusable topology, both stationary replay and changed geometry.
        with calc.prepare_batch(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        ) as batch:
            start = perf_counter()
            checked(
                batch.execute(strict=True).items[0],
                0,
                "prepared_initial",
                1000 * (perf_counter() - start),
            )
            for stage, coords, which in (
                ("prepared_warm", xyz, 0),
                ("changed_geometry", moved_xyz, 1),
            ):
                for repeat in range(args.repeats):
                    if stage == "changed_geometry":
                        restored = batch.execute(coordinates=[xyz], strict=True).items[
                            0
                        ]
                        checked(restored, 0, "restore", 0.0)
                    start = perf_counter()
                    result = batch.execute(coordinates=[coords], strict=True).items[0]
                    checked(result, which, stage, 1000 * (perf_counter() - start))
        plan = calc.estimate_resources(
            [atoms], charges=[spin], multiplicities=[spin + 1]
        )
        case["resource_peak_bytes"] = plan.peak_bytes
        if mol.nao_nr() <= 16:
            plan.require_feasible()
            bounded = Calculator(
                basis=basis,
                device="cuda",
                method="uhf" if spin else "rhf",
                resource_budget=ResourceBudget(
                    host_bytes=plan.peak_bytes["host"],
                    device_bytes=plan.peak_bytes["device"],
                ),
            )
            with bounded.prepare_batch(
                [atoms], charges=[spin], multiplicities=[spin + 1]
            ) as batch:
                start = perf_counter()
                checked(
                    batch.execute(strict=True).items[0],
                    0,
                    "budgeted",
                    1000 * (perf_counter() - start),
                )
                ledger = batch.resource_diagnostics["observation"]["device_ledger"]
                assert (
                    ledger["rejected_allocations"] == 0
                    and ledger["peak_bytes"] <= ledger["limit_bytes"]
                )
                case["device_ledger"] = ledger
        start = perf_counter()
        with calc.prepare_batch(
            [atoms, moved_atoms],
            charges=[spin, spin],
            multiplicities=[spin + 1, spin + 1],
        ) as batch:
            results = batch.execute(strict=True)
        elapsed = 1000 * (perf_counter() - start)
        for index, result in enumerate(results.items):
            checked(result, index, "batch2", elapsed)
        for stage in (
            "first_call",
            "prepared_warm",
            "changed_geometry",
            "budgeted",
            "batch2",
        ):
            samples = [s["ms"] for s in case["samples"] if s["stage"] == stage]
            if samples:
                case["timings"][stage] = {
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
