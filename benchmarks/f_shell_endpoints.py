"""Audit real f-shell bases and compare FPPS dispatch with the generic fallback.

Run under a finite Slurm GPU allocation. Timing uses the existing frozen-dm0
ABBA protocol; the independent CPU PySCF oracle is outside the timed region.
Use --profile with Nsight Systems' cudaProfilerApi capture to record one warm
candidate replay separately from the unprofiled performance acceptance run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "water-def2-tzvp",
    "water-dimer-def2-tzvp-spherical",
    "water-tetramer-def2-tzvp-spherical",
)


def release_library_identity() -> dict:
    """Reject fast/unknown builds and bind endpoint results to the measured library."""
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    directory = library.parent
    settings = {}
    for line in (directory / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            name, value = line.split("=", 1)
            settings[name.split(":", 1)[0]] = value
    if (
        settings.get("VIBEQC_CUDA_FAST_COMPILE") != "OFF"
        or settings.get("CMAKE_BUILD_TYPE") != "Release"
    ):
        raise ValueError(
            "endpoint acceptance requires a verified Release build with fast compilation OFF"
        )
    if settings.get("VIBEQC_CUDA_COMPILE_ARCHITECTURES") not in ("120", "120-real"):
        raise ValueError(
            "endpoint acceptance requires the actual sm_120 release target"
        )

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    generated = directory / "generated/production_shell_kernels"
    sources = {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(generated.rglob("*"))
        if path.is_file()
    }
    if not sources:
        raise ValueError("endpoint build has no recorded generated class sources")
    return {
        "library_sha256": digest(library),
        "library_bytes": library.stat().st_size,
        "cmake_cache_sha256": digest(directory / "CMakeCache.txt"),
        "generated_source_hashes": sources,
        "settings": {
            k: v
            for k, v in settings.items()
            if k.startswith("VIBEQC_")
            or k in ("CMAKE_BUILD_TYPE", "CMAKE_CUDA_COMPILER")
        },
    }


def inspected_basis(case, calculator, atoms) -> tuple[dict, object]:
    """Inspect the loaded native and oracle shells, including contraction sizes."""
    from pyscf import gto
    from vibeqc import Atom

    shells = calculator._shells_for_atoms(tuple(Atom.from_value(a) for a in atoms))
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=case.pyscf_basis,
        charge=case.charge,
        spin=case.multiplicity - 1,
        cart=case.basis_representation == "cartesian",
        verbose=0,
    )
    native = [(s.atom_index, s.angular_momentum, len(s.primitives)) for s in shells]
    reference = [
        (mol.bas_atom(i), mol.bas_angular(i), mol.bas_nprim(i))
        for i in range(mol.nbas)
        for _ in range(mol.bas_nctr(i))
    ]
    if sorted(native) != sorted(reference):
        raise ValueError("loaded VibeQC/PySCF shell and contraction catalogs differ")
    f_count = sum(angular == 3 for _, angular, _ in native)
    if f_count == 0:
        raise ValueError("endpoint does not contain loaded l=3 shells")
    if case.expected_ao_count is not None and mol.nao_nr() != case.expected_ao_count:
        raise ValueError("endpoint AO count differs from its declared workload")
    return {
        "shells": native,
        "reference_shells": reference,
        "f_shell_count": f_count,
        "ao_count": mol.nao_nr(),
        "representation": case.basis_representation,
    }, mol


def independent_result(mol, method: str) -> dict:
    """Use CPU libcint SCF and all analytic force terms without density fitting."""
    from pyscf import scf

    engine = scf.RHF(mol) if method == "rhf" else scf.UHF(mol)
    engine.conv_tol = 1e-12
    engine.conv_tol_grad = 1e-10
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 100
    energy = engine.kernel()
    if not engine.converged:
        raise ValueError("independent PySCF endpoint did not converge")
    force = -engine.nuc_grad_method().kernel()
    return {
        "energy_hartree": float(energy),
        "forces_hartree_per_bohr": force.tolist(),
        "iterations": int(engine.cycles),
        "converged": bool(engine.converged),
        "orbital_gradient_norm": float(
            np.linalg.norm(engine.get_grad(engine.mo_coeff, engine.mo_occ))
        ),
    }


def run_endpoint(
    name, batch_size, *, repeats=6, profile=False, profile_side="candidate"
) -> dict:
    """Measure current FPPS force selection against identical non-f dispatch."""
    import cupy as cp
    from _cases import benchmark_cases
    from _support import cuda_accelerator_metadata, environment_metadata
    from aot_shell_batch_gate import _execute_once, _fixed_dm0_measurement
    from compare_gpu4pyscf_batch import scaled_geometries
    from shell_class_histogram import (
        ShellWork,
        summarize_active_shell_classes,
        summarize_shell_classes,
    )
    from vibeqc import Calculator

    case = benchmark_cases()[name]
    library_identity = release_library_identity()
    systems = scaled_geometries(case.atoms, batch_size)
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    basis, _ = inspected_basis(case, calculator, systems[0])
    manifest = json.loads(
        (ROOT / "tools/vibeqc_codegen/production_shell_classes.json").read_text()
    )
    kernels = manifest["architectures"]["sm_120"]["kernels"]
    force = tuple(k["shell_class"] for k in kernels if "force" in k["consumers"])
    fock = tuple(k["shell_class"] for k in kernels if "fock" in k["consumers"])
    baseline = tuple(n for n in force if "f" not in n)
    candidate = (*baseline, "fpps")
    if "fpps" not in force:
        raise ValueError("FPPS candidate must be compiled into the endpoint library")
    topology = summarize_shell_classes(
        [
            ShellWork(angular=l, ao_count=(l + 1) * (l + 2) // 2, primitive_count=n)
            for _, l, n in basis["shells"]
        ],
        None,
    )
    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * batch_size,
        multiplicities=[case.multiplicity] * batch_size,
        warm_start=True,
        shell_class_profiling=True,
    ) as prepared:
        # The union primes the immutable generated-task capacity before either
        # measured side. Clearing warm state then supplies one common cold dm0.
        _execute_once(prepared, cp, candidate, fock_classes=fock)
        prepared.clear_warm_starts()
        measurement, _ = _fixed_dm0_measurement(
            prepared,
            cp,
            baseline,
            candidate,
            repeats,
            order_style="abba",
            warmups=1,
            baseline_fock_classes=fock,
            candidate_fock_classes=fock,
            maximum_energy_error=1e-10,
            maximum_force_error=1e-9,
            minimum_speedup=1 / 1.02,
        )
        _execute_once(prepared, cp, candidate, fock_classes=fock)
        active = summarize_active_shell_classes(
            prepared.last_shell_class_profile(), None
        )
        if profile:
            cp.cuda.Stream.null.synchronize()
            cp.cuda.profiler.start()
            try:
                _execute_once(
                    prepared,
                    cp,
                    baseline if profile_side == "baseline" else candidate,
                    fock_classes=fock,
                )
            finally:
                cp.cuda.profiler.stop()
    references = [
        independent_result(inspected_basis(case, calculator, atoms)[1], case.method)
        for atoms in systems
    ]
    errors = []
    for side in ("baseline", "candidate"):
        for sample in measurement[f"{side}_samples"]:
            errors.append(
                {
                    "side": side,
                    "energy": float(
                        np.max(
                            np.abs(
                                np.asarray(sample["energies_hartree"])
                                - [r["energy_hartree"] for r in references]
                            )
                        )
                    ),
                    "force": float(
                        np.max(
                            np.abs(
                                np.asarray(sample["forces_hartree_per_bohr"])
                                - [r["forces_hartree_per_bohr"] for r in references]
                            )
                        )
                    ),
                }
            )
    total_work = sum(row["primitive_quartets"] for row in active)
    f_work = sum(row["primitive_quartets"] for row in active if "f" in row["class"])
    if f_work == 0 or total_work == 0:
        raise ValueError("loaded f shells produced no measured active primitive work")
    oracle_passed = all(e["energy"] <= 1e-9 and e["force"] <= 1e-7 for e in errors)
    return {
        "schema": "vibeqc.f_shell_endpoint",
        "schema_version": 1,
        "case": name,
        "batch_size": batch_size,
        "basis": basis,
        "atoms_bohr": case.atoms,
        "profiled": profile,
        "profile_side": profile_side if profile else None,
        "baseline_force": baseline,
        "candidate_force": candidate,
        "fock": fock,
        "settings": {
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-14,
            "reference_gradient_tolerance": 1e-10,
            "non_regression_budget": 0.02,
        },
        "topology_before_screening": topology,
        "active_shell_classes": active,
        "f_primitive_work_fraction": f_work / total_work,
        "measurement": measurement,
        "reference": references,
        "reference_errors": errors,
        "independent_numerical_passed": oracle_passed,
        "promotion_passed": oracle_passed and measurement["passed"] and not profile,
        "environment": environment_metadata(
            distributions={
                "numpy": ("numpy",),
                "pyscf": ("pyscf",),
                "cupy": ("cupy-cuda12x", "cupy"),
            },
            accelerator=cuda_accelerator_metadata(cp),
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "release_library": library_identity,
    }


def main() -> int:
    """Fail on numerical errors; record a slower candidate as a rejected promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-side", choices=("baseline", "candidate"), default="candidate"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 2:
        parser.error("positive batch and at least two ABBA repeats are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_endpoint(
            args.case,
            args.batch,
            repeats=args.repeats,
            profile=args.profile,
            profile_side=args.profile_side,
        )
    except RuntimeError as error:
        # Retain a failed endpoint instead of leaving a missing file that could
        # later be mistaken for an unrequested or successful batch-size gate.
        failure = {
            "schema": "vibeqc.f_shell_endpoint",
            "schema_version": 1,
            "case": args.case,
            "batch_size": args.batch,
            "status": "fail",
            "reason": str(error),
            "release_library": release_library_identity(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        args.case,
        args.batch,
        "oracle",
        result["independent_numerical_passed"],
        "speedup",
        result["measurement"]["speedup"],
        flush=True,
    )
    failures = set(result["measurement"]["failures"]) - {"performance"}
    return int(not result["independent_numerical_passed"] or bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
