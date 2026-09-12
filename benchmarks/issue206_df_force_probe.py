"""Measure the DF force-response cost separately from the SCF endpoint.

This is a diagnostic companion to ``issue206_df_matrix.py``.  It intentionally
uses fresh single-system calculations for each observable so the reported
energy-only versus energy-plus-force difference cannot be mistaken for a
matched warm-solve speed claim.  Run it inside a finite Slurm allocation on the
target GPU; the resulting JSON is a bottleneck ledger input for issue #206.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from _cases import benchmark_cases
except ModuleNotFoundError:  # imported as ``benchmarks.issue206_df_force_probe``
    from benchmarks._cases import benchmark_cases
from vibeqc import Calculator

CASES = (
    "water-tetramer-def2-svp-spherical",
    "water-octamer-s4-def2-svp-spherical",
)
ENERGY_PARITY_TOLERANCE = 1.0e-10  # Hartree, absolute; identical SCF settings.


def _git_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_git_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_metadata(library: Path) -> dict:
    """Identify the checkout and exact binary independently, before timing.

    A checkout revision alone cannot identify an externally supplied build.
    Preserve dirty state and a patch hash as well as the library's content hash.
    """
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=_git_root(), text=True
    )
    patch = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=_git_root()
    )
    return {
        "git_head": _git_revision(),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "repository": str(_git_root()),
        "native_library": str(library),
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def _validate_pair(energy: dict, energy_force: dict) -> dict:
    """Reject scientifically incomparable solves before publishing a delta."""
    for sample in (energy, energy_force):
        if sample["converged"] is not True:
            raise ValueError("force ledger requires both solves to converge")
        if not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
            raise ValueError("force ledger requires finite positive timings")
        if not math.isfinite(sample["energy_hartree"]):
            raise ValueError("force ledger requires finite energies")
    if energy["has_forces"] is not False or energy_force["has_forces"] is not True:
        raise ValueError("force ledger requires exactly the requested force outputs")
    if energy["iterations"] != energy_force["iterations"]:
        raise ValueError("force ledger requires matching SCF iteration counts")
    error = abs(energy["energy_hartree"] - energy_force["energy_hartree"])
    if error > ENERGY_PARITY_TOLERANCE:
        raise ValueError("force ledger energy parity tolerance exceeded")
    return {
        "valid": True,
        "energy_difference_hartree": error,
        "energy_tolerance_hartree": ENERGY_PARITY_TOLERANCE,
    }


def _sample(case_name: str, properties: tuple[str, ...], library: Path) -> dict:
    """Time one fresh solve using the explicitly selected native library."""
    case = benchmark_cases()[case_name]
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        screening_tolerance=1.0e-12,
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
    )
    if Path(calculator._library._name).resolve() != library:
        raise RuntimeError("Calculator loaded a different native library")
    started = time.perf_counter()
    result = calculator.singlepoint(
        case.atoms,
        charge=case.charge,
        multiplicity=case.multiplicity,
        properties=properties,
    )
    elapsed = time.perf_counter() - started
    if result.forces is not None and (
        result.forces.shape != (len(case.atoms), 3)
        or not np.isfinite(result.forces).all()
    ):
        raise ValueError("force ledger requires finite atom/xyz forces")
    return {
        "seconds": elapsed,
        "iterations": result.iterations,
        "converged": result.converged,
        "energy_hartree": result.energy,
        "has_forces": result.forces is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", dest="cases")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/issue206-force-ledger.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run requires a finite Slurm allocation (SLURM_JOB_ID)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run requires Slurm-provided CUDA_VISIBLE_DEVICES")
    library = args.library.resolve(strict=True)
    if not library.is_file():
        parser.error("--library must select a native shared library file")
    os.environ["VIBEQC_LIBRARY"] = str(library)
    source = _source_metadata(library)

    cases = args.cases or list(CASES)
    records = []
    for case_name in cases:
        for repeat in range(args.repeats):
            energy = _sample(case_name, ("energy",), library)
            energy_force = _sample(case_name, ("energy", "forces"), library)
            validation = _validate_pair(energy, energy_force)
            records.append(
                {
                    "case": case_name,
                    "repeat": repeat,
                    "energy_only": energy,
                    "energy_plus_force": energy_force,
                    "validation": validation,
                    "force_increment_seconds": energy_force["seconds"]
                    - energy["seconds"],
                }
            )

    if _source_metadata(library) != source:
        raise RuntimeError("source or native library changed during the benchmark")
    payload = {
        "schema": "vibeqc.issue206.df_force_ledger",
        "version": 2,
        "source": source,
        "execution": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "python": sys.executable,
            "platform": platform.platform(),
            "repeats": args.repeats,
            "warning": (
                "fresh single-system calculations; the difference diagnoses "
                "force cost and is not a warm-solve speed claim"
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
