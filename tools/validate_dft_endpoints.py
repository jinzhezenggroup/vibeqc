"""Record method-resolved CPU/CUDA DFT energy endpoint evidence.

The runner exercises prepared batch-1 and ragged workloads through cold,
fixed-geometry warm, changed-geometry and changed-geometry warm phases. It is a
correctness and execution-boundary artifact, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import typing
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc import Calculator
from vibeqc.autotune import source_identity

METHODS = ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
UKS_METHODS = frozenset(("lda-uks", "pbe-uks"))


def capture(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def timed(call: typing.Any) -> typing.Any:
    start = perf_counter()
    value = call()
    return value, (perf_counter() - start) * 1000.0


def systems_for(method: str, batch: int) -> typing.Any:
    h2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    if method in UKS_METHODS:
        inventory = ((h2, -1, 2), ([("H", (0.0, 0.0, 0.0))], 0, 2))
    else:
        inventory = ((h2, 0, 1), ([("He", (0.0, 0.0, 0.0))], 0, 1))
    systems = []
    charges = []
    multiplicities = []
    for item in range(batch):
        template, charge, multiplicity = inventory[item % len(inventory)]
        systems.append(
            [
                (element, (x, y, z + 0.01 * (item // len(inventory))))
                for element, (x, y, z) in template
            ]
        )
        charges.append(charge)
        multiplicities.append(multiplicity)
    return systems, charges, multiplicities


def changed_coordinates(systems: typing.Any) -> typing.Any:
    changed = [
        np.asarray([xyz for _, xyz in system], dtype=np.float64) for system in systems
    ]
    for item, coordinates in enumerate(changed):
        coordinates[-1, 2] += 0.025 + 0.005 * item
    return changed


def transport_payload(value: typing.Any) -> typing.Any:
    if value is None:
        return None
    return value.to_payload()


def transport_delta(previous: typing.Any, current: typing.Any) -> typing.Any:
    if previous is None or current is None:
        return None
    return {
        field: getattr(current, field) - getattr(previous, field)
        for field in current.__dataclass_fields__
    }


def result_record(
    result: typing.Any,
    milliseconds: float,
    transport_before: typing.Any,
    transport_after: typing.Any,
) -> typing.Any:
    """Keep physical convergence and actual prepared grids with every phase."""
    return {
        "milliseconds": milliseconds,
        "energies": result.energies.tolist(),
        "iterations": [item.iterations for item in result.items],
        "executed_backends": [item.executed_backend for item in result.items],
        "warm_start_used": [item.warm_start_used for item in result.items],
        "warm_start_fallback": [item.warm_start_fallback for item in result.items],
        "bucket_ids": [item.bucket_id for item in result.items],
        "converged": [item.converged for item in result.items],
        "physical_residuals": [item.physical_residual_rms for item in result.items],
        "grid_points": [item.ks_diagnostic.grid_points for item in result.items],
        "transport": [transport_payload(item) for item in transport_after],
        "transport_delta": [
            transport_delta(before, after)
            for before, after in zip(transport_before, transport_after, strict=True)
        ],
    }


def independent_energies(
    method: typing.Any,
    systems: typing.Any,
    charges: typing.Any,
    multiplicities: typing.Any,
    coordinates: typing.Any = None,
) -> typing.Any:
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cpu",
        max_iterations=200,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    values = []
    for index, system in enumerate(systems):
        atoms = system
        if coordinates is not None:
            atoms = [
                (element, tuple(coordinates[index][atom]))
                for atom, (element, _) in enumerate(system)
            ]
        result = calculator.singlepoint(
            atoms,
            charge=charges[index],
            multiplicity=multiplicities[index],
            properties=("energy",),
        )
        if not result.converged:
            raise RuntimeError(f"independent CPU {method} endpoint did not converge")
        values.append(result.energy)
    return np.asarray(values)


def validate_phase(record: typing.Any, expected: typing.Any) -> None:
    actual = np.asarray(record["energies"])
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(
            f"CPU/CUDA DFT energy mismatch: maximum error {np.max(np.abs(actual - expected))}"
        )
    if any(backend != "cuda" for backend in record["executed_backends"]):
        raise RuntimeError("DFT endpoint did not execute on CUDA")
    if not all(record["converged"]) or not all(
        residual < 1.0e-9 for residual in record["physical_residuals"]
    ):
        raise RuntimeError("DFT endpoint did not pass the physical convergence gate")


def run_case(method: str, batch: int) -> typing.Any:
    systems, charges, multiplicities = systems_for(method, batch)
    changed = changed_coordinates(systems)
    expected_initial = independent_energies(method, systems, charges, multiplicities)
    expected_changed = independent_energies(
        method, systems, charges, multiplicities, changed
    )
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        max_iterations=200,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    prepared, prepare_ms = timed(
        lambda: calculator.prepare_batch(
            systems,
            charges=charges,
            multiplicities=multiplicities,
            warm_start=True,
        )
    )
    try:
        phases = {}
        previous_transport = tuple(prepared.ks_transport_diagnostics)
        for name, coordinates in (
            ("cold", None),
            ("fixed_geometry", None),
            ("changed_geometry", changed),
            ("changed_geometry_fixed", changed),
        ):
            result, elapsed = timed(
                lambda coordinates=coordinates: prepared.execute(
                    coordinates=coordinates, properties=("energy",), strict=True
                )
            )
            current_transport = tuple(prepared.ks_transport_diagnostics)
            phases[name] = result_record(
                result, elapsed, previous_transport, current_transport
            )
            previous_transport = current_transport
        validate_phase(phases["cold"], expected_initial)
        validate_phase(phases["fixed_geometry"], expected_initial)
        validate_phase(phases["changed_geometry"], expected_changed)
        validate_phase(phases["changed_geometry_fixed"], expected_changed)
        if any(phases["cold"]["warm_start_used"]):
            raise RuntimeError("cold DFT batch unexpectedly used a warm density")
        if not all(phases["fixed_geometry"]["warm_start_used"]):
            raise RuntimeError("fixed-geometry DFT replay did not use warm densities")
        # Resident KS rebuilds geometry-bound owners and normalizes the last
        # good density in the target overlap metric. Independent target energies
        # and physical residuals above guard against reuse of the old operator.
        if not all(phases["changed_geometry"]["warm_start_used"]):
            raise RuntimeError("changed geometry lost compatible DFT warm seeds")
        if not all(phases["changed_geometry_fixed"]["warm_start_used"]):
            raise RuntimeError(
                "changed-geometry replay did not retain new warm densities"
            )
    finally:
        prepared.close()
    return {
        "method": method,
        "batch": batch,
        "grid_points": phases["cold"]["grid_points"],
        "setup_milliseconds": prepare_ms,
        "phases": phases,
        "component_costs": {
            "endpoint_phase_totals_recorded": True,
            "internal_j_xc_breakdown": None,
            "transport_fields": [
                "setup_h2d_bytes",
                "density_h2d_bytes",
                "scalar_d2h_bytes",
                "matrix_d2h_bytes",
                "synchronizations",
            ],
            "reason": "native CUDA KS exposes measured transport counters; separate J/XC event timing remains unavailable",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches):
        parser.error("batch sizes must be positive")
    if capture(["git", "-C", str(ROOT), "status", "--porcelain"]):
        parser.error("DFT endpoint evidence requires a clean source checkout")
    library_path = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    library = ctypes.CDLL(str(library_path))
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    native_identity = library.vibeqc_get_source_identity().decode()
    if native_identity != source_identity(ROOT):
        parser.error("selected native library does not match this source checkout")
    report = {
        "schema": "vibeqc.dft-endpoints.v1",
        "revision": capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "native_source_identity": native_identity,
        "library": str(library_path),
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "scope": {
            "properties": ["energy"],
            "coulomb": "conventional exact J",
            "density_fitting": "unsupported",
            "scf_control": "host scalar control with resident CUDA J/XC/Fock/DIIS/eigensolve state",
            "performance_claim": False,
        },
        "runs": [],
    }
    for method in args.methods:
        for batch in args.batches:
            report["runs"].append(run_case(method, batch))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"DFT endpoint validation passed: {len(report['runs'])} cases")


if __name__ == "__main__":
    main()
