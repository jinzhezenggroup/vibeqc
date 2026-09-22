"""Measure current cold, warm, and changed-geometry DF endpoints.

The ordinary matched runner already records cold and unchanged-geometry warm
replays.  This companion keeps the changed-geometry boundary explicit: each
timed changed replay starts from the same post-cold engine-local density after
an untimed original-geometry prime.  It supports the same canonical orbital /
auxiliary basis snapshots as the matched runner, including unequal ``Naux``.

The result is an evidence record, not an automatic performance claim.  Cold,
warm, and changed samples have different preparation semantics and are kept in
separate sections so a warm result cannot be reused as an MD/optimization
throughput claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
# Keep protocol preflight usable from a checkout even when the caller has not
# exported PYTHONPATH; GPU package imports still occur only after Slurm checks.
for import_root in (PYTHON_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmarks._cases import benchmark_cases
from benchmarks._retention import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import (
    GpuCycleTracker,
    convergence_payload,
    gpu_convergence_payload,
    load_comparison_basis,
    native_build_metadata,
    scaled_geometries,
)


def _sync(cp: Any) -> None:
    """Drain the default stream at every timing boundary."""

    cp.cuda.Stream.null.synchronize()


def _coordinates(
    case: Any, batch_size: int, displacement: float
) -> tuple[Any, list[np.ndarray], list[np.ndarray]]:
    """Return fixed-topology original and displaced coordinate arrays."""

    systems = scaled_geometries(case.atoms, batch_size)
    original = [
        np.asarray([position for _, position in atoms], dtype=np.float64)
        for atoms in systems
    ]
    changed = [coordinates.copy() for coordinates in original]
    if any(len(coordinates) < 2 for coordinates in changed):
        raise ValueError("changed-geometry evidence requires at least two atoms")
    for coordinates in changed:
        coordinates[1, 0] += displacement
    return systems, original, changed


def _native_sample(batch: Any, cp: Any, coordinates: Any = None) -> dict[str, Any]:
    """Execute one synchronized native endpoint and serialize its result."""

    _sync(cp)
    started = time.perf_counter()
    result = batch.execute(
        None if coordinates is None else coordinates,
        strict=True,
        properties=("energy", "forces"),
    )
    _sync(cp)
    return {
        "seconds": time.perf_counter() - started,
        "energies_hartree": result.energies.tolist(),
        "forces_hartree_per_bohr": [item.forces.tolist() for item in result.items],
        "convergence": convergence_payload(result),
    }


def _stock_sample(
    engines: Any,
    densities: Any,
    cp: Any,
    *,
    systems: Any = None,
    coordinates: Any = None,
) -> dict[str, Any]:
    """Execute one synchronized GPU4PySCF endpoint from fixed density seeds."""

    trackers = [GpuCycleTracker() for _ in engines]
    for engine, tracker in zip(engines, trackers, strict=True):
        engine.callback = tracker
    _sync(cp)
    started = time.perf_counter()
    if coordinates is not None:
        if systems is None:
            raise ValueError("changed reference sample requires system identities")
        _reset_stock(engines, systems, coordinates)
    energies = [
        engine.kernel(dm0=density.copy())
        for engine, density in zip(engines, densities, strict=True)
    ]
    _require_converged_reference(engines)
    gradients = [engine.nuc_grad_method().kernel() for engine in engines]
    _sync(cp)
    return {
        "seconds": time.perf_counter() - started,
        "energies_hartree": [float(energy) for energy in energies],
        "forces_hartree_per_bohr": [
            cp.asnumpy(-gradient).tolist() for gradient in gradients
        ],
        "convergence": gpu_convergence_payload(engines, trackers),
    }


def _require_converged_reference(engines: Any) -> None:
    """A finite reference energy is not evidence of a stationary SCF state."""
    if not all(bool(engine.converged) for engine in engines):
        raise RuntimeError("reference SCF did not converge; endpoint evidence rejected")


def _reset_stock(engines: Any, systems: Any, coordinates: Any) -> None:
    """Reset stock molecules; changed-geometry samples include this in their timer."""

    for engine, atoms, xyz in zip(engines, systems, coordinates, strict=True):
        engine.reset(
            engine.mol.set_geom_(
                [
                    (element, tuple(position))
                    for (element, _), position in zip(atoms, xyz, strict=True)
                ],
                unit="Bohr",
                inplace=True,
            )
        )


def _paired_errors(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Return complete endpoint errors without selecting a favorable repeat."""

    def arrays(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        try:
            energy = np.asarray(sample["energies_hartree"])
            force = np.asarray(sample["forces_hartree_per_bohr"])
        except (TypeError, ValueError) as error:
            raise ValueError("malformed endpoint result arrays") from error
        if any(value.dtype.kind not in "iuf" for value in (energy, force)):
            raise TypeError("endpoint arrays must contain real numeric values")
        if energy.ndim != 1 or not energy.size:
            raise ValueError("endpoint energies must be a nonempty batch vector")
        if (
            force.ndim != 3
            or force.shape[0] != energy.size
            or force.shape[1] == 0
            or force.shape[2] != 3
        ):
            raise ValueError(
                "endpoint forces must have matching batch-by-atom-by-3 shape"
            )
        energy = energy.astype(np.float64, copy=False)
        force = force.astype(np.float64, copy=False)
        if not np.isfinite(energy).all() or not np.isfinite(force).all():
            raise ValueError("endpoint arrays must contain finite values")
        return energy, force

    left_energy, left_force = arrays(left)
    right_energy, right_force = arrays(right)
    if left_energy.shape != right_energy.shape or left_force.shape != right_force.shape:
        raise ValueError(
            "paired endpoint shapes differ; broadcasting is not qualification"
        )
    energy = float(np.max(np.abs(left_energy - right_energy)))
    force = float(np.max(np.abs(left_force - right_force)))
    return {
        "maximum_energy_error_hartree": energy,
        "maximum_force_error_hartree_per_bohr": force,
    }


def _case_inputs(args: Any, case: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Resolve shared native/PySCF basis inputs before GPU package setup."""

    native_orbital, reference_orbital = case.vibeqc_basis, case.pyscf_basis
    overrides: dict[str, Any] = {}
    if args.orbital_basis_file:
        native_orbital, reference_orbital = load_comparison_basis(
            args.orbital_basis_file, case, role="orbital", compute_forces=True
        )
        overrides["orbital"] = native_orbital.to_payload()
    native_auxiliary, reference_auxiliary = native_orbital, reference_orbital
    if args.auxiliary_basis_file:
        native_auxiliary, reference_auxiliary = load_comparison_basis(
            args.auxiliary_basis_file, case, role="auxiliary", compute_forces=True
        )
        overrides["auxiliary"] = native_auxiliary.to_payload()
    return (
        native_orbital,
        reference_orbital,
        native_auxiliary,
        reference_auxiliary,
        overrides,
    )


def _stock_engines(
    case: Any,
    systems: Any,
    reference_orbital: Any,
    reference_auxiliary: Any,
    cp: Any,
    scf: Any,
    gto: Any,
    gpu_uhf: Any,
) -> list[Any]:
    """Construct one GPU4PySCF engine per fixed-topology batch item."""

    engines = []
    for atoms in systems:
        molecule = gto.M(
            atom=atoms,
            unit="Bohr",
            charge=case.charge,
            spin=case.multiplicity - 1,
            cart=case.basis_representation == "cartesian",
            basis=reference_orbital,
            verbose=0,
        )
        engine = gpu_uhf.UHF(molecule) if case.method == "uhf" else scf.RHF(molecule)
        engine = engine.density_fit(auxbasis=reference_auxiliary).to_gpu()
        engine.conv_tol = 1.0e-12
        engine.conv_tol_grad = 1.0e-10
        engine.direct_scf_tol = 1.0e-14
        engine.max_cycle = 100
        engines.append(engine)
    return engines


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw timings while retaining every sample in the result."""

    values = [float(row["seconds"]) for row in samples]
    median = statistics.median(values)
    return {
        "samples_seconds": values,
        "median_seconds": median,
        "relative_median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        )
        / median,
    }


def main() -> None:
    """Run one current cold/warm/changed endpoint matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    cases = benchmark_cases()
    parser.add_argument("--case", choices=sorted(cases), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--displacement", type=float, default=1.0e-3)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--orbital-basis-file", type=Path)
    parser.add_argument("--auxiliary-basis-file", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--maximum-energy-error", type=float, default=3.0e-11)
    parser.add_argument("--maximum-force-error", type=float, default=3.0e-11)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("at least five repeats are required")
    if args.output.exists():
        parser.error("refusing to overwrite an existing evidence file")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run inside a finite Slurm GPU allocation")
    library = args.library.resolve(strict=True)
    os.environ["VIBEQC_LIBRARY"] = str(library)

    # Import GPU packages only after protocol validation, as in the matched
    # runner; this makes --help and basis preflight safe on login nodes.
    import cupy as cp
    from gpu4pyscf.scf import uhf as gpu_uhf
    from pyscf import df, gto, scf
    from vibeqc import Calculator

    case = cases[args.case]
    (
        native_orbital,
        reference_orbital,
        native_auxiliary,
        reference_auxiliary,
        overrides,
    ) = _case_inputs(args, case)
    systems, original, changed = _coordinates(case, args.batch, args.displacement)
    serialized_original = [
        [
            {"element": element, "coordinates_bohr": list(position)}
            for element, position in atoms
        ]
        for atoms in systems
    ]

    calculator = Calculator(
        method=case.method,
        basis=native_orbital,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        screening_tolerance=1.0e-12,
        density_fitting="cuda",
        auxiliary_basis=native_auxiliary,
    )
    build = native_build_metadata(calculator)
    metric = []
    native_cold = None
    native_warm: list[dict[str, Any]] = []
    native_changed: list[dict[str, Any]] = []
    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
        warm_start=True,
    ) as batch:
        _sync(cp)
        started = time.perf_counter()
        cold_result = batch.execute(strict=True, properties=("energy", "forces"))
        _sync(cp)
        native_cold = {
            "seconds": time.perf_counter() - started,
            "energies_hartree": cold_result.energies.tolist(),
            "forces_hartree_per_bohr": [
                item.forces.tolist() for item in cold_result.items
            ],
            "convergence": convergence_payload(cold_result),
        }
        metric = [
            row.to_dict() for row in batch.last_density_fitting_metric_diagnostics()
        ]
        batch.set_warm_start_updates(False)
        _native_sample(batch, cp)
        for _ in range(args.repeats):
            native_warm.append(_native_sample(batch, cp))
        for _ in range(args.repeats):
            # Restore the original input outside the changed timer.  The
            # retained post-cold density remains frozen by the explicit flag.
            batch.execute(original, strict=True, properties=("energy", "forces"))
            native_changed.append(_native_sample(batch, cp, changed))

    _sync(cp)
    stock_cold = None
    stock_warm: list[dict[str, Any]] = []
    stock_changed: list[dict[str, Any]] = []
    engines = _stock_engines(
        case, systems, reference_orbital, reference_auxiliary, cp, scf, gto, gpu_uhf
    )
    ao_count = int(engines[0].mol.nao_nr())
    naux_count = int(
        df.addons.make_auxmol(engines[0].mol, reference_auxiliary).nao_nr()
    )
    started = time.perf_counter()
    cold_trackers = [GpuCycleTracker() for _ in engines]
    for engine, tracker in zip(engines, cold_trackers, strict=True):
        engine.callback = tracker
    cold_energies = [engine.kernel() for engine in engines]
    _require_converged_reference(engines)
    cold_gradients = [engine.nuc_grad_method().kernel() for engine in engines]
    _sync(cp)
    stock_cold = {
        "seconds": time.perf_counter() - started,
        "energies_hartree": [float(value) for value in cold_energies],
        "forces_hartree_per_bohr": [
            cp.asnumpy(-value).tolist() for value in cold_gradients
        ],
        "convergence": gpu_convergence_payload(engines, cold_trackers),
    }
    densities = [engine.make_rdm1().copy() for engine in engines]
    _stock_sample(engines, densities, cp)
    for _ in range(args.repeats):
        stock_warm.append(_stock_sample(engines, densities, cp))
    for _ in range(args.repeats):
        _reset_stock(engines, systems, original)
        _stock_sample(engines, densities, cp)
        stock_changed.append(
            _stock_sample(engines, densities, cp, systems=systems, coordinates=changed)
        )

    def paired(
        left: list[dict[str, Any]], right: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        return [_paired_errors(a, b) for a, b in zip(left, right, strict=True)]

    warm_pairs = paired(native_warm, stock_warm)
    changed_pairs = paired(native_changed, stock_changed)
    package_versions = {}
    for package in ("gpu4pyscf-cuda12x", "cupy-cuda12x", "pyscf", "numpy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    payload = {
        "schema": "vibeqc.issue206.rebuild",
        "version": 2,
        "status": "pass"
        if all(
            pair["maximum_energy_error_hartree"] <= args.maximum_energy_error
            and pair["maximum_force_error_hartree_per_bohr"] <= args.maximum_force_error
            for pair in (*warm_pairs, *changed_pairs)
        )
        else "numerical-gate-failed",
        "source_identity": build["probe"]["source_identity"],
        "library_sha256": build["library_sha256"],
        "library": str(library),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "host": platform.platform(),
        "packages": package_versions,
        "workload": {
            "case": args.case,
            "method": case.method,
            "ao_count": ao_count,
            "naux_count": naux_count,
            "batch_size": args.batch,
            "basis_representation": case.basis_representation,
            "basis_overrides": overrides,
            "auxiliary_basis": native_auxiliary.name
            if args.auxiliary_basis_file
            else "same as orbital basis",
            "geometries": serialized_original,
            "changed_displacement_bohr": args.displacement,
            "energy_tolerance": 1.0e-12,
            "density_tolerance": 1.0e-10,
            "reference_gradient_tolerance": 1.0e-10,
            "maximum_energy_error_hartree": args.maximum_energy_error,
            "maximum_force_error_hartree_per_bohr": args.maximum_force_error,
        },
        "native_build": build,
        "metric": metric,
        "protocol": {
            "repeats": args.repeats,
            "warm_policy": "fixed post-cold engine-local density snapshot with one untimed original-geometry prime",
            "changed_policy": "restore original geometry and prime outside each changed timer; time geometry reset, SCF and complete forces",
            "endpoint_boundary": "synchronized host energies and complete host forces",
        },
        "vibeqc": {
            "cold": native_cold,
            "warm": native_warm,
            "warm_summary": _summary(native_warm),
            "changed": native_changed,
            "changed_summary": _summary(native_changed),
        },
        "gpu4pyscf": {
            "cold": stock_cold,
            "warm": stock_warm,
            "warm_summary": _summary(stock_warm),
            "changed": stock_changed,
            "changed_summary": _summary(stock_changed),
        },
        "accuracy": {
            "warm_pairs": warm_pairs,
            "changed_pairs": changed_pairs,
            "maximum_warm_energy_error_hartree": max(
                row["maximum_energy_error_hartree"] for row in warm_pairs
            ),
            "maximum_warm_force_error_hartree_per_bohr": max(
                row["maximum_force_error_hartree_per_bohr"] for row in warm_pairs
            ),
            "maximum_changed_energy_error_hartree": max(
                row["maximum_energy_error_hartree"] for row in changed_pairs
            ),
            "maximum_changed_force_error_hartree_per_bohr": max(
                row["maximum_force_error_hartree_per_bohr"] for row in changed_pairs
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
