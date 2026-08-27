"""Interleaved homogeneous-batch VibeQC/GPU4PySCF parity benchmark.

VibeQC executes one native fixed-topology bucket. GPU4PySCF currently exposes
a single-molecule SCF interface, so one initialized GPU object and warm density
are retained per system. Warm samples are interleaved in a deterministic ABBA
order to reduce clock and thermal drift, and every timing remains paired with
the SCF branch and numerical result that produced it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import statistics
import time
from typing import Any, Iterator, Sequence

import numpy as np

from vibeqc import Calculator

from _cases import benchmark_cases
from _support import (
    benchmark_gate_failures,
    cuda_accelerator_metadata,
    environment_metadata,
    write_result,
)


VIBEQC_ENGINE = "vibeqc"
GPU4PYSCF_ENGINE = "gpu4pyscf"


def convergence_payload(result) -> list[dict[str, object]]:
    """Serialize one VibeQC replay's per-system convergence diagnostics."""

    return [
        {
            "converged": item.converged,
            "iterations": item.iterations,
            # Retain the schema-v1 flat fields for readers that have not yet
            # adopted the explicit residual/warm-start groups.
            "energy_change_hartree": item.energy_change,
            "density_rms": item.density_rms,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "final_residuals": {
                "energy_change_hartree": item.energy_change,
                "density_rms": item.density_rms,
                "orbital_gradient_norm": None,
            },
            "warm_start": {
                "used": item.warm_start_used,
                "fallback": item.warm_start_fallback,
            },
        }
        for item in result.items
    ]


class GpuCycleTracker:
    """Collect GPU4PySCF cycle count and final callback residuals.

    PySCF callbacks receive a backend-defined locals dictionary. The tracker
    intentionally tolerates missing optional norms so benchmark collection
    remains compatible across pinned GPU4PySCF/PySCF patch releases.
    """

    def __init__(self) -> None:
        self.iterations = 0
        self.energy_change_hartree: float | None = None
        self.density_rms: float | None = None
        self.orbital_gradient_norm: float | None = None
        self._previous_energy: float | None = None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def __call__(self, environment: dict[str, Any]) -> None:
        """Record the latest explicitly reported SCF cycle and residuals."""

        cycle = environment.get("cycle")
        if cycle is not None:
            self.iterations = max(self.iterations, int(cycle) + 1)
        else:
            self.iterations += 1

        energy = self._optional_float(environment.get("e_tot"))
        reported_change = self._optional_float(environment.get("de"))
        if reported_change is not None:
            self.energy_change_hartree = abs(reported_change)
        elif energy is not None and self._previous_energy is not None:
            self.energy_change_hartree = abs(energy - self._previous_energy)
        if energy is not None:
            self._previous_energy = energy

        self.density_rms = self._optional_float(environment.get("norm_ddm"))
        self.orbital_gradient_norm = self._optional_float(
            environment.get("norm_gorb")
        )


def gpu_convergence_payload(
    engines: Sequence[Any], trackers: Sequence[GpuCycleTracker]
) -> list[dict[str, object]]:
    """Serialize per-system GPU4PySCF diagnostics for one batch sample."""

    payload = []
    for engine, tracker in zip(engines, trackers, strict=True):
        # Newer PySCF releases expose ``cycles`` after ``kernel``. Prefer it
        # when present, while retaining callback counting as the portable path.
        reported_cycles = getattr(engine, "cycles", None)
        iterations = (
            int(reported_cycles)
            if reported_cycles is not None
            else tracker.iterations
        )
        payload.append(
            {
                "converged": bool(engine.converged),
                "iterations": iterations,
                "final_residuals": {
                    "energy_change_hartree": tracker.energy_change_hartree,
                    "density_rms": tracker.density_rms,
                    "orbital_gradient_norm": tracker.orbital_gradient_norm,
                },
                "warm_start": {
                    "used": True,
                    "fallback": False,
                },
            }
        )
    return payload


def interleaved_engine_order(repeats: int) -> tuple[str, ...]:
    """Return exactly ``repeats`` samples per engine in ABBA blocks."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    order: list[str] = []
    counts = {VIBEQC_ENGINE: 0, GPU4PYSCF_ENGINE: 0}
    block = (VIBEQC_ENGINE, GPU4PYSCF_ENGINE, GPU4PYSCF_ENGINE, VIBEQC_ENGINE)
    while counts[VIBEQC_ENGINE] < repeats or counts[GPU4PYSCF_ENGINE] < repeats:
        for engine in block:
            if counts[engine] >= repeats:
                continue
            order.append(engine)
            counts[engine] += 1
    return tuple(order)


def iteration_branch(sample: dict[str, Any]) -> tuple[int, ...]:
    """Return the per-system iteration tuple identifying one SCF branch."""

    return tuple(int(item["iterations"]) for item in sample["convergence"])


def iteration_matched_summary(
    vibeqc_samples: Sequence[dict[str, Any]],
    gpu_samples: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize the best-supported SCF branch shared by both engines."""

    vibeqc_by_branch: dict[tuple[int, ...], list[float]] = {}
    gpu_by_branch: dict[tuple[int, ...], list[float]] = {}
    for sample in vibeqc_samples:
        vibeqc_by_branch.setdefault(iteration_branch(sample), []).append(
            float(sample["seconds"])
        )
    for sample in gpu_samples:
        gpu_by_branch.setdefault(iteration_branch(sample), []).append(
            float(sample["seconds"])
        )
    shared = set(vibeqc_by_branch) & set(gpu_by_branch)
    if not shared:
        return None
    branch = min(
        shared,
        key=lambda item: (
            -min(len(vibeqc_by_branch[item]), len(gpu_by_branch[item])),
            item,
        ),
    )
    vibeqc_seconds = vibeqc_by_branch[branch]
    gpu_seconds = gpu_by_branch[branch]
    vibeqc_median = statistics.median(vibeqc_seconds)
    gpu_median = statistics.median(gpu_seconds)
    return {
        "iteration_branch": list(branch),
        "vibeqc_sample_count": len(vibeqc_seconds),
        "gpu4pyscf_sample_count": len(gpu_seconds),
        "vibeqc_median_seconds": vibeqc_median,
        "gpu4pyscf_median_seconds": gpu_median,
        "speedup": gpu_median / vibeqc_median,
    }


def pair_repeat_accuracy(
    vibeqc_samples: Sequence[dict[str, Any]],
    gpu_samples: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each engine's nth warm result and calculate numerical parity."""

    if len(vibeqc_samples) != len(gpu_samples):
        raise ValueError("warm sample counts must match")
    pairs = []
    for repeat, (vibeqc, gpu) in enumerate(
        zip(vibeqc_samples, gpu_samples, strict=True)
    ):
        vibeqc_energies = np.asarray(vibeqc["energies_hartree"])
        gpu_energies = np.asarray(gpu["energies_hartree"])
        vibeqc_forces = np.asarray(vibeqc["forces_hartree_per_bohr"])
        gpu_forces = np.asarray(gpu["forces_hartree_per_bohr"])
        pairs.append(
            {
                "repeat": repeat,
                "iteration_branches_match": (
                    iteration_branch(vibeqc) == iteration_branch(gpu)
                ),
                "maximum_energy_error_hartree": float(
                    np.max(np.abs(vibeqc_energies - gpu_energies))
                ),
                "maximum_force_error_hartree_per_bohr": float(
                    np.max(np.abs(vibeqc_forces - gpu_forces))
                ),
            }
        )
    return pairs


def accuracy_gate_summary(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select branch-matched accuracy rows when the engines share them.

    Every repeat remains published. The gate uses matched rows when available
    because a looser SCF branch can move analytic forces at the same scale as
    the tight benchmark tolerance even though both engines report convergence.
    """

    matched = [item for item in pairs if item["iteration_branches_match"]]
    # Schema v1 gated the final warm result. Preserve that established
    # accuracy contract when no ordinal repeat shares an iteration branch,
    # while publishing the larger all-repeat maximum immediately beside it.
    selected = matched or [pairs[-1]]
    return {
        "selection": (
            "iteration_matched_pairs" if matched else "final_pair_unmatched_labeled"
        ),
        "pair_count": len(selected),
        "maximum_energy_error_hartree": max(
            item["maximum_energy_error_hartree"] for item in selected
        ),
        "maximum_force_error_hartree_per_bohr": max(
            item["maximum_force_error_hartree_per_bohr"] for item in selected
        ),
    }


@contextmanager
def nvtx_range(cupy_module: Any, label: str) -> Iterator[None]:
    """Annotate profiler captures without making NVTX a hard dependency."""

    nvtx = getattr(cupy_module.cuda, "nvtx", None)
    if nvtx is None:
        yield
        return
    nvtx.RangePush(label)
    try:
        yield
    finally:
        nvtx.RangePop()


def scaled_geometries(atoms, batch_size: int):
    """Create nearby fixed-topology geometries without changing the centroid."""

    coordinates = np.asarray([position for _, position in atoms], dtype=np.float64)
    centroid = coordinates.mean(axis=0)
    systems = []
    for index in range(batch_size):
        centered_index = index - 0.5 * (batch_size - 1)
        scale = 1.0 + 0.002 * centered_index
        displaced = centroid + scale * (coordinates - centroid)
        systems.append(
            tuple(
                (
                    atoms[atom][0],
                    tuple(float(component) for component in displaced[atom]),
                )
                for atom in range(len(atoms))
            )
        )
    return systems


def _vibeqc_sample(batch: Any, cupy_module: Any, sequence_index: int) -> dict[str, Any]:
    """Execute and serialize one synchronized VibeQC warm sample."""

    cupy_module.cuda.Stream.null.synchronize()
    with nvtx_range(cupy_module, "vibeqc/warm/energy-plus-force"):
        start = time.perf_counter()
        result = batch.execute(strict=True)
        cupy_module.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "sequence_index": sequence_index,
        "seconds": elapsed,
        "component_seconds": {"energy_plus_force": elapsed},
        "convergence": convergence_payload(result),
        "energies_hartree": result.energies.tolist(),
        "forces_hartree_per_bohr": np.stack(
            [item.forces for item in result.items]
        ).tolist(),
    }


def _gpu_sample(
    engines: Sequence[Any],
    warm_densities: Sequence[Any],
    cupy_module: Any,
    sequence_index: int,
) -> dict[str, Any]:
    """Execute one synchronized GPU4PySCF warm energy-plus-force sample."""

    # Reuse the same post-cold converged density for every repeat. Advancing
    # dm0 from the previous warm result makes one nondeterministic SCF branch
    # contaminate every later sample and can turn a transient direct-J/K
    # reduction difference into a 100-cycle failure at 192 AOs.
    densities = [density.copy() for density in warm_densities]
    trackers = [GpuCycleTracker() for _ in engines]
    for engine, tracker in zip(engines, trackers, strict=True):
        engine.callback = tracker

    cupy_module.cuda.Stream.null.synchronize()
    total_start = time.perf_counter()
    with nvtx_range(cupy_module, "gpu4pyscf/warm/scf"):
        scf_start = time.perf_counter()
        energies = [
            engine.kernel(dm0=density)
            for engine, density in zip(engines, densities, strict=True)
        ]
        cupy_module.cuda.Stream.null.synchronize()
        scf_seconds = time.perf_counter() - scf_start
    with nvtx_range(cupy_module, "gpu4pyscf/warm/force"):
        force_start = time.perf_counter()
        gradients = [engine.nuc_grad_method().kernel() for engine in engines]
        cupy_module.cuda.Stream.null.synchronize()
        force_seconds = time.perf_counter() - force_start
    elapsed = time.perf_counter() - total_start

    return {
        "sequence_index": sequence_index,
        "seconds": elapsed,
        "component_seconds": {
            "scf": scf_seconds,
            "force": force_seconds,
        },
        "convergence": gpu_convergence_payload(engines, trackers),
        "energies_hartree": [float(energy) for energy in energies],
        "forces_hartree_per_bohr": np.stack(
            [cupy_module.asnumpy(-gradient) for gradient in gradients]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    cases = benchmark_cases()
    parser.add_argument("--case", choices=cases, default="sp8")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--reference-gradient-tolerance",
        type=float,
        default=1.0e-10,
        help="GPU4PySCF orbital-gradient convergence threshold",
    )
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--minimum-speedup", type=float)
    parser.add_argument("--maximum-energy-error", type=float)
    parser.add_argument("--maximum-force-error", type=float)
    parser.add_argument(
        "--capture-warm-range",
        action="store_true",
        help=(
            "delimit all interleaved warm samples with the CUDA profiler API "
            "for Nsight Systems capture-range profiling"
        ),
    )
    parser.add_argument(
        "--output",
        help="optional JSON path for raw timings and reproducibility metadata",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1 or args.max_iterations < 1:
        raise ValueError("--batch, --repeats, and --max-iterations must be positive")
    if (
        args.energy_tolerance <= 0.0
        or args.density_tolerance <= 0.0
        or args.reference_gradient_tolerance <= 0.0
        or args.screening_tolerance <= 0.0
    ):
        raise ValueError("SCF tolerances must be positive")
    if args.minimum_speedup is not None and args.minimum_speedup <= 0.0:
        raise ValueError("--minimum-speedup must be positive")
    if args.maximum_energy_error is not None and args.maximum_energy_error < 0.0:
        raise ValueError("--maximum-energy-error must be non-negative")
    if args.maximum_force_error is not None and args.maximum_force_error < 0.0:
        raise ValueError("--maximum-force-error must be non-negative")

    # Import GPU packages only after argument parsing so workload construction
    # and --help remain usable on login nodes without an allocated device.
    import cupy as cp
    from pyscf import gto, scf
    from gpu4pyscf.scf import uhf as gpu_uhf

    case = cases[args.case]
    systems = scaled_geometries(case.atoms, args.batch)
    reference_molecule = gto.M(
        atom=systems[0],
        unit="Bohr",
        charge=case.charge,
        spin=case.multiplicity - 1,
        cart=case.basis_representation == "cartesian",
        basis=case.pyscf_basis,
        verbose=0,
    )
    ao_count = int(reference_molecule.nao_nr())
    if case.expected_ao_count is not None and ao_count != case.expected_ao_count:
        raise ValueError(
            f"{args.case} expected {case.expected_ao_count} AOs, "
            f"but PySCF constructed {ao_count}"
        )

    gpu_objects = []
    for atoms in systems:
        molecule = gto.M(
            atom=atoms,
            unit="Bohr",
            charge=case.charge,
            spin=case.multiplicity - 1,
            cart=case.basis_representation == "cartesian",
            basis=case.pyscf_basis,
            verbose=0,
        )
        if molecule.nao_nr() != ao_count:
            raise ValueError("scaled fixed-topology geometry changed AO count")
        engine = (
            gpu_uhf.UHF(molecule)
            if case.method == "uhf"
            else scf.RHF(molecule).to_gpu()
        )
        engine.conv_tol = args.energy_tolerance
        engine.conv_tol_grad = args.reference_gradient_tolerance
        engine.direct_scf_tol = 1.0e-14
        engine.max_cycle = args.max_iterations
        gpu_objects.append(engine)

    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=args.max_iterations,
        energy_tolerance=args.energy_tolerance,
        density_tolerance=args.density_tolerance,
        screening_tolerance=args.screening_tolerance,
    )
    vibeqc_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    measurement_order = interleaved_engine_order(args.repeats)

    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
        warm_start=True,
    ) as batch:
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        vibeqc_cold_result = batch.execute(strict=True)
        cp.cuda.Stream.null.synchronize()
        vibeqc_cold = time.perf_counter() - start

        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu_cold_trackers = [GpuCycleTracker() for _ in gpu_objects]
        for engine, tracker in zip(gpu_objects, gpu_cold_trackers, strict=True):
            engine.callback = tracker
        gpu_cold_energies = [engine.kernel() for engine in gpu_objects]
        gpu_cold_gradients = [
            engine.nuc_grad_method().kernel() for engine in gpu_objects
        ]
        cp.cuda.Stream.null.synchronize()
        gpu_cold = time.perf_counter() - start
        gpu_cold_convergence = gpu_convergence_payload(
            gpu_objects, gpu_cold_trackers
        )
        gpu_warm_densities = [engine.make_rdm1().copy() for engine in gpu_objects]

        if args.capture_warm_range:
            cp.cuda.profiler.start()
        try:
            for sequence_index, engine in enumerate(measurement_order):
                if engine == VIBEQC_ENGINE:
                    vibeqc_samples.append(
                        _vibeqc_sample(batch, cp, sequence_index)
                    )
                else:
                    gpu_samples.append(
                        _gpu_sample(
                            gpu_objects, gpu_warm_densities, cp, sequence_index
                        )
                    )
        finally:
            if args.capture_warm_range:
                cp.cuda.profiler.stop()

    repeat_accuracy = pair_repeat_accuracy(vibeqc_samples, gpu_samples)
    maximum_energy_error = max(
        item["maximum_energy_error_hartree"] for item in repeat_accuracy
    )
    maximum_force_error = max(
        item["maximum_force_error_hartree_per_bohr"] for item in repeat_accuracy
    )
    gate_accuracy = accuracy_gate_summary(repeat_accuracy)
    vibeqc_warm = [float(sample["seconds"]) for sample in vibeqc_samples]
    gpu_warm = [float(sample["seconds"]) for sample in gpu_samples]
    vibeqc_warm_median = statistics.median(vibeqc_warm)
    gpu_warm_median = statistics.median(gpu_warm)
    ordinary_speedup = gpu_warm_median / vibeqc_warm_median
    matched = iteration_matched_summary(vibeqc_samples, gpu_samples)
    matched_speedup = None if matched is None else float(matched["speedup"])
    speedup_for_gate = matched_speedup or ordinary_speedup

    vibeqc_converged = all(
        item["converged"]
        for sample in vibeqc_samples
        for item in sample["convergence"]
    )
    reference_converged = all(
        item["converged"]
        for sample in gpu_samples
        for item in sample["convergence"]
    )
    gate_failures = benchmark_gate_failures(
        speedup=speedup_for_gate,
        maximum_energy_error=gate_accuracy["maximum_energy_error_hartree"],
        maximum_force_error=gate_accuracy[
            "maximum_force_error_hartree_per_bohr"
        ],
        vibeqc_converged=vibeqc_converged,
        reference_converged=reference_converged,
        minimum_speedup=args.minimum_speedup,
        maximum_energy_error_limit=args.maximum_energy_error,
        maximum_force_error_limit=args.maximum_force_error,
    )
    print(
        f"scope: {case.description}, {ao_count} AOs, "
        f"homogeneous batch {args.batch}"
    )
    print("warm measurement order: " + " ".join(measurement_order))
    print(f"maximum warm energy difference: {maximum_energy_error:.3e} Eh")
    print(f"maximum warm force difference: {maximum_force_error:.3e} Eh/bohr")
    print(
        f"accuracy gate ({gate_accuracy['selection']}): "
        f"{gate_accuracy['maximum_energy_error_hartree']:.3e} Eh, "
        f"{gate_accuracy['maximum_force_error_hartree_per_bohr']:.3e} Eh/bohr"
    )
    print(f"VibeQC/reference converged: {vibeqc_converged}/{reference_converged}")
    print(f"VibeQC cold batch: {vibeqc_cold * 1e3:.3f} ms")
    print(
        f"VibeQC warm median/min: {vibeqc_warm_median * 1e3:.3f}/"
        f"{min(vibeqc_warm) * 1e3:.3f} ms"
    )
    print(
        "VibeQC warm SCF iterations: "
        + "; ".join(
            ",".join(str(item["iterations"]) for item in sample["convergence"])
            for sample in vibeqc_samples
        )
    )
    print(f"GPU4PySCF cold batch: {gpu_cold * 1e3:.3f} ms")
    print(
        f"GPU4PySCF warm median/min: {gpu_warm_median * 1e3:.3f}/"
        f"{min(gpu_warm) * 1e3:.3f} ms"
    )
    print(
        "GPU4PySCF warm SCF iterations: "
        + "; ".join(
            ",".join(str(item["iterations"]) for item in sample["convergence"])
            for sample in gpu_samples
        )
    )
    print(f"ordinary scoped warm speedup: {ordinary_speedup:.2f}x")
    if matched is None:
        print("iteration-matched speedup: unavailable (branches do not overlap)")
    else:
        branch = ",".join(str(value) for value in matched["iteration_branch"])
        print(
            f"iteration-matched speedup: {matched_speedup:.2f}x "
            f"(branch {branch})"
        )
    print("warning: GPU4PySCF is measured through its single-system interface")

    if args.output:
        final_vibeqc = vibeqc_samples[-1]
        final_gpu = gpu_samples[-1]
        payload = {
            "schema_version": 2,
            "benchmark": "compare_gpu4pyscf_batch",
            "environment": environment_metadata(
                distributions={
                    "cupy": ("cupy-cuda12x", "cupy"),
                    "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                    "numpy": ("numpy",),
                    "pyscf": ("pyscf",),
                },
                accelerator=cuda_accelerator_metadata(cp),
            ),
            "workload": {
                "case": args.case,
                "description": case.description,
                "method": case.method,
                "ao_count": ao_count,
                "batch_size": args.batch,
                "geometries": [
                    [
                        {"element": element, "coordinates_bohr": list(position)}
                        for element, position in atoms
                    ]
                    for atoms in systems
                ],
                "charge": case.charge,
                "multiplicity": case.multiplicity,
                "basis_representation": case.basis_representation,
                "energy_tolerance": args.energy_tolerance,
                "density_tolerance": args.density_tolerance,
                "reference_gradient_tolerance": args.reference_gradient_tolerance,
                "max_iterations": args.max_iterations,
                "vibeqc_screening_tolerance": args.screening_tolerance,
                "direct_scf_tolerance": 1.0e-14,
            },
            "settings": {
                "repeats_per_engine": args.repeats,
                "interleave_policy": "deterministic ABBA",
                "measurement_order": list(measurement_order),
                "warm_start_policy": {
                    "vibeqc": "retained native fixed-topology density",
                    "gpu4pyscf": "fixed post-cold converged density snapshot",
                },
                "gates": {
                    "minimum_iteration_matched_speedup": args.minimum_speedup,
                    "maximum_energy_error_hartree": args.maximum_energy_error,
                    "maximum_force_error_hartree_per_bohr": args.maximum_force_error,
                },
            },
            "accuracy": {
                "maximum_energy_error_hartree": maximum_energy_error,
                "maximum_force_error_hartree_per_bohr": maximum_force_error,
                "paired_warm_repeats": repeat_accuracy,
                "gate_selection": gate_accuracy,
            },
            "timing_summary": {
                "ordinary": {
                    "vibeqc_median_seconds": vibeqc_warm_median,
                    "gpu4pyscf_median_seconds": gpu_warm_median,
                    "speedup": ordinary_speedup,
                    "iteration_branches_match_for_every_pair": all(
                        item["iteration_branches_match"]
                        for item in repeat_accuracy
                    ),
                },
                "iteration_matched": matched,
                "speed_claim_uses": (
                    "iteration_matched" if matched is not None else "unmatched_labeled"
                ),
            },
            "vibeqc": {
                "energies_hartree": final_vibeqc["energies_hartree"],
                "forces_hartree_per_bohr": final_vibeqc[
                    "forces_hartree_per_bohr"
                ],
                "convergence": final_vibeqc["convergence"],
                "cold_seconds": vibeqc_cold,
                "cold_convergence": convergence_payload(vibeqc_cold_result),
                "warm_samples": vibeqc_samples,
                "warm_seconds": vibeqc_warm,
                "warm_median_seconds": vibeqc_warm_median,
                "warm_systems_per_second": args.batch / vibeqc_warm_median,
            },
            "gpu4pyscf": {
                "energies_hartree": final_gpu["energies_hartree"],
                "forces_hartree_per_bohr": final_gpu[
                    "forces_hartree_per_bohr"
                ],
                "convergence": final_gpu["convergence"],
                "cold_seconds": gpu_cold,
                "cold_energies_hartree": [
                    float(energy) for energy in gpu_cold_energies
                ],
                "cold_forces_hartree_per_bohr": np.stack(
                    [cp.asnumpy(-gradient) for gradient in gpu_cold_gradients]
                ).tolist(),
                "cold_convergence": gpu_cold_convergence,
                "warm_samples": gpu_samples,
                "warm_seconds": gpu_warm,
                "warm_median_seconds": gpu_warm_median,
                "warm_systems_per_second": args.batch / gpu_warm_median,
                "interface": "sequential single-system objects",
            },
            "gate": {
                "passed": not gate_failures,
                "failures": gate_failures,
            },
        }
        destination = write_result(args.output, payload)
        print(f"JSON result: {destination}")

    if gate_failures:
        for failure in gate_failures:
            print(f"gate failure: {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
