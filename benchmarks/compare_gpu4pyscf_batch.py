"""Homogeneous-batch QCE versus conventional GPU4PySCF throughput.

QCE executes one native fixed-topology bucket. GPU4PySCF currently exposes a
single-molecule SCF interface, so the comparison retains one initialized GPU
object and warm density per system and executes them sequentially inside the
same synchronized batch timing boundary.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from qce import Calculator

from _cases import benchmark_cases
from _support import (
    benchmark_gate_failures,
    cuda_accelerator_metadata,
    environment_metadata,
    write_result,
)


def convergence_payload(result) -> list[dict[str, object]]:
    """Serialize one QCE replay's per-system SCF convergence diagnostics."""

    return [
        {
            "converged": item.converged,
            "iterations": item.iterations,
            "energy_change_hartree": item.energy_change,
            "density_rms": item.density_rms,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
        }
        for item in result.items
    ]


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
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        help="fail after recording results when warm speedup is below this",
    )
    parser.add_argument(
        "--maximum-energy-error",
        type=float,
        help="fail after recording results when max error exceeds this Eh limit",
    )
    parser.add_argument(
        "--maximum-force-error",
        type=float,
        help=(
            "fail after recording results when max error exceeds this "
            "Eh/bohr limit"
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
    if (
        case.expected_ao_count is not None
        and ao_count != case.expected_ao_count
    ):
        raise ValueError(
            f"{args.case} expected {case.expected_ao_count} AOs, "
            f"but PySCF constructed {ao_count}"
        )
    calculator = Calculator(
        method=case.method,
        basis=case.qce_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=args.max_iterations,
        energy_tolerance=args.energy_tolerance,
        density_tolerance=args.density_tolerance,
        screening_tolerance=args.screening_tolerance,
    )
    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
        warm_start=True,
    ) as batch:
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        qce_result = batch.execute(strict=True)
        cp.cuda.Stream.null.synchronize()
        qce_cold = time.perf_counter() - start
        qce_warm = []
        qce_warm_convergence = []
        for _ in range(args.repeats):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            qce_result = batch.execute(strict=True)
            cp.cuda.Stream.null.synchronize()
            qce_warm.append(time.perf_counter() - start)
            # Keep timing and convergence state paired per replay. Direct-J/K
            # atomic reduction order can move a system across a tight energy
            # threshold, so retaining only the final repeat hides stragglers.
            qce_warm_convergence.append(convergence_payload(qce_result))

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

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    gpu_energies = [engine.kernel() for engine in gpu_objects]
    gpu_gradients = [engine.nuc_grad_method().kernel() for engine in gpu_objects]
    cp.cuda.Stream.null.synchronize()
    gpu_cold = time.perf_counter() - start

    gpu_warm = []
    for _ in range(args.repeats):
        densities = [engine.make_rdm1() for engine in gpu_objects]
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu_energies = [
            engine.kernel(dm0=density)
            for engine, density in zip(gpu_objects, densities, strict=True)
        ]
        gpu_gradients = [
            engine.nuc_grad_method().kernel() for engine in gpu_objects
        ]
        cp.cuda.Stream.null.synchronize()
        gpu_warm.append(time.perf_counter() - start)

    qce_energies = qce_result.energies
    qce_forces = np.stack([item.forces for item in qce_result.items])
    gpu_energy_array = np.asarray([float(energy) for energy in gpu_energies])
    gpu_force_array = np.stack(
        [cp.asnumpy(-gradient) for gradient in gpu_gradients]
    )
    maximum_energy_error = float(np.max(np.abs(qce_energies - gpu_energy_array)))
    maximum_force_error = float(np.max(np.abs(qce_forces - gpu_force_array)))
    qce_warm_median = statistics.median(qce_warm)
    gpu_warm_median = statistics.median(gpu_warm)
    warm_speedup = gpu_warm_median / qce_warm_median
    qce_converged = all(item.converged for item in qce_result.items)
    reference_converged = all(engine.converged for engine in gpu_objects)
    gate_failures = benchmark_gate_failures(
        speedup=warm_speedup,
        maximum_energy_error=maximum_energy_error,
        maximum_force_error=maximum_force_error,
        qce_converged=qce_converged,
        reference_converged=reference_converged,
        minimum_speedup=args.minimum_speedup,
        maximum_energy_error_limit=args.maximum_energy_error,
        maximum_force_error_limit=args.maximum_force_error,
    )

    print(
        f"scope: {case.description}, {ao_count} AOs, "
        f"homogeneous batch {args.batch}"
    )
    print(f"maximum energy difference: {maximum_energy_error:.3e} Eh")
    print(f"maximum force difference: {maximum_force_error:.3e} Eh/bohr")
    print(
        "QCE final max density RMS: "
        f"{max(item.density_rms for item in qce_result.items):.3e}"
    )
    print(f"QCE/reference converged: {qce_converged}/{reference_converged}")
    print(f"QCE cold batch: {qce_cold * 1e3:.3f} ms")
    print(f"QCE warm median/min: {qce_warm_median * 1e3:.3f}/"
          f"{min(qce_warm) * 1e3:.3f} ms")
    print(
        "QCE warm SCF iterations: "
        + "; ".join(
            ",".join(str(item["iterations"]) for item in replay)
            for replay in qce_warm_convergence
        )
    )
    print(f"QCE warm throughput: {args.batch / qce_warm_median:.2f} systems/s")
    print(f"GPU4PySCF cold batch: {gpu_cold * 1e3:.3f} ms")
    print(f"GPU4PySCF warm median/min: {gpu_warm_median * 1e3:.3f}/"
          f"{min(gpu_warm) * 1e3:.3f} ms")
    print(
        "GPU4PySCF warm throughput: "
        f"{args.batch / gpu_warm_median:.2f} systems/s"
    )
    print(f"scoped warm speedup: {warm_speedup:.2f}x")
    print("warning: GPU4PySCF is measured through its single-system interface")

    if args.output:
        payload = {
            "schema_version": 1,
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
                "reference_gradient_tolerance":
                    args.reference_gradient_tolerance,
                "max_iterations": args.max_iterations,
                "qce_screening_tolerance": args.screening_tolerance,
                "direct_scf_tolerance": 1.0e-14,
            },
            "settings": {
                "repeats": args.repeats,
                "gates": {
                    "minimum_speedup": args.minimum_speedup,
                    "maximum_energy_error_hartree": args.maximum_energy_error,
                    "maximum_force_error_hartree_per_bohr":
                        args.maximum_force_error,
                },
            },
            "accuracy": {
                "maximum_energy_error_hartree": maximum_energy_error,
                "maximum_force_error_hartree_per_bohr": maximum_force_error,
            },
            "qce": {
                "energies_hartree": qce_energies.tolist(),
                "forces_hartree_per_bohr": qce_forces.tolist(),
                # ``convergence`` remains the final replay for schema
                # compatibility; ``warm_convergence`` pairs every raw timing
                # sample with the state that produced it.
                "convergence": convergence_payload(qce_result),
                "warm_convergence": qce_warm_convergence,
                "cold_seconds": qce_cold,
                "warm_seconds": qce_warm,
                "warm_median_seconds": qce_warm_median,
                "warm_systems_per_second": args.batch / qce_warm_median,
            },
            "gpu4pyscf": {
                "energies_hartree": gpu_energy_array.tolist(),
                "forces_hartree_per_bohr": gpu_force_array.tolist(),
                "cold_seconds": gpu_cold,
                "warm_seconds": gpu_warm,
                "warm_median_seconds": gpu_warm_median,
                "warm_systems_per_second": args.batch / gpu_warm_median,
                "converged": [bool(engine.converged) for engine in gpu_objects],
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
