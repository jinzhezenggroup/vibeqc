"""Benchmark homogeneous-batch Direct-J/K pooling for issue #992."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import typing

from _support import environment_metadata, raw_output_path, write_result

WATER = [
    ("O", (0.000000, 0.000000, 0.000000)),
    ("H", (0.000000, -0.757160, 0.586260)),
    ("H", (0.000000, 0.757160, 0.586260)),
]


def _sizes(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(","))
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return parsed


def _validated_energies(result: typing.Any, batch_size: int, phase: str) -> list[float]:
    """Require complete, converged, finite evidence before computing throughput."""
    if len(result.items) != batch_size:
        raise RuntimeError(f"{phase} batch item count differs from request")
    if any(not item.converged for item in result.items):
        raise RuntimeError(f"{phase} batch did not converge")
    energies = [float(item.energy) for item in result.items]
    if not all(math.isfinite(energy) for energy in energies):
        raise RuntimeError(f"{phase} batch returned nonfinite energies")
    return energies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_sizes, default=(1, 4, 16, 64))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--output", type=raw_output_path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("this real-GPU benchmark must run inside a Slurm allocation")

    import cupy as cp
    from vibeqc import Calculator

    records: dict[str, object] = {}
    reference_energy: float | None = None
    for batch_size in arguments.batch_sizes:
        calculator = Calculator(
            method="rhf",
            basis=arguments.basis,
            basis_representation="spherical",
            device="cuda",
            max_iterations=100,
            energy_tolerance=1.0e-10,
            density_tolerance=1.0e-8,
            screening_tolerance=1.0e-12,
        )
        systems = [WATER for _ in range(batch_size)]
        with calculator.prepare_batch(systems, warm_start=True) as batch:
            cold = batch.execute(strict=True, properties=("energy",))
            _validated_energies(cold, batch_size, "cold")
            batch.set_warm_start_updates(False)
            setup = batch.execute(strict=True, properties=("energy",))
            _validated_energies(setup, batch_size, "warm setup")
            samples: list[float] = []
            iterations: list[list[int]] = []
            for _ in range(arguments.repeats):
                cp.cuda.Stream.null.synchronize()
                started = time.perf_counter()
                replay = batch.execute(strict=True, properties=("energy",))
                cp.cuda.Stream.null.synchronize()
                samples.append(time.perf_counter() - started)
                iterations.append([item.iterations for item in replay.items])
                energies = _validated_energies(replay, batch_size, "warm replay")
                spread = max(energies) - min(energies)
                if spread > 2.0e-10:
                    raise RuntimeError(f"pooled batch energy spread {spread:.3e} Eh")
                if reference_energy is None:
                    reference_energy = energies[0]
                elif abs(energies[0] - reference_energy) > 2.0e-10:
                    raise RuntimeError("batch size changed the converged energy")

        median = statistics.median(samples)
        records[str(batch_size)] = {
            "batch_size": batch_size,
            "warm_seconds": samples,
            "warm_median_seconds": median,
            "warm_systems_per_second": batch_size / median,
            "iterations": iterations,
            "executed_backend": setup.items[0].executed_backend,
        }
    payload = {
        "schema_version": 1,
        "benchmark": "issue992_pooled_direct_jk",
        "settings": {
            "basis": arguments.basis,
            "batch_sizes": list(arguments.batch_sizes),
            "repeats": arguments.repeats,
            "properties": ["energy"],
            "warm_start_updates": False,
        },
        "results": records,
        "environment": environment_metadata(
            distributions={"numpy": ("numpy",), "cupy": ("cupy-cuda12x", "cupy")}
        ),
        "limitations": [
            "This endpoint benchmark measures warm complete energy-only SCF replay.",
            "The system-major compaction route is selected only for exact fixed-topology batches with equal per-item shell-quartet counts.",
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        destination = write_result(arguments.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
