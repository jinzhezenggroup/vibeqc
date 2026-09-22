from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import time
from typing import TYPE_CHECKING

import numpy as np
from _support import (
    cuda_accelerator_metadata,
    environment_metadata,
    raw_output_path,
    write_result,
)
from vibeqc import D4CorrectionBatch

if TYPE_CHECKING:
    from collections.abc import Callable


def _small() -> tuple[np.ndarray, np.ndarray, float]:
    return (
        np.array([8, 1, 1], dtype=np.int32),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4304281900354767, 0.0, 1.0999887901572898],
                [-1.4304281900354767, 0.0, 1.0999887901572898],
            ],
            dtype=np.float64,
        ),
        0.0,
    )


def _medium() -> tuple[np.ndarray, np.ndarray, float]:
    atoms = 32
    z = np.array([6 if i % 3 == 0 else 1 for i in range(atoms)], dtype=np.int32)
    xyz = np.zeros((atoms, 3), dtype=np.float64)
    for i in range(atoms):
        xyz[i] = (2.15 * i, 0.37 * (i % 4), 0.23 * ((i * 3) % 5))
    return z, xyz, 0.0


def _changed(
    systems: tuple[tuple[np.ndarray, np.ndarray, float], ...],
) -> list[np.ndarray]:
    changed = []
    for index, (_, xyz, _) in enumerate(systems):
        geometry = xyz.copy()
        geometry[index % len(geometry), 0] += 2.0e-3
        changed.append(geometry)
    return changed


def _timed(callable_: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def _measure(
    systems: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    *,
    device: str,
    repeats: int,
    maximum_bytes: int,
) -> dict:
    prepare_seconds, batch = _timed(
        lambda: D4CorrectionBatch(
            "PBE-D4(BJ-EEQ-ATM)",
            systems,
            device=device,
            maximum_bytes=maximum_bytes,
        )
    )
    with batch:
        cold_seconds, cold = _timed(lambda: batch.execute())
        replay_samples = []
        replay = cold
        for _ in range(repeats):
            elapsed, replay = _timed(lambda: batch.execute())
            replay_samples.append(elapsed)
        changed = _changed(systems)
        changed_seconds, changed_result = _timed(lambda: batch.execute(changed))
        changed_replay_samples = []
        changed_replay = changed_result
        for _ in range(repeats):
            elapsed, changed_replay = _timed(lambda: batch.execute(changed))
            changed_replay_samples.append(elapsed)
        diagnostic = batch.diagnostic()

    for result_set in (cold, replay, changed_result, changed_replay):
        if any(not item.ok or not np.isfinite(item.energy) for item in result_set):
            raise RuntimeError("D4 performance gate observed a failed/nonfinite item")
    return {
        "systems": len(systems),
        "atoms": [len(system[0]) for system in systems],
        "timing_seconds": {
            "prepare": prepare_seconds,
            "cold": cold_seconds,
            "unchanged_replays": replay_samples,
            "unchanged_replay_median": statistics.median(replay_samples),
            "changed_geometry": changed_seconds,
            "changed_replays": changed_replay_samples,
            "changed_replay_median": statistics.median(changed_replay_samples),
        },
        "diagnostic": dataclasses.asdict(diagnostic),
        "endpoint": {
            "cold_energy_hartree": [float(item.energy) for item in cold],
            "cold_gradient_norm": [
                float(np.linalg.norm(item.gradient))
                if item.gradient is not None
                else None
                for item in cold
            ],
            "executed_backend": sorted({item.backend for item in cold}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-copies", type=int, default=8)
    parser.add_argument("--maximum-bytes", type=int, default=256 << 20)
    parser.add_argument(
        "--output",
        type=raw_output_path,
        default=".artifacts/benchmarks/d4-production-gate.json",
    )
    args = parser.parse_args()
    if min(args.repeats, args.batch_copies, args.maximum_bytes) < 1:
        parser.error("repeats, batch-copies and maximum-bytes must be positive")
    accelerator = None
    allocation_id = os.environ.get("VIBEQC_GPU_ALLOCATION_ID")
    if args.device == "cuda":
        if not allocation_id:
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                allocation_id = f"slurm:{slurm_job_id}"
        if not allocation_id:
            parser.error(
                "real CUDA performance evidence requires VIBEQC_GPU_ALLOCATION_ID "
                "or a finite Slurm allocation"
            )
        import cupy as cp

        accelerator = cuda_accelerator_metadata(cp)

    small = _small()
    medium = _medium()
    ragged = tuple(
        small if index % 2 == 0 else medium for index in range(args.batch_copies)
    )
    payload = {
        "schema": "vibeqc.d4-production-gate.v1",
        "environment": environment_metadata(
            distributions={"numpy": ("numpy",), "cupy": ("cupy-cuda12x", "cupy")},
            accelerator=accelerator,
        ),
        "settings": {
            "device": args.device,
            "method": "PBE-D4(BJ-EEQ-ATM)",
            "repeats": args.repeats,
            "batch_copies": args.batch_copies,
            "maximum_bytes": args.maximum_bytes,
            "gradient": True,
            "charges": True,
            "gpu_allocation_id": allocation_id,
        },
        "small": _measure(
            (small,),
            device=args.device,
            repeats=args.repeats,
            maximum_bytes=args.maximum_bytes,
        ),
        "medium": _measure(
            (medium,),
            device=args.device,
            repeats=args.repeats,
            maximum_bytes=args.maximum_bytes,
        ),
        "ragged_batch": _measure(
            ragged,
            device=args.device,
            repeats=args.repeats,
            maximum_bytes=args.maximum_bytes,
        ),
    }
    destination = write_result(args.output, payload)
    print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
