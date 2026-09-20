"""Interleaved CPU SCF + complete-gradient diagnostic benchmark.

Both routes warm their compiled artifacts before the measured repetitions.
This is not a cold-compile comparison or a public Calculator force benchmark.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import typing
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

SYSTEMS = {
    "h2": [("H", (0.1, 0.2, -0.6)), ("H", (0.2, -0.1, 0.8))],
    "water": [
        ("O", (0.1, -0.1, 0.0)),
        ("H", (0.1, 0.2, 1.7)),
        ("H", (1.6, -0.2, -0.5)),
    ],
}


def benchmark(cache: typing.Any, repeats: typing.Any) -> typing.Any:
    records, summaries, warmups = [], [], []
    grid = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    for name, atoms in SYSTEMS.items():
        coordinates = np.array([a[1] for a in atoms])
        calculator = Calculator(
            method="pbe-rks",
            device="cpu",
            ks_options=KsOptions(grid=grid),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with ExitStack() as stack:
            batches = {
                mode: stack.enter_context(calculator.prepare_batch([atoms]))
                for mode in ("reference", "native")
            }

            def run(
                mode: typing.Any,
                coords: typing.Any,
                *,
                atoms: typing.Any = atoms,
                batches: typing.Any = batches,
            ) -> typing.Any:
                moved = [(a[0], p) for a, p in zip(atoms, coords, strict=True)]
                start = time.perf_counter()
                item = batches[mode].execute(coordinates=[coords], strict=True).items[0]
                with NativeAO(moved) as basis:
                    state = StationaryKsState.from_native(batches[mode], basis)
                    result = complete_rks_gradient_diagnostic(
                        state,
                        basis,
                        cache=cache,
                        execution=mode,
                    )
                return time.perf_counter() - start, item, result

            for mode in ("reference", "native"):
                elapsed, _, _ = run(mode, coordinates)
                warmups.append({"system": name, "mode": mode, "seconds": elapsed})
            for phase in ("fixed_geometry", "changed_geometry"):
                for repeat in range(repeats):
                    coords = coordinates.copy()
                    if phase == "changed_geometry":
                        coords[1] += (repeat + 1) * np.array([0.001, -0.0005, 0.0003])
                    pair = {}
                    order = (
                        ("reference", "native")
                        if repeat % 2 == 0
                        else ("native", "reference")
                    )
                    for mode in order:
                        elapsed, item, result = run(mode, coords)
                        pair[mode] = (item, result)
                        records.append(
                            {
                                "system": name,
                                "phase": phase,
                                "repeat": repeat,
                                "mode": mode,
                                "seconds": elapsed,
                                "iterations": item.iterations,
                                "energy": item.energy,
                            }
                        )
                    energy_error = abs(
                        pair["native"][0].energy - pair["reference"][0].energy
                    )
                    gradient_error = float(
                        np.max(
                            np.abs(
                                pair["native"][1].gradient
                                - pair["reference"][1].gradient
                            )
                        )
                    )
                    if (
                        energy_error > 1e-10
                        or gradient_error > 1e-7
                        or pair["native"][0].iterations
                        != pair["reference"][0].iterations
                    ):
                        raise RuntimeError(
                            "timed routes changed energy, gradient or SCF work"
                        )
                    for record in records[-2:]:
                        record.update(
                            energy_error=energy_error, gradient_error=gradient_error
                        )
                summary = {
                    mode: float(
                        np.median(
                            [
                                r["seconds"]
                                for r in records
                                if r["system"] == name
                                and r["phase"] == phase
                                and r["mode"] == mode
                            ]
                        )
                    )
                    for mode in ("reference", "native")
                }
                summary.update(
                    system=name,
                    phase=phase,
                    speed_ratio=summary["reference"] / summary["native"],
                )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    return {
        "records": records,
        "summaries": summaries,
        "warmups_not_cold_comparisons": warmups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/stationary-consumers")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 50:
        parser.error("--repeats must be between 1 and 50")
    result = benchmark(args.cache, args.repeats)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, timeout=10
            )
        )
    except (subprocess.SubprocessError, OSError):
        head, dirty = None, None
    library = Path(os.environ.get("VIBEQC_LIBRARY", ""))
    result["provenance"] = {
        "head": head,
        "dirty": dirty,
        "numpy": np.__version__,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()
        if library.is_file()
        else None,
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": {
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "method": "pbe-rks",
        "basis": "sto-3g",
        "grid": [24, 8, 16],
        "batch_size": 1,
        "scope": "prepared CPU SCF + current state export + AO owner + complete gradient; warmed artifacts",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
