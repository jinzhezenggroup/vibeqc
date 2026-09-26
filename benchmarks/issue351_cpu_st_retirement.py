"""Measure the complete CPU RHF endpoint used to qualify #351 S/T retirement."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from vibeqc import Calculator, Primitive, Shell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    if args.samples <= args.warmup or args.warmup < 0:
        raise ValueError("samples must exceed a nonnegative warmup count")
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    timings: list[float] = []
    result = None
    for _ in range(args.samples):
        start = time.perf_counter()
        result = Calculator(
            method="rhf",
            basis=basis,
            device="cpu",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        ).singlepoint(atoms, charge=1)
        timings.append(time.perf_counter() - start)
    assert result is not None
    retained = timings[args.warmup :]
    print(
        json.dumps(
            {
                "schema": "vibeqc.issue351.cpu-st-endpoint.v1",
                "label": args.label,
                "library": os.environ.get("VIBEQC_LIBRARY"),
                "samples_s": timings,
                "warmup": args.warmup,
                "median_s": statistics.median(retained),
                "mean_s": statistics.mean(retained),
                "energy_hartree": result.energy,
                "forces_hartree_per_bohr": result.forces.tolist(),
                "iterations": result.iterations,
                "density_rms": result.density_rms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
