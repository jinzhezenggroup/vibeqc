from __future__ import annotations

import argparse
import statistics
import time

from qce import Calculator

from _support import environment_metadata, write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        help="optional JSON path for raw timings and reproducibility metadata",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    calculator = Calculator(method="rhf", basis="sto-3g", device=args.device)
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    timings: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = calculator.singlepoint(atoms)
        timings.append(time.perf_counter() - start)
    print(f"energy: {result.energy:.12f} Eh")
    print(f"scientific backend: {result.executed_backend}")
    print(f"median latency: {statistics.median(timings) * 1e3:.3f} ms")
    print(f"minimum latency: {min(timings) * 1e3:.3f} ms")
    if args.output:
        payload = {
            "schema_version": 1,
            "benchmark": "h2_latency",
            "environment": environment_metadata(
                distributions={"numpy": ("numpy",)}
            ),
            "settings": {
                "iterations": args.iterations,
                "requested_device": args.device,
                "method": "rhf",
                "basis": "sto-3g",
            },
            "result": {
                "executed_backend": result.executed_backend,
                "energy_hartree": result.energy,
                "timings_seconds": timings,
                "summary": {
                    "median_seconds": statistics.median(timings),
                    "minimum_seconds": min(timings),
                },
            },
        }
        destination = write_result(args.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
