from __future__ import annotations

import argparse
import statistics
import time

from qce import Calculator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
