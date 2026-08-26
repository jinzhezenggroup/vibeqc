from __future__ import annotations

import argparse
import statistics
import time

from vibeqc import Calculator

from _support import environment_metadata, write_result


def make_system(index: int):
    if index % 3 == 0:
        distance = 1.2 + 0.01 * (index % 10)
        return [
            ("H", (0.0, 0.0, -0.5 * distance)),
            ("H", (0.0, 0.0, 0.5 * distance)),
        ], 0
    if index % 3 == 1:
        return [("He", (0.0, 0.0, 0.0))], 0
    return [
        ("H", (-1.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.0)),
        ("H", (1.0, 0.0, 0.0)),
    ], 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        help="optional JSON path for raw timings and reproducibility metadata",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1:
        raise ValueError("--batch and --repeats must be positive")

    generated = [make_system(index) for index in range(args.batch)]
    systems = [item[0] for item in generated]
    charges = [item[1] for item in generated]
    calculator = Calculator(device=args.device)

    start = time.perf_counter()
    independent = [
        calculator.singlepoint(system, charge=charge)
        for system, charge in zip(systems, charges, strict=True)
    ]
    independent_time = time.perf_counter() - start

    with calculator.prepare_batch(systems, charges=charges) as batch:
        start = time.perf_counter()
        cold = batch.execute(strict=True)
        cold_time = time.perf_counter() - start
        warm_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            warm = batch.execute(strict=True)
            warm_times.append(time.perf_counter() - start)

    assert all(
        abs(cold.items[index].energy - independent[index].energy) < 2.0e-10
        for index in range(args.batch)
    )
    best_warm = min(warm_times)
    print(f"batch size: {args.batch}")
    print(f"requested device: {args.device}")
    print(f"scientific backend: {cold.items[0].executed_backend}")
    print(f"independent Python calls: {independent_time * 1e3:.3f} ms")
    print(f"native cold batch: {cold_time * 1e3:.3f} ms")
    print(f"native warm batch best: {best_warm * 1e3:.3f} ms")
    print(f"cold throughput: {args.batch / cold_time:.2f} systems/s")
    print(f"warm throughput: {args.batch / best_warm:.2f} systems/s")
    if args.output:
        payload = {
            "schema_version": 1,
            "benchmark": "batch_throughput",
            "environment": environment_metadata(
                distributions={"numpy": ("numpy",)}
            ),
            "settings": {
                "batch_size": args.batch,
                "repeats": args.repeats,
                "requested_device": args.device,
            },
            "result": {
                "executed_backend": cold.items[0].executed_backend,
                "energies_hartree": [item.energy for item in cold.items],
                "timings_seconds": {
                    "independent_calls": independent_time,
                    "cold_batch": cold_time,
                    "warm_batches": warm_times,
                },
                "summary": {
                    "warm_median_seconds": statistics.median(warm_times),
                    "warm_minimum_seconds": best_warm,
                    "cold_systems_per_second": args.batch / cold_time,
                    "warm_systems_per_second": args.batch / best_warm,
                },
            },
        }
        destination = write_result(args.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
