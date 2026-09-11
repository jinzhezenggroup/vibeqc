"""Alternate baseline/candidate order for the small CPU warm-batch endpoint.

The existing runner retains normal preparation/warmup and verifies actual
backend and convergence. Every round also repeats the fixed-density probes.
Run with the same production Release builds used for the complete endpoint
study, inside a finite Slurm allocation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--build-relative", default=".artifacts/overhead-cuda-build")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a finite Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    runner = args.head.resolve() / "tools/benchmark_fock_strategies.py"
    sys.path.insert(0, str(runner.parent))
    from benchmark_fock_strategies import compare

    rounds = []
    for cycle in range(6):
        order = [("baseline", args.baseline.resolve()), ("head", args.head.resolve())]
        if cycle % 2:
            order.reverse()
        values = {}
        for label, root in order:
            build = root / args.build_relative
            output = args.output.resolve() / f"focus-{cycle}-{label}.json"
            env = dict(
                os.environ,
                VIBEQC_LIBRARY=str(build / "libvibeqc.so"),
                VIBEQC_PROFILE="off",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--worker",
                    "--root",
                    str(root),
                    "--build",
                    str(build),
                    "--device",
                    "cpu",
                    "--label",
                    f"focus-{cycle}-{label}",
                    "--samples",
                    "9",
                    "--output",
                    str(output),
                    "--case",
                    "h2",
                    "--spin",
                    "uhf",
                    "--approximation",
                    "density_fitted",
                    "--endpoint",
                    "batch_four_warm",
                ],
                env=env,
                check=True,
            )
            values[label] = json.loads(output.read_text())
        comparison = compare(values["baseline"], values["head"])
        assert comparison["accuracy_passed"]
        rounds.append(
            {"cycle": cycle, "order": [p[0] for p in order], "comparison": comparison}
        )
        print(cycle, comparison["endpoints"][0]["median_ratio"], flush=True)
    (args.output / "focused-comparison.json").write_text(
        json.dumps(rounds, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
