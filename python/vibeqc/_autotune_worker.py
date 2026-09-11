"""Fresh-process workload execution for local autotune library comparisons.

One process loads exactly one native build, avoiding shared-library symbol or
warm-state contamination. Batch.execute returns host energies/forces and thus
completes its GPU work before the timer stops; process startup is not timed.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .calculator import Calculator
from .profiles import atomic_json


def execute_workload(workload: dict, *, profile: bool = False) -> dict:
    """Freeze a converged density, then measure one complete energy/force replay."""
    calculator = Calculator(
        method=workload["method"],
        basis=workload["basis"],
        device="cuda",
        device_id=workload["device_id"],
        basis_representation=workload["representation"],
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    systems = [workload["atoms"] for _ in range(workload["batch"])]
    with calculator.prepare_batch(
        systems,
        charges=[workload["charge"]] * len(systems),
        multiplicities=[workload["multiplicity"]] * len(systems),
        warm_start=True,
        shell_class_profiling=profile,
    ) as prepared:
        cold = prepared.execute(strict=True)
        prepared.set_warm_start_updates(False)
        prepared.execute(strict=True)
        started = time.perf_counter()
        result = prepared.execute(strict=True)
        elapsed = time.perf_counter() - started
        work, profiling_reason = [], None
        if profile:
            try:
                work = [asdict(row) for row in prepared.last_shell_class_profile()]
            except NotImplementedError:
                # Small cached-ERI routes have no direct shell-class work to
                # accelerate. Never substitute unscreened static counts for
                # absent measured counters or compile every class speculatively.
                profiling_reason = "runtime did not report direct shell-class work for this execution path"
    return {
        "seconds": elapsed,
        "cold_iterations": [item.iterations for item in cold.items],
        "iterations": [item.iterations for item in result.items],
        "energies": [item.energy for item in result.items],
        "forces": [item.forces.tolist() for item in result.items],
        "converged": all(item.converged for item in result.items),
        "backend": [item.executed_backend for item in result.items],
        "work": work,
        "profiling_reason": profiling_reason,
        "profile": calculator.profile_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    atomic_json(
        args.output,
        execute_workload(json.loads(args.input.read_text()), profile=args.profile),
    )


if __name__ == "__main__":
    main()
