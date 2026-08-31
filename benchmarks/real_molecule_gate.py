"""Run the explicit real-molecule acceptance matrix.

The direct matrix retains its historical GPU4PySCF parity requirements. The
separate CUDA-DF matrix covers 96-, 192-, and 384-AO correctness/scaling points;
external GPU4PySCF availability and provider-specific Graph replay are recorded
in each artifact rather than silently changing the direct gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from _cases import (
    BenchmarkGatePoint,
    benchmark_cases,
    density_fitting_gate_points,
    real_molecule_gate_points,
)


_ENERGY_TOLERANCE = 1.0e-12
_DENSITY_TOLERANCE = 1.0e-10
_SCREENING_TOLERANCE = 1.0e-14
_MAX_ITERATIONS = 100


def _point_command(
    point: BenchmarkGatePoint,
    *,
    repeats: int,
    output: Path,
    density_fitting_memory_budget_bytes: int,
) -> list[str]:
    """Build one child command without importing GPU packages in this runner."""

    command = [
        sys.executable,
        str(Path(__file__).with_name("compare_gpu4pyscf_batch.py")),
        "--case",
        point.case,
        "--batch",
        str(point.batch_size),
        "--repeats",
        str(repeats),
        "--max-iterations",
        str(_MAX_ITERATIONS),
        "--energy-tolerance",
        str(_ENERGY_TOLERANCE),
        "--density-tolerance",
        str(_DENSITY_TOLERANCE),
        "--reference-gradient-tolerance",
        str(point.reference_gradient_tolerance),
        "--screening-tolerance",
        str(_SCREENING_TOLERANCE),
        "--maximum-energy-error",
        str(point.maximum_energy_error),
        "--maximum-force-error",
        str(point.maximum_force_error),
        "--output",
        str(output),
    ]
    if point.minimum_speedup is not None:
        command.extend(("--minimum-speedup", str(point.minimum_speedup)))
    if density_fitting_memory_budget_bytes:
        command.extend((
            "--density-fitting-memory-budget-bytes",
            str(density_fitting_memory_budget_bytes),
        ))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        choices=("all", "96", "192", "384"),
        default="all",
        help="run the full matrix or only one AO-size gate",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="interleaved warm samples collected per engine and gate point",
    )
    parser.add_argument(
        "--density-fitting",
        choices=("none", "cuda"),
        default="none",
        help="run the unchanged direct matrix or the CUDA DF acceptance matrix",
    )
    parser.add_argument(
        "--density-fitting-memory-budget-bytes",
        type=int,
        default=0,
        help="positive CUDA-DF planner budget forwarded to each gate point",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print exact child commands without importing CUDA packages",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.density_fitting_memory_budget_bytes < 0:
        raise ValueError(
            "--density-fitting-memory-budget-bytes must be non-negative"
        )

    cases = benchmark_cases()
    gate_points = (
        density_fitting_gate_points()
        if args.density_fitting == "cuda"
        else real_molecule_gate_points()
    )
    points = tuple(
        point
        for point in gate_points
        if args.size == "all" or point.expected_ao_count == int(args.size)
    )
    if not points:
        raise ValueError(
            f"no acceptance point for --size {args.size} with "
            f"--density-fitting {args.density_fitting}"
        )
    for point in points:
        case = cases[point.case]
        if case.expected_ao_count != point.expected_ao_count:
            raise ValueError(
                f"{point.case} case/gate AO counts disagree: "
                f"{case.expected_ao_count} versus {point.expected_ao_count}"
            )

    commands = []
    for point in points:
        output = args.output_directory / (
            f"{point.case}-b{point.batch_size}-{args.density_fitting}.json"
        )
        commands.append((
            point,
            output,
            _point_command(
                point,
                repeats=args.repeats,
                output=output,
                density_fitting_memory_budget_bytes=(
                    args.density_fitting_memory_budget_bytes
                    if args.density_fitting == "cuda" else 0
                ),
            )
            + ["--density-fitting", args.density_fitting],
        ))
    if args.dry_run:
        for _, _, command in commands:
            print(shlex.join(command))
        return

    args.output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for point, output, command in commands:
        completed = subprocess.run(command, check=False)
        payload = None
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
        passed = completed.returncode == 0
        failed = failed or not passed
        results.append({
            "case": point.case,
            "ao_count": point.expected_ao_count,
            "batch_size": point.batch_size,
            "minimum_speedup": point.minimum_speedup,
            "reference_gradient_tolerance":
                point.reference_gradient_tolerance,
            "maximum_energy_error_hartree": point.maximum_energy_error,
            "maximum_force_error_hartree_per_bohr": point.maximum_force_error,
            "returncode": completed.returncode,
            "passed": passed,
            "artifact": str(output.resolve()),
            "gate": None if payload is None else payload.get("gate"),
        })

    summary = {
        "schema_version": 1,
        "benchmark": "real_molecule_gate",
        "repeats": args.repeats,
        "max_iterations": _MAX_ITERATIONS,
        "energy_tolerance": _ENERGY_TOLERANCE,
        "density_tolerance": _DENSITY_TOLERANCE,
        "screening_tolerance": _SCREENING_TOLERANCE,
        "gpu4pyscf_interface": "sequential single-system objects",
        "density_fitting": args.density_fitting,
        "density_fitting_memory_budget_bytes": (
            args.density_fitting_memory_budget_bytes
        ),
        "points": results,
        "passed": not failed,
    }
    summary_path = args.output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"gate summary: {summary_path.resolve()}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
