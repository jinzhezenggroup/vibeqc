"""Freeze and optionally execute the matched DF completion matrix for #206.

The existing ``compare_gpu4pyscf_batch`` benchmark owns one scientifically
matched endpoint.  This driver owns the cross-case protocol: it freezes the
four required 96/192-AO cases, records the exact command and source identity,
and executes cases sequentially so one RTX 5090 is never shared by concurrent
measurements.  It intentionally preserves failed runs in the manifest rather
than turning a partial matrix into a passing result.

Run the ``--run`` mode inside a finite Slurm allocation.  A manifest-only run
is safe on a login node and is useful for review before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "compare_gpu4pyscf_batch.py"


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """One mandatory #206 DF endpoint and its fixed topology."""

    name: str
    ao_count: int
    batch_size: int


MATRIX: tuple[MatrixCase, ...] = (
    MatrixCase("water-tetramer-def2-svp-spherical", 96, 1),
    MatrixCase("water-tetramer-def2-svp-spherical", 96, 4),
    MatrixCase("water-octamer-s4-def2-svp-spherical", 192, 1),
    MatrixCase("water-octamer-s4-def2-svp-spherical", 192, 4),
)


def _git(*command: str) -> str:
    """Return a repository value, or an explicit empty value if unavailable."""

    completed = subprocess.run(
        ("git", "-C", str(ROOT), *command),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _matrix(selected: str | None) -> tuple[MatrixCase, ...]:
    """Select one case for a smoke run or retain the complete required matrix."""

    if selected is None:
        return MATRIX
    return tuple(item for item in MATRIX if item.name == selected)


def manifest_payload(
    *,
    cases: tuple[MatrixCase, ...],
    repeats: int,
    python: str,
    library: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a reviewable protocol record before any GPU work starts."""

    return {
        "schema": "vibeqc.issue206.df_matrix",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "repository": str(ROOT),
        },
        "execution": {
            "slurm_required_for_run": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": python,
            "library": str(library),
            "repeats_per_engine": repeats,
            "output_dir": str(output_dir),
            "benchmark": str(BENCHMARK),
        },
        "matched_contract": {
            "method": "rhf",
            "density_fitting": "cuda",
            "auxiliary_basis": "same as orbital basis",
            "energy_tolerance": 1.0e-12,
            "density_tolerance": 1.0e-10,
            "reference_gradient_tolerance": 1.0e-10,
            "screening_tolerance": 1.0e-12,
            "direct_scf_tolerance": 1.0e-14,
            "warm_policy": "fixed post-cold engine-local density snapshot",
            "comparison": "VibeQC DF versus GPU4PySCF DF; no mixed direct/DF claim",
        },
        "component_ledger": {
            "status": "pending_measurement",
            "required_components": [
                "raw_metric_and_three_center_generation",
                "metric_factorization",
                "metric_transform",
                "ri_j",
                "ri_k",
                "eigensolver_and_diis",
                "one_electron_force",
                "df_two_and_three_center_response",
                "host_device_transfer",
                "synchronization_and_control",
            ],
            "measurement_source": (
                "endpoint JSON plus synchronized Nsight NVTX/CUPTI capture; "
                "missing components remain null"
            ),
        },
        "matrix": [
            {
                **asdict(case),
                "result": None,
                "status": "pending",
            }
            for case in cases
        ],
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Write atomically enough for an interrupted benchmark to remain readable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_matrix(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    python: str,
    library: Path,
    output_dir: Path,
) -> None:
    """Retain every attempt, then fail the job if any endpoint failed.

    Publish the command before launching so an interrupted job still identifies
    its active endpoint. A failed attempt must not claim a previous run's JSON.
    """

    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("--run requires a finite Slurm allocation (SLURM_JOB_ID)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("--run requires Slurm-provided CUDA_VISIBLE_DEVICES")

    environment = os.environ.copy()
    environment["VIBEQC_LIBRARY"] = str(library)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in payload["matrix"]:
        case = MatrixCase(
            name=entry["name"],
            ao_count=int(entry["ao_count"]),
            batch_size=int(entry["batch_size"]),
        )
        stem = f"{case.ao_count}ao-b{case.batch_size}"
        result_path = output_dir / f"{stem}.json"
        log_path = output_dir / f"{stem}.log"

        def result_identity(path=result_path):
            """Distinguish a new endpoint artifact from an earlier attempt."""
            try:
                stat = path.stat()
            except FileNotFoundError:
                return None
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        previous_result = result_identity()
        command = [
            python,
            str(BENCHMARK),
            "--case",
            case.name,
            "--batch",
            str(case.batch_size),
            "--repeats",
            str(payload["execution"]["repeats_per_engine"]),
            "--density-fitting",
            "cuda",
            "--output",
            str(result_path),
        ]
        entry.update({"status": "running", "command": command, "result": None})
        _write(manifest_path, payload)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            returncode = completed.returncode
            log = completed.stdout + "\n--- stderr ---\n" + completed.stderr
        except OSError as error:
            # A missing interpreter is an attempted endpoint too; retain its
            # failure and finish the other cases before reporting job failure.
            returncode = None
            log = str(error) + "\n"
        current_result = result_identity()
        fresh_result = current_result is not None and current_result != previous_result
        passed = returncode == 0 and fresh_result
        if returncode == 0 and not passed:
            log += "\nEndpoint exited successfully without a result JSON.\n"
        log_path.write_text(log)
        entry.update(
            {
                "status": "passed" if passed else "failed",
                "returncode": returncode,
                # Numerical gate failures also produce useful endpoint JSON;
                # retain it when this attempt wrote it, regardless of exit code.
                "result": str(result_path) if fresh_result else None,
                "log": str(log_path),
            }
        )
        _write(manifest_path, payload)
    if any(entry["status"] != "passed" for entry in payload["matrix"]):
        raise SystemExit("DF matrix failed; see endpoint logs in " + str(manifest_path))


def main() -> None:
    """Create the #206 protocol manifest and optionally run its matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute cases in Slurm")
    parser.add_argument("--case", choices=sorted({item.name for item in MATRIX}))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT / "build" / "cuda-dev-fast" / "libvibeqc.so",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".artifacts" / "issue206-df"
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    cases = _matrix(args.case)
    # Children run from ROOT; resolve caller-relative paths before changing cwd.
    output_dir = args.output_dir.resolve()
    library = args.library.resolve()
    manifest_path = (
        args.manifest.resolve() if args.manifest else output_dir / "manifest.json"
    )
    payload = manifest_payload(
        cases=cases,
        repeats=args.repeats,
        python=args.python,
        library=library,
        output_dir=output_dir,
    )
    _write(manifest_path, payload)
    if args.run:
        run_matrix(
            payload,
            manifest_path=manifest_path,
            python=args.python,
            library=library,
            output_dir=output_dir,
        )
    print(manifest_path)


if __name__ == "__main__":
    main()
