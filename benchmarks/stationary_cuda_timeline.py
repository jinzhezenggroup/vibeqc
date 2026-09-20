"""Issue #662 stationary CUDA force timeline and scaling evidence.

Run inside a finite Slurm GPU allocation. The benchmark times only final-state
export plus stationary force execution; SCF convergence is preparation and is
outside each recorded endpoint. The diagnostic timeline is opt-in and does not
change scientific tolerances or execution policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cuda import complete_rks_cuda_gradient_diagnostic
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.dft import NativeAO

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

METHODS = (
    "lda-rks",
    "lda-uks",
    "pbe-rks",
    "pbe-uks",
    "r2scan-rks",
    "r2scan-uks",
)
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
WATER = [
    ("O", (0.1, -0.1, 0.0)),
    ("H", (0.1, 0.2, 1.7)),
    ("H", (1.6, -0.2, -0.5)),
]
FIXTURES = {
    "h2": [("H", (0.1, -0.2, -0.7)), ("H", (0.2, 0.1, 0.8))],
    "water": WATER,
    "water-dimer": WATER
    + [(name, tuple(np.asarray(xyz) + (4.0, 0.4, -0.3))) for name, xyz in WATER],
}


def _spin(method: str) -> tuple[int, int]:
    return (1, 2) if method.endswith("uks") else (0, 1)


def _moved(atoms: list[tuple[str, typing.Any]]) -> list[tuple[str, tuple[float, ...]]]:
    moved = [(name, tuple(map(float, xyz))) for name, xyz in atoms]
    name, xyz = moved[-1]
    delta = np.asarray((0.013, -0.007, 0.011))
    moved[-1] = (name, tuple(np.asarray(xyz) + delta))
    return moved


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _gpu_metadata() -> dict[str, str]:
    fields = "name,uuid,driver_version,memory.total"
    output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    row = output[0].split(", ")
    return dict(zip(fields.split(","), row))


def _sample(
    *,
    mode: str,
    atoms: list[tuple[str, typing.Any]],
    method: str,
    compiler: CudaCompilerAdapter,
    cache: Path,
    reuse_state: tuple[typing.Any, typing.Any] | None = None,
) -> tuple[dict[str, typing.Any], tuple[typing.Any, typing.Any] | None]:
    charge, multiplicity = _spin(method)
    if reuse_state is not None:
        state, basis = reuse_state
        started = perf_counter()
        result = complete_rks_cuda_gradient_diagnostic(
            state,
            basis,
            compiler=compiler,
            cache=cache,
            measure_timeline=True,
        )
        endpoint = perf_counter() - started
        export_seconds = 0.0
        return _record(mode, endpoint, export_seconds, basis, result), reuse_state

    calc = _calculator(method)
    with (
        calc.prepare_batch(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            warm_start=False,
        ) as batch,
        NativeAO(atoms, charge=charge, multiplicity=multiplicity) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        started = perf_counter()
        export_started = perf_counter()
        state = StationaryKsState.from_native(batch, basis)
        export_seconds = perf_counter() - export_started
        result = complete_rks_cuda_gradient_diagnostic(
            state,
            basis,
            compiler=compiler,
            cache=cache,
            measure_timeline=True,
        )
        endpoint = perf_counter() - started
        record = _record(mode, endpoint, export_seconds, basis, result)
        if mode == "artifact-warm":
            replay = _sample(
                mode="same-state-warm",
                atoms=atoms,
                method=method,
                compiler=compiler,
                cache=cache,
                reuse_state=(state, basis),
            )[0]
            record["same_state_replay"] = replay
        return record, None


def _record(
    mode: str,
    endpoint: float,
    export_seconds: float,
    basis: typing.Any,
    result: typing.Any,
) -> dict[str, typing.Any]:
    work = dict(result.work)
    timeline = dict(work["timeline"])
    phases = {"state_export": export_seconds, **timeline["exclusive_seconds"]}
    durations = (endpoint, export_seconds, *phases.values())
    if any(not math.isfinite(value) or value < 0.0 for value in durations):
        raise ValueError("timeline durations must be finite and nonnegative")
    gradient = np.asarray(result.gradient)
    if gradient.shape != (basis.natom, 3) or not np.isfinite(gradient).all():
        raise ValueError("timeline gradient must be finite with shape (natom, 3)")
    phase_total = sum(phases.values())
    if not math.isfinite(phase_total):
        raise ValueError("timeline duration sum must be finite")
    residual = endpoint - phase_total
    return {
        "status": "ok",
        "mode": mode,
        "nao": int(basis.nao),
        "natom": int(basis.natom),
        "endpoint_seconds": endpoint,
        "phase_seconds": phases,
        "reconciliation": {
            "phase_seconds": phase_total,
            "residual_seconds": residual,
            "residual_fraction": residual / endpoint if endpoint else 0.0,
            "within_5_percent": (abs(residual) <= 0.05 * endpoint)
            if endpoint
            else True,
        },
        "nested_seconds": timeline.get("nested_seconds", {}),
        "native_call_seconds": timeline.get("native_call_seconds", {}),
        "native_call_counts": timeline.get("native_call_counts", {}),
        "operation_counts": timeline.get("operation_counts", {}),
        "transfer_bytes": timeline.get("transfer_bytes", {}),
        "work": {
            key: work[key]
            for key in (
                "launches",
                "primitive_records",
                "xc_points",
                "grid_pair_visits",
                "tensor_executions",
                "tensor_work",
                "snapshot_export_work",
                "additional_device_peak_bound",
                "additional_host_numeric_bound",
                "endpoint_seconds",
            )
        },
    }


def _run_case_in_cache(
    fixture: str,
    method: str,
    compiler: CudaCompilerAdapter,
    cache_root: Path,
    include_cold: bool,
) -> dict[str, typing.Any]:
    atoms = FIXTURES[fixture]
    cache = cache_root
    case: dict[str, typing.Any] = {"fixture": fixture, "method": method, "samples": []}
    try:
        if include_cold:
            cold, _ = _sample(
                mode="cold",
                atoms=atoms,
                method=method,
                compiler=compiler,
                cache=cache,
            )
            case["samples"].append(cold)
        else:
            warmup, _ = _sample(
                mode="warmup-unreported",
                atoms=atoms,
                method=method,
                compiler=compiler,
                cache=cache,
            )
            case["warmup_endpoint_seconds"] = warmup["endpoint_seconds"]

        warm, _ = _sample(
            mode="artifact-warm",
            atoms=atoms,
            method=method,
            compiler=compiler,
            cache=cache,
        )
        same_state = warm.pop("same_state_replay")
        case["samples"].extend((warm, same_state))
        changed, _ = _sample(
            mode="changed-geometry",
            atoms=_moved(atoms),
            method=method,
            compiler=compiler,
            cache=cache,
        )
        case["samples"].append(changed)
    except NotImplementedError as exc:
        case["status"] = "unsupported"
        case["reason"] = str(exc)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        case["status"] = "error"
        case["reason"] = f"{type(exc).__name__}: {exc}"
    return case


def _run_case(
    fixture: str,
    method: str,
    compiler: CudaCompilerAdapter,
    cache_root: Path,
    include_cold: bool,
) -> dict[str, typing.Any]:
    if not include_cold:
        return _run_case_in_cache(fixture, method, compiler, cache_root, False)
    # A cold sample owns one new child; never recursively remove a caller path.
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="stationary-timeline-", dir=cache_root) as cache:
        return _run_case_in_cache(fixture, method, compiler, Path(cache), True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/stationary-662")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--fixtures", nargs="+", choices=tuple(FIXTURES), default=list(FIXTURES)
    )
    parser.add_argument("--include-cold", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "stationary CUDA timeline benchmark requires a Slurm allocation"
        )
    toolkit = Path(os.environ.get("CUDACXX", "/group/software/cuda-12.9.1/bin/nvcc"))
    compiler = CudaCompilerAdapter(
        toolkit, cuda_target_info("sm_120"), compile_timeout=900
    )
    library = Path(os.environ["VIBEQC_LIBRARY"])
    payload: dict[str, typing.Any] = {
        "schema": "vibeqc.stationary-cuda-benchmark/v1",
        "issue": 662,
        "source_commit": _git_head(),
        "native_library": str(library),
        "native_sha256": file_hash(library),
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "gpu": _gpu_metadata(),
        "parameters": {
            "tile_points": 256,
            "integral_terms": 32,
            "primitive_tile": 128,
            "grid": {
                "radial_points": GRID.radial_points,
                "angular_polar": GRID.angular_polar,
                "angular_azimuth": GRID.angular_azimuth,
            },
        },
        "cases": [],
    }
    for fixture in args.fixtures:
        for method in args.methods:
            case = _run_case(fixture, method, compiler, args.cache, args.include_cold)
            payload["cases"].append(case)
            status = case.get("status", "ok")
            print(f"{fixture} {method}: {status}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if args.require_all:
        for case in payload["cases"]:
            if case.get("status", "ok") != "ok":
                return 2
            if any(
                not sample["reconciliation"]["within_5_percent"]
                for sample in case["samples"]
            ):
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
