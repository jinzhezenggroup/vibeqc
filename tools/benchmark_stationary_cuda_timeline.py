"""Benchmark complete stationary CUDA force phases without adding profiler synchronization.

The JSON schema keeps additive host-wall phases separate from CUDA transfer,
launch and synchronization counters.  Run this only inside the existing Slurm
GPU profile; it is measurement evidence, not a public-force implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import typing
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cuda import (
    PreparedStationaryCudaExecution,
    complete_rks_cuda_gradient_diagnostic,
)
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.dft import NativeAO

if typing.TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

SYSTEMS = {
    "h2": [
        ("H", (0.10, -0.20, -0.70)),
        ("H", (0.20, 0.10, 0.80)),
    ],
    "water": [
        ("O", (0.10, -0.10, 0.00)),
        ("H", (0.10, 0.20, 1.70)),
        ("H", (1.60, -0.20, -0.50)),
    ],
    # 12 AO in the default minimal basis: large enough to expose n^4 source
    # scaling while staying below the checked 2M primitive-record cap.
    "acetylene": [
        ("H", (-3.20, 0.00, 0.00)),
        ("C", (-1.20, 0.00, 0.00)),
        ("C", (1.20, 0.00, 0.00)),
        ("H", (3.20, 0.00, 0.00)),
    ],
}
DEFAULT_METHODS = (
    "lda-rks",
    "lda-uks",
    "pbe-rks",
    "pbe-uks",
    "r2scan-rks",
    "r2scan-uks",
)
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
# SCF is fixture preparation, not part of the force timeline. These stricter
# settings qualify the exported physical state without relaxing native checks.
SCF_PREPARATION = MappingProxyType(
    {"energy_tolerance": 1e-12, "density_tolerance": 1e-12, "max_iterations": 200}
)
# Keep measurement provenance and the executed production schedule on one owner.
# These are the public stationary CUDA defaults; benchmark metadata is derived
# from this mapping instead of maintaining an independent copy.
PRODUCTION_EXECUTION = MappingProxyType(
    {"tile_points": 256, "integral_terms": 32, "primitive_tile": 4096}
)


def _spin(method: str) -> tuple[int, int]:
    return (1, 2) if method.endswith("-uks") else (0, 1)


def _jsonable(value: typing.Any) -> typing.Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *command], text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_identity() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        **SCF_PREPARATION,
    )


def _timeline_record(
    scenario: str,
    state_export_seconds: float,
    work: typing.Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    diagnostic = work["timeline"]
    phases = {
        "state_export": state_export_seconds,
        **diagnostic["exclusive_wall_seconds"],
    }
    durations = (state_export_seconds, diagnostic["endpoint_seconds"], *phases.values())
    if any(not math.isfinite(value) or value < 0.0 for value in durations):
        raise ValueError("stationary timeline requires finite nonnegative durations")
    endpoint = state_export_seconds + diagnostic["endpoint_seconds"]
    reconciled = sum(phases.values())
    if not math.isfinite(endpoint) or not math.isfinite(reconciled):
        raise ValueError("stationary timeline duration sum is not finite")
    return {
        "schema": "vibeqc.stationary-cuda-force-timeline.v1",
        "scenario": scenario,
        "exclusive_wall_seconds": phases,
        "reconciled_seconds": reconciled,
        "endpoint_seconds": endpoint,
        "reconciliation_error_seconds": endpoint - reconciled,
        "state_export_reused": state_export_seconds == 0.0,
        "measurement_policy": diagnostic["measurement_policy"],
    }


def _diagnostic(
    state: typing.Any,
    basis: typing.Any,
    compiler: CudaCompilerAdapter | None,
    cache: Path,
    *,
    prepared: PreparedStationaryCudaExecution | None = None,
    target: typing.Any = None,
    library: Path | None = None,
) -> tuple[typing.Any, float]:
    kwargs: dict[str, typing.Any] = {
        "compiler": compiler,
        "cache": cache,
        "profile_device": True,
    }
    if prepared is not None:
        if compiler is not None:
            raise ValueError(
                "prepared AOT timeline must not enable runtime compilation"
            )
        if target is None or library is None:
            raise ValueError("prepared AOT timeline requires target and native library")
        kwargs.update(
            aot_directory=library.parent,
            native_grid_library=library,
            target=target,
            prepared=prepared,
            **PRODUCTION_EXECUTION,
        )
    started = perf_counter()
    result = complete_rks_cuda_gradient_diagnostic(state, basis, **kwargs)
    observed = perf_counter() - started
    return result, observed


def _successful_record(
    *,
    system: str,
    method: str,
    scenario: str,
    repeat: int,
    state_export_seconds: float,
    observed_diagnostic_seconds: float,
    result: typing.Any,
    energy: float,
    basis: typing.Any,
) -> dict[str, typing.Any]:
    work = dict(result.work)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (observed_diagnostic_seconds, work["endpoint_seconds"])
    ):
        raise ValueError("stationary diagnostic requires finite nonnegative durations")
    if not math.isfinite(energy):
        raise ValueError("stationary benchmark energy must be finite")
    timeline = _timeline_record(scenario, state_export_seconds, work)
    tolerance = max(2e-6, 2e-6 * timeline["endpoint_seconds"])
    if abs(timeline["reconciliation_error_seconds"]) > tolerance:
        raise RuntimeError("exclusive stationary timeline did not reconcile")
    if abs(observed_diagnostic_seconds - work["endpoint_seconds"]) > max(
        0.005, 0.02 * observed_diagnostic_seconds
    ):
        raise RuntimeError("diagnostic timer disagrees with external wall clock")
    gradient = np.asarray(result.gradient)
    if gradient.shape != (basis.natom, 3) or not np.isfinite(gradient).all():
        raise RuntimeError("stationary gradient requires finite atom-by-three values")
    return {
        "status": "ok",
        "system": system,
        "method": method,
        "spin": "uks" if method.endswith("-uks") else "rks",
        "scenario": scenario,
        "repeat": repeat,
        "atoms": basis.natom,
        "ao_count": basis.nao,
        "primitive_count": basis.nprimitive,
        "energy": energy,
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "timeline": timeline,
        "diagnostic_external_wall_seconds": observed_diagnostic_seconds,
        "work": _jsonable(work),
    }


def _failed_record(
    *,
    system: str,
    method: str,
    scenario: str,
    repeat: int,
    error: Exception,
) -> dict[str, typing.Any]:
    return {
        "status": "unsupported" if isinstance(error, NotImplementedError) else "error",
        "system": system,
        "method": method,
        "spin": "uks" if method.endswith("-uks") else "rks",
        "scenario": scenario,
        "repeat": repeat,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _export_state(batch: typing.Any, basis: typing.Any) -> tuple[typing.Any, float]:
    started = perf_counter()
    state = StationaryKsState.from_native(batch, basis)
    return state, perf_counter() - started


def benchmark_case(
    *,
    system: str,
    atoms: list[tuple[str, tuple[float, float, float]]],
    method: str,
    compiler: CudaCompilerAdapter | None,
    cache: Path,
    same_state_repeats: int,
    records: list[dict[str, typing.Any]] | None = None,
    target: typing.Any = None,
    library: Path | None = None,
) -> list[dict[str, typing.Any]]:
    charge, multiplicity = _spin(method)
    calc = _calculator(method)
    records = [] if records is None else records
    cache.mkdir(parents=True, exist_ok=True)
    # A cold measurement owns a fresh child, never deletes caller cache/evidence.
    cache = Path(tempfile.mkdtemp(prefix="cold-", dir=cache))
    if (target is None) != (library is None):
        raise ValueError(
            "prepared AOT timeline requires both target and native library"
        )
    prepared_context: typing.ContextManager[PreparedStationaryCudaExecution | None] = (
        PreparedStationaryCudaExecution() if target is not None else nullcontext(None)
    )

    with (
        calc.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(atoms, charge=charge, multiplicity=multiplicity) as basis,
        prepared_context as prepared,
    ):

        def diagnostic(
            state: typing.Any, current_basis: typing.Any
        ) -> tuple[typing.Any, float]:
            if prepared is None:
                return _diagnostic(state, current_basis, compiler, cache)
            return _diagnostic(
                state,
                current_basis,
                compiler,
                cache,
                prepared=prepared,
                target=target,
                library=library,
            )

        cold_item = batch.execute(strict=True, properties=("energy",)).items[0]
        cold_state, export_seconds = _export_state(batch, basis)
        try:
            result, wall = diagnostic(cold_state, basis)
            records.append(
                _successful_record(
                    system=system,
                    method=method,
                    scenario="cold",
                    repeat=0,
                    state_export_seconds=export_seconds,
                    observed_diagnostic_seconds=wall,
                    result=result,
                    energy=cold_item.energy,
                    basis=basis,
                )
            )
        except Exception as error:
            records.append(
                _failed_record(
                    system=system,
                    method=method,
                    scenario="cold",
                    repeat=0,
                    error=error,
                )
            )
            if isinstance(error, NotImplementedError):
                return records
            raise

        warm_item = batch.execute(strict=True, properties=("energy",)).items[0]
        warm_state, export_seconds = _export_state(batch, basis)
        result, wall = diagnostic(warm_state, basis)
        records.append(
            _successful_record(
                system=system,
                method=method,
                scenario="artifact_warm",
                repeat=0,
                state_export_seconds=export_seconds,
                observed_diagnostic_seconds=wall,
                result=result,
                energy=warm_item.energy,
                basis=basis,
            )
        )

        for repeat in range(same_state_repeats):
            result, wall = diagnostic(warm_state, basis)
            records.append(
                _successful_record(
                    system=system,
                    method=method,
                    scenario="same_state_warm",
                    repeat=repeat,
                    state_export_seconds=0.0,
                    observed_diagnostic_seconds=wall,
                    result=result,
                    energy=warm_item.energy,
                    basis=basis,
                )
            )

        xyz = np.asarray([atom[1] for atom in atoms], dtype=np.float64)
        xyz[-1] += np.asarray((0.0010, -0.0005, 0.0003))
        changed_atoms = [
            (atom[0], tuple(float(x) for x in coord))
            for atom, coord in zip(atoms, xyz, strict=True)
        ]
        changed_item = batch.execute(
            coordinates=[xyz], strict=True, properties=("energy",)
        ).items[0]
        with NativeAO(
            changed_atoms, charge=charge, multiplicity=multiplicity
        ) as changed_basis:
            changed_state, export_seconds = _export_state(batch, changed_basis)
            result, wall = diagnostic(changed_state, changed_basis)
            records.append(
                _successful_record(
                    system=system,
                    method=method,
                    scenario="changed_geometry",
                    repeat=0,
                    state_export_seconds=export_seconds,
                    observed_diagnostic_seconds=wall,
                    result=result,
                    energy=changed_item.energy,
                    basis=changed_basis,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/issue-662-stationary-timeline")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--same-state-repeats", type=int, default=3)
    parser.add_argument(
        "--target",
        default=os.environ.get("VIBEQC_STATIONARY_CUDA_TARGET", "sm_120"),
    )
    args = parser.parse_args()
    methods = tuple(x for x in args.methods.split(",") if x)
    systems = tuple(x for x in args.systems.split(",") if x)
    unknown = set(systems) - SYSTEMS.keys()
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    if not 1 <= args.same_state_repeats <= 20:
        parser.error("--same-state-repeats must be in [1,20]")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("issue #662 GPU benchmark requires a Slurm allocation")
    library_text = os.environ.get("VIBEQC_LIBRARY")
    if not library_text:
        parser.error("set VIBEQC_LIBRARY to the qualified native CUDA library")
    library = Path(library_text).resolve()
    if not library.is_file():
        parser.error(f"VIBEQC_LIBRARY is not a file: {library}")
    target = cuda_target_info(args.target)

    records: list[dict[str, typing.Any]] = []
    payload = {
        "schema": "vibeqc.stationary-cuda-force-benchmark.v1",
        "issue": 662,
        "completed": False,
        "records": records,
        "provenance": {
            "head": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
            "library": str(library),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "gpu": _gpu_identity(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "target": args.target,
            "execution_route": "prepared-aot",
            "runtime_compilation": False,
            "grid": [24, 8, 16],
            "scf_preparation": dict(SCF_PREPARATION),
            "production_defaults": dict(PRODUCTION_EXECUTION),
        },
    }

    try:
        for method in methods:
            for system in systems:
                print(f"issue662: {method} {system}", flush=True)
                benchmark_case(
                    system=system,
                    atoms=SYSTEMS[system],
                    method=method,
                    compiler=None,
                    target=target,
                    library=library,
                    cache=args.cache / method / system,
                    same_state_repeats=args.same_state_repeats,
                    records=records,
                )
        payload["completed"] = True
    except Exception as error:
        payload["failure"] = _failed_record(
            system=system, method=method, scenario="case", repeat=0, error=error
        )
        raise
    finally:
        # Retain completed and partial-case rows even if a later SCF/export fails.
        # Publication does not swallow the failure or relabel the campaign complete.
        atomic_json(args.output, payload)
    ok = sum(record["status"] == "ok" for record in records)
    unsupported = sum(record["status"] == "unsupported" for record in records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "successful_records": ok,
                "unsupported_records": unsupported,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
