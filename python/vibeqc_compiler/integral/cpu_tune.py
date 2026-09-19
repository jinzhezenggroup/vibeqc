"""Bounded CPU schedule search, static resource models, and measured promotion.

This tuner consumes the #469 lane lowering and #470 explicit CPU targets. It
does not own Gaussian science. Every executable candidate comes from the same
IntegralIR / ShellClassComponentKernel path and is checked against a caller-
supplied independent reference before timing can influence selection.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import islice, product
from math import ceil, prod
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from vibeqc_compiler.common.cpu_dispatch import (
    CpuRuntimeFeatures,
    cpu_target_supported,
    detect_cpu_features,
)
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
    GENERIC_CPU_TARGET,
    CpuTargetInfo,
)
from vibeqc_compiler.common.performance import assess_comparison, measure_interleaved
from vibeqc_compiler.common.provenance import atomic_json, canonical_hash

from .cpu_lane import emit_first_components_cpu_lanes
from .cpu_lane_execute import (
    FirstDerivativeCpuLaneShellEvaluator,
    compile_first_derivative_cpu_lane_shell,
)
from .cpu_schedule import (
    CpuAlgebraPlacement,
    CpuScheduleIR,
    default_cpu_schedule,
)
from .expr import AlgebraFusion, AlgebraOrdering, PowerLowering
from .first_derivatives_execute import first_derivative_component_tiles
from .ir_serialization import integral_to_payload
from .shell_class import build_shell_class_component_kernel
from .shell_spec import cartesian_components


@dataclass(frozen=True, slots=True)
class CpuCacheInfo:
    l1_data_bytes: int | None
    l2_bytes: int | None
    source: str

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _parse_cache_size(text: str) -> int:
    value = text.strip().upper()
    scale = 1
    if value.endswith("K"):
        value, scale = value[:-1], 1024
    elif value.endswith("M"):
        value, scale = value[:-1], 1024**2
    result = int(value) * scale
    if result <= 0:
        raise ValueError("cache size must be positive")
    return result


def detect_cpu_cache_info() -> CpuCacheInfo:
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    l1 = None
    l2 = None
    if root.is_dir():
        for index in sorted(root.glob("index*")):
            try:
                level = int((index / "level").read_text().strip())
                kind = (index / "type").read_text().strip().lower()
                size = _parse_cache_size((index / "size").read_text())
            except (OSError, ValueError):
                continue
            if level == 1 and kind in ("data", "unified"):
                l1 = max(l1 or 0, size)
            elif level == 2 and kind in ("data", "unified"):
                l2 = max(l2 or 0, size)
        if l1 is not None or l2 is not None:
            return CpuCacheInfo(l1, l2, "linux-sysfs")
    return CpuCacheInfo(None, None, "unavailable")


@dataclass(frozen=True, slots=True)
class CpuTuneSchedule:
    target: CpuTargetInfo
    lane: CpuScheduleIR
    component_tile_size: int

    def __post_init__(self) -> None:
        self.lane.validate_for(self.target)
        if (
            type(self.component_tile_size) is not int
            or not 1 <= self.component_tile_size <= 64
        ):
            raise ValueError("CPU tune component tile size must be in 1..64")

    def to_payload(self) -> dict[str, object]:
        return {
            "target": self.target.to_payload(),
            "lane": self.lane.to_payload(),
            "component_tile_size": self.component_tile_size,
        }


@dataclass(frozen=True, slots=True)
class CpuTuneLimits:
    maximum_candidates: int = 16
    repeats: int = 5
    maximum_source_bytes: int = 2 * 1024**2
    maximum_working_set_bytes: int = 2 * 1024**2
    maximum_compile_seconds: float = 30.0
    parallel_tasks: int = 8
    parallel_workers: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_candidates <= 64:
            raise ValueError("CPU tune candidate limit must be in 1..64")
        if not 5 <= self.repeats <= 30:
            raise ValueError("CPU tune repeats must be in 5..30")
        if self.maximum_source_bytes < 1 or self.maximum_working_set_bytes < 1:
            raise ValueError("CPU tune static byte budgets must be positive")
        if self.maximum_compile_seconds <= 0:
            raise ValueError("CPU tune compile budget must be positive")
        if self.parallel_tasks < 2:
            raise ValueError("CPU tune parallel task count must be at least two")
        workers = tuple(self.parallel_workers)
        if (
            not workers
            or any(type(value) is not int or value < 1 for value in workers)
            or len(set(workers)) != len(workers)
        ):
            raise ValueError("CPU tune worker counts must be unique positive integers")
        object.__setattr__(self, "parallel_workers", workers)


DEFAULT_CPU_TUNE_LIMITS = CpuTuneLimits()


def _schedule_variants(target: CpuTargetInfo) -> tuple[CpuScheduleIR, ...]:
    baseline = default_cpu_schedule(target)
    rows = [
        baseline,
        replace(
            baseline,
            algebra_placement=CpuAlgebraPlacement.MATERIALIZED_CSE,
            algebra_ordering=AlgebraOrdering.TOPOLOGICAL,
        ),
        replace(
            baseline,
            algebra_placement=CpuAlgebraPlacement.PRESSURE_REMATERIALIZED,
        ),
        replace(baseline, power_lowering=PowerLowering.NATIVE),
    ]
    if "fma" in target.features:
        rows.append(replace(baseline, algebra_fusion=AlgebraFusion.FMA))
    seen: set[tuple[object, ...]] = set()
    result = []
    for row in rows:
        key = (
            row.algebra_placement,
            row.algebra_ordering,
            row.algebra_fusion,
            row.algebra_form,
            row.power_lowering,
        )
        if key not in seen:
            row.validate_for(target)
            result.append(row)
            seen.add(key)
    return tuple(result)


def cpu_tune_candidates(
    integral,
    *,
    targets: Sequence[CpuTargetInfo],
    limits: CpuTuneLimits = DEFAULT_CPU_TUNE_LIMITS,
) -> tuple[CpuTuneSchedule, ...]:
    """Enumerate a deterministic bounded schedule prefix for one complete shell."""

    component_count = integral.signature.component_count
    tile_sizes = tuple(
        dict.fromkeys(
            (
                min(component_count, 64),
                min(component_count, 16),
                min(component_count, 8),
            )
        )
    )

    def rows():
        # Put one directly comparable default from every target first so a
        # bounded prefix can never accidentally become a scalar-only search.
        for target in targets:
            yield CpuTuneSchedule(
                target,
                default_cpu_schedule(target),
                tile_sizes[0],
            )
        # Then vary algebra policy at a fixed complete-shell call boundary.
        for target in targets:
            variants = _schedule_variants(target)
            for lane in variants[1:]:
                yield CpuTuneSchedule(target, lane, tile_sizes[0])
        # Finally vary ABI/component batching with the target default schedule.
        for target in targets:
            for tile in tile_sizes[1:]:
                yield CpuTuneSchedule(target, default_cpu_schedule(target), tile)

    return tuple(islice(rows(), limits.maximum_candidates))


@functools.cache
def _component_static_plan(
    integral, component, schedule: CpuScheduleIR
) -> dict[str, int]:
    kernel = build_shell_class_component_kernel(
        integral.spec,
        component,
        integral=integral,
    )
    roots = (kernel.value,) + tuple(
        value for axes in kernel.gradients for value in axes
    )
    graph, optimized_roots = kernel.graph.apply_algebra_form(
        roots,
        schedule.algebra_form,
        schedule.power_lowering,
    )
    plan = graph.materialization_plan(
        optimized_roots,
        schedule.algebra_placement.policy(),
        schedule.algebra_ordering,
        schedule.algebra_fusion,
    )
    return {
        "arithmetic_operations": plan.arithmetic_operation_count,
        "materialized_values": plan.materialized_value_count,
        "peak_live_values": plan.peak_live_values,
        "rematerialized_values": plan.rematerialized_value_count,
        "fma_operations": plan.fma_operation_count,
    }


def cpu_static_cost(
    integral,
    tune_schedule: CpuTuneSchedule,
    *,
    record_count: int,
    cache_info: CpuCacheInfo | None = None,
) -> dict[str, object]:
    """Return labeled static resource estimates for a complete-shell candidate."""

    if type(record_count) is not int or record_count < 1:
        raise ValueError("CPU static cost requires a positive record count")
    target, schedule = tune_schedule.target, tune_schedule.lane
    tiles = first_derivative_component_tiles(
        integral,
        tile_size=tune_schedule.component_tile_size,
    )
    labels = tuple(
        product(*(cartesian_components(l) for l in integral.signature.angular))
    )
    metrics = [
        _component_static_plan(integral, labels[index], schedule)
        for index in range(integral.signature.component_count)
    ]
    maxima = {key: max(row[key] for row in metrics) for key in metrics[0]}
    lanes = target.vector_lanes
    lane_tiles = ceil(record_count / lanes)
    total_lanes = lane_tiles * lanes
    utilization = record_count / total_lanes
    ninput = len(integral.signature.shells) + 3 * len(integral.operator.centers)
    noutput = 1 + 3 * len(integral.operator.centers)
    component_count = integral.signature.component_count
    vector_live_bytes = maxima["peak_live_values"] * lanes * 8
    # Match the complete-shell evaluator's conservative numeric reservation,
    # including the AoS record buffer and both shell/tile result storage. The
    # lane-only estimate omitted these owned arrays and admitted undersized
    # caller budgets. Keep record capacity identical between planning/execution.
    record_capacity = max(128, record_count)
    tile_output_values = max(len(tile) for tile in tiles) * noutput
    native_stack_values = lanes * (ninput + 1 + noutput) + tile_output_values
    numeric_storage_bytes = 8 * (
        record_capacity * (ninput + 1)
        + 4 * tile_output_values
        + native_stack_values
        + component_count * noutput
    )
    runtime_working_set = numeric_storage_bytes + vector_live_bytes
    logical_traffic = (
        record_count * (ninput + 1) * 8
        + lane_tiles * lanes * (ninput + 1) * 8
        + component_count * lane_tiles * lanes * (ninput + noutput) * 8
        + component_count * record_count * noutput * 8
    )
    sources = [
        emit_first_components_cpu_lanes(integral, tile, target, schedule)
        for tile in tiles
    ]
    source_bytes = sum(len(source.encode("utf-8")) for source in sources)
    cache = cache_info or detect_cpu_cache_info()
    return {
        "schema": "vibeqc.cpu.static-cost.v2",
        "target": target.to_payload(),
        "schedule": schedule.to_payload(),
        "component_tile_size": tune_schedule.component_tile_size,
        "component_tiles": len(tiles),
        "component_count": component_count,
        "record_count": record_count,
        "record_capacity": record_capacity,
        "estimated_numeric_storage_bytes": numeric_storage_bytes,
        "vector_width_fp64": lanes,
        "lane_utilization": utilization,
        "tail_fraction": 1.0 - utilization,
        "estimated_peak_live_vector_values": maxima["peak_live_values"],
        "estimated_peak_live_scalar_values": (
            maxima["peak_live_values"] if lanes == 1 else 0
        ),
        "estimated_vector_live_bytes": vector_live_bytes,
        "maximum_component_arithmetic_operations": maxima["arithmetic_operations"],
        "maximum_component_materialized_values": maxima["materialized_values"],
        "maximum_component_rematerialized_values": maxima["rematerialized_values"],
        "maximum_component_fma_operations": maxima["fma_operations"],
        "estimated_runtime_working_set_bytes": runtime_working_set,
        "estimated_logical_load_store_bytes": logical_traffic,
        "aos_to_soa_bytes_per_lane_tile": lanes * (ninput + 1) * 8,
        "generated_source_bytes": source_bytes,
        "cache": cache.to_payload(),
        "estimated_l1_fit": (
            runtime_working_set <= cache.l1_data_bytes
            if cache.l1_data_bytes is not None
            else None
        ),
        "estimated_l2_fit": (
            runtime_working_set <= cache.l2_bytes
            if cache.l2_bytes is not None
            else None
        ),
        "compiled_register_spill_evidence": None,
        "register_spill_scope": (
            "portable C++ compiler path exposes no stable register/spill report; "
            "live-value metrics are static estimates, not measured registers"
        ),
    }


def _workload_identity(integral, primitives, centers, reference_identity: str) -> str:
    payload = {
        "schema": "vibeqc.cpu-tune-workload.v1",
        "integral": integral_to_payload(integral),
        "primitives": [
            [[float(exponent), float(coefficient)] for exponent, coefficient in shell]
            for shell in primitives
        ],
        "centers": np.asarray(centers, dtype=np.float64).tolist(),
        "reference_identity": reference_identity,
    }
    return canonical_hash(payload)


def _measure_pair(
    baseline,
    candidate,
    primitives,
    centers,
    *,
    repeats: int,
    inputs_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def evaluate(selection):
        executor = baseline if selection == "baseline" else candidate
        value = executor.contract(primitives, centers)
        return {
            "finite": bool(np.isfinite(value).all()),
            "selected_target": executor.artifact.target.name,
            "schedule_identity": executor.artifact.program_identity,
        }

    rows = measure_interleaved(
        evaluate,
        lambda: None,
        workload="unchanged-geometry",
        inputs_hash=inputs_hash,
        repeats=repeats,
    )
    return rows, assess_comparison(rows)


def _parallel_measurement(
    evaluator,
    primitives,
    centers,
    *,
    workers: int,
    tasks: int,
    repeats: int,
) -> dict[str, object]:
    samples = []
    if workers == 1:
        for _ in range(repeats):
            start = time.perf_counter()
            for _task in range(tasks):
                evaluator.contract(primitives, centers)
            samples.append(time.perf_counter() - start)
    else:
        # Retain the pool across samples: this is warm throughput evidence,
        # not Python worker-construction latency.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(repeats):
                start = time.perf_counter()
                futures = [
                    pool.submit(evaluator.contract, primitives, centers)
                    for _task in range(tasks)
                ]
                for future in futures:
                    result = future.result()
                    if not np.isfinite(result).all():
                        raise FloatingPointError(
                            "parallel CPU tune result is non-finite"
                        )
                samples.append(time.perf_counter() - start)
    middle = float(median(samples))
    return {
        "workers": workers,
        "tasks": tasks,
        "samples": samples,
        "median_seconds": middle,
        "systems_per_second": tasks / middle,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def tune_cpu_first_derivative_shell(
    integral,
    compiler,
    cache,
    *,
    primitives,
    centers,
    independent_reference,
    reference_identity: str,
    runtime: CpuRuntimeFeatures | None = None,
    targets: Sequence[CpuTargetInfo] | None = None,
    limits: CpuTuneLimits = DEFAULT_CPU_TUNE_LIMITS,
    atol: float = 5e-11,
    rtol: float = 2e-10,
) -> dict[str, object]:
    """Compile, validate, time, and conservatively select one CPU shell schedule."""

    if not reference_identity:
        raise ValueError("CPU tuning requires an independent reference identity")
    reference = np.asarray(independent_reference, dtype=np.float64)
    expected_shape = (
        integral.signature.component_count,
        1 + 3 * len(integral.operator.centers),
    )
    if reference.shape != expected_shape or not np.isfinite(reference).all():
        raise ValueError("CPU tuning independent reference has invalid shape/values")
    runtime = runtime or detect_cpu_features()
    selected_targets = tuple(
        targets
        or (
            GENERIC_CPU_TARGET,
            AVX2_FMA_TARGET,
            AVX512F_FMA_TARGET,
        )
    )
    candidates = cpu_tune_candidates(integral, targets=selected_targets, limits=limits)
    workload_identity = _workload_identity(
        integral, primitives, centers, reference_identity
    )
    record_count = prod(len(shell) for shell in primitives)
    cache_info = detect_cpu_cache_info()
    rows: list[dict[str, object]] = []
    ready = []
    cache = Path(cache).resolve()
    baseline_executor = None
    baseline_row = None

    for index, candidate in enumerate(candidates):
        static = cpu_static_cost(
            integral,
            candidate,
            record_count=record_count,
            cache_info=cache_info,
        )
        row: dict[str, object] = {
            "index": index,
            "candidate": candidate.to_payload(),
            "static_resources": static,
            "status": "static",
        }
        reasons = []
        source_bytes = static["generated_source_bytes"]
        working_set_bytes = static["estimated_runtime_working_set_bytes"]
        if type(source_bytes) is not int or type(working_set_bytes) is not int:
            raise TypeError("CPU static cost byte metrics must be integers")
        if source_bytes > limits.maximum_source_bytes:
            reasons.append("generated source exceeds static compile budget")
        if working_set_bytes > limits.maximum_working_set_bytes:
            reasons.append("estimated working set exceeds tuning budget")
        if reasons:
            row.update(
                status="pruned", stage="static-resource", reason="; ".join(reasons)
            )
            rows.append(row)
            continue
        if not cpu_target_supported(candidate.target, runtime):
            row.update(
                status="rejected",
                stage="target-compatibility",
                reason="runtime does not advertise the candidate ISA",
            )
            rows.append(row)
            continue
        try:
            artifact = compile_first_derivative_cpu_lane_shell(
                integral,
                compiler,
                cache / f"candidate-{index:02d}",
                target=candidate.target,
                schedule=candidate.lane,
                tile_size=candidate.component_tile_size,
            )
        except (RuntimeError, ValueError) as error:
            row.update(status="rejected", stage="compilation", reason=str(error))
            rows.append(row)
            continue
        compile_seconds = sum(
            float(tile.native.metadata["compile_seconds"]) for tile in artifact.tiles
        )
        binary_bytes = sum(
            tile.native.library.stat().st_size for tile in artifact.tiles
        )
        compiled_resources: dict[str, object] = {
            "cold_compile_seconds_sum": compile_seconds,
            "shared_library_bytes_sum": binary_bytes,
            "artifact_keys": [tile.native.metadata["key"] for tile in artifact.tiles],
            "shell_program_identity": artifact.program_identity,
        }
        row["compiled_resources"] = compiled_resources
        if compile_seconds > limits.maximum_compile_seconds:
            row.update(
                status="rejected",
                stage="compile-budget",
                reason="candidate exceeds cold compilation budget",
            )
            rows.append(row)
            continue
        load_start = time.perf_counter()
        evaluator = FirstDerivativeCpuLaneShellEvaluator(
            artifact,
            record_capacity=max(128, record_count),
            budget_bytes=limits.maximum_working_set_bytes,
        )
        compiled_resources["cold_load_seconds"] = time.perf_counter() - load_start
        compiled_resources["numeric_storage_bytes"] = evaluator.numeric_bytes
        actual = evaluator.contract(primitives, centers)
        error = float(np.max(np.abs(actual - reference)))
        numerical = {
            "maximum_absolute_error": error,
            "atol": atol,
            "rtol": rtol,
            "passed": bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
        }
        row["numerical"] = numerical
        if not numerical["passed"]:
            row.update(
                status="rejected",
                stage="independent-numerical",
                reason="candidate failed the independent reference gate",
            )
            rows.append(row)
            continue
        row.update(status="ready", stage="numerical")
        rows.append(row)
        ready.append((row, evaluator))

        is_baseline = (
            candidate.target == GENERIC_CPU_TARGET
            and candidate.lane == default_cpu_schedule(GENERIC_CPU_TARGET)
            and candidate.component_tile_size
            == min(integral.signature.component_count, 64)
        )
        if is_baseline:
            baseline_executor = evaluator
            baseline_row = row

    if baseline_executor is None or baseline_row is None:
        raise RuntimeError(
            "CPU tuning could not establish the required scalar baseline"
        )

    timed = []
    measured = []
    for row, evaluator in ready:
        if row is baseline_row:
            continue
        samples, comparison = _measure_pair(
            baseline_executor,
            evaluator,
            primitives,
            centers,
            repeats=limits.repeats,
            inputs_hash=workload_identity,
        )
        row["timing_samples"] = samples
        row["comparison"] = comparison
        summary = comparison.get("workloads", {}).get("unchanged-geometry", {})
        candidate_summary = summary.get("candidate", {})
        median_seconds = candidate_summary.get("median_seconds")
        if isinstance(median_seconds, (int, float)):
            measured.append((float(median_seconds), row, evaluator))
            if comparison.get("status") == "pass":
                timed.append((float(median_seconds), row, evaluator))

    selected_row = baseline_row
    selected_executor = baseline_executor
    selection_reason = (
        "scalar baseline retained; no candidate cleared the benefit/noise gate"
    )
    if timed:
        _, selected_row, selected_executor = min(timed, key=lambda item: item[0])
        selection_reason = (
            "fastest candidate clearing paired benefit/noise and numerical gates"
        )

    parallel: dict[str, object] = {
        "baseline": [
            _parallel_measurement(
                baseline_executor,
                primitives,
                centers,
                workers=workers,
                tasks=limits.parallel_tasks,
                repeats=limits.repeats,
            )
            for workers in limits.parallel_workers
        ],
        "selected": [
            _parallel_measurement(
                selected_executor,
                primitives,
                centers,
                workers=workers,
                tasks=limits.parallel_tasks,
                repeats=limits.repeats,
            )
            for workers in limits.parallel_workers
        ],
    }
    simd_measured = [
        item for item in measured if item[2].artifact.target.vector_lanes > 1
    ]
    if simd_measured:
        _, simd_row, simd_executor = min(simd_measured, key=lambda item: item[0])
        parallel["best_simd"] = {
            "candidate": simd_row["candidate"],
            "program_identity": simd_executor.artifact.program_identity,
            "measurements": [
                _parallel_measurement(
                    simd_executor,
                    primitives,
                    centers,
                    workers=workers,
                    tasks=limits.parallel_tasks,
                    repeats=limits.repeats,
                )
                for workers in limits.parallel_workers
            ],
        }
    else:
        parallel["best_simd"] = {
            "status": "not-run",
            "reason": "no numerically valid runtime-compatible SIMD candidate",
        }
    selected_candidate = selected_row["candidate"]
    selected_compiled = selected_row["compiled_resources"]
    if not isinstance(selected_compiled, dict):
        raise TypeError("selected CPU candidate lacks compiled resource metadata")
    selected_program_identity = selected_compiled.get("shell_program_identity")
    selected_artifact_keys = selected_compiled.get("artifact_keys")
    if not isinstance(selected_program_identity, str):
        raise TypeError("selected CPU candidate lacks a program identity")
    if not isinstance(selected_artifact_keys, list) or not all(
        isinstance(value, str) for value in selected_artifact_keys
    ):
        raise TypeError("selected CPU candidate lacks compiler artifact keys")
    selection_identity = canonical_hash(
        {
            "schema": "vibeqc.cpu-tune-selection.v1",
            "integral": integral_to_payload(integral),
            "workload_identity": workload_identity,
            "reference_identity": reference_identity,
            "selected_candidate": selected_candidate,
            "selected_program_identity": selected_program_identity,
            "selected_compiler_artifact_keys": selected_artifact_keys,
        }
    )
    return {
        "schema": "vibeqc.cpu-autotune.v1",
        "integral": integral_to_payload(integral),
        "workload_identity": workload_identity,
        "reference_identity": reference_identity,
        "runtime": runtime.to_payload(),
        "cache": cache_info.to_payload(),
        "limits": asdict(limits),
        "candidates": rows,
        "selected": {
            "candidate": selected_candidate,
            "program_identity": selected_program_identity,
            "compiler_artifact_keys": selected_artifact_keys,
            "selection_identity": selection_identity,
            "reason": selection_reason,
        },
        "parallel_interaction": parallel,
    }


def write_cpu_tuning_manifest(path: str | Path, result: dict[str, object]) -> None:
    if result.get("schema") != "vibeqc.cpu-autotune.v1":
        raise ValueError("not a CPU autotuning result")
    atomic_json(Path(path), result)
