"""Bounded TensorIR schedule search, with no compiler or device side effects.

Only dimensions implemented by the ordinary-stream emitter are searched. GEMM
panels are global allocations, not shared-memory tiles. Static register pressure
and occupancy are explicitly heuristics for generated kernels, never cuBLAS
resource measurements or grounds for performance promotion.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, fields
from itertools import combinations, islice, product
from math import prod

from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.provenance import canonical_hash

from .cuda_emit import emit_cuda
from .cuda_gemm import gemm_contract
from .cuda_plan import (
    TensorPlan,
    TensorSchedule,
    estimated_cuda_launches,
    plan_cuda,
)
from .precision import describe_precision
from .program import Program


@dataclass(frozen=True)
class TensorScheduleSpace:
    """Ordered axes; visit single-axis changes before higher-order interactions.

    Enumeration is a bounded Hamming-radius walk, not a materialized Cartesian
    product. Axis order and values are part of the reproducible search identity.
    A new axis needs no handwritten list of complete schedule combinations.
    """

    views: tuple[bool, ...] = (True, False)
    fuse: tuple[bool, ...] = (True, False)
    recompute: tuple[bool, ...] = (False, True)
    direct_gemm: tuple[bool, ...] = (True, False)
    layouts: tuple[bool, ...] = (False, True)
    threads: tuple[int, ...] = (128, 64, 256)
    tile_m: tuple[int, ...] = (128, 64, 256, 512, 32)
    tile_n: tuple[int, ...] = (128, 64, 256, 512, 32)
    tile_k: tuple[int, ...] = (128, 64, 256, 512, 32)
    elements_per_thread: tuple[int, ...] = (1, 2, 4)
    reduction_unroll: tuple[int, ...] = (1, 2, 4)
    staging_width: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        for field in fields(self):
            values = tuple(getattr(self, field.name))
            if not 1 <= len(values) <= 16:
                raise ValueError("schedule axes require 1..16 values")
            for value in values:
                TensorSchedule(**{field.name: value})
            if len(set(values)) != len(values):
                raise ValueError("schedule axes must not contain duplicates")
            object.__setattr__(self, field.name, values)

    @property
    def cardinality(self) -> int:
        return prod(len(getattr(self, field.name)) for field in fields(self))

    def generate(self, maximum: int = 128) -> tuple[TensorSchedule, ...]:
        """Return a deterministic prefix without enumerating the full space."""
        _positive_int(maximum, "candidate limit")
        axes = asdict(self)
        names = tuple(axes)
        anchor = {name: values[0] for name, values in axes.items()}

        def walk() -> typing.Any:
            yield TensorSchedule(**anchor)
            for radius in range(1, len(names) + 1):
                for changed in combinations(names, radius):
                    for values in product(*(axes[name][1:] for name in changed)):
                        yield TensorSchedule(**(anchor | dict(zip(changed, values))))

        return tuple(islice(walk(), maximum))


def _positive_int(value: typing.Any, label: typing.Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class TensorSearchLimits:
    """Independent generation, compilation and static-source budgets.

    The existing tuner wall-clock deadline and compiler process timeout remain
    separate hard stop mechanisms. The mandatory baseline is not a candidate.
    """

    maximum_candidates: int = 256
    maximum_compilations: int = 12
    maximum_source_bytes: int = 2 * 1024**2
    minimum_resident_blocks: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_int(value, name)
        if self.maximum_candidates > 4096:
            raise ValueError("candidate limit must not exceed 4096")
        if self.maximum_compilations > self.maximum_candidates:
            raise ValueError("compilation limit must not exceed candidate limit")


DEFAULT_SEARCH_LIMITS = TensorSearchLimits()


@dataclass(frozen=True)
class TensorScreeningPolicy:
    """Bound full qualification using ranking-only representative measurements.

    Fixture indices select existing inputs without changing their shapes or
    values. A screen never grants a performance guard. Every finalist must pass
    fresh complete-endpoint measurements on every original fixture. Searches
    small enough to fit the finalist budget bypass screening entirely.
    """

    maximum_finalists: int = 3
    repeats: int = 5
    fixture_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        _positive_int(self.maximum_finalists, "finalist limit")
        if self.maximum_finalists > 4096:
            raise ValueError("finalist limit must not exceed 4096")
        if type(self.repeats) is not int or not 5 <= self.repeats <= 30:
            raise ValueError("screening repeats must be in 5..30")
        indices = tuple(self.fixture_indices)
        if not 1 <= len(indices) <= 8 or any(
            type(index) is not int or not 0 <= index < 8 for index in indices
        ):
            raise ValueError("screening requires 1..8 fixture indices in 0..7")
        if len(set(indices)) != len(indices):
            raise ValueError("screening fixture indices must be unique")
        object.__setattr__(self, "fixture_indices", indices)


DEFAULT_SCREENING_POLICY = TensorScreeningPolicy()


def execution_key(plan: TensorPlan) -> str:
    """Ignore requested knobs that do not alter this plan's executable work.

    In particular, tile sizes do not affect direct GEMM; oversized packing
    panels are clamped to the contraction dimensions by the existing emitter.
    Aliases, lifetimes, allocations, outputs and target identity remain checked.
    This key only avoids redundant tuning, never replaces an artifact key.
    """
    payload = plan.to_payload()
    payload.pop("schedule")
    # Planning diagnostics are provenance, not executable work. Physical layout
    # changes remain represented by step layouts, GEMM kinds and layout_identity.
    payload.pop("layout_planning", None)
    active_steps = tuple(
        step
        for step in plan.steps
        if not step.virtual
        and step.node.op not in ("input", "constant")
        and step.node.spec.size
    )
    generic_steps = tuple(step for step in active_steps if step.gemm == "none")
    packed_steps = tuple(step for step in active_steps if step.gemm == "packed")
    payload["threads"] = plan.schedule.threads if active_steps else None
    payload["elements_per_thread"] = (
        plan.schedule.elements_per_thread if generic_steps else None
    )
    payload["reduction_unroll"] = (
        plan.schedule.reduction_unroll
        if any(step.node.op in ("reduce", "einsum") for step in generic_steps)
        else None
    )
    payload["staging_width"] = plan.schedule.staging_width if packed_steps else None
    payload["packing_tiles"] = [
        tuple(
            min(tile, size)
            for tile, size in zip(
                (plan.schedule.tile_m, plan.schedule.tile_n, plan.schedule.tile_k),
                (g.m, g.n, g.k),
                strict=True,
            )
        )
        for step in plan.steps
        if step.gemm == "packed" and (g := gemm_contract(step.node)) is not None
    ]
    return canonical_hash(payload)


def _resident_blocks(
    plan: typing.Any, registers: typing.Any, shared_bytes: typing.Any
) -> typing.Any:
    target, threads = plan.target, plan.schedule.threads
    limits = [target.maximum_blocks_per_sm, target.maximum_threads_per_sm // threads]
    if registers:
        limits.append(target.registers_per_sm // (registers * threads))
    if shared_bytes:
        limits.append(target.shared_memory_per_sm // shared_bytes)
    # Allocation granularity is deliberately not modeled: this is an upper
    # bound, not a claim of achievable occupancy or measured SM utilization.
    return min(limits)


def _fp64_accumulation_terms(plan: TensorPlan) -> int:
    """Count scalar contributions widened from FP32 into qualified FP64 reductions."""
    total = 0
    for step in plan.steps:
        value = plan.precision_by_node[step.node]
        if value.compute_dtype == value.accumulation_dtype:
            continue
        node = step.node
        if node.op == "reduce":
            domain = prod(
                node.inputs[0].spec.shape[axis] for axis in node.attrs["axes"]
            )
            total += node.spec.size * domain
        elif node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            reduction = prod(
                size
                for label, size in domains.items()
                if label not in node.attrs["output"]
            )
            total += node.spec.size * reduction
        else:  # pragma: no cover - precision admission owns this invariant
            raise AssertionError(f"unexpected mixed-accumulation op: {node.op}")
    return total


def estimate_schedule(plan: TensorPlan) -> dict:
    """Reuse exact capacity accounting and expose bounded, calibratable cost proxies."""
    live_values, registers = [], 0
    materialized = 0
    for step in plan.steps:
        live = 1 + sum(
            live_values[child] if plan.steps[child].virtual else 1
            for child in step.inputs
        )
        live_values.append(live)
        if not step.virtual and step.node.op not in ("input", "constant"):
            materialized += step.node.spec.size * step.node.spec.itemsize
            estimate = 16 + 2 * live + 2 * len(step.node.spec.shape)
            if step.gemm == "none":
                estimate += 2 * (plan.schedule.elements_per_thread - 1)
                if step.node.op in ("reduce", "einsum"):
                    estimate += plan.schedule.reduction_unroll - 1
                value_precision = plan.precision_by_node[step.node]
                if value_precision.compute_dtype != value_precision.accumulation_dtype:
                    # One wider live accumulator plus conversion temporary.
                    estimate += 2
            elif step.gemm == "packed":
                estimate += 2 * (plan.schedule.staging_width - 1)
            registers = max(registers, estimate)
    source_bytes = len(emit_cuda(plan, embed_static_data=False).encode("utf-8"))
    resident = _resident_blocks(plan, registers, 0)
    traffic = plan.semantic_traffic
    occupancy = resident * plan.schedule.threads / plan.target.maximum_threads_per_sm
    launches = estimated_cuda_launches(plan)
    profitability = GpuProfitability(
        semantic_traffic_bytes=traffic["total_bytes"],
        estimated_registers_per_thread=registers,
        estimated_occupancy_upper_bound=occupancy,
        launch_count=launches,
        source_bytes=source_bytes,
    )
    return {
        "schema": "vibeqc.tensor.cuda.static-cost.v4",
        "peak_numeric_bytes": plan.peak_bytes,
        "device_bytes": plan.device_bytes,
        "host_bytes": plan.host_bytes,
        "panel_bytes": plan.panel_bytes,
        "materialization_bytes": materialized,
        "estimated_logical_traffic_bytes": traffic["logical_tensor_bytes"],
        "estimated_layout_conversion_bytes": traffic["layout_conversion_bytes"],
        "estimated_precision_cast_read_bytes": traffic["precision_cast_read_bytes"],
        "estimated_precision_cast_write_bytes": traffic["precision_cast_write_bytes"],
        "estimated_precision_cast_simultaneous_bytes": traffic[
            "precision_cast_simultaneous_bytes"
        ],
        "precision_schedule_identity": plan.precision_schedule.identity,
        "estimated_host_to_device_bytes": traffic["host_to_device_bytes"],
        "estimated_device_to_host_bytes": traffic["device_to_host_bytes"],
        "estimated_endpoint_semantic_traffic_bytes": traffic["total_bytes"],
        "traffic_scope": traffic["scope"],
        "estimated_flops": plan.estimated_flops,
        "estimated_fp64_accumulation_terms": _fp64_accumulation_terms(plan),
        "estimated_registers_per_thread": registers,
        "estimated_shared_bytes": 0,
        "estimated_local_bytes": None,
        "register_scope": "scalar-liveness/work-per-thread heuristic for generated kernels; excludes cuBLAS",
        "resident_blocks_upper_bound": resident,
        "occupancy_upper_bound": occupancy,
        "estimated_kernel_launches": launches,
        "generated_source_bytes": source_bytes,
        "generated_static_data_bytes": plan.static_data_bytes,
        "profitability": profitability.to_payload(),
        "compile_cost_proxy": "generated_source_bytes calibrated against compiler-reported seconds; immutable static payload is external and is not parsed by NVCC",
    }


@dataclass(frozen=True)
class ScheduleCandidate:
    """One attempted plan, including negative evidence before compilation."""

    requested: TensorSchedule
    plan: TensorPlan | None
    status: str
    stage: str
    reason: str | None = None
    estimates: dict | None = None
    equivalent_to: str | None = None
    precision_schedule: dict | None = None

    def to_payload(self) -> typing.Any:
        row = {
            "requested_schedule": asdict(self.requested),
            "status": self.status,
            "stage": self.stage,
        }
        if self.plan is not None:
            row.update(plan=self.plan.to_payload(), plan_identity=self.plan.identity)
        if self.reason is not None:
            row["reason"] = self.reason
        if self.estimates is not None:
            row["static_resources"] = self.estimates
        if self.equivalent_to is not None:
            row["equivalent_to"] = self.equivalent_to
        if self.precision_schedule is not None:
            row["precision_schedule"] = self.precision_schedule
        return row


def _program_abi(program: Program) -> tuple:
    inputs = {
        node.attrs["name"]: node.spec
        for node in program.live_nodes
        if node.op == "input"
    }
    outputs = {name: node.spec for name, node in program.outputs.items()}
    return tuple(sorted(inputs.items())), tuple(sorted(outputs.items()))


def plan_schedule_search(
    baseline: typing.Any,
    schedules: typing.Any,
    limits: typing.Any = DEFAULT_SEARCH_LIMITS,
    *,
    precision_programs: typing.Any = None,
) -> typing.Any:
    """Plan, deduplicate and statically prune without compiling or allocating."""
    if not isinstance(limits, TensorSearchLimits):
        raise TypeError("limits must be TensorSearchLimits")
    schedules = tuple(islice(schedules, limits.maximum_candidates + 1))
    programs = (
        (baseline.program,)
        if precision_programs is None
        else tuple(islice(precision_programs, limits.maximum_candidates + 1))
    )
    if (
        not schedules
        or not programs
        or len(schedules) * len(programs) > limits.maximum_candidates
    ):
        raise ValueError(
            "schedule/precision product exceeds the candidate limit or is empty"
        )
    if any(not isinstance(s, TensorSchedule) for s in schedules):
        raise TypeError("search requires TensorSchedule candidates")
    if any(not isinstance(program, Program) for program in programs):
        raise TypeError("precision variants must be TensorIR Programs")
    baseline_abi = _program_abi(baseline.program)
    resolved = []
    for program in programs:
        if _program_abi(program) != baseline_abi:
            raise ValueError(
                "precision variants must preserve the baseline input/output ABI"
            )
        precision = describe_precision(program)
        if precision.source_equation != baseline.program.logical_hash:
            raise ValueError(
                "precision variant must retain the baseline scientific equation identity"
            )
        resolved.append((program, precision.to_payload()))

    seen = {execution_key(baseline): baseline.identity}
    candidates = []
    for program, precision in resolved:
        for requested in schedules:
            try:
                plan = plan_cuda(
                    program,
                    baseline.target,
                    max_bytes=baseline.max_bytes,
                    schedule=requested,
                    reservations=baseline.reservations,
                    library_bytes=baseline.library_bytes,
                    provider_bytes=baseline.provider_bytes,
                )
            except ValueError as error:
                candidates.append(
                    ScheduleCandidate(
                        requested,
                        None,
                        "pruned",
                        "legality",
                        str(error),
                        precision_schedule=precision,
                    )
                )
                continue
            key = execution_key(plan)
            if key in seen:
                candidates.append(
                    ScheduleCandidate(
                        requested,
                        plan,
                        "pruned",
                        "duplicate",
                        "equivalent executable plan",
                        equivalent_to=seen[key],
                        precision_schedule=precision,
                    )
                )
                continue
            seen[key] = plan.identity
            try:
                estimates = estimate_schedule(plan)
            except ValueError as error:
                candidates.append(
                    ScheduleCandidate(
                        requested,
                        plan,
                        "pruned",
                        "legality",
                        str(error),
                        precision_schedule=precision,
                    )
                )
                continue
            reasons = []
            if estimates["generated_source_bytes"] > limits.maximum_source_bytes:
                reasons.append("generated source exceeds compile-cost budget")
            if estimates["estimated_registers_per_thread"] > min(
                plan.target.tuning_maximum_registers,
                plan.target.maximum_registers_per_thread,
            ):
                reasons.append("estimated register pressure exceeds target policy")
            if (
                estimates["resident_blocks_upper_bound"]
                < limits.minimum_resident_blocks
            ):
                reasons.append(
                    "estimated occupancy cannot satisfy resident-block policy"
                )
            candidates.append(
                ScheduleCandidate(
                    requested,
                    plan,
                    "pruned" if reasons else "ready",
                    "static-resource",
                    "; ".join(reasons) if reasons else None,
                    estimates,
                    precision_schedule=precision,
                )
            )
    return tuple(candidates)


def compiled_resource_calibration(
    plan: typing.Any, estimates: typing.Any, resources: typing.Any
) -> dict:
    """Compare static heuristics with compiler-reported resources.

    This record calibrates the human-readable cost model; promotion still uses
    the hard PTXAS gate below and endpoint evidence rather than this ratio.
    """
    if not isinstance(estimates, dict):
        raise TypeError("resource calibration requires static estimates")
    if not resources:
        raise ValueError("resource calibration requires compiled resources")
    required = (
        "registers",
        "stack_bytes",
        "spill_store_bytes",
        "spill_load_bytes",
        "shared_bytes",
    )
    if any(
        not isinstance(row, dict)
        or any(type(row.get(key)) is not int or row[key] < 0 for key in required)
        for row in resources
    ):
        raise ValueError("resource calibration requires complete compiled resources")
    registers = max(row["registers"] for row in resources)
    estimated = estimates["estimated_registers_per_thread"]
    local = [
        row.get("local_bytes")
        for row in resources
        if type(row.get("local_bytes")) is int and row["local_bytes"] >= 0
    ]
    resident = min(
        _resident_blocks(plan, row["registers"], row["shared_bytes"])
        for row in resources
    )
    return {
        "schema": "vibeqc.tensor.cuda.resource-calibration.v1",
        "kernel_count": len(resources),
        "estimated_registers_per_thread": estimated,
        "compiled_max_registers_per_thread": registers,
        "register_calibration_ratio": None if estimated == 0 else registers / estimated,
        "compiled_max_stack_bytes": max(row["stack_bytes"] for row in resources),
        "compiled_max_spill_store_bytes": max(
            row["spill_store_bytes"] for row in resources
        ),
        "compiled_max_spill_load_bytes": max(
            row["spill_load_bytes"] for row in resources
        ),
        "compiled_max_shared_bytes": max(row["shared_bytes"] for row in resources),
        "compiled_max_local_bytes": max(local) if local else None,
        "compiled_resident_blocks_upper_bound": resident,
        "compiled_occupancy_upper_bound": (
            resident * plan.schedule.threads / plan.target.maximum_threads_per_sm
        ),
        "local_memory_scope": "PTXAS lmem when reported; stack/spill bytes are retained separately and never inferred as lmem",
    }


def require_compiled_resources(
    plan: typing.Any, resources: typing.Any, *, minimum_resident_blocks: typing.Any = 1
) -> None:
    """Fail closed on missing PTXAS data, spills or infeasible block resources."""
    required = (
        "registers",
        "stack_bytes",
        "spill_store_bytes",
        "spill_load_bytes",
        "shared_bytes",
    )
    if not resources:
        raise ValueError("candidate lacks compiled resource evidence")
    for row in resources:
        if (
            not isinstance(row, dict)
            or any(type(row.get(key)) is not int or row[key] < 0 for key in required)
            or (
                row.get("local_bytes") is not None
                and (type(row["local_bytes"]) is not int or row["local_bytes"] < 0)
            )
        ):
            raise ValueError("candidate has incomplete compiled resource evidence")
        target = plan.target
        if (
            row["registers"]
            > min(target.tuning_maximum_registers, target.maximum_registers_per_thread)
            or row["stack_bytes"] > target.tuning_maximum_stack_bytes
            or row["spill_store_bytes"]
            or row["spill_load_bytes"]
            or row["shared_bytes"]
            > min(target.tuning_maximum_shared_bytes, target.shared_memory_per_block)
            or _resident_blocks(plan, row["registers"], row["shared_bytes"])
            < minimum_resident_blocks
        ):
            raise ValueError("candidate fails the compiled resource gate")
