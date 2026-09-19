"""Bounded TensorIR schedule search, with no compiler or device side effects.

Only dimensions implemented by the ordinary-stream emitter are searched. GEMM
panels are global allocations, not shared-memory tiles. Static register pressure
and occupancy are explicitly heuristics for generated kernels, never cuBLAS
resource measurements or grounds for performance promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from itertools import combinations, islice, product
from math import prod

from vibeqc_compiler.common.provenance import canonical_hash

from .cuda_emit import emit_cuda
from .cuda_gemm import gemm_contract
from .cuda_plan import TensorPlan, TensorSchedule, plan_cuda


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
    tile_m: tuple[int, ...] = (128, 64, 32)
    tile_n: tuple[int, ...] = (128, 64, 32)
    tile_k: tuple[int, ...] = (128, 64, 32)

    def __post_init__(self):
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

        def walk():
            yield TensorSchedule(**anchor)
            for radius in range(1, len(names) + 1):
                for changed in combinations(names, radius):
                    for values in product(*(axes[name][1:] for name in changed)):
                        yield TensorSchedule(**(anchor | dict(zip(changed, values))))

        return tuple(islice(walk(), maximum))


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class TensorSearchLimits:
    """Independent generation, compilation and static-source budgets.

    The existing tuner wall-clock deadline and compiler process timeout remain
    separate hard stop mechanisms. The mandatory baseline is not a candidate.
    """

    maximum_candidates: int = 128
    maximum_compilations: int = 12
    maximum_source_bytes: int = 2 * 1024**2
    minimum_resident_blocks: int = 1

    def __post_init__(self):
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

    def __post_init__(self):
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
    payload["threads"] = (
        plan.schedule.threads
        if any(
            not step.virtual
            and step.node.op not in ("input", "constant")
            and step.node.spec.size
            for step in plan.steps
        )
        else None
    )
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


def _resident_blocks(plan, registers, shared_bytes):
    target, threads = plan.target, plan.schedule.threads
    limits = [target.maximum_blocks_per_sm, target.maximum_threads_per_sm // threads]
    if registers:
        limits.append(target.registers_per_sm // (registers * threads))
    if shared_bytes:
        limits.append(target.shared_memory_per_sm // shared_bytes)
    # Allocation granularity is deliberately not modeled: this is an upper
    # bound, not a claim of achievable occupancy or measured SM utilization.
    return min(limits)


def estimate_schedule(plan: TensorPlan) -> dict:
    """Reuse exact numeric-buffer accounting and expose labeled cost proxies."""
    live_values, registers = [], 0
    materialized = 0
    for step in plan.steps:
        live = 1 + sum(
            live_values[child] if plan.steps[child].virtual else 1
            for child in step.inputs
        )
        live_values.append(live)
        if not step.virtual and step.node.op not in ("input", "constant"):
            materialized += step.node.spec.size * 8
            registers = max(registers, 16 + 2 * live + 2 * len(step.node.spec.shape))
    source_bytes = len(emit_cuda(plan).encode("utf-8"))
    resident = _resident_blocks(plan, registers, 0)
    return {
        "schema": "vibeqc.tensor.cuda.static-cost.v1",
        "peak_numeric_bytes": plan.peak_bytes,
        "device_bytes": plan.device_bytes,
        "host_bytes": plan.host_bytes,
        "panel_bytes": plan.panel_bytes,
        "materialization_bytes": materialized,
        "estimated_logical_traffic_bytes": plan.estimated_traffic_bytes,
        "traffic_scope": "planner logical traffic; excludes packing/provider/cache traffic",
        "estimated_flops": plan.estimated_flops,
        "estimated_registers_per_thread": registers,
        "estimated_shared_bytes": 0,
        "estimated_local_bytes": None,
        "register_scope": "scalar-liveness heuristic for generated kernels; excludes cuBLAS",
        "resident_blocks_upper_bound": resident,
        "occupancy_upper_bound": resident
        * plan.schedule.threads
        / plan.target.maximum_threads_per_sm,
        "generated_source_bytes": source_bytes,
        "compile_cost_proxy": "generated_source_bytes; not predicted seconds",
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

    def to_payload(self):
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
        return row


def plan_schedule_search(baseline, schedules, limits=DEFAULT_SEARCH_LIMITS):
    """Plan, deduplicate and statically prune without compiling or allocating."""
    if not isinstance(limits, TensorSearchLimits):
        raise TypeError("limits must be TensorSearchLimits")
    schedules = tuple(islice(schedules, limits.maximum_candidates + 1))
    if not 1 <= len(schedules) <= limits.maximum_candidates:
        raise ValueError("schedule count exceeds the candidate limit or is empty")
    if any(not isinstance(s, TensorSchedule) for s in schedules):
        raise TypeError("search requires TensorSchedule candidates")
    seen = {execution_key(baseline): baseline.identity}
    candidates = []
    for requested in schedules:
        try:
            plan = plan_cuda(
                baseline.program,
                baseline.target,
                max_bytes=baseline.max_bytes,
                schedule=requested,
                reservations=baseline.reservations,
                library_bytes=baseline.library_bytes,
                provider_bytes=baseline.provider_bytes,
            )
        except ValueError as error:
            candidates.append(
                ScheduleCandidate(requested, None, "pruned", "legality", str(error))
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
                )
            )
            continue
        seen[key] = plan.identity
        try:
            estimates = estimate_schedule(plan)
        except ValueError as error:
            candidates.append(
                ScheduleCandidate(requested, plan, "pruned", "legality", str(error))
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
        if estimates["resident_blocks_upper_bound"] < limits.minimum_resident_blocks:
            reasons.append("estimated occupancy cannot satisfy resident-block policy")
        candidates.append(
            ScheduleCandidate(
                requested,
                plan,
                "pruned" if reasons else "ready",
                "static-resource",
                "; ".join(reasons) if reasons else None,
                estimates,
            )
        )
    return tuple(candidates)


def require_compiled_resources(plan, resources, *, minimum_resident_blocks=1):
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
        if not isinstance(row, dict) or any(
            type(row.get(key)) is not int or row[key] < 0 for key in required
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
