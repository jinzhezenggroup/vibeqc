"""GPU-free CUDA resource screening for compiler-generated candidates.

The model is intentionally an optimistic resource/parallelism bound, not a
runtime simulator.  It consumes architecture limits plus compiler-visible cost
facts and exposes deterministic screening evidence that can be used before a
real-device benchmark.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .cuda_target import CudaTargetInfo
from .gpu_profitability import GpuProfitability


def _optional_count(value: int | None, name: str, *, positive: bool = False) -> None:
    if value is None:
        return
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer or None")


def _minimize(value: float | None) -> tuple[bool, float]:
    return value is None, 0.0 if value is None else float(value)


def _maximize(value: float | None) -> tuple[bool, float]:
    return value is None, 0.0 if value is None else -float(value)


@dataclass(frozen=True, slots=True)
class StaticCudaCost:
    """One GPU-free screening report for a legal CUDA launch shape.

    ``occupancy_upper_bound`` is theoretical occupancy from the known resource
    constraints.  Unknown constraints are omitted, so the value is deliberately
    optimistic. ``grid_saturation_upper_bound`` reports the fraction of one
    resident wave supplied by the grid. ``device_occupancy_upper_bound`` combines
    that fraction with per-SM occupancy when grid size and SM count are known.
    ``spill_bytes`` retains static PTXAS spill evidence; it is not total dynamic
    memory traffic across threads, loop iterations, or repeated launches.
    """

    architecture: str
    block_threads: int
    evidence_stage: str
    registers_per_thread: int | None
    shared_bytes_per_block: int | None
    spill_bytes: int | None
    resident_blocks_per_sm_upper_bound: int
    occupancy_upper_bound: float
    grid_blocks: int | None
    sm_count: int | None
    grid_saturation_upper_bound: float | None
    device_occupancy_upper_bound: float | None
    semantic_traffic_bytes: int | None
    arithmetic_operation_count: int | None
    arithmetic_intensity_ops_per_byte: float | None
    launch_count: int | None
    source_bytes: int | None
    limiting_resources: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def screening_priority(self, generation_index: int) -> tuple[object, ...]:
        """Return a deterministic shortlist key, never a predicted runtime.

        Compare candidates from the same evidence stage. Missing evidence sorts
        behind known evidence, matching :class:`GpuProfitability` conventions.
        """

        _optional_count(generation_index, "generation_index")
        return (
            self.resident_blocks_per_sm_upper_bound == 0,
            _minimize(self.spill_bytes),
            _maximize(self.device_occupancy_upper_bound),
            _maximize(self.occupancy_upper_bound),
            _minimize(self.semantic_traffic_bytes),
            _minimize(self.registers_per_thread),
            _minimize(self.shared_bytes_per_block),
            _minimize(self.launch_count),
            _minimize(self.arithmetic_operation_count),
            _minimize(self.source_bytes),
            generation_index,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize derived facts with an explicit non-timing scope."""

        return {
            "schema": "generativeqc.compiler.cuda-static-cost.v1",
            "scope": (
                "screening only; theoretical resource/parallelism bounds, "
                "not a runtime or speedup prediction"
            ),
            **asdict(self),
        }


def static_cuda_cost(
    profitability: GpuProfitability,
    target: CudaTargetInfo,
    block_threads: int,
    *,
    grid_blocks: int | None = None,
    sm_count: int | None = None,
) -> StaticCudaCost:
    """Build a GPU-free CUDA screening report from available compiler evidence.

    Compiled PTXAS resource facts take precedence over pre-compilation register
    estimates.  The function never probes CUDA. A caller that knows the intended
    GPU SKU may pass ``sm_count`` explicitly; otherwise a runtime-enriched target
    is used when available and global-grid saturation stays unknown.
    """

    if not isinstance(profitability, GpuProfitability):
        raise TypeError("static CUDA cost requires GpuProfitability")
    if not isinstance(target, CudaTargetInfo):
        raise TypeError("static CUDA cost requires CudaTargetInfo")
    if (
        type(block_threads) is not int
        or not 0 < block_threads <= target.maximum_threads_per_block
    ):
        raise ValueError("block_threads must fit the CUDA target")
    _optional_count(grid_blocks, "grid_blocks")
    _optional_count(sm_count, "sm_count", positive=True)

    compiled = (
        profitability.compiled_registers_per_thread is not None
        or profitability.shared_bytes is not None
    )
    registers = (
        profitability.compiled_registers_per_thread
        if profitability.compiled_registers_per_thread is not None
        else profitability.estimated_registers_per_thread
    )
    shared = profitability.shared_bytes if compiled else None

    limits: dict[str, int] = {
        "resident-block-cap": target.maximum_blocks_per_sm,
        "threads": target.maximum_threads_per_sm // block_threads,
    }
    diagnostics: list[str] = []

    if registers is not None:
        if registers > target.maximum_registers_per_thread:
            limits["registers-per-thread"] = 0
        elif registers:
            limits["registers"] = target.registers_per_sm // (registers * block_threads)
    else:
        diagnostics.append(
            "register pressure is unknown and omitted from the occupancy bound"
        )

    if shared is not None:
        if shared > target.shared_memory_per_block_optin:
            limits["shared-memory-per-block"] = 0
        elif shared:
            limits["shared-memory"] = target.shared_memory_per_sm // shared
    else:
        diagnostics.append(
            "shared-memory pressure is unknown and omitted from the occupancy bound"
        )

    if not compiled and profitability.estimated_occupancy_upper_bound is not None:
        estimated_resident = math.floor(
            profitability.estimated_occupancy_upper_bound
            * target.maximum_threads_per_sm
            / block_threads
            + 1.0e-12
        )
        limits["caller-estimated-occupancy"] = max(0, estimated_resident)

    resident = max(0, min(limits.values()))
    if resident == 0:
        diagnostics.append("known resource limits admit no resident block")
    limiting = tuple(
        sorted(name for name, value in limits.items() if value == resident)
    )
    occupancy = min(
        1.0,
        resident * block_threads / target.maximum_threads_per_sm,
    )

    effective_sm_count = target.sm_count if sm_count is None else sm_count
    saturation = None
    device_occupancy = None
    if grid_blocks is not None:
        if effective_sm_count is None:
            diagnostics.append("grid saturation unavailable without a device SM count")
        elif resident == 0:
            saturation = 0.0
            device_occupancy = 0.0
        else:
            one_wave = resident * effective_sm_count
            saturation = min(1.0, grid_blocks / one_wave)
            # Use integer work/capacity directly: reducing residency must not
            # improve priority merely by shrinking the wave's denominator.
            device_occupancy = (
                min(grid_blocks, one_wave)
                * block_threads
                / (effective_sm_count * target.maximum_threads_per_sm)
            )
            if grid_blocks < one_wave:
                diagnostics.append("grid exposes fewer blocks than one resident wave")

    traffic = profitability.semantic_traffic_bytes
    operations = profitability.arithmetic_operation_count
    intensity = None
    if traffic is not None and operations is not None:
        if traffic:
            intensity = operations / traffic
        else:
            diagnostics.append(
                "arithmetic intensity is undefined because semantic traffic is zero"
            )

    spill_bytes = profitability.spill_bytes
    if spill_bytes:
        diagnostics.append("compiled spill traffic is non-zero")
    if not compiled:
        diagnostics.append(
            "resource occupancy uses static estimates; PTXAS evidence is unavailable"
        )

    return StaticCudaCost(
        architecture=target.architecture,
        block_threads=block_threads,
        evidence_stage="compiled" if compiled else "static",
        registers_per_thread=registers,
        shared_bytes_per_block=shared,
        spill_bytes=spill_bytes,
        resident_blocks_per_sm_upper_bound=resident,
        occupancy_upper_bound=occupancy,
        grid_blocks=grid_blocks,
        sm_count=effective_sm_count,
        grid_saturation_upper_bound=saturation,
        device_occupancy_upper_bound=device_occupancy,
        semantic_traffic_bytes=traffic,
        arithmetic_operation_count=operations,
        arithmetic_intensity_ops_per_byte=intensity,
        launch_count=profitability.launch_count,
        source_bytes=profitability.source_bytes,
        limiting_resources=limiting,
        diagnostics=tuple(diagnostics),
    )
