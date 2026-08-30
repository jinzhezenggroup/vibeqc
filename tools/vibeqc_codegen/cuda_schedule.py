"""CUDA target schedule IR and target-derived candidate enumeration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import comb

from .backend import TargetScheduleShape
from .cuda_target import DEFAULT_CUDA_TARGET, CudaTargetInfo
from .expr import (
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    RematerializationPolicy,
)
from .ir import IntegralIR, KernelConsumer


class ScheduleKind(str, Enum):
    """Supported CUDA mappings for shell work."""

    PACKED_TASKS = "packed_tasks"
    THREAD_TASKS = "thread_tasks"
    SUBGROUP_TASKS = "subgroup_tasks"
    SHELL_TASK = "shell_task"
    COMPONENT_LANES = "component_lanes"
    TILED_COMPONENTS = "tiled_components"


class PairOrientation(str, Enum):
    """Compile-time Gaussian-pair orientation used by the recurrence."""

    CANONICAL = "canonical"
    SWAPPED = "swapped"


class PairStorage(str, Enum):
    """Whether one Gaussian-pair term table is cached or recomputed."""

    MATERIALIZED = "materialized"
    RECOMPUTED = "recomputed"


class AlgebraPlacement(str, Enum):
    """Source-level scalar CSE and rematerialization strategy."""

    MATERIALIZED_CSE = "materialized_cse"
    INLINE_SINGLE_USE = "inline_single_use"
    PRESSURE_REMATERIALIZED = "pressure_rematerialized"

    def materialization_policy(self) -> RematerializationPolicy:
        """Return the expression-layer cost model for this schedule choice."""

        if self == AlgebraPlacement.MATERIALIZED_CSE:
            return RematerializationPolicy.materialized_cse()
        if self == AlgebraPlacement.INLINE_SINGLE_USE:
            return RematerializationPolicy.inline_single_use_values()
        return RematerializationPolicy.pressure_rematerialized()


@dataclass(frozen=True, slots=True)
class CudaScheduleIR:
    """CUDA execution policy validated independently of integral intent."""

    kind: ScheduleKind
    block_threads: int
    component_tile: int
    tasks_per_warp: int = 1
    shared_coulomb: bool = True
    pair_orientation: PairOrientation = PairOrientation.CANONICAL
    pair_storage: PairStorage = PairStorage.MATERIALIZED
    algebra_placement: AlgebraPlacement = AlgebraPlacement.MATERIALIZED_CSE
    algebra_ordering: AlgebraOrdering = AlgebraOrdering.TOPOLOGICAL
    algebra_fusion: AlgebraFusion = AlgebraFusion.SEPARATE
    algebra_form: AlgebraForm = AlgebraForm.BINARY
    unroll_pair_terms: bool = True
    minimum_blocks_per_sm: int = 0
    maximum_registers: int = 0
    warp_size: int = 32

    def __post_init__(self) -> None:
        if self.warp_size < 1:
            raise ValueError("CUDA warp size must be positive")
        if self.block_threads < self.warp_size or self.block_threads > 1024:
            raise ValueError("CUDA block size must be between one warp and 1024")
        if self.block_threads % self.warp_size != 0:
            raise ValueError("CUDA block size must contain complete warps")
        if self.component_tile < 1:
            raise ValueError("component tile must be positive")
        if not 1 <= self.tasks_per_warp <= self.warp_size:
            raise ValueError("tasks per warp must fit the target warp")
        if not 0 <= self.minimum_blocks_per_sm <= 32:
            raise ValueError("minimum blocks per SM must be between zero and 32")
        if self.maximum_registers != 0 and not 32 <= self.maximum_registers <= 255:
            raise ValueError("maximum registers must be zero or between 32 and 255")
        if self.minimum_blocks_per_sm and self.maximum_registers:
            raise ValueError("launch bounds and maximum registers are mutually exclusive")
        if self.maximum_registers and self.kind != ScheduleKind.PACKED_TASKS:
            raise ValueError("maximum-register lowering currently supports packed tasks")
        if (
            self.algebra_placement != AlgebraPlacement.MATERIALIZED_CSE
            and self.kind != ScheduleKind.PACKED_TASKS
        ):
            raise ValueError(
                "non-baseline algebra placement currently supports packed tasks"
            )
        if (
            self.algebra_ordering != AlgebraOrdering.TOPOLOGICAL
            and self.kind != ScheduleKind.PACKED_TASKS
        ):
            raise ValueError(
                "non-baseline algebra ordering currently supports packed tasks"
            )
        if (
            self.algebra_fusion != AlgebraFusion.SEPARATE
            and self.kind != ScheduleKind.PACKED_TASKS
        ):
            raise ValueError(
                "non-baseline algebra fusion currently supports packed tasks"
            )
        if (
            self.algebra_form != AlgebraForm.BINARY
            and self.kind != ScheduleKind.PACKED_TASKS
        ):
            raise ValueError(
                "non-baseline algebra form currently supports packed tasks"
            )
        if self.kind == ScheduleKind.PACKED_TASKS:
            if self.block_threads != self.warp_size:
                raise ValueError("packed-task schedules currently use one warp")
            if self.tasks_per_warp == 1:
                raise ValueError("packed-task schedules require multiple tasks")
        elif self.kind == ScheduleKind.THREAD_TASKS:
            if self.tasks_per_warp != self.warp_size:
                raise ValueError("thread-task schedules require one task per lane")
            if self.shared_coulomb:
                raise ValueError("thread-task schedules require lane-local Coulomb data")
        elif self.kind == ScheduleKind.SUBGROUP_TASKS:
            if self.tasks_per_warp not in (1, 2, 4, 8):
                raise ValueError(
                    "subgroup-task schedules require 1, 2, 4, or 8 tasks"
                )
            if self.warp_size % self.tasks_per_warp != 0:
                raise ValueError("subgroup tasks must divide the target warp")
            if not self.shared_coulomb:
                raise ValueError("subgroup tasks require task-local shared Coulomb data")
        elif self.tasks_per_warp != 1:
            raise ValueError("only packed, thread, or subgroup schedules own many tasks")

    def validate_for(self, target: CudaTargetInfo) -> None:
        """Validate block, occupancy, register, and shared target limits."""

        TargetScheduleShape(self.block_threads, self.warp_size).validate_for(
            target.target_info
        )
        if self.minimum_blocks_per_sm > target.maximum_blocks_per_sm:
            raise ValueError("launch bounds exceed the target resident-block limit")
        if self.maximum_registers > target.maximum_registers_per_thread:
            raise ValueError("register cap exceeds the target per-thread limit")

    @property
    def warp_count(self) -> int:
        return self.block_threads // self.warp_size

    @property
    def subgroup_lanes(self) -> int:
        if self.kind != ScheduleKind.SUBGROUP_TASKS:
            raise ValueError("only subgroup-task schedules define subgroup lanes")
        return self.warp_size // self.tasks_per_warp

    @property
    def tasks_per_block(self) -> int:
        if self.kind == ScheduleKind.PACKED_TASKS:
            return self.warp_size
        if self.kind == ScheduleKind.THREAD_TASKS:
            return self.block_threads
        if self.kind == ScheduleKind.SUBGROUP_TASKS:
            return self.warp_count * self.tasks_per_warp
        return 1


# Keep the established public name while making the backend boundary explicit.
ScheduleIR = CudaScheduleIR


@dataclass(frozen=True, slots=True)
class CudaKernelIR:
    """CUDA lowering input: mathematical intent plus target schedule."""

    integral: IntegralIR
    schedule: CudaScheduleIR
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET

    def __post_init__(self) -> None:
        if (
            KernelConsumer.FORCE in self.integral.consumers
            and self.integral.derivative is not None
            and self.integral.derivative.order != 1
        ):
            # Keep higher-order mathematical intent representable in
            # IntegralIR, but never let a first-gradient CUDA ABI consume it
            # as if it were an atomic force result.
            raise ValueError(
                "CUDA force result ABI currently exposes only order-one "
                "derivatives"
            )
        self.schedule.validate_for(self.target)
        component_count = self.integral.spec.component_count
        if (
            self.schedule.kind != ScheduleKind.TILED_COMPONENTS
            and self.schedule.component_tile < component_count
        ):
            raise ValueError("non-tiled schedules must cover every shell component")


KernelIR = CudaKernelIR


def schedule_candidates(
    integral: IntegralIR,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
) -> tuple[CudaScheduleIR, ...]:
    """Enumerate legal CUDA schedules from explicit target capabilities."""

    component_count = integral.spec.component_count
    warp_size = target.warp_size
    candidates: list[CudaScheduleIR] = []

    if component_count <= 9:
        candidates.append(
            CudaScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=warp_size,
                component_tile=component_count,
                tasks_per_warp=warp_size,
                shared_coulomb=False,
                warp_size=warp_size,
            )
        )
        candidates.append(
            CudaScheduleIR(
                kind=ScheduleKind.SHELL_TASK,
                block_threads=warp_size,
                component_tile=component_count,
                shared_coulomb=False,
                warp_size=warp_size,
            )
        )

    subgroup_block_threads = min(
        target.maximum_threads_per_block,
        target.maximum_threads_per_sm,
        warp_size * 8,
    )
    subgroup_block_threads -= subgroup_block_threads % warp_size
    if component_count <= 64 and subgroup_block_threads >= warp_size:
        for tasks_per_warp in (2, 4):
            if warp_size % tasks_per_warp == 0:
                candidates.append(
                    CudaScheduleIR(
                        kind=ScheduleKind.SUBGROUP_TASKS,
                        block_threads=subgroup_block_threads,
                        component_tile=component_count,
                        tasks_per_warp=tasks_per_warp,
                        shared_coulomb=True,
                        warp_size=warp_size,
                    )
                )

    coulomb_state_count = comb(integral.maximum_coulomb_order + 3, 3)
    cooperative_threads = (
        (max(component_count, coulomb_state_count, 12) + warp_size - 1)
        // warp_size
        * warp_size
    )
    if cooperative_threads <= target.maximum_threads_per_block:
        candidates.append(
            CudaScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=cooperative_threads,
                component_tile=component_count,
                warp_size=warp_size,
            )
        )
        if component_count <= warp_size and cooperative_threads > warp_size:
            candidates.append(
                CudaScheduleIR(
                    kind=ScheduleKind.COMPONENT_LANES,
                    block_threads=warp_size,
                    component_tile=component_count,
                    warp_size=warp_size,
                )
            )

    for tile in (warp_size * 2, warp_size * 4, warp_size * 8):
        if component_count > tile and tile <= target.maximum_threads_per_block:
            candidates.append(
                CudaScheduleIR(
                    kind=ScheduleKind.TILED_COMPONENTS,
                    block_threads=tile,
                    component_tile=tile,
                    warp_size=warp_size,
                )
            )
    for candidate in candidates:
        candidate.validate_for(target)
    return tuple(candidates)


def default_schedule(
    integral: IntegralIR,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
) -> CudaScheduleIR:
    """Return the conservative component schedule for ``target``."""

    candidates = schedule_candidates(integral, target)
    for candidate in candidates:
        if candidate.kind == ScheduleKind.COMPONENT_LANES:
            return candidate
    for candidate in candidates:
        if candidate.kind == ScheduleKind.TILED_COMPONENTS:
            return candidate
    raise ValueError(f"{integral.spec.name} has no schedule legal on {target.architecture}")


def tuning_schedule_candidates(
    integral: IntegralIR,
    target: CudaTargetInfo = DEFAULT_CUDA_TARGET,
) -> tuple[CudaScheduleIR, ...]:
    """Expand target-legal schedule families into a bounded tuning search."""

    candidates = []
    for schedule in schedule_candidates(integral, target):
        if schedule.kind in (ScheduleKind.COMPONENT_LANES, ScheduleKind.TILED_COMPONENTS):
            shared_options = (
                (True, False)
                if schedule.kind == ScheduleKind.COMPONENT_LANES
                else (True,)
            )
            pair_storage_options = (
                tuple(PairStorage)
                if schedule.kind == ScheduleKind.TILED_COMPONENTS
                else (schedule.pair_storage,)
            )
            for pair_storage in pair_storage_options:
                for pair_orientation in PairOrientation:
                    for shared_coulomb in shared_options:
                        for unroll_pair_terms in (True, False):
                            candidates.append(
                                replace(
                                    schedule,
                                    shared_coulomb=shared_coulomb,
                                    unroll_pair_terms=unroll_pair_terms,
                                    pair_orientation=pair_orientation,
                                    pair_storage=pair_storage,
                                )
                            )
        elif schedule.kind == ScheduleKind.PACKED_TASKS:
            occupancy_candidates = tuple(
                dict.fromkeys((2, min(12, target.maximum_blocks_per_sm)))
            )
            for minimum_blocks_per_sm in occupancy_candidates:
                for unroll_pair_terms in (True, False):
                    for algebra_placement in AlgebraPlacement:
                        for algebra_ordering in AlgebraOrdering:
                            for algebra_fusion in AlgebraFusion:
                                for algebra_form in AlgebraForm:
                                    candidates.append(
                                        replace(
                                            schedule,
                                            algebra_placement=algebra_placement,
                                            algebra_ordering=algebra_ordering,
                                            algebra_fusion=algebra_fusion,
                                            algebra_form=algebra_form,
                                            unroll_pair_terms=unroll_pair_terms,
                                            minimum_blocks_per_sm=(
                                                minimum_blocks_per_sm
                                            ),
                                        )
                                    )
            for maximum_registers in target.packed_register_caps:
                if maximum_registers > target.maximum_registers_per_thread:
                    continue
                for unroll_pair_terms in (True, False):
                    for algebra_placement in AlgebraPlacement:
                        for algebra_ordering in AlgebraOrdering:
                            for algebra_fusion in AlgebraFusion:
                                for algebra_form in AlgebraForm:
                                    candidates.append(
                                        replace(
                                            schedule,
                                            algebra_placement=algebra_placement,
                                            algebra_ordering=algebra_ordering,
                                            algebra_fusion=algebra_fusion,
                                            algebra_form=algebra_form,
                                            unroll_pair_terms=unroll_pair_terms,
                                            maximum_registers=maximum_registers,
                                        )
                                    )
        elif schedule.kind == ScheduleKind.SUBGROUP_TASKS:
            for pair_orientation in PairOrientation:
                for unroll_pair_terms in (True, False):
                    candidates.append(
                        replace(
                            schedule,
                            pair_orientation=pair_orientation,
                            unroll_pair_terms=unroll_pair_terms,
                        )
                    )
        else:
            candidates.append(schedule)
    for candidate in candidates:
        candidate.validate_for(target)
    return tuple(candidates)
