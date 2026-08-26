"""Backend-independent integral and execution-schedule IR.

The code generator deliberately separates mathematical intent from CUDA
mapping.  ``IntegralIR`` describes which shell integral observables are
required, while ``ScheduleIR`` describes how complete shell tasks and their
Cartesian components are assigned to CUDA lanes.  CUDA emitters consume the
combined ``KernelIR`` instead of inferring policy from a shell-class name.

Keeping these layers explicit is important for two reasons:

* Fock values and analytic forces share the same primitive recurrence and can
  therefore be emitted from one mathematical definition.
* several legal CUDA schedules can implement the same integral.  Resource and
  timing gates may select among them without changing the correctness oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import comb

from .shell_spec import ShellClassSpec


class KernelConsumer(str, Enum):
    """Observable contracted by a generated shell kernel."""

    FOCK = "fock"
    FORCE = "force"


class ScheduleKind(str, Enum):
    """Supported families for mapping shell work onto CUDA lanes."""

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


@dataclass(frozen=True, slots=True)
class IntegralIR:
    """Mathematical definition shared by Fock and force CUDA consumers.

    The fourth force center is intentionally absent from
    ``independent_force_centers``: its derivative is reconstructed from exact
    translation invariance.  This is both a scientific invariant and an
    important register-pressure reduction in the generated kernels.
    """

    spec: ShellClassSpec
    consumers: frozenset[KernelConsumer]
    independent_force_centers: tuple[int, ...] = (0, 1, 2)
    recurrence: str = "subset_wick"

    def __post_init__(self) -> None:
        if not self.consumers:
            raise ValueError("an integral IR requires at least one consumer")
        if not self.consumers <= frozenset(KernelConsumer):
            raise ValueError("integral IR contains an unsupported consumer")
        if self.recurrence != "subset_wick":
            raise ValueError(f"unsupported integral recurrence {self.recurrence!r}")
        if (
            KernelConsumer.FORCE in self.consumers
            and self.independent_force_centers != (0, 1, 2)
        ):
            raise ValueError(
                "first-force lowering requires centers 0/1/2 and "
                "translation recovery for center 3"
            )

    @property
    def value_coulomb_order(self) -> int:
        """Largest Cartesian Coulomb derivative needed for ERI values."""

        return sum(self.spec.angular)

    @property
    def maximum_coulomb_order(self) -> int:
        """Largest derivative required by every requested consumer."""

        force_increment = int(KernelConsumer.FORCE in self.consumers)
        return self.value_coulomb_order + force_increment


@dataclass(frozen=True, slots=True)
class ScheduleIR:
    """CUDA-independent cooperative execution policy.

    ``component_tile`` is the maximum number of Cartesian components handled
    by one cooperative task.  ``tasks_per_warp`` also defines the width of a
    ``SUBGROUP_TASKS`` worker: each power-of-two subgroup owns one complete
    shell task and has ``32 / tasks_per_warp`` cooperating lanes.
    """

    kind: ScheduleKind
    block_threads: int
    component_tile: int
    tasks_per_warp: int = 1
    shared_coulomb: bool = True
    pair_orientation: PairOrientation = PairOrientation.CANONICAL
    pair_storage: PairStorage = PairStorage.MATERIALIZED
    unroll_pair_terms: bool = True
    minimum_blocks_per_sm: int = 0
    maximum_registers: int = 0

    def __post_init__(self) -> None:
        if self.block_threads < 32 or self.block_threads > 1024:
            raise ValueError("CUDA block size must be between 32 and 1024")
        if self.block_threads % 32 != 0:
            raise ValueError("CUDA block size must contain complete warps")
        if self.component_tile < 1:
            raise ValueError("component tile must be positive")
        if not 1 <= self.tasks_per_warp <= 32:
            raise ValueError("tasks per warp must be between one and 32")
        if not 0 <= self.minimum_blocks_per_sm <= 32:
            raise ValueError("minimum blocks per SM must be between zero and 32")
        if self.maximum_registers != 0 and not 32 <= self.maximum_registers <= 255:
            raise ValueError("maximum registers must be zero or between 32 and 255")
        if self.minimum_blocks_per_sm and self.maximum_registers:
            raise ValueError("launch bounds and maximum registers are mutually exclusive")
        if self.maximum_registers and self.kind != ScheduleKind.PACKED_TASKS:
            raise ValueError("maximum-register lowering currently supports packed tasks")
        if self.kind == ScheduleKind.PACKED_TASKS:
            if self.block_threads != 32:
                raise ValueError("packed-task schedules currently use one warp")
            if self.tasks_per_warp == 1:
                raise ValueError("packed-task schedules require multiple tasks")
        elif self.kind == ScheduleKind.THREAD_TASKS:
            if self.tasks_per_warp != 32:
                raise ValueError("thread-task schedules require one task per lane")
            if self.shared_coulomb:
                raise ValueError("thread-task schedules require lane-local Coulomb data")
        elif self.kind == ScheduleKind.SUBGROUP_TASKS:
            if self.tasks_per_warp not in (2, 4, 8):
                raise ValueError(
                    "subgroup-task schedules require 2, 4, or 8 tasks per warp"
                )
            if not self.shared_coulomb:
                raise ValueError(
                    "subgroup-task schedules require task-local shared Coulomb data"
                )
        elif self.tasks_per_warp != 1:
            raise ValueError(
                "only packed-, thread-, or subgroup-task schedules may own "
                "multiple tasks"
            )

    @property
    def warp_count(self) -> int:
        """Return the number of full warps in one cooperative block."""

        return self.block_threads // 32

    @property
    def subgroup_lanes(self) -> int:
        """Return the cooperating lane count for one subgroup shell task."""

        if self.kind != ScheduleKind.SUBGROUP_TASKS:
            raise ValueError("only subgroup-task schedules define subgroup lanes")
        return 32 // self.tasks_per_warp

    @property
    def tasks_per_block(self) -> int:
        """Return complete shell tasks advanced by one generated block."""

        if self.kind == ScheduleKind.PACKED_TASKS:
            return 32
        if self.kind == ScheduleKind.THREAD_TASKS:
            return self.block_threads
        if self.kind == ScheduleKind.SUBGROUP_TASKS:
            return self.warp_count * self.tasks_per_warp
        return 1


@dataclass(frozen=True, slots=True)
class KernelIR:
    """Complete backend input: mathematical integral plus CUDA schedule."""

    integral: IntegralIR
    schedule: ScheduleIR

    def __post_init__(self) -> None:
        component_count = self.integral.spec.component_count
        if (
            self.schedule.kind != ScheduleKind.TILED_COMPONENTS
            and self.schedule.component_tile < component_count
        ):
            raise ValueError(
                "non-tiled schedules must cover every shell component"
            )


def build_integral_ir(
    spec: ShellClassSpec,
    consumers: tuple[KernelConsumer | str, ...] = (KernelConsumer.FORCE,),
) -> IntegralIR:
    """Normalize requested observables into one mathematical integral IR."""

    return IntegralIR(
        spec=spec,
        consumers=frozenset(KernelConsumer(item) for item in consumers),
    )


def schedule_candidates(integral: IntegralIR) -> tuple[ScheduleIR, ...]:
    """Enumerate bounded schedule families for compile-and-measure tuning.

    The candidates intentionally describe a small search space.  They encode
    the qualitatively different mappings that matter for shell kernels without
    turning build-time tuning into an unconstrained compiler search.
    """

    component_count = integral.spec.component_count
    candidates: list[ScheduleIR] = []

    # One independent complete shell task per lane is attractive only when the
    # lane can cheaply evaluate every component itself.  Retain both packed and
    # one-task-per-warp variants so resource/timing gates make the final choice.
    if component_count <= 9:
        candidates.append(
            ScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=32,
                component_tile=component_count,
                tasks_per_warp=32,
                shared_coulomb=False,
            )
        )
        candidates.append(
            ScheduleIR(
                kind=ScheduleKind.SHELL_TASK,
                block_threads=32,
                component_tile=component_count,
                shared_coulomb=False,
            )
        )

    # Multiple task-local subgroups preserve primitive/Coulomb reuse without
    # dedicating an entire block to a low- or medium-component shell quartet.
    # The emitter loops components within each subgroup, so this schedule also
    # remains legal when a task owns more components than subgroup lanes.
    if component_count <= 64:
        for tasks_per_warp in (2, 4):
            candidates.append(
                ScheduleIR(
                    kind=ScheduleKind.SUBGROUP_TASKS,
                    block_threads=256,
                    component_tile=component_count,
                    tasks_per_warp=tasks_per_warp,
                    shared_coulomb=True,
                )
            )

    coulomb_state_count = comb(integral.maximum_coulomb_order + 3, 3)
    cooperative_threads = (
        (max(component_count, coulomb_state_count, 12) + 31)
        // 32
        * 32
    )
    if cooperative_threads <= 1024:
        candidates.append(
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=cooperative_threads,
                component_tile=component_count,
            )
        )
        # Coulomb setup is a short cooperative prelude, whereas every
        # component lane remains live through all primitive quartets.  When a
        # shell fits in one warp, also let that warp generate a larger Coulomb
        # table in strides.  This avoids keeping a second warp resident solely
        # to initialize a handful of states before it becomes idle.
        if component_count <= 32 and cooperative_threads > 32:
            candidates.append(
                ScheduleIR(
                    kind=ScheduleKind.COMPONENT_LANES,
                    block_threads=32,
                    component_tile=component_count,
                )
            )

    # Tiled schedules bound block size and live ranges for large d/f classes.
    # Multiple tile sizes are kept because register pressure and shared-state
    # reuse trade off differently across GPU architectures.
    for tile in (64, 128, 256):
        if component_count > tile:
            candidates.append(
                ScheduleIR(
                    kind=ScheduleKind.TILED_COMPONENTS,
                    block_threads=tile,
                    component_tile=tile,
                )
            )
    return tuple(candidates)


def default_schedule(integral: IntegralIR) -> ScheduleIR:
    """Return the compatibility schedule used before architecture tuning."""

    candidates = schedule_candidates(integral)
    for candidate in candidates:
        if candidate.kind == ScheduleKind.COMPONENT_LANES:
            return candidate
    for candidate in candidates:
        if candidate.kind == ScheduleKind.TILED_COMPONENTS:
            return candidate
    raise ValueError(
        f"{integral.spec.name} has no schedule supported by the current IR"
    )


def tuning_schedule_candidates(integral: IntegralIR) -> tuple[ScheduleIR, ...]:
    """Expand schedule families into a bounded compile-and-measure search.

    Cooperative component kernels expose four profitable code-shape knobs:
    sharing all Coulomb states versus recomputing lane-local states, aggressive
    versus compiler-directed loop unrolling, and which Gaussian pair is
    outermost in the Cartesian contraction.  Tiled kernels also choose whether
    to materialize that pair table or recompute terms to bound stack use.
    Packed schedules retain their occupancy/register search while shell-task
    schedules remain one structural candidate until profiling justifies a
    larger search space.
    """

    candidates = []
    for schedule in schedule_candidates(integral):
        if schedule.kind in (
            ScheduleKind.COMPONENT_LANES,
            ScheduleKind.TILED_COMPONENTS,
        ):
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
            for minimum_blocks_per_sm in (2, 12):
                for unroll_pair_terms in (True, False):
                    candidates.append(
                        replace(
                            schedule,
                            unroll_pair_terms=unroll_pair_terms,
                            minimum_blocks_per_sm=minimum_blocks_per_sm,
                        )
                    )
            for maximum_registers in (192, 208):
                for unroll_pair_terms in (True, False):
                    candidates.append(
                        replace(
                            schedule,
                            unroll_pair_terms=unroll_pair_terms,
                            minimum_blocks_per_sm=0,
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
    return tuple(candidates)
