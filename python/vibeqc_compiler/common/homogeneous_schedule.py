"""Compiler-owned bounded scheduling for homogeneous runtime task ranges.

Scientific owners provide exact-class/signature ranges and legal execution
modes. This module only supplies deterministic packetization, schedule identity,
and common GPU profitability evidence. It never changes equations, screening,
precision, convergence, or provider legality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .gpu_profitability import GpuProfitability
from .provenance import canonical_hash


class HomogeneousExecution(str, Enum):
    """Backend execution families represented without selecting one implicitly."""

    ORDINARY = "ordinary"
    COOPERATIVE = "cooperative"
    PERSISTENT = "persistent"


@dataclass(frozen=True, slots=True)
class HomogeneousTaskRange:
    """One exact class/signature range with a bounded logical task count."""

    exact_class: tuple[int, ...]
    signature: tuple[int, ...]
    task_count: int
    work_per_task: int = 1

    def __post_init__(self) -> None:
        exact_class = tuple(self.exact_class)
        signature = tuple(self.signature)
        if not exact_class or any(
            type(value) is not int or value < 0 for value in exact_class
        ):
            raise ValueError(
                "homogeneous task class must be non-empty nonnegative integers"
            )
        if not signature or any(
            type(value) is not int or value < 0 for value in signature
        ):
            raise ValueError(
                "homogeneous task signature must be non-empty nonnegative integers"
            )
        if type(self.task_count) is not int or self.task_count < 1:
            raise ValueError("homogeneous task count must be a positive integer")
        if type(self.work_per_task) is not int or self.work_per_task < 1:
            raise ValueError("homogeneous work per task must be a positive integer")
        object.__setattr__(self, "exact_class", exact_class)
        object.__setattr__(self, "signature", signature)

    @property
    def weighted_work(self) -> int:
        return self.task_count * self.work_per_task

    def to_payload(self) -> dict[str, object]:
        return {
            "exact_class": self.exact_class,
            "signature": self.signature,
            "task_count": self.task_count,
            "work_per_task": self.work_per_task,
        }


@dataclass(frozen=True, slots=True)
class HomogeneousTaskSchedule:
    """Bounded packet/page policy plus auditable profitability evidence."""

    packet_capacity: int
    execution: HomogeneousExecution = HomogeneousExecution.ORDINARY
    profitability: GpuProfitability = field(default_factory=GpuProfitability)
    target_identity: str | None = None

    def __post_init__(self) -> None:
        if type(self.packet_capacity) is not int or self.packet_capacity < 1:
            raise ValueError("homogeneous packet capacity must be a positive integer")
        if not isinstance(self.execution, HomogeneousExecution):
            raise TypeError("homogeneous execution must be a HomogeneousExecution")
        if not isinstance(self.profitability, GpuProfitability):
            raise TypeError("homogeneous profitability must use GpuProfitability")
        if self.target_identity is not None and (
            type(self.target_identity) is not str or not self.target_identity
        ):
            raise ValueError("target identity must be a non-empty string or None")

    def ordered(
        self, ranges: tuple[HomogeneousTaskRange, ...]
    ) -> tuple[HomogeneousTaskRange, ...]:
        """Order heavy homogeneous ranges first while preserving stable ties."""
        rows = tuple(ranges)
        if any(not isinstance(row, HomogeneousTaskRange) for row in rows):
            raise TypeError("homogeneous schedules require HomogeneousTaskRange rows")
        return tuple(
            row
            for _, row in sorted(
                enumerate(rows),
                key=lambda item: (-item[1].work_per_task, item[0]),
            )
        )

    def packets(
        self, ranges: tuple[HomogeneousTaskRange, ...]
    ) -> tuple[tuple[HomogeneousTaskRange, ...], ...]:
        ordered = self.ordered(ranges)
        return tuple(
            ordered[index : index + self.packet_capacity]
            for index in range(0, len(ordered), self.packet_capacity)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.compiler.homogeneous-task-schedule.v1",
            "packet_capacity": self.packet_capacity,
            "execution": self.execution.value,
            "target_identity": self.target_identity,
            "profitability": self.profitability.to_payload(),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())
