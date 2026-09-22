"""Integral-domain lane-group ownership for cooperative accelerator schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vibeqc_compiler.common.backend import TargetInfo, TargetScheduleShape
from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.precision import (
    ExecutionPrecisionSchedule,
    uniform_precision_schedule,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
)


@dataclass(frozen=True, slots=True)
class CooperativeLaneSchedule:
    """One logical task per lane group inside a hardware workgroup.

    The contract describes ownership only. Scientific equations, task meaning,
    shared-state contents and reduction values remain owned by each consumer.
    """

    subgroup_size: int
    lanes_per_group: int
    groups_per_workgroup: int
    shared_state: bool = True
    group_reduction: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.subgroup_size, "subgroup size"),
            (self.lanes_per_group, "lanes per group"),
            (self.groups_per_workgroup, "groups per workgroup"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported cooperative schedule schema")
        if self.lanes_per_group > self.subgroup_size:
            raise ValueError("cooperative group cannot exceed one hardware subgroup")
        if self.subgroup_size % self.lanes_per_group:
            raise ValueError("cooperative group lanes must divide the subgroup")
        if self.group_reduction and self.lanes_per_group & (self.lanes_per_group - 1):
            raise ValueError("reduced cooperative groups require power-of-two lanes")
        groups_per_subgroup = self.subgroup_size // self.lanes_per_group
        if self.groups_per_workgroup % groups_per_subgroup:
            raise ValueError("workgroup must contain complete hardware subgroups")

    @property
    def workgroup_threads(self) -> int:
        return self.lanes_per_group * self.groups_per_workgroup

    @property
    def groups_per_subgroup(self) -> int:
        return self.subgroup_size // self.lanes_per_group

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def validate_for(self, target: TargetInfo) -> None:
        """Validate subgroup geometry against an explicit backend target."""
        if target.subgroup_size is None:
            raise ValueError(
                "cooperative schedule requires a known target subgroup size"
            )
        TargetScheduleShape(self.workgroup_threads, self.subgroup_size).validate_for(
            target
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "group_reduction": self.group_reduction,
            "groups_per_workgroup": self.groups_per_workgroup,
            "lanes_per_group": self.lanes_per_group,
            "schema_version": self.schema_version,
            "shared_state": self.shared_state,
            "subgroup_size": self.subgroup_size,
            "workgroup_threads": self.workgroup_threads,
        }


def cooperative_schedule_contract(
    schedule: CooperativeLaneSchedule,
    *,
    consumer: str,
    target: TargetInfo,
    workload_hash: str | None = None,
    profile_key: str | None = None,
    precision_schedule: ExecutionPrecisionSchedule | None = None,
    fallback: bool = False,
    provenance: tuple[tuple[str, str], ...] = (),
) -> ScheduleContract:
    """Project lane-group ownership into the shared #833/#874 schedule contract.

    Domain consumers still own task meaning and scientific legality. This adapter
    only exposes portable topology, target identity, and profile/provenance hooks
    so integral schedules participate in the same compiler diagnostics/tuning
    vocabulary as TensorIR and DFT. Unknown resource/profitability facts remain
    explicit None rather than being guessed from target limits.
    """

    if not isinstance(schedule, CooperativeLaneSchedule):
        raise TypeError("cooperative contract requires CooperativeLaneSchedule")
    if not isinstance(target, TargetInfo):
        raise TypeError("cooperative contract requires TargetInfo")
    if precision_schedule is None:
        precision_schedule = uniform_precision_schedule(consumer)
    if not isinstance(precision_schedule, ExecutionPrecisionSchedule):
        raise TypeError(
            "cooperative precision schedule requires ExecutionPrecisionSchedule"
        )
    schedule.validate_for(target)
    return ScheduleContract(
        consumer=consumer,
        schedule_hash=schedule.identity,
        workload_hash=workload_hash,
        profile_key=profile_key,
        target_hash=canonical_hash(asdict(target)),
        precision_schedule_hash=precision_schedule.identity,
        fallback=fallback,
        topology=ScheduleTopology(
            workgroup_threads=schedule.workgroup_threads,
            subgroup_size=schedule.subgroup_size,
            fusion="domain-owned",
            materialization="shared-state"
            if schedule.shared_state
            else "private-state",
            residency="device",
            reduction="lane-group" if schedule.group_reduction else "none",
            cooperative=True,
            bucket="lane-groups",
        ),
        resources=ScheduleResources(),
        profitability=GpuProfitability(
            precision_cast_read_bytes=0 if precision_schedule.is_strict_fp64 else None,
            precision_cast_write_bytes=0 if precision_schedule.is_strict_fp64 else None,
            precision_cast_simultaneous_bytes=(
                0 if precision_schedule.is_strict_fp64 else None
            ),
            precision_widened_accumulation_terms=(
                0 if precision_schedule.is_strict_fp64 else None
            ),
        ),
        provenance=provenance
        + (
            ("precision_contract", "common.precision"),
            ("groups_per_workgroup", str(schedule.groups_per_workgroup)),
            ("lanes_per_group", str(schedule.lanes_per_group)),
        ),
    )
