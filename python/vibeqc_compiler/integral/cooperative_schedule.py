"""Integral-domain lane-group ownership for cooperative accelerator schedules."""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.backend import TargetInfo, TargetScheduleShape
from vibeqc_compiler.common.provenance import canonical_hash


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
