"""CPU schedule IR for lane-parallel generated integral kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vibeqc_compiler.common.cpu_target import CpuTargetInfo

from .expr import (
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    PowerLowering,
    RematerializationPolicy,
)


class CpuLanePacking(str, Enum):
    PRIMITIVE_RECORDS = "primitive_records"


class CpuTailPolicy(str, Enum):
    SAFE_PAD = "safe_pad"


class CpuAlgebraPlacement(str, Enum):
    MATERIALIZED_CSE = "materialized_cse"
    INLINE_SINGLE_USE = "inline_single_use"
    PRESSURE_REMATERIALIZED = "pressure_rematerialized"

    def policy(self) -> RematerializationPolicy:
        if self == CpuAlgebraPlacement.MATERIALIZED_CSE:
            return RematerializationPolicy.materialized_cse()
        if self == CpuAlgebraPlacement.INLINE_SINGLE_USE:
            return RematerializationPolicy.inline_single_use_values()
        return RematerializationPolicy.pressure_rematerialized()


@dataclass(frozen=True, slots=True)
class CpuScheduleIR:
    """CPU execution policy; no scientific recurrence belongs here."""

    vector_lanes: int
    lane_packing: CpuLanePacking = CpuLanePacking.PRIMITIVE_RECORDS
    tail_policy: CpuTailPolicy = CpuTailPolicy.SAFE_PAD
    algebra_placement: CpuAlgebraPlacement = CpuAlgebraPlacement.INLINE_SINGLE_USE
    algebra_ordering: AlgebraOrdering = AlgebraOrdering.PRESSURE_AWARE
    algebra_fusion: AlgebraFusion = AlgebraFusion.SEPARATE
    algebra_form: AlgebraForm = AlgebraForm.BINARY
    power_lowering: PowerLowering = PowerLowering.SMALL_INTEGER

    def __post_init__(self) -> None:
        if type(self.vector_lanes) is not int or self.vector_lanes not in (1, 4, 8):
            raise ValueError("CPU schedule lanes must be one, four, or eight")

    def validate_for(self, target: CpuTargetInfo) -> None:
        if self.vector_lanes != target.vector_lanes:
            raise ValueError("CPU schedule vector width does not match target")
        if self.lane_packing != CpuLanePacking.PRIMITIVE_RECORDS:
            raise ValueError("only primitive-record CPU lane packing is implemented")
        if self.tail_policy != CpuTailPolicy.SAFE_PAD:
            raise ValueError("only safe padded CPU tails are implemented")
        if self.algebra_fusion == AlgebraFusion.FMA and "fma" not in target.features:
            raise ValueError("FMA schedule requires a target with FMA")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.cpu-schedule.v1",
            "vector_lanes": self.vector_lanes,
            "lane_packing": self.lane_packing.value,
            "tail_policy": self.tail_policy.value,
            "algebra_placement": self.algebra_placement.value,
            "algebra_ordering": self.algebra_ordering.value,
            "algebra_fusion": self.algebra_fusion.value,
            "algebra_form": self.algebra_form.value,
            "power_lowering": self.power_lowering.value,
        }


def default_cpu_schedule(target: CpuTargetInfo) -> CpuScheduleIR:
    schedule = CpuScheduleIR(vector_lanes=target.vector_lanes)
    schedule.validate_for(target)
    return schedule
