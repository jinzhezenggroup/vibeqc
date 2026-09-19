"""Explicit CPU ISA targets for generated native kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CpuTargetInfo:
    """Compiler-visible CPU feature contract, independent of CPU model names."""

    name: str
    architecture: str
    features: tuple[str, ...]
    vector_lanes: int
    compiler_options: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.architecture:
            raise ValueError("CPU target name and architecture are required")
        if type(self.vector_lanes) is not int or self.vector_lanes not in (1, 4, 8):
            raise ValueError("CPU FP64 vector lanes must be one, four, or eight")
        if tuple(sorted(set(self.features))) != self.features:
            raise ValueError("CPU target features must be unique and sorted")
        if self.vector_lanes > 1 and self.architecture != "x86_64":
            raise ValueError("initial SIMD CPU targets are x86_64 only")
        required = {
            1: (),
            4: ("avx2", "fma"),
            8: ("avx512f", "fma"),
        }[self.vector_lanes]
        if any(feature not in self.features for feature in required):
            raise ValueError("CPU vector target is missing a required ISA feature")
        # This initial target contract describes ISA only, not arbitrary compiler
        # arithmetic. Extra flags can override -ffp-contract=off and even erase
        # the runtime's finite-input checks, while -march=native hides an ISA.
        expected_options = {
            1: (),
            4: ("-mavx2", "-mfma"),
            8: ("-mavx512f", "-mfma"),
        }[self.vector_lanes]
        if self.compiler_options != expected_options:
            raise ValueError("CPU compiler options must exactly match the declared ISA")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.cpu-target.v1",
            "name": self.name,
            "architecture": self.architecture,
            "features": self.features,
            "vector_lanes": self.vector_lanes,
            "compiler_options": self.compiler_options,
        }


GENERIC_CPU_TARGET = CpuTargetInfo(
    name="generic",
    architecture="portable",
    features=(),
    vector_lanes=1,
    compiler_options=(),
)

AVX2_FMA_TARGET = CpuTargetInfo(
    name="x86_64-avx2-fma",
    architecture="x86_64",
    features=("avx2", "fma"),
    vector_lanes=4,
    compiler_options=("-mavx2", "-mfma"),
)

AVX512F_FMA_TARGET = CpuTargetInfo(
    name="x86_64-avx512f-fma",
    architecture="x86_64",
    features=("avx512f", "fma"),
    vector_lanes=8,
    compiler_options=("-mavx512f", "-mfma"),
)

CPU_TARGETS = (
    GENERIC_CPU_TARGET,
    AVX2_FMA_TARGET,
    AVX512F_FMA_TARGET,
)


def cpu_target(value: str | CpuTargetInfo) -> CpuTargetInfo:
    if isinstance(value, CpuTargetInfo):
        return value
    for target in CPU_TARGETS:
        if target.name == value:
            return target
    raise ValueError(f"unknown CPU target {value!r}")
