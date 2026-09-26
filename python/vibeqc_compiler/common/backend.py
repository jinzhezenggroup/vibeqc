"""Backend contracts shared by lowering, tuning, and validation.

The mathematical integral representation deliberately does not import this
module.  Backends consume that representation and add their own target and
schedule records at the lowering boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Backend-neutral execution limits used for schedule-shape validation.

    Portable runtimes may not expose a subgroup width or residency limit.
    ``None`` keeps those unknown instead of importing CUDA's warp/SM limits.
    Scalar schedules can still execute; subgroup schedules require a known width.
    """

    backend: str
    architecture: str
    subgroup_size: int | None
    maximum_workgroup_threads: int
    maximum_resident_workgroups: int | None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("target backend must be named")
        if not self.architecture:
            raise ValueError("target architecture must be named")
        if (
            type(self.maximum_workgroup_threads) is not int
            or self.maximum_workgroup_threads < 1
        ):
            raise ValueError("target workgroup limit must be positive")
        if self.subgroup_size is not None and (
            type(self.subgroup_size) is not int or self.subgroup_size < 1
        ):
            raise ValueError("target subgroup size must be positive when known")
        if (
            self.subgroup_size is not None
            and self.maximum_workgroup_threads < self.subgroup_size
        ):
            raise ValueError("workgroup limit must contain one subgroup")
        if self.maximum_resident_workgroups is not None and (
            type(self.maximum_resident_workgroups) is not int
            or self.maximum_resident_workgroups < 1
        ):
            raise ValueError("resident workgroup limit must be positive when known")


@dataclass(frozen=True, slots=True)
class TargetScheduleShape:
    """Minimal schedule geometry that any accelerator backend can validate."""

    workgroup_threads: int
    subgroup_size: int | None

    def validate_for(self, target: TargetInfo) -> None:
        """Reject geometry that cannot execute on ``target``."""

        if self.subgroup_size is not None and (
            type(self.subgroup_size) is not int or self.subgroup_size < 1
        ):
            raise ValueError("schedule subgroup size must be positive when requested")
        if (
            self.subgroup_size is not None
            and self.subgroup_size != target.subgroup_size
        ):
            raise ValueError(
                f"schedule subgroup size {self.subgroup_size} does not match "
                f"target subgroup size {target.subgroup_size}"
            )
        if (
            type(self.workgroup_threads) is not int
            or not 1 <= self.workgroup_threads <= target.maximum_workgroup_threads
        ):
            raise ValueError("schedule workgroup exceeds the target thread limit")
        if (
            self.subgroup_size is not None
            and self.workgroup_threads % self.subgroup_size != 0
        ):
            raise ValueError("schedule workgroup must contain complete subgroups")


@runtime_checkable
class SourceEmitter(Protocol):
    """Lower a backend-specific kernel representation to source text."""

    def emit(self, kernel: object) -> str: ...


@runtime_checkable
class CompilerAdapter(Protocol):
    """Compile and link source without exposing a vendor CLI to the pipeline."""

    def compile(self, source: str, target: TargetInfo) -> object: ...


@runtime_checkable
class ResourceParser(Protocol):
    """Convert compiler diagnostics into backend-independent resource data."""

    def parse(self, diagnostics: str) -> object: ...


@runtime_checkable
class DeviceProbe(Protocol):
    """Query the device assigned to a benchmark execution environment."""

    def probe(self) -> TargetInfo: ...


@runtime_checkable
class BenchmarkExecutor(Protocol):
    """Execute one compiled benchmark artifact on its selected target."""

    def run(self, artifact: object, target: TargetInfo) -> object: ...


@runtime_checkable
class RegistryEmitter(Protocol):
    """Emit production dispatch metadata for one or more backend profiles."""

    def emit_registry(self, profiles: object) -> str: ...
