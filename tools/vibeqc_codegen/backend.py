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
    """Backend-neutral execution limits used for schedule-shape validation."""

    backend: str
    architecture: str
    subgroup_size: int
    maximum_workgroup_threads: int
    maximum_resident_workgroups: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("target backend must be named")
        if not self.architecture:
            raise ValueError("target architecture must be named")
        if self.subgroup_size < 1:
            raise ValueError("target subgroup size must be positive")
        if self.maximum_workgroup_threads < self.subgroup_size:
            raise ValueError("workgroup limit must contain one subgroup")
        if self.maximum_resident_workgroups < 1:
            raise ValueError("resident workgroup limit must be positive")


@dataclass(frozen=True, slots=True)
class TargetScheduleShape:
    """Minimal schedule geometry that any accelerator backend can validate."""

    workgroup_threads: int
    subgroup_size: int

    def validate_for(self, target: TargetInfo) -> None:
        """Reject geometry that cannot execute on ``target``."""

        if self.subgroup_size != target.subgroup_size:
            raise ValueError(
                f"schedule subgroup size {self.subgroup_size} does not match "
                f"target subgroup size {target.subgroup_size}"
            )
        if not 1 <= self.workgroup_threads <= target.maximum_workgroup_threads:
            raise ValueError("schedule workgroup exceeds the target thread limit")
        if self.workgroup_threads % self.subgroup_size != 0:
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
