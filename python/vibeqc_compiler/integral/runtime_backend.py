"""Explicit compiler/runtime/library contracts for optional accelerator backends.

These records belong to execution policy, never scientific operator IR. Missing
features raise an unsupported result before allocation or submission; a backend
cannot satisfy an accelerator request by silently evaluating it on the CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from vibeqc_compiler.common.evidence import canonical_hash


class UnsupportedBackendFeature(NotImplementedError):
    """A selected backend has no implementation of the requested operation."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Queried capabilities with explicit unknown subgroup and resource values.

    OpenCL 3.0 does not imply every 2.x feature. Native FP64, floating-point
    atomics, device enqueue and graph execution are independent capabilities.
    ``subgroup_size=None`` never means 32 and does not prohibit scalar kernels.
    """

    backend: str
    fp64: bool
    maximum_workgroup_threads: int
    local_memory_bytes: int
    subgroup_size: int | None = None
    fp64_atomic_add: bool = False
    native_events: bool = True
    event_profiling: bool = True
    device_enqueue: bool = False
    graphs: bool = False

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("runtime backend must be named")
        for name in (
            "fp64",
            "fp64_atomic_add",
            "native_events",
            "event_profiling",
            "device_enqueue",
            "graphs",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean capability")
        for name in ("maximum_workgroup_threads", "local_memory_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (
                1 if name.startswith("maximum") else 0
            ):
                raise ValueError(f"invalid {name}")
        if self.subgroup_size is not None and (
            type(self.subgroup_size) is not int
            or not 1 <= self.subgroup_size <= self.maximum_workgroup_threads
        ):
            raise ValueError("invalid queried subgroup size")


@dataclass(frozen=True)
class ExecutionShape:
    """A padded workgroup schedule with optional explicit subgroup requirements."""

    workgroup_threads: int
    local_bytes: int = 0
    subgroup_size: int | None = None
    requires_fp64: bool = True
    requires_fp64_atomic_add: bool = False
    requires_device_enqueue: bool = False
    requires_graphs: bool = False

    def validate_for(self, target: RuntimeCapabilities):
        """Check actual queried limits without inserting CUDA warp assumptions."""
        for name in (
            "requires_fp64",
            "requires_fp64_atomic_add",
            "requires_device_enqueue",
            "requires_graphs",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean requirement")
        if (
            type(self.workgroup_threads) is not int
            or not 1 <= self.workgroup_threads <= target.maximum_workgroup_threads
        ):
            raise ValueError("workgroup exceeds target limits")
        if (
            type(self.local_bytes) is not int
            or not 0 <= self.local_bytes <= target.local_memory_bytes
        ):
            raise ValueError("local workspace exceeds target limits")
        if self.subgroup_size is not None:
            if type(self.subgroup_size) is not int or self.subgroup_size < 1:
                raise ValueError("invalid subgroup request")
            if target.subgroup_size is None:
                raise UnsupportedBackendFeature("target subgroup width is not known")
            if (
                self.subgroup_size != target.subgroup_size
                or self.workgroup_threads % self.subgroup_size
            ):
                raise ValueError("schedule does not match complete native subgroups")
        for required, available, name in (
            (self.requires_fp64, target.fp64, "FP64"),
            (
                self.requires_fp64_atomic_add,
                target.fp64_atomic_add,
                "native FP64 atomic add",
            ),
            (self.requires_device_enqueue, target.device_enqueue, "device enqueue"),
            (self.requires_graphs, target.graphs, "graph execution"),
        ):
            if required and not available:
                raise UnsupportedBackendFeature(
                    f"{target.backend} does not support {name}"
                )

    def padded_items(self, count):
        """OpenCL 1.2 global sizes must be multiples of the local workgroup size."""
        if (
            type(count) is not int
            or count < 0
            or type(self.workgroup_threads) is not int
            or self.workgroup_threads < 1
        ):
            raise ValueError("nonnegative count and positive workgroup size required")
        return (
            (count + self.workgroup_threads - 1) // self.workgroup_threads
        ) * self.workgroup_threads


@dataclass(frozen=True)
class CompiledArtifactIdentity:
    """Executable compatibility is stricter than scientific equation identity.

    Include the actual compiler/runtime/driver and device UUID or vendor device
    identity. CUDA's existing keys and official/local profile precedence remain
    unchanged. This identity does not authorize downloading or loading binaries.
    """

    backend: str
    device: str
    runtime: str
    compiler: str
    driver: str
    language: str
    options: tuple[str, ...]
    scientific_hash: str
    source_hash: str
    schedule_hash: str
    abi: int = 1

    def __post_init__(self):
        for name in ("backend", "device", "runtime", "compiler", "driver", "language"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"verified {name} identity is required")
        for name in ("scientific_hash", "source_hash", "schedule_hash"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if type(self.abi) is not int or self.abi < 1:
            raise ValueError("positive ABI version required")
        object.__setattr__(self, "options", tuple(self.options))
        if any(not isinstance(option, str) for option in self.options):
            raise ValueError("compiler options must be strings")

    @property
    def key(self):
        return canonical_hash(asdict(self))

    def require_compatible(self, other):
        """Reject executable reuse across any compilation/runtime boundary."""
        if self != other:
            raise ValueError(
                "compiled artifact identity is incompatible with this execution"
            )


@dataclass(frozen=True)
class LibraryRequest:
    """Explicit FP64 library call convention, including workspace and residual gates.

    GEMM uses A(m,k), B(k,n), C(m,n); symmetric eigensolve and Cholesky use
    square A(m,m). Layout applies to every operand. Providers must report errors
    and required workspace before enqueueing, and eigensolves/factorizations
    must expose a residual rather than assuming success from a status code.
    """

    operation: str
    shape: tuple[int, ...]
    layout: str = "row_major"
    dtype: str = "fp64"
    workspace_limit_bytes: int = 0
    residual_tolerance: float = 1e-11

    def __post_init__(self):
        import math

        rank = 3 if self.operation == "gemm" else 1
        if self.operation not in ("gemm", "symmetric_eigh", "cholesky"):
            raise ValueError("unknown library operation")
        if len(self.shape) != rank or any(
            type(n) is not int or n < 1 for n in self.shape
        ):
            raise ValueError("invalid library operation dimensions")
        if self.layout not in ("row_major", "column_major") or self.dtype != "fp64":
            raise ValueError("explicit supported layout and FP64 dtype required")
        if (
            type(self.workspace_limit_bytes) is not int
            or self.workspace_limit_bytes < 0
        ):
            raise ValueError("workspace limit must be nonnegative")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0:
            raise ValueError("positive finite residual tolerance required")


@dataclass(frozen=True)
class UnsupportedLibraryProvider:
    """An honest missing-provider adapter, with no CPU fallback path."""

    backend: str
    reason: str

    def plan(self, request: LibraryRequest):
        raise UnsupportedBackendFeature(
            f"{self.backend} {request.operation}: {self.reason}"
        )


@runtime_checkable
class AcceleratorRuntime(Protocol):
    """Ownership-aware memory, streams/events, compilation and launch boundary.

    Buffer/program/event handles must belong to the runtime's own context.
    Implementations check byte bounds before transfers and release resources
    deterministically; asynchronous submissions retain their input resources
    until completion. Graph/device-enqueue APIs are optional capabilities.
    """

    def capabilities(self) -> RuntimeCapabilities: ...
    def create_stream(self) -> object: ...
    def allocate(self, nbytes: int) -> object: ...
    def release(self, resource: object) -> None: ...
    def write(self, buffer: object, data: bytes, *, offset: int = 0) -> None: ...
    def read(self, buffer: object, nbytes: int, *, offset: int = 0) -> bytes: ...
    def compile(self, source: str, *, options: tuple[str, ...] = ()) -> object: ...
    def link(self, programs: tuple[object, ...]) -> object: ...
    def launch(
        self,
        program: object,
        kernel: str,
        arguments: tuple[object, ...],
        *,
        items: int,
        shape: ExecutionShape,
    ) -> object: ...
    def wait(self, event: object) -> None: ...
    def reduce_sum(
        self, buffer: object, count: int, *, workgroup: int = 64
    ) -> object: ...
