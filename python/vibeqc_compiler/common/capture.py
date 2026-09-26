"""Versioned, pure eligibility/identity contract for compiled replay regions.

This references #459 specialization records and existing artifact hashes. It
neither probes a device nor creates a second executable/profile cache.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass

from .provenance import canonical_hash
from .specialization import (
    CompilationIdentity,
    SpecializationGuard,
    TargetCapabilities,
    WorkloadSignature,
)

CAPTURE_SCHEMA = "vibeqc.capture.v1"
MAX_CAPTURE_LAUNCHES = 4096


@dataclass(frozen=True, slots=True)
class CaptureContract:
    """Qualification identity, excluding numeric inputs and owner-local pointers.

    Native ownership additionally binds the stream, arena and library handle.
    Layout, precision, shape and method state belong in workload facts. The
    runtime hash includes the device UUID, driver, CUDA and library versions.
    Unsupported effects are explicit, never silently assumed capturable.
    """

    compilation: CompilationIdentity
    workload: WorkloadSignature
    target: TargetCapabilities
    guard: SpecializationGuard
    artifact_key: str
    schedule_hash: str
    runtime_hash: str
    unsupported_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, cls in (
            ("compilation", CompilationIdentity),
            ("workload", WorkloadSignature),
            ("target", TargetCapabilities),
            ("guard", SpecializationGuard),
        ):
            if not isinstance(getattr(self, field), cls):
                raise TypeError(f"{field} must be a {cls.__name__}")
        for field in ("artifact_key", "schedule_hash", "runtime_hash"):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{field} must be a SHA-256 digest")
        effects = tuple(self.unsupported_effects)
        if any(type(v) is not str or not v.strip() for v in effects):
            raise ValueError("unsupported effects must be nonempty strings")
        object.__setattr__(self, "unsupported_effects", tuple(sorted(set(effects))))

    @property
    def failures(self) -> tuple[str, ...]:
        facts = {
            "workload": {"kind": self.workload.kind, **dict(self.workload.features)},
            "target": {**asdict(self.target.target), **dict(self.target.features)},
        }
        return self.unsupported_effects + self.guard.failures(facts)

    @property
    def eligible(self) -> bool:
        return not self.failures

    @property
    def identity(self) -> str:
        return canonical_hash({"schema": CAPTURE_SCHEMA, **asdict(self)})


class _GraphMetrics(ctypes.Structure):
    """Matches runtime/cuda_graph_region.cuh; separate from the shared Metrics ABI."""

    _fields_ = [
        *[
            (name, ctypes.c_uint64)
            for name in (
                "capture_attempts",
                "captures",
                "replays",
                "fallbacks",
                "invalidations",
                "node_count",
                "retained_device_bytes",
            )
        ],
        *[
            (name, ctypes.c_double)
            for name in (
                "capture_ms",
                "instantiate_ms",
                "submission_ms",
            )
        ],
        ("mode", ctypes.c_int32),
    ]


GRAPH_MODES = ("ordinary", "warmup", "captured", "replay", "fallback", "profiling")
