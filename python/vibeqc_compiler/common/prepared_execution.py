"""Shared identity and lifecycle contract for retained compiled executions.

The contract owns no scientific state and performs no compilation. Consumers bind
their existing artifact, target, schedule, specialization and optional capture
identities, then retain method-owned buffers/workspaces behind that immutable
request. Dynamic geometry/state refresh remains the consumer's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .provenance import canonical_hash

PREPARED_EXECUTION_SCHEMA = "vibeqc.prepared-execution.v1"


def _name(value: str, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value: str | None, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PreparedArtifactBinding:
    """Existing artifact identity retained by a prepared execution owner."""

    key: str
    binary_sha256: str

    def __post_init__(self) -> None:
        _digest(self.key, "artifact key")
        _digest(self.binary_sha256, "artifact binary hash")

    @classmethod
    def from_artifact(cls, artifact: object) -> PreparedArtifactBinding:
        try:
            metadata = getattr(artifact, "metadata", None)
            if not isinstance(metadata, Mapping):
                raise TypeError("artifact metadata must be a mapping")
            return cls(metadata["key"], metadata["binary_sha256"])
        except (AttributeError, KeyError, TypeError) as error:
            raise TypeError("prepared artifact requires key/binary metadata") from error


@dataclass(frozen=True, slots=True)
class PreparedExecutionRequest:
    """Compatibility identity excluding mutable numeric state and artifact bytes.

    specialization_identity binds a selected #459 profile when one exists.
    capture_identity binds the #507 capture contract when replay is enabled.
    Omitting either means that capability is not part of this prepared region.
    """

    kind: str
    scientific_identity: str
    target_identity: str
    schedule_identity: str
    workspace_identity: str
    device: int = 0
    specialization_identity: str | None = None
    capture_identity: str | None = None

    def __post_init__(self) -> None:
        _name(self.kind, "prepared execution kind")
        for field in (
            "scientific_identity",
            "target_identity",
            "schedule_identity",
            "workspace_identity",
        ):
            _digest(getattr(self, field), field)
        for field in ("specialization_identity", "capture_identity"):
            _digest(getattr(self, field), field, optional=True)
        if type(self.device) is not int or self.device < 0:
            raise ValueError("prepared execution device must be a nonnegative integer")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {"schema": PREPARED_EXECUTION_SCHEMA, "request": asdict(self)}
        )


@dataclass(frozen=True, slots=True)
class PreparedExecutionContract:
    """Installed request plus immutable artifact and retained-storage ownership."""

    request: PreparedExecutionRequest
    artifacts: tuple[PreparedArtifactBinding, ...]
    host_bytes: int
    device_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, PreparedExecutionRequest):
            raise TypeError("prepared contract requires a PreparedExecutionRequest")
        artifacts = tuple(self.artifacts)
        if not artifacts or any(
            not isinstance(item, PreparedArtifactBinding) for item in artifacts
        ):
            raise ValueError("prepared contract requires artifact bindings")
        if len({item.key for item in artifacts}) != len(artifacts):
            raise ValueError("prepared contract contains duplicate artifact keys")
        object.__setattr__(self, "artifacts", artifacts)
        for field in ("host_bytes", "device_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": PREPARED_EXECUTION_SCHEMA,
                "contract": {
                    "request": asdict(self.request),
                    "artifacts": [asdict(item) for item in self.artifacts],
                    "host_bytes": self.host_bytes,
                    "device_bytes": self.device_bytes,
                },
            }
        )


class PreparedExecutionMismatch(ValueError):
    """A retained compiled execution cannot serve the requested domain."""


class PreparedExecutionLease:
    """Shared state machine around method-owned prepared resources.

    The lease never refreshes scientific state itself. A failure marks the next
    compatible execution as requiring a consumer-owned refresh/reset before it can
    be published successful. Invalidation leaves resource destruction to the
    consumer that owns buffers and libraries.
    """

    def __init__(self) -> None:
        self._contract: PreparedExecutionContract | None = None
        self._failed = False
        self._executions = 0
        self._refreshes = 0
        self._invalidations = 0

    @property
    def contract(self) -> PreparedExecutionContract | None:
        return self._contract

    @property
    def identity(self) -> str | None:
        return None if self._contract is None else self._contract.identity

    @property
    def request_identity(self) -> str | None:
        return None if self._contract is None else self._contract.request.identity

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def needs_refresh(self) -> bool:
        return self._failed

    @property
    def executions(self) -> int:
        return self._executions

    @property
    def refreshes(self) -> int:
        return self._refreshes

    @property
    def invalidations(self) -> int:
        return self._invalidations

    def install(
        self,
        request: PreparedExecutionRequest,
        artifacts: tuple[PreparedArtifactBinding, ...],
        *,
        host_bytes: int,
        device_bytes: int,
    ) -> PreparedExecutionContract:
        if self._contract is not None:
            raise RuntimeError("prepared execution lease is already installed")
        contract = PreparedExecutionContract(
            request, artifacts, host_bytes, device_bytes
        )
        self._contract = contract
        self._failed = False
        return contract

    def require(
        self,
        request: PreparedExecutionRequest,
        *,
        max_host_bytes: int,
        max_device_bytes: int,
    ) -> PreparedExecutionContract:
        contract = self._contract
        if contract is None:
            raise RuntimeError("prepared execution lease is not installed")
        if not isinstance(request, PreparedExecutionRequest):
            raise TypeError("prepared execution requires a typed request")
        if request.identity != contract.request.identity:
            raise PreparedExecutionMismatch("prepared execution identity changed")
        for value, name in (
            (max_host_bytes, "host budget"),
            (max_device_bytes, "device budget"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if contract.host_bytes > max_host_bytes:
            raise ValueError("prepared execution host budget exceeded")
        if contract.device_bytes > max_device_bytes:
            raise ValueError("prepared execution device budget exceeded")
        return contract

    def mark_refresh(self) -> None:
        if self._contract is None:
            raise RuntimeError("cannot refresh an uninstalled prepared execution")
        self._failed = False
        self._refreshes += 1

    def mark_failure(self) -> None:
        if self._contract is not None:
            self._failed = True

    def mark_success(self) -> None:
        if self._contract is None:
            raise RuntimeError("cannot publish an uninstalled prepared execution")
        if self._failed:
            raise RuntimeError(
                "failed prepared execution requires refresh before success"
            )
        self._executions += 1

    def invalidate(self) -> None:
        if self._contract is not None:
            self._invalidations += 1
        self._contract = None
        self._failed = False
