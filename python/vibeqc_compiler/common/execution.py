"""Pure identity contract for prepared compiled-execution owners."""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from .provenance import canonical_hash

EXECUTION_SCHEMA = "vibeqc.compiled-execution.v1"


@dataclass(frozen=True, slots=True)
class CompiledExecutionIdentity:
    """Bind scientific request, compiled artifacts and runtime without pointers."""

    owner: str
    request_hash: str
    artifact_hash: str
    runtime_hash: str

    def __post_init__(self) -> None:
        if type(self.owner) is not str or not self.owner.strip():
            raise ValueError("compiled execution owner must be a nonempty string")
        for name in ("request_hash", "artifact_hash", "runtime_hash"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a SHA-256 digest")

    @classmethod
    def from_payloads(
        cls,
        *,
        owner: str,
        request: typing.Any,
        artifacts: typing.Any,
        runtime: typing.Any,
    ) -> CompiledExecutionIdentity:
        return cls(
            owner=owner,
            request_hash=canonical_hash(request),
            artifact_hash=canonical_hash(artifacts),
            runtime_hash=canonical_hash(runtime),
        )

    @property
    def identity(self) -> str:
        return canonical_hash({"schema": EXECUTION_SCHEMA, **asdict(self)})
