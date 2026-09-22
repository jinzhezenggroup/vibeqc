"""Backend-neutral lowering-provider request and candidate contracts.

Scientific IR owns operation semantics.  This module only records how an
already-defined operation may be lowered, including composite generated/library
implementations, resource requirements, numerical mode, and negative evidence.
It performs no device probing, compilation, promotion, or method selection.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import Literal

from .provenance import canonical_hash
from .specialization import TargetCapabilities

LOWERING_REQUEST_SCHEMA = "vibeqc.compiler.lowering-request.v1"
LOWERING_PROVIDER_SCHEMA = "vibeqc.compiler.lowering-provider.v1"
LOWERING_CANDIDATE_SCHEMA = "vibeqc.compiler.lowering-candidate.v1"
LOWERING_DIAGNOSTICS_SCHEMA = "vibeqc.compiler.lowering-diagnostics.v1"

Scalar = bool | int | float | str


def _name(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _scalar(value: typing.Any, label: str) -> Scalar:
    if type(value) not in (bool, int, float, str) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite immutable JSON scalar")
    return value


def _pairs(
    values: typing.Iterable[tuple[str, Scalar]], label: str
) -> tuple[tuple[str, Scalar], ...]:
    result: list[tuple[str, Scalar]] = []
    names: set[str] = set()
    for pair in values:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{label} must contain name/value pairs")
        name, value = pair
        _name(name, f"{label} name")
        _scalar(value, f"{label} value")
        if name in names:
            raise ValueError(f"duplicate {label} name {name!r}")
        names.add(name)
        result.append((name, value))
    return tuple(sorted(result))


def _nonnegative_bytes(value: typing.Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class _TypedRecord:
    """Keep equality consistent with type-sensitive canonical JSON identities."""

    __slots__ = ()

    def to_payload(self) -> dict[str, typing.Any]:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        peer = typing.cast("_TypedRecord", other)
        return canonical_hash(self.to_payload()) == canonical_hash(peer.to_payload())

    def __hash__(self) -> int:
        return hash((type(self), canonical_hash(self.to_payload())))


@dataclass(frozen=True, slots=True, eq=False)
class LoweringRequest(_TypedRecord):
    """One compiler operation requesting an implementation on a target backend."""

    consumer: str
    operation: str
    backend: str
    dtype: str
    accumulation_dtype: str
    shape: tuple[int, ...]
    semantics: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        for label in (
            "consumer",
            "operation",
            "backend",
            "dtype",
            "accumulation_dtype",
        ):
            _name(getattr(self, label), label)
        shape = tuple(self.shape)
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError("lowering shape extents must be non-negative integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "semantics", _pairs(self.semantics, "semantic"))

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": LOWERING_REQUEST_SCHEMA,
            "consumer": self.consumer,
            "operation": self.operation,
            "backend": self.backend,
            "dtype": self.dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "shape": list(self.shape),
            "semantics": dict(self.semantics),
        }


@dataclass(frozen=True, slots=True, eq=False)
class ProviderDescriptor(_TypedRecord):
    """Stable provider identity independent of one particular lowering request."""

    name: str
    kind: Literal["generated", "library", "runtime"]
    implementation: str
    version: str | None = None
    required_features: tuple[str, ...] = ()
    provenance: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        _name(self.name, "provider name")
        _name(self.implementation, "provider implementation")
        if self.kind not in ("generated", "library", "runtime"):
            raise ValueError("unknown lowering provider kind")
        if self.version is not None:
            _name(self.version, "provider version")
        features = tuple(self.required_features)
        if any(type(feature) is not str or not feature for feature in features):
            raise ValueError("provider features must be nonempty strings")
        if len(set(features)) != len(features):
            raise ValueError("provider features must be unique")
        object.__setattr__(self, "required_features", tuple(sorted(features)))
        object.__setattr__(
            self, "provenance", _pairs(self.provenance, "provider provenance")
        )

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": LOWERING_PROVIDER_SCHEMA,
            "name": self.name,
            "kind": self.kind,
            "implementation": self.implementation,
            "version": self.version,
            "required_features": list(self.required_features),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True, eq=False)
class LoweringCandidate(_TypedRecord):
    """One legal or rejected lowering, retaining explicit negative evidence."""

    request: LoweringRequest
    implementation: str
    providers: tuple[ProviderDescriptor, ...]
    status: Literal["ready", "unsupported"]
    numerical_mode: str
    workspace_bytes: int = 0
    provider_bytes: int = 0
    reason: str | None = None
    provenance: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request, LoweringRequest):
            raise TypeError("lowering candidate requires a LoweringRequest")
        _name(self.implementation, "lowering implementation")
        _name(self.numerical_mode, "lowering numerical mode")
        providers = tuple(self.providers)
        if not providers or any(
            not isinstance(provider, ProviderDescriptor) for provider in providers
        ):
            raise ValueError("lowering candidate requires provider descriptors")
        identities = [provider.identity for provider in providers]
        if len(set(identities)) != len(identities):
            raise ValueError("lowering candidate providers must be unique")
        object.__setattr__(self, "providers", providers)
        _nonnegative_bytes(self.workspace_bytes, "lowering workspace bytes")
        _nonnegative_bytes(self.provider_bytes, "lowering provider bytes")
        if self.status not in ("ready", "unsupported"):
            raise ValueError("unknown lowering candidate status")
        if self.status == "ready" and self.reason is not None:
            raise ValueError("ready lowering candidate cannot carry a rejection reason")
        if self.status == "unsupported":
            _name(self.reason, "unsupported lowering rejection reason")
        object.__setattr__(
            self, "provenance", _pairs(self.provenance, "lowering provenance")
        )

    @property
    def request_hash(self) -> str:
        return self.request.identity

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": LOWERING_CANDIDATE_SCHEMA,
            "request": self.request.to_payload(),
            "request_hash": self.request_hash,
            "implementation": self.implementation,
            "providers": [provider.to_payload() for provider in self.providers],
            "status": self.status,
            "numerical_mode": self.numerical_mode,
            "workspace_bytes": self.workspace_bytes,
            "provider_bytes": self.provider_bytes,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


class LoweringProvider(typing.Protocol):
    """Side-effect-free candidate producer; selection and promotion live elsewhere."""

    descriptor: ProviderDescriptor

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        """Return ready and/or explicit unsupported candidates for one request."""
        ...


def collect_lowering_candidates(
    request: LoweringRequest,
    target: TargetCapabilities,
    providers: typing.Iterable[LoweringProvider],
) -> tuple[LoweringCandidate, ...]:
    """Collect provider offers without ranking, probing, compiling, or fallback."""

    if not isinstance(request, LoweringRequest):
        raise TypeError("provider collection requires a LoweringRequest")
    if not isinstance(target, TargetCapabilities):
        raise TypeError("provider collection requires TargetCapabilities")
    if request.backend != target.target.backend:
        raise ValueError(
            f"lowering backend {request.backend!r} does not match target "
            f"{target.target.backend!r}"
        )

    collected: list[LoweringCandidate] = []
    identities: set[str] = set()
    for provider in providers:
        descriptor = provider.descriptor
        if not isinstance(descriptor, ProviderDescriptor):
            raise TypeError("lowering provider descriptor has the wrong type")
        offered = tuple(provider.candidates(request, target))
        if not offered:
            raise ValueError(
                f"provider {descriptor.name!r} must return explicit unsupported evidence"
            )
        for candidate in offered:
            if not isinstance(candidate, LoweringCandidate):
                raise TypeError("lowering provider returned a non-candidate")
            if candidate.request != request:
                raise ValueError(
                    f"provider {descriptor.name!r} returned a candidate for another request"
                )
            if descriptor.identity not in {
                dependency.identity for dependency in candidate.providers
            }:
                raise ValueError(
                    f"candidate from {descriptor.name!r} does not name its provider"
                )
            if candidate.identity in identities:
                raise ValueError("duplicate lowering candidate identity")
            identities.add(candidate.identity)
            collected.append(candidate)
    return tuple(collected)


def lowering_diagnostics(
    candidates: typing.Iterable[LoweringCandidate],
) -> dict[str, typing.Any]:
    """Return deterministic provider/candidate provenance without selecting a winner."""

    materialized = tuple(candidates)
    if any(not isinstance(candidate, LoweringCandidate) for candidate in materialized):
        raise TypeError("lowering diagnostics require LoweringCandidate records")
    ordered = sorted(materialized, key=lambda candidate: candidate.identity)
    payloads = [candidate.to_payload() for candidate in ordered]
    providers = sorted(
        {
            provider.name
            for candidate in materialized
            for provider in candidate.providers
        }
    )
    return {
        "schema": LOWERING_DIAGNOSTICS_SCHEMA,
        "identity": canonical_hash(payloads),
        "providers": providers,
        "requests": [
            candidate.request.to_payload()
            for candidate in sorted(
                {
                    candidate.request.identity: candidate for candidate in materialized
                }.values(),
                key=lambda candidate: candidate.request.identity,
            )
        ],
        "candidates": payloads,
    }
