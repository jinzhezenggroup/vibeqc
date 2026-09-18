"""Pure workload guards and selection over existing artifact/profile identities.

This is a selection contract, not another cache or executable loader. Consumers
supply verified scientific/compiler/profile hashes, capability facts and ordered
profiles; existing loaders still verify the selected artifact key and binary.
No method policy, empirical thresholds, device probing or compilation lives here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .backend import TargetInfo
from .provenance import canonical_hash

FeatureValue = bool | int | float | str
Features = tuple[tuple[str, FeatureValue], ...]
_SCHEMA = "vibeqc.specialization.v1"


def _name(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _digest(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _scalar(value: FeatureValue) -> None:
    if type(value) not in (bool, int, float, str) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError("features require finite immutable JSON scalars")


def _features(values: Features, reserved: set[str]) -> Features:
    result = []
    names = set(reserved)
    for pair in values:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("features must be name/value pairs")
        name, value = pair
        _name(name, "feature name")
        if name in names:
            raise ValueError(f"duplicate or reserved feature {name!r}")
        _scalar(value)
        names.add(name)
        result.append((name, value))
    return tuple(sorted(result))


class _TypedRecord:
    """Keep record equality consistent with type-sensitive JSON identities."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return canonical_hash(asdict(self)) == canonical_hash(asdict(other))

    def __hash__(self) -> int:
        return hash((type(self), canonical_hash(asdict(self))))


@dataclass(frozen=True, slots=True, eq=False)
class WorkloadSignature(_TypedRecord):
    """Named consumer and general workload facts, not a benchmark endpoint ID.

    Omit unknown facts. Input pairs are copied and sorted; mutable/nested values
    and duplicate names are rejected so a frozen record is genuinely immutable.
    """

    kind: str
    features: Features = ()

    def __post_init__(self) -> None:
        _name(self.kind, "workload kind")
        object.__setattr__(self, "features", _features(self.features, {"kind"}))


@dataclass(frozen=True, slots=True, eq=False)
class TargetCapabilities(_TypedRecord):
    """Reuse TargetInfo plus explicit feature/resource facts, without probing.

    Product name, UUID and measurement provenance stay with the existing device
    and artifact records, not the capability predicates. Unknown facts are omitted
    (or remain None in TargetInfo); they never satisfy a required capability.
    """

    target: TargetInfo
    features: Features = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, TargetInfo):
            raise TypeError("target must be a TargetInfo")
        _name(self.target.backend, "target backend")
        _name(self.target.architecture, "target architecture")
        object.__setattr__(
            self, "features", _features(self.features, set(asdict(self.target)))
        )


@dataclass(frozen=True, slots=True)
class CompilationIdentity:
    """References to existing identity owners, not replacement hash recipes.

    scientific_hash includes method/precision/output intent. compiler_hash must
    cover relevant source, IR/generator/ABI versions, toolchain and compile options
    as defined by the consumer's existing compilation identity. Schedule/profile
    and executable hashes are carried separately by ImplementationProfile.
    """

    scientific_hash: str
    compiler_hash: str

    def __post_init__(self) -> None:
        _digest(self.scientific_hash, "scientific_hash")
        _digest(self.compiler_hash, "compiler_hash")


@dataclass(frozen=True, slots=True, eq=False)
class GuardPredicate(_TypedRecord):
    """One declarative equality or inclusive bound; no callables or eval.

    Equality is type-sensitive (True, 1 and 1.0 are distinct). Bounds accept
    finite int/float facts, never booleans or strings. Missing facts fail closed.
    """

    scope: Literal["workload", "target"]
    feature: str
    operator: Literal["eq", "ge", "le"]
    value: FeatureValue

    def __post_init__(self) -> None:
        if self.scope not in ("workload", "target"):
            raise ValueError("unknown guard scope")
        _name(self.feature, "guard feature")
        if self.operator not in ("eq", "ge", "le"):
            raise ValueError("unknown guard operator")
        _scalar(self.value)
        if self.operator != "eq" and type(self.value) not in (int, float):
            raise ValueError("guard bounds require a number, not a bool or string")

    def failure(self, facts: dict) -> str | None:
        actual = facts[self.scope].get(self.feature)
        label = f"{self.scope}.{self.feature}"
        if actual is None:
            return f"{label}: missing required fact"
        try:
            _scalar(actual)
        except ValueError:
            return f"{label}: invalid non-scalar or non-finite fact"
        if self.operator == "eq":
            matches = type(actual) is type(self.value) and actual == self.value
        elif type(actual) not in (int, float):
            matches = False
        elif self.operator == "ge":
            matches = actual >= self.value
        else:
            matches = actual <= self.value
        if not matches:
            return f"{label}: expected {self.operator} {self.value!r}, got {actual!r}"
        return None


@dataclass(frozen=True, slots=True)
class SpecializationGuard:
    """A canonical conjunction; an explicit empty guard has no restrictions."""

    predicates: tuple[GuardPredicate, ...] = ()

    def __post_init__(self) -> None:
        predicates = tuple(self.predicates)
        if any(not isinstance(p, GuardPredicate) for p in predicates):
            raise ValueError("guard predicates must be GuardPredicate records")
        predicates = tuple(
            sorted(
                predicates,
                key=lambda p: (p.scope, p.feature, p.operator, canonical_hash(p.value)),
            )
        )
        object.__setattr__(self, "predicates", predicates)

    def failures(self, facts: dict) -> tuple[str, ...]:
        return tuple(
            failure
            for p in self.predicates
            if (failure := p.failure(facts)) is not None
        )


@dataclass(frozen=True, slots=True)
class ImplementationProfile:
    """An existing implementation and independent correctness/promotion guards.

    None means *not promoted*, not an unrestricted promotion. Supplying a
    performance guard asserts that the consumer has qualified that domain;
    this module neither creates evidence nor promotes an experimental profile.
    artifact_key is the original owner's key and is returned unchanged.
    """

    name: str
    identity: CompilationIdentity
    artifact_key: str
    schedule_hash: str
    profile_hash: str
    correctness: SpecializationGuard
    performance: SpecializationGuard | None = None

    def __post_init__(self) -> None:
        _name(self.name, "profile name")
        if not isinstance(self.identity, CompilationIdentity):
            raise TypeError("profile requires a CompilationIdentity")
        for label in ("artifact_key", "schedule_hash", "profile_hash"):
            _digest(getattr(self, label), label)
        if not isinstance(self.correctness, SpecializationGuard) or (
            self.performance is not None
            and not isinstance(self.performance, SpecializationGuard)
        ):
            raise ValueError("profile requires explicit specialization guards")


@dataclass(frozen=True, slots=True)
class ProfileEvaluation:
    """Correctness and performance results remain separately inspectable."""

    name: str
    eligibility_failures: tuple[str, ...]
    promotion_failures: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.eligibility_failures

    @property
    def promoted(self) -> bool:
        return not self.promotion_failures


@dataclass(frozen=True, slots=True)
class SpecializationDecision:
    """Selected original artifact and deterministic, provenance-visible reasons."""

    selected: ImplementationProfile | None
    evaluations: tuple[ProfileEvaluation, ...]
    fallback: ProfileEvaluation
    selection_key: str

    @property
    def status(self) -> Literal["specialized", "fallback", "unsupported"]:
        if self.selected is None:
            return "unsupported"
        return "fallback" if self.selected.name == self.fallback.name else "specialized"

    def to_payload(self) -> dict:
        """A detached JSON record; selection_key is not an executable cache key."""
        return {
            "schema": _SCHEMA,
            "selection_key": self.selection_key,
            "status": self.status,
            "selected_profile": self.selected.name if self.selected else None,
            "artifact_key": self.selected.artifact_key if self.selected else None,
            "implementation": asdict(self.selected) if self.selected else None,
            "evaluations": [asdict(e) for e in self.evaluations],
            "fallback": asdict(self.fallback),
        }


def select_specialization(
    *,
    workload: WorkloadSignature,
    target: TargetCapabilities,
    identity: CompilationIdentity,
    profiles: Sequence[ImplementationProfile],
    fallback: ImplementationProfile,
) -> SpecializationDecision:
    """Choose the first eligible promoted profile, or a correctness-checked fallback.

    Caller order is explicit priority and participates in selection identity.
    Unknown workloads/devices can use the supplied generic implementation only
    when its own identity and correctness guard pass. No implicit CPU fallback,
    binary loading, compilation or on-disk cache mutation is performed.
    """
    if (
        not isinstance(workload, WorkloadSignature)
        or not isinstance(target, TargetCapabilities)
        or not isinstance(identity, CompilationIdentity)
    ):
        raise TypeError(
            "selection requires typed workload, target and identity records"
        )
    profiles = tuple(profiles)
    if any(not isinstance(p, ImplementationProfile) for p in (*profiles, fallback)):
        raise ValueError("selection requires ImplementationProfile records")
    names = [p.name for p in (*profiles, fallback)]
    if len(names) != len(set(names)):
        raise ValueError("profile names, including the fallback, must be unique")
    facts = {
        "workload": {"kind": workload.kind, **dict(workload.features)},
        "target": {**asdict(target.target), **dict(target.features)},
    }

    def evaluate(profile: ImplementationProfile) -> ProfileEvaluation:
        mismatches = tuple(
            f"identity.{field}: mismatch"
            for field in ("scientific_hash", "compiler_hash")
            if getattr(profile.identity, field) != getattr(identity, field)
        )
        return ProfileEvaluation(
            profile.name,
            mismatches + profile.correctness.failures(facts),
            ("profile is not promoted",)
            if profile.performance is None
            else profile.performance.failures(facts),
        )

    evaluations = tuple(evaluate(p) for p in profiles)
    fallback_evaluation = evaluate(fallback)
    selected = next(
        (p for p, e in zip(profiles, evaluations) if e.eligible and e.promoted), None
    )
    if selected is None and fallback_evaluation.eligible:
        selected = fallback
    key = canonical_hash(
        {
            "schema": _SCHEMA,
            "workload": asdict(workload),
            "target": asdict(target),
            "identity": asdict(identity),
            "profiles": [asdict(p) for p in profiles],
            "fallback": asdict(fallback),
        }
    )
    return SpecializationDecision(selected, evaluations, fallback_evaluation, key)
