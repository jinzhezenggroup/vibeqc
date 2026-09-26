"""Shared schedule identity, resource, profitability, and provenance contract.

Domain IRs retain their own legal schedule primitives. This module only
normalizes execution facts needed by tuning, profile binding, and
cross-consumer diagnostics, so adding a new consumer does not require another
autotune/profile vocabulary.
"""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass

from .gpu_profitability import ENDPOINT_NOISE_FRACTION, GpuProfitability
from .provenance import canonical_hash

SCHEDULE_CONTRACT_SCHEMA = "vibeqc.compiler.schedule-contract.v1"
SCHEDULE_DIAGNOSTICS_SCHEMA = "vibeqc.compiler.schedule-diagnostics.v1"


def _text(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value: typing.Any, label: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest or None")
    return value


def _optional_count(value: typing.Any, label: str, *, positive: bool = False) -> None:
    if value is None:
        return
    lower = 1 if positive else 0
    if type(value) is not int or value < lower:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer or None")


@dataclass(frozen=True, slots=True)
class ScheduleTopology:
    """Portable execution choices; unsupported dimensions remain explicit None."""

    tiles: tuple[int, ...] = ()
    workgroup_threads: int | None = None
    subgroup_size: int | None = None
    fusion: str = "unfused"
    materialization: str = "materialize"
    residency: str = "streamed"
    staging_width: int | None = None
    reduction: str = "default"
    persistent: bool = False
    cooperative: bool = False
    page_size: int | None = None
    bucket: str | None = None

    def __post_init__(self) -> None:
        tiles = tuple(self.tiles)
        if any(type(value) is not int or value < 1 for value in tiles):
            raise ValueError("schedule tiles must be positive integers")
        object.__setattr__(self, "tiles", tiles)
        _optional_count(self.workgroup_threads, "workgroup_threads", positive=True)
        _optional_count(self.subgroup_size, "subgroup_size", positive=True)
        _optional_count(self.staging_width, "staging_width", positive=True)
        _optional_count(self.page_size, "page_size", positive=True)
        for label in ("fusion", "materialization", "residency", "reduction"):
            _text(getattr(self, label), label)
        if self.bucket is not None:
            _text(self.bucket, "bucket")
        if type(self.persistent) is not bool or type(self.cooperative) is not bool:
            raise TypeError("persistent/cooperative schedule flags must be boolean")

    def to_payload(self) -> dict[str, typing.Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScheduleResources:
    """Comparable planned/compiled resource facts without inventing unknowns."""

    device_bytes: int | None = None
    host_bytes: int | None = None
    workspace_bytes: int | None = None
    peak_live_values: int | None = None
    registers_per_thread: int | None = None
    shared_bytes: int | None = None
    resident_workgroups: int | None = None
    source_bytes: int | None = None

    def __post_init__(self) -> None:
        for label, value in asdict(self).items():
            _optional_count(value, label)

    def to_payload(self) -> dict[str, typing.Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScheduleResourceLimits:
    """Shared fail-closed resource limits for schedule admission.

    Domain schedulers own scientific legality and resource estimation.  This
    record only centralizes comparable compiler resource bounds so consumers do
    not reimplement the same maximum/minimum checks with local policy.
    """

    maximum_device_bytes: int | None = None
    maximum_host_bytes: int | None = None
    maximum_workspace_bytes: int | None = None
    maximum_peak_live_values: int | None = None
    maximum_registers_per_thread: int | None = None
    maximum_shared_bytes: int | None = None
    maximum_source_bytes: int | None = None
    minimum_resident_workgroups: int | None = None

    def __post_init__(self) -> None:
        for label in (
            "maximum_device_bytes",
            "maximum_host_bytes",
            "maximum_workspace_bytes",
            "maximum_peak_live_values",
            "maximum_registers_per_thread",
            "maximum_shared_bytes",
            "maximum_source_bytes",
        ):
            _optional_count(getattr(self, label), label)
        _optional_count(
            self.minimum_resident_workgroups,
            "minimum_resident_workgroups",
            positive=True,
        )


def schedule_resource_rejections(
    resources: ScheduleResources,
    limits: ScheduleResourceLimits,
) -> tuple[str, ...]:
    """Return deterministic shared resource-limit failures.

    A requested limit requires the corresponding resource fact.  Unknown facts
    fail closed rather than silently passing a finite admission bound.
    """

    if not isinstance(resources, ScheduleResources):
        raise TypeError("schedule resource admission requires ScheduleResources")
    if not isinstance(limits, ScheduleResourceLimits):
        raise TypeError("schedule resource admission requires ScheduleResourceLimits")
    reasons: list[str] = []
    maxima = (
        ("device_bytes", "maximum_device_bytes", "device bytes"),
        ("host_bytes", "maximum_host_bytes", "host bytes"),
        ("workspace_bytes", "maximum_workspace_bytes", "workspace bytes"),
        ("peak_live_values", "maximum_peak_live_values", "peak live values"),
        (
            "registers_per_thread",
            "maximum_registers_per_thread",
            "registers per thread",
        ),
        ("shared_bytes", "maximum_shared_bytes", "shared bytes"),
        ("source_bytes", "maximum_source_bytes", "source bytes"),
    )
    for resource_name, limit_name, label in maxima:
        limit = getattr(limits, limit_name)
        if limit is None:
            continue
        value = getattr(resources, resource_name)
        if value is None:
            reasons.append(f"{label} unavailable for required limit {limit}")
        elif value > limit:
            reasons.append(f"{label} {value} exceeds limit {limit}")
    minimum = limits.minimum_resident_workgroups
    if minimum is not None:
        value = resources.resident_workgroups
        if value is None:
            reasons.append(
                f"resident workgroups unavailable for required minimum {minimum}"
            )
        elif value < minimum:
            reasons.append(f"resident workgroups {value} below minimum {minimum}")
    return tuple(reasons)


def _profitability_from_payload(payload: typing.Any) -> GpuProfitability:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "vibeqc.compiler.gpu-profitability.v1"
    ):
        raise ValueError("invalid shared profitability payload")
    if set(payload) != {"schema", "static", "compiled", "endpoint_seconds"}:
        raise ValueError("invalid shared profitability fields")
    static = payload["static"]
    compiled = payload["compiled"]
    if not isinstance(static, dict) or not isinstance(compiled, dict):
        raise TypeError("profitability static/compiled records must be mappings")
    return GpuProfitability(
        **static,
        **compiled,
        endpoint_seconds=payload["endpoint_seconds"],
    )


@dataclass(frozen=True, slots=True)
class ScheduleContract:
    """One domain schedule projected into the shared compiler vocabulary.

    schedule_hash remains the owning subsystem's schedule identity. The
    contract identity additionally includes workload/profile/resource evidence;
    it never replaces an artifact key or specialization/profile identity owner.
    """

    consumer: str
    schedule_hash: str
    topology: ScheduleTopology
    resources: ScheduleResources
    profitability: GpuProfitability
    workload_hash: str | None = None
    profile_key: str | None = None
    target_hash: str | None = None
    precision_schedule_hash: str | None = None
    fallback: bool = False
    legal: bool = True
    reasons: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _text(self.consumer, "schedule consumer")
        _text(self.schedule_hash, "schedule_hash")
        _digest(self.schedule_hash, "schedule_hash")
        for label in (
            "workload_hash",
            "profile_key",
            "target_hash",
            "precision_schedule_hash",
        ):
            _digest(getattr(self, label), label)
        if not isinstance(self.topology, ScheduleTopology):
            raise TypeError("schedule contract requires ScheduleTopology")
        if not isinstance(self.resources, ScheduleResources):
            raise TypeError("schedule contract requires ScheduleResources")
        if not isinstance(self.profitability, GpuProfitability):
            raise TypeError("schedule contract requires GpuProfitability")
        if type(self.fallback) is not bool or type(self.legal) is not bool:
            raise TypeError("fallback/legal schedule flags must be boolean")
        reasons = tuple(self.reasons)
        if any(type(reason) is not str or not reason for reason in reasons):
            raise ValueError("schedule rejection reasons must be nonempty strings")
        if self.legal and reasons:
            raise ValueError("legal schedule contract cannot carry rejection reasons")
        if not self.legal and not reasons:
            raise ValueError("illegal schedule contract requires rejection reasons")
        object.__setattr__(self, "reasons", reasons)
        provenance = tuple(self.provenance)
        names: set[str] = set()
        normalized = []
        for pair in provenance:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("schedule provenance must be name/value pairs")
            name, value = pair
            _text(name, "provenance name")
            _text(value, "provenance value")
            if name in names:
                raise ValueError(f"duplicate schedule provenance {name!r}")
            names.add(name)
            normalized.append((name, value))
        object.__setattr__(self, "provenance", tuple(sorted(normalized)))

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": SCHEDULE_CONTRACT_SCHEMA,
            "consumer": self.consumer,
            "schedule_hash": self.schedule_hash,
            "workload_hash": self.workload_hash,
            "profile_key": self.profile_key,
            "target_hash": self.target_hash,
            "precision_schedule_hash": self.precision_schedule_hash,
            "fallback": self.fallback,
            "legal": self.legal,
            "reasons": list(self.reasons),
            "topology": self.topology.to_payload(),
            "resources": self.resources.to_payload(),
            "profitability": self.profitability.to_payload(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: typing.Any) -> ScheduleContract:
        expected = {
            "schema",
            "consumer",
            "schedule_hash",
            "workload_hash",
            "profile_key",
            "target_hash",
            "precision_schedule_hash",
            "fallback",
            "legal",
            "reasons",
            "topology",
            "resources",
            "profitability",
            "provenance",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema") != SCHEDULE_CONTRACT_SCHEMA
        ):
            raise ValueError("invalid shared schedule contract payload")
        topology = payload["topology"]
        resources = payload["resources"]
        provenance = payload["provenance"]
        if not isinstance(topology, dict) or set(topology) != set(
            ScheduleTopology.__dataclass_fields__
        ):
            raise ValueError("invalid shared schedule topology payload")
        if not isinstance(resources, dict) or set(resources) != set(
            ScheduleResources.__dataclass_fields__
        ):
            raise ValueError("invalid shared schedule resource payload")
        if not isinstance(provenance, dict):
            raise TypeError("schedule provenance payload must be a mapping")
        return cls(
            consumer=payload["consumer"],
            schedule_hash=payload["schedule_hash"],
            workload_hash=payload["workload_hash"],
            profile_key=payload["profile_key"],
            target_hash=payload["target_hash"],
            precision_schedule_hash=payload["precision_schedule_hash"],
            fallback=payload["fallback"],
            legal=payload["legal"],
            reasons=tuple(payload["reasons"]),
            topology=ScheduleTopology(**topology),
            resources=ScheduleResources(**resources),
            profitability=_profitability_from_payload(payload["profitability"]),
            provenance=tuple(provenance.items()),
        )


def rank_schedule_contracts(
    contracts: typing.Iterable[ScheduleContract], maximum: int | None = None
) -> tuple[ScheduleContract, ...]:
    """Order legal schedules with the same profitability policy for every consumer.

    This only allocates a finite investigation/compile budget. It is not a
    performance promotion decision and never replaces a subsystem's fallback.
    """

    materialized = tuple(contracts)
    if any(not isinstance(contract, ScheduleContract) for contract in materialized):
        raise TypeError("schedule ranking requires ScheduleContract records")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("schedule ranking limit must be positive or None")
    ranked = sorted(
        (
            (contract.profitability.static_compile_priority(index), index, contract)
            for index, contract in enumerate(materialized)
            if contract.legal
        ),
        key=lambda row: (row[0], row[1]),
    )
    if maximum is not None:
        ranked = ranked[:maximum]
    return tuple(contract for _, _, contract in ranked)



def select_measured_schedule_contract(
    contracts: typing.Iterable[ScheduleContract],
    *,
    minimum_speedup: float,
    endpoint_noise_fraction: float = ENDPOINT_NOISE_FRACTION,
) -> ScheduleContract:
    """Select one measured legal schedule, failing closed to the declared fallback.

    Domain owners still define scientific legality and the measurement protocol.
    This shared layer only compares schedules for the same consumer/workload/
    target/precision identity. A candidate without complete endpoint timing is
    never promoted. Measured candidates must clear both the caller's speedup gate
    and the shared resource-regression gate; schedules inside the timing-noise
    band are ordered by compiled resource pressure.
    """

    materialized = tuple(contracts)
    if not materialized:
        raise ValueError("measured schedule selection requires candidates")
    if any(not isinstance(contract, ScheduleContract) for contract in materialized):
        raise TypeError("measured schedule selection requires ScheduleContract records")
    for value, label, lower, upper in (
        (minimum_speedup, "minimum_speedup", 1.0, None),
        (
            endpoint_noise_fraction,
            "endpoint_noise_fraction",
            0.0,
            1.0,
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < lower or (
            upper is not None and numeric > upper
        ):
            if upper is None:
                raise ValueError(f"{label} must be finite and at least {lower}")
            raise ValueError(f"{label} must be finite and in [{lower}, {upper}]")

    fallbacks = tuple(contract for contract in materialized if contract.fallback)
    if len(fallbacks) != 1:
        raise ValueError("measured schedule selection requires exactly one fallback")
    baseline = fallbacks[0]
    if not baseline.legal:
        raise ValueError("measured schedule fallback must be legal")

    comparison_identity = (
        baseline.consumer,
        baseline.workload_hash,
        baseline.profile_key,
        baseline.target_hash,
        baseline.precision_schedule_hash,
    )
    for contract in materialized:
        if not contract.legal:
            continue
        candidate_identity = (
            contract.consumer,
            contract.workload_hash,
            contract.profile_key,
            contract.target_hash,
            contract.precision_schedule_hash,
        )
        if candidate_identity != comparison_identity:
            raise ValueError(
                "measured schedule candidates must share consumer/workload/profile/"
                "target/precision identity"
            )

    if baseline.profitability.endpoint_seconds is None:
        return baseline

    admitted: list[ScheduleContract] = []
    for contract in materialized:
        if contract is baseline or not contract.legal:
            continue
        if contract.profitability.endpoint_seconds is None:
            continue
        if contract.profitability.endpoint_regressions_against(
            baseline.profitability,
            minimum_speedup=float(minimum_speedup),
        ):
            continue
        if contract.profitability.resource_regressions_against(
            baseline.profitability,
            endpoint_noise_fraction=float(endpoint_noise_fraction),
        ):
            continue
        admitted.append(contract)

    if not admitted:
        return baseline

    fastest = min(
        typing.cast(float, contract.profitability.endpoint_seconds)
        for contract in admitted
    )
    tied = tuple(
        contract
        for contract in admitted
        if typing.cast(float, contract.profitability.endpoint_seconds)
        <= fastest * (1.0 + float(endpoint_noise_fraction))
    )
    return min(
        tied,
        key=lambda contract: contract.profitability.compiled_resource_priority(),
    )


def schedule_diagnostics(
    contracts: typing.Iterable[ScheduleContract],
) -> dict[str, typing.Any]:
    """Emit one comparison vocabulary across heterogeneous compiler consumers."""

    materialized = tuple(contracts)
    if not materialized:
        raise ValueError("schedule diagnostics require at least one contract")
    ranked = rank_schedule_contracts(materialized)
    return {
        "schema": SCHEDULE_DIAGNOSTICS_SCHEMA,
        "consumers": sorted({contract.consumer for contract in materialized}),
        "contracts": [contract.to_payload() for contract in materialized],
        "static_order": [contract.identity for contract in ranked],
        "fallbacks": [
            contract.identity for contract in materialized if contract.fallback
        ],
    }
