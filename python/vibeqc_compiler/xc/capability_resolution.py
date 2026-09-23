"""Fail-closed resolution for evidence-backed bulk Libxc capabilities.

The resolver deliberately consumes the existing capability owner instead of
creating a second support registry. A successful resolution records the exact
scientific/source identity together with the exact capability stages requested
by the caller. Missing, ready-only, or blocked stages fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .libxc_bulk_capabilities import (
    CAPABILITY_STAGES,
    STAGE_REQUIREMENTS,
    BulkFunctionalCapability,
    functional_capability,
)

RESOLUTION_SCHEMA = "vibeqc.libxc-capability-resolution.v1"


@dataclass(frozen=True)
class CapabilityResolution:
    """One exact, successfully qualified capability request."""

    name: str
    identity: str
    required_stages: tuple[str, ...]
    qualified_stages: tuple[str, ...]
    public_dft: bool

    def to_payload(self) -> dict[str, Any]:
        """Return a detached machine-readable resolution record."""
        return {
            "schema": RESOLUTION_SCHEMA,
            "name": self.name,
            "identity": self.identity,
            "required_stages": list(self.required_stages),
            "qualified_stages": list(self.qualified_stages),
            "public_dft": self.public_dft,
        }


class CapabilityNotQualified(LookupError):
    """Raised when an exact capability request lacks qualified evidence."""

    def __init__(
        self,
        *,
        name: str,
        identity: str,
        blockers: tuple[tuple[str, str], ...],
    ) -> None:
        self.name = name
        self.identity = identity
        self.blockers = blockers
        self.missing_stages = tuple(stage for stage, _ in blockers)
        details = "; ".join(f"{stage}: {reason}" for stage, reason in blockers)
        super().__init__(f"{name} is not qualified for requested stages ({details})")


def _normalize_required_stages(required_stages: Sequence[str]) -> tuple[str, ...]:
    if isinstance(required_stages, (str, bytes)) or not isinstance(
        required_stages, Sequence
    ):
        raise TypeError("required_stages must be a sequence of capability stage names")
    requested = tuple(required_stages)
    if not requested:
        raise ValueError("required_stages must not be empty")
    if any(not isinstance(stage, str) or not stage for stage in requested):
        raise TypeError("required_stages must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("required_stages must not contain duplicates")
    unknown = sorted(set(requested) - set(CAPABILITY_STAGES))
    if unknown:
        raise ValueError(f"unknown capability stages: {unknown!r}")
    requested_set = set(requested)
    return tuple(stage for stage in CAPABILITY_STAGES if stage in requested_set)


def _missing_stage_reason(
    capability: BulkFunctionalCapability,
    stage: str,
) -> str:
    evidence_by_stage = {item.stage: item for item in capability.stage_evidence}
    evidence = evidence_by_stage.get(stage)
    if evidence is not None and evidence.status != "pass":
        return evidence.reason or f"{evidence.status} evidence"

    if stage in capability.ready_stages:
        return "missing pass evidence"

    qualified = set(capability.qualified_stages)
    unmet = []
    for alternatives in STAGE_REQUIREMENTS[stage]:
        if not any(required in qualified for required in alternatives):
            unmet.append(" or ".join(alternatives))
    if unmet:
        return f"unmet prerequisites: {'; '.join(unmet)}"
    return "not qualified"


def resolve_capability(
    name: str,
    *,
    required_stages: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
) -> CapabilityResolution:
    """Resolve one functional only when every requested stage is qualified.

    Stage requests are exact. For example, ``compiled-cpu`` evidence does not
    satisfy ``compiled-cuda`` and a ready stage does not count as qualified.
    """
    required = _normalize_required_stages(required_stages)
    capability = functional_capability(name, evidence=evidence)
    qualified = set(capability.qualified_stages)
    missing = tuple(stage for stage in required if stage not in qualified)
    if missing:
        raise CapabilityNotQualified(
            name=capability.name,
            identity=capability.identity,
            blockers=tuple(
                (stage, _missing_stage_reason(capability, stage)) for stage in missing
            ),
        )
    return CapabilityResolution(
        name=capability.name,
        identity=capability.identity,
        required_stages=required,
        qualified_stages=capability.qualified_stages,
        public_dft=capability.public_dft,
    )
