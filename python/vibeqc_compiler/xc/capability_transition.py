"""Fail-closed transition checks for persisted Libxc capability snapshots.

This module does not create qualification evidence.  It only classifies changes
between two already validated capability snapshots and requires regressions to
be acknowledged explicitly before a CI/update workflow can accept them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vibeqc_compiler.common.provenance import canonical_hash

from .capability_catalog import capability_changes
from .libxc_bulk_capabilities import CAPABILITY_STAGES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

RegressionKind = Literal["removed-functional", "identity-change", "stage-demotion"]


@dataclass(frozen=True, slots=True)
class CapabilityRegression:
    """One exact capability regression that requires explicit acknowledgement."""

    kind: RegressionKind
    functional: str
    transition_identity: str
    stage: str | None = None

    @property
    def token(self) -> str:
        """Bind a readable CI token to this functional's exact transition."""
        prefix = (
            f"{self.kind}:{self.functional}"
            if self.stage is None
            else f"{self.kind}:{self.stage}:{self.functional}"
        )
        return f"{prefix}:{self.transition_identity}"


def _transition_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize validated stage sets without binding unrelated catalog entries."""
    if record is None:
        return None
    return {
        "identity": record["identity"],
        "qualified_stages": sorted(record["qualified_stages"]),
        "ready_stages": sorted(record["ready_stages"]),
        "blocked_stages": dict(record["blocked_stages"]),
        "public_dft": record["public_dft"],
    }


def capability_regressions(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[CapabilityRegression, ...]:
    """Return deterministic regressions between two validated snapshots.

    Promotions and newly added functionals are not regressions.  Removing a
    functional, changing its exact scientific identity, or losing a qualified
    stage is surfaced independently so callers cannot silently carry stale
    claims across revisions.
    """
    changes = capability_changes(previous, current)

    def regression(
        kind: RegressionKind, name: str, stage: str | None = None
    ) -> CapabilityRegression:
        identity = canonical_hash(
            {
                "schema": "vibeqc.libxc-capability-transition.v1",
                "functional": name,
                "previous": _transition_record(previous["functionals"].get(name)),
                "current": _transition_record(current["functionals"].get(name)),
            }
        )
        return CapabilityRegression(kind, name, identity, stage)

    regressions = [
        regression("removed-functional", name)
        for name in changes["removed_functionals"]
    ]
    regressions.extend(
        regression("identity-change", name)
        for name in changes["identity_changes"]
    )
    for stage in CAPABILITY_STAGES:
        regressions.extend(
            regression("stage-demotion", name, stage)
            for name in changes["demotions"].get(stage, ())
        )
    return tuple(regressions)


def require_acknowledged_capability_regressions(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    acknowledged: Iterable[str] = (),
) -> tuple[CapabilityRegression, ...]:
    """Reject unacknowledged capability regressions and stale acknowledgements.

    Acknowledgements are exact tokens from :attr:`CapabilityRegression.token`,
    bound to the functional's previous/current identity and qualification state.
    A token for one revision transition cannot acknowledge a later transition.
    Unknown or duplicate tokens are rejected so a broad/stale CI exception
    cannot silently survive after the underlying transition changes.
    """
    regressions = capability_regressions(previous, current)
    expected = {item.token for item in regressions}

    supplied_list = list(acknowledged)
    if any(not isinstance(token, str) or not token for token in supplied_list):
        raise TypeError(
            "capability regression acknowledgements must be non-empty strings"
        )
    if len(set(supplied_list)) != len(supplied_list):
        raise ValueError("duplicate capability regression acknowledgement")

    supplied = set(supplied_list)
    stale = sorted(supplied - expected)
    if stale:
        raise ValueError(
            "stale or unknown capability regression acknowledgement: "
            + ", ".join(stale)
        )

    missing = sorted(expected - supplied)
    if missing:
        raise ValueError("unacknowledged capability regression: " + ", ".join(missing))
    return regressions
