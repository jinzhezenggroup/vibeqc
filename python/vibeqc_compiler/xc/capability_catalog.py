"""Deterministic summaries for evidence-backed bulk Libxc capability promotion.

This module is deliberately reporting-only. It consumes the qualification DAG
owned by :mod:`libxc_bulk_capabilities`; it never turns a summary, a count, or a
previous snapshot into new qualification evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .libxc_bulk_capabilities import (
    CAPABILITY_STAGES,
    BulkFunctionalCapability,
    available_capabilities,
)

SUMMARY_SCHEMA = "vibeqc.libxc-bulk-capability-summary.v1"
CHANGE_SCHEMA = "vibeqc.libxc-bulk-capability-changes.v1"


def _blocked_stages(capability: BulkFunctionalCapability) -> dict[str, str | None]:
    """Return only stages carrying explicit non-pass evidence."""
    return {
        item.stage: item.reason
        for item in capability.stage_evidence
        if item.status != "pass"
    }


def capability_summary(
    evidence_by_functional: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, detached qualification snapshot.

    Counts are observations of the existing evidence graph. In particular,
    branch-like stages such as CPU and CUDA compilation are not reinterpreted as
    an ordered ladder, and missing evidence never becomes an inferred pass.
    """
    capabilities = available_capabilities(evidence_by_functional)
    by_functional = {
        capability.name: {
            "identity": capability.identity,
            "qualified_stages": list(capability.qualified_stages),
            "ready_stages": list(capability.ready_stages),
            "blocked_stages": _blocked_stages(capability),
            "public_dft": capability.public_dft,
        }
        for capability in capabilities
    }
    qualified_counts = {
        stage: sum(stage in capability.qualified_stages for capability in capabilities)
        for stage in CAPABILITY_STAGES
    }
    ready_counts = {
        stage: sum(stage in capability.ready_stages for capability in capabilities)
        for stage in CAPABILITY_STAGES
    }
    blocked_counts = {
        stage: sum(stage in _blocked_stages(capability) for capability in capabilities)
        for stage in CAPABILITY_STAGES
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "stage_order": list(CAPABILITY_STAGES),
        "total_functionals": len(capabilities),
        "qualified_counts": qualified_counts,
        "ready_counts": ready_counts,
        "blocked_counts": blocked_counts,
        "functionals": by_functional,
    }


def _validate_summary(summary: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(summary, Mapping):
        raise TypeError(f"{label} capability summary must be a mapping")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"{label} capability summary has unsupported schema")
    if summary.get("stage_order") != list(CAPABILITY_STAGES):
        raise ValueError(f"{label} capability summary stage order mismatch")
    functionals = summary.get("functionals")
    if not isinstance(functionals, Mapping):
        raise TypeError(f"{label} capability summary functionals must be a mapping")
    if type(summary.get("total_functionals")) is not int or summary[
        "total_functionals"
    ] != len(functionals):
        raise ValueError(f"{label} capability summary total does not match inventory")

    stages = set(CAPABILITY_STAGES)
    for name, record in functionals.items():
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            raise TypeError(
                f"{label} capability summary has malformed functional record"
            )
        identity = record.get("identity")
        qualified = record.get("qualified_stages")
        ready = record.get("ready_stages")
        blocked = record.get("blocked_stages")
        if not isinstance(identity, str) or not identity:
            raise ValueError(
                f"{label} capability summary has invalid identity for {name}"
            )
        if not isinstance(qualified, list) or not isinstance(ready, list):
            raise TypeError(f"{label} capability summary has invalid stages for {name}")
        if not isinstance(blocked, Mapping):
            raise TypeError(
                f"{label} capability summary has invalid blockers for {name}"
            )
        if (
            not set(qualified) <= stages
            or not set(ready) <= stages
            or not set(blocked) <= stages
        ):
            raise ValueError(f"{label} capability summary has unknown stage for {name}")
        if len(set(qualified)) != len(qualified) or len(set(ready)) != len(ready):
            raise ValueError(
                f"{label} capability summary has duplicate stages for {name}"
            )
        if set(qualified) & set(blocked):
            raise ValueError(
                f"{label} capability summary has qualified/blocked overlap for {name}"
            )
        public = record.get("public_dft")
        if type(public) is not bool or public != ("public-method" in qualified):
            raise ValueError(
                f"{label} capability summary has invalid public flag for {name}"
            )
        if set(qualified) & set(ready):
            raise ValueError(
                f"{label} capability summary has qualified/ready overlap for {name}"
            )

    # Persisted summaries are redundant: every counter must agree with the
    # detailed inventory. Do not accept an internally contradictory snapshot.
    for field, inventory in (
        ("qualified_counts", "qualified_stages"),
        ("ready_counts", "ready_stages"),
        ("blocked_counts", "blocked_stages"),
    ):
        counts = summary.get(field)
        if not isinstance(counts, Mapping) or set(counts) != stages:
            raise ValueError(
                f"{label} capability summary has invalid {field} inventory"
            )
        for stage in CAPABILITY_STAGES:
            expected = sum(
                stage in record[inventory] for record in functionals.values()
            )
            if type(counts[stage]) is not int or counts[stage] != expected:
                raise ValueError(
                    f"{label} capability summary has inconsistent {field}: {stage}"
                )


def capability_changes(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two exact snapshots without promoting either snapshot.

    The result exposes promotions, demotions, inventory changes, and scientific
    identity changes separately. Consumers can therefore fail closed on a
    demotion or an identity change instead of preserving a stale public claim.
    """
    _validate_summary(previous, label="previous")
    _validate_summary(current, label="current")
    old = previous["functionals"]
    new = current["functionals"]
    old_names = set(old)
    new_names = set(new)
    common = sorted(old_names & new_names)

    promotions: dict[str, list[str]] = {}
    demotions: dict[str, list[str]] = {}
    for stage in CAPABILITY_STAGES:
        promoted = [
            name
            for name in common
            if stage not in old[name]["qualified_stages"]
            and stage in new[name]["qualified_stages"]
        ]
        demoted = [
            name
            for name in common
            if stage in old[name]["qualified_stages"]
            and stage not in new[name]["qualified_stages"]
        ]
        if promoted:
            promotions[stage] = promoted
        if demoted:
            demotions[stage] = demoted

    return {
        "schema": CHANGE_SCHEMA,
        "added_functionals": sorted(new_names - old_names),
        "removed_functionals": sorted(old_names - new_names),
        "identity_changes": [
            name for name in common if old[name]["identity"] != new[name]["identity"]
        ],
        "promotions": promotions,
        "demotions": demotions,
    }
