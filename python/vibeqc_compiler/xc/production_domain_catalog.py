"""Deterministic production-domain admission summaries for bulk Libxc XC.

This module is reporting-only. It consumes the exact qualification profiles and
stage evidence owned by :mod:`libxc_bulk_capabilities`; it never turns a report
or count into production evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    available_capabilities,
)

SUMMARY_SCHEMA = "vibeqc.libxc-production-domain-summary.v1"
STATUS_ORDER = ("qualified", "ready", "blocked", "pending")


def _stage_blocker(capability: BulkFunctionalCapability) -> str | None:
    """Return an explicit production-domain blocker, if one is recorded."""
    profile = capability.production_domain_profile
    if not profile.eligible:
        return profile.blocker
    for item in capability.stage_evidence:
        if item.stage == "production-domain" and item.status != "pass":
            return item.reason or f"evidence-status:{item.status}"
    return None


def _status(capability: BulkFunctionalCapability) -> tuple[str, str | None]:
    """Classify one registration without inferring unrecorded admission."""
    if "production-domain" in capability.qualified_stages:
        return "qualified", None
    blocker = _stage_blocker(capability)
    if blocker is not None:
        return "blocked", blocker
    if "production-domain" in capability.ready_stages:
        return "ready", None
    return "pending", None


def _empty_counts() -> dict[str, int]:
    return {status: 0 for status in STATUS_ORDER}


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row[key]
        group = groups.setdefault(
            name,
            {"total": 0, "status_counts": _empty_counts()},
        )
        group["total"] += 1
        group["status_counts"][row["status"]] += 1
    return {name: groups[name] for name in sorted(groups)}


def production_domain_summary(
    evidence_by_functional: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize exact production-domain state by family and ingredient set.

    ``ready`` means the structural prerequisites are satisfied and the current
    boundary profile is eligible for evidence. It is deliberately distinct from
    ``qualified``. Explicit numerical failures and structurally unsupported
    ingredients remain visible as ``blocked`` records with stable reasons.
    """
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    for capability in sorted(
        available_capabilities(evidence_by_functional), key=lambda item: item.name
    ):
        status, blocker = _status(capability)
        ingredients = "+".join(capability.required_ingredients)
        row = {
            "name": capability.name,
            "family": capability.family,
            "ingredients": ingredients,
            "required_ingredients": list(capability.required_ingredients),
            "profile_identity": capability.production_domain_profile.identity,
            "status": status,
            "blocker": blocker,
        }
        rows.append(row)
        if blocker is not None:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

    status_counts = _empty_counts()
    for row in rows:
        status_counts[row["status"]] += 1

    return {
        "schema": SUMMARY_SCHEMA,
        "status_order": list(STATUS_ORDER),
        "total_functionals": len(rows),
        "status_counts": status_counts,
        "by_family": _group_summary(rows, "family"),
        "by_ingredients": _group_summary(rows, "ingredients"),
        "blocker_reasons": {
            reason: blocker_counts[reason] for reason in sorted(blocker_counts)
        },
        "functionals": {row["name"]: row for row in rows},
    }


def render_production_domain_summary(summary: Mapping[str, Any]) -> str:
    """Render a concise deterministic report from a generated summary."""
    if not isinstance(summary, Mapping) or summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("unsupported production-domain summary schema")
    if summary.get("status_order") != list(STATUS_ORDER):
        raise ValueError("production-domain summary status order mismatch")

    counts = summary.get("status_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(STATUS_ORDER):
        raise ValueError("production-domain summary has invalid status counts")
    total = summary.get("total_functionals")
    if type(total) is not int or sum(counts.values()) != total:
        raise ValueError("production-domain summary total does not match status counts")

    lines = [
        f"Libxc production-domain summary ({total} functionals)",
        "status: " + "  ".join(f"{status}={counts[status]}" for status in STATUS_ORDER),
    ]
    for label, field in (("family", "by_family"), ("ingredients", "by_ingredients")):
        groups = summary.get(field)
        if not isinstance(groups, Mapping):
            raise ValueError(f"production-domain summary has invalid {field}")
        lines.append(f"{label}:")
        for name in sorted(groups):
            group = groups[name]
            if not isinstance(group, Mapping):
                raise ValueError(f"production-domain summary has invalid {field} row")
            group_counts = group.get("status_counts")
            group_total = group.get("total")
            if (
                not isinstance(group_counts, Mapping)
                or set(group_counts) != set(STATUS_ORDER)
                or type(group_total) is not int
                or sum(group_counts.values()) != group_total
            ):
                raise ValueError(
                    f"production-domain summary has inconsistent {field} row"
                )
            lines.append(
                f"  {name}: total={group_total} "
                + " ".join(
                    f"{status}={group_counts[status]}" for status in STATUS_ORDER
                )
            )

    blockers = summary.get("blocker_reasons")
    if not isinstance(blockers, Mapping):
        raise ValueError("production-domain summary has invalid blocker inventory")
    if blockers:
        lines.append("blockers:")
        for reason in sorted(blockers):
            count = blockers[reason]
            if type(count) is not int or count <= 0:
                raise ValueError("production-domain summary has invalid blocker count")
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("blockers: none")
    return "\n".join(lines)
