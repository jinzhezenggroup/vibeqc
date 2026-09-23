"""Validate the historical compiler optimization ownership ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "compiler_historical_optimization_ledger.json"

_SCHEMA = "vibeqc.compiler-historical-optimization-ledger"
_REQUIRED_GROUPS = {
    "active-domain-pruning",
    "specialization-workspace",
    "multi-output-fusion",
    "persistent-resident-schedules",
    "precision-placement",
    "data-residency-lifetime",
    "dft-tile-batch-fusion",
    "runtime-controller-boundary",
}
_ALLOWED_OWNER_KINDS = {
    "compiler",
    "schedule",
    "runtime",
    "numerical-policy",
    "method-planner",
    "evidence",
}
_ALLOWED_DECISIONS = {"rejected", "retained-negative"}
_COMMIT = re.compile(r"^[0-9a-f]{8,40}$")
_ISSUE = re.compile(r"^#[1-9][0-9]*$")


class LedgerError(ValueError):
    """Raised when the ownership ledger is incomplete or inconsistent."""


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{label} must be a non-empty string")
    return value


def _require_string_list(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise LedgerError(f"{label} must be {qualifier}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise LedgerError(f"{label} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise LedgerError(f"{label} contains duplicate entries")
    return value


def _validate_commits(value: Any, label: str, *, allow_empty: bool = False) -> None:
    commits = _require_string_list(value, label, allow_empty=allow_empty)
    for commit in commits:
        if not _COMMIT.fullmatch(commit):
            raise LedgerError(f"{label} contains invalid commit identity {commit!r}")


def _validate_issue_refs(value: Any, label: str) -> None:
    refs = _require_string_list(value, label)
    for ref in refs:
        if not _ISSUE.fullmatch(ref):
            raise LedgerError(f"{label} contains invalid issue reference {ref!r}")


def validate(payload: Any) -> dict[str, int]:
    """Validate *payload* and return deterministic ledger counts."""
    if not isinstance(payload, dict):
        raise LedgerError("ledger root must be an object")
    if payload.get("schema") != _SCHEMA:
        raise LedgerError("unknown ledger schema")
    if payload.get("schema_version") != 1:
        raise LedgerError("unsupported ledger schema_version")
    _require_text(payload.get("coverage"), "coverage")

    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise LedgerError("groups must be a list")

    seen: set[str] = set()
    mechanism_count = 0
    for index, group in enumerate(groups):
        label = f"groups[{index}]"
        if not isinstance(group, dict):
            raise LedgerError(f"{label} must be an object")
        group_id = _require_text(group.get("id"), f"{label}.id")
        if group_id in seen:
            raise LedgerError(f"duplicate group id {group_id!r}")
        seen.add(group_id)
        _require_text(group.get("name"), f"{label}.name")
        _validate_commits(
            group.get("historical_commits"), f"{label}.historical_commits"
        )
        mechanisms = _require_string_list(
            group.get("mechanisms"), f"{label}.mechanisms"
        )
        mechanism_count += len(mechanisms)
        owners = group.get("current_owners")
        if not isinstance(owners, list) or not owners:
            raise LedgerError(f"{label}.current_owners must be a non-empty list")
        for owner_index, owner in enumerate(owners):
            owner_label = f"{label}.current_owners[{owner_index}]"
            if not isinstance(owner, dict):
                raise LedgerError(f"{owner_label} must be an object")
            kind = _require_text(owner.get("kind"), f"{owner_label}.kind")
            if kind not in _ALLOWED_OWNER_KINDS:
                raise LedgerError(f"{owner_label}.kind is unknown: {kind!r}")
            _validate_issue_refs(owner.get("refs"), f"{owner_label}.refs")
            _require_text(owner.get("responsibility"), f"{owner_label}.responsibility")
        _require_text(group.get("retained_boundary"), f"{label}.retained_boundary")

    missing = _REQUIRED_GROUPS - seen
    extra = seen - _REQUIRED_GROUPS
    if missing or extra:
        raise LedgerError(
            "group inventory mismatch: "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )

    negative = payload.get("negative_evidence")
    if not isinstance(negative, list) or not negative:
        raise LedgerError("negative_evidence must be a non-empty list")
    negative_ids: set[str] = set()
    for index, item in enumerate(negative):
        label = f"negative_evidence[{index}]"
        if not isinstance(item, dict):
            raise LedgerError(f"{label} must be an object")
        item_id = _require_text(item.get("id"), f"{label}.id")
        if item_id in negative_ids:
            raise LedgerError(f"duplicate negative-evidence id {item_id!r}")
        negative_ids.add(item_id)
        _validate_commits(
            item.get("historical_commits"),
            f"{label}.historical_commits",
            allow_empty=True,
        )
        _validate_issue_refs(item.get("related_issues"), f"{label}.related_issues")
        decision = _require_text(item.get("decision"), f"{label}.decision")
        if decision not in _ALLOWED_DECISIONS:
            raise LedgerError(f"{label}.decision is unknown: {decision!r}")
        _require_text(item.get("reason"), f"{label}.reason")
        _require_text(item.get("promotion_rule"), f"{label}.promotion_rule")

    return {
        "groups": len(groups),
        "mechanisms": mechanism_count,
        "negative_evidence": len(negative),
    }


def main() -> int:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    counts = validate(payload)
    print(
        "compiler optimization ledger: "
        f"groups={counts['groups']} "
        f"mechanisms={counts['mechanisms']} "
        f"negative_evidence={counts['negative_evidence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
