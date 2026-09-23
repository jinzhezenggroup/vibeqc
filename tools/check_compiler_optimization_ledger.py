"""Validate the historical compiler optimization ownership/adoption ledger."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "compiler_historical_optimization_ledger.json"

_SCHEMA = "vibeqc.compiler-historical-optimization-ledger"
_SCHEMA_VERSION = 2
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
_REQUIRED_METHODS = ("HF", "DFT", "MP2", "CCSD(T)")
_REQUIRED_BACKENDS = ("cpu", "cuda")
_ALLOWED_ADOPTION_STATUSES = (
    "not-applicable",
    "unverified",
    "represented",
    "wired",
    "production",
    "benchmark-qualified",
)
_STATUS_REQUIRES_REFS = {
    "represented",
    "wired",
    "production",
    "benchmark-qualified",
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
    """Raised when the ownership/adoption ledger is incomplete or inconsistent."""


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


def _validate_issue_refs(value: Any, label: str, *, allow_empty: bool = False) -> None:
    refs = _require_string_list(value, label, allow_empty=allow_empty)
    for ref in refs:
        if not _ISSUE.fullmatch(ref):
            raise LedgerError(f"{label} contains invalid issue reference {ref!r}")


def _validate_contract(payload: dict[str, Any]) -> Path:
    contract = payload.get("adoption_contract")
    if not isinstance(contract, dict):
        raise LedgerError("adoption_contract must be an object")

    methods = _require_string_list(contract.get("methods"), "adoption_contract.methods")
    if methods != list(_REQUIRED_METHODS):
        raise LedgerError(
            "adoption_contract.methods must exactly match the required method inventory"
        )

    backends = _require_string_list(
        contract.get("backends"), "adoption_contract.backends"
    )
    if backends != list(_REQUIRED_BACKENDS):
        raise LedgerError(
            "adoption_contract.backends must exactly match the required backend inventory"
        )

    statuses = _require_string_list(
        contract.get("statuses"), "adoption_contract.statuses"
    )
    if statuses != list(_ALLOWED_ADOPTION_STATUSES):
        raise LedgerError(
            "adoption_contract.statuses must exactly match the required status inventory"
        )

    semantics = contract.get("status_semantics")
    if not isinstance(semantics, dict):
        raise LedgerError("adoption_contract.status_semantics must be an object")
    if set(semantics) != set(_ALLOWED_ADOPTION_STATUSES):
        raise LedgerError(
            "adoption_contract.status_semantics must describe every adoption status"
        )
    for status in _ALLOWED_ADOPTION_STATUSES:
        _require_text(
            semantics.get(status), f"adoption_contract.status_semantics[{status!r}]"
        )

    evidence_root_text = _require_text(
        contract.get("benchmark_evidence_root"),
        "adoption_contract.benchmark_evidence_root",
    )
    if evidence_root_text != "benchmarks/results":
        raise LedgerError(
            "adoption_contract.benchmark_evidence_root must be 'benchmarks/results'"
        )
    return (ROOT / evidence_root_text).resolve()


def _validate_evidence_paths(value: Any, label: str, evidence_root: Path) -> None:
    paths = _require_string_list(value, label)
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise LedgerError(f"{label} contains unsafe evidence path {raw!r}")
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(evidence_root)
        except ValueError as exc:
            raise LedgerError(
                f"{label} must stay under benchmarks/results: {raw!r}"
            ) from exc
        if not candidate.is_file():
            raise LedgerError(f"{label} references missing evidence file {raw!r}")


def _validate_adoption(
    group: dict[str, Any], label: str, evidence_root: Path
) -> tuple[int, int]:
    adoption = group.get("adoption")
    if not isinstance(adoption, list) or not adoption:
        raise LedgerError(f"{label}.adoption must be a non-empty list")

    seen_cells: set[tuple[str, str]] = set()
    qualified_cells = 0

    for row_index, row in enumerate(adoption):
        row_label = f"{label}.adoption[{row_index}]"
        if not isinstance(row, dict):
            raise LedgerError(f"{row_label} must be an object")

        methods = _require_string_list(row.get("methods"), f"{row_label}.methods")
        unknown_methods = set(methods) - set(_REQUIRED_METHODS)
        if unknown_methods:
            raise LedgerError(
                f"{row_label}.methods contains unknown methods {sorted(unknown_methods)!r}"
            )

        backends = _require_string_list(row.get("backends"), f"{row_label}.backends")
        unknown_backends = set(backends) - set(_REQUIRED_BACKENDS)
        if unknown_backends:
            raise LedgerError(
                f"{row_label}.backends contains unknown backends "
                f"{sorted(unknown_backends)!r}"
            )

        status = _require_text(row.get("status"), f"{row_label}.status")
        if status not in _ALLOWED_ADOPTION_STATUSES:
            raise LedgerError(f"{row_label}.status is unknown: {status!r}")

        _require_text(row.get("note"), f"{row_label}.note")

        refs = row.get("refs")
        if status in _STATUS_REQUIRES_REFS:
            _validate_issue_refs(refs, f"{row_label}.refs")
        elif refs is not None:
            _validate_issue_refs(refs, f"{row_label}.refs", allow_empty=True)

        if status == "benchmark-qualified":
            if len(methods) != 1 or len(backends) != 1:
                raise LedgerError(
                    f"{row_label} benchmark-qualified evidence must identify exactly "
                    "one method/backend cell"
                )
            _validate_evidence_paths(
                row.get("evidence"), f"{row_label}.evidence", evidence_root
            )
            budget = row.get("max_regression_percent")
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0.0
                or float(budget) > 100.0
            ):
                raise LedgerError(
                    f"{row_label}.max_regression_percent must be finite and in (0, 100]"
                )
        elif "evidence" in row or "max_regression_percent" in row:
            raise LedgerError(
                f"{row_label} may carry benchmark evidence only when "
                "status is 'benchmark-qualified'"
            )

        row_cells = 0
        for method in methods:
            for backend in backends:
                cell = (method, backend)
                if cell in seen_cells:
                    raise LedgerError(
                        f"{row_label} overlaps existing adoption cell "
                        f"{method}/{backend}"
                    )
                seen_cells.add(cell)
                row_cells += 1
        if status == "benchmark-qualified":
            qualified_cells += row_cells

    expected = {
        (method, backend)
        for method in _REQUIRED_METHODS
        for backend in _REQUIRED_BACKENDS
    }
    missing = expected - seen_cells
    extra = seen_cells - expected
    if missing or extra:
        raise LedgerError(
            f"{label}.adoption cell inventory mismatch: "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )

    return len(seen_cells), qualified_cells


def validate(payload: Any) -> dict[str, int]:
    """Validate *payload* and return deterministic ledger counts."""
    if not isinstance(payload, dict):
        raise LedgerError("ledger root must be an object")
    if payload.get("schema") != _SCHEMA:
        raise LedgerError("unknown ledger schema")
    version = payload.get("schema_version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise LedgerError("unsupported ledger schema_version")
    _require_text(payload.get("coverage"), "coverage")
    evidence_root = _validate_contract(payload)

    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise LedgerError("groups must be a list")

    seen: set[str] = set()
    mechanism_count = 0
    adoption_cells = 0
    benchmark_qualified = 0
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

        cells, qualified = _validate_adoption(group, label, evidence_root)
        adoption_cells += cells
        benchmark_qualified += qualified

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
        "adoption_cells": adoption_cells,
        "benchmark_qualified": benchmark_qualified,
        "negative_evidence": len(negative),
    }


def main() -> int:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    counts = validate(payload)
    print(
        "compiler optimization ledger: "
        f"groups={counts['groups']} "
        f"mechanisms={counts['mechanisms']} "
        f"adoption_cells={counts['adoption_cells']} "
        f"benchmark_qualified={counts['benchmark_qualified']} "
        f"negative_evidence={counts['negative_evidence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
