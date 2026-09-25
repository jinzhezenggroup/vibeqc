"""Identity-bound production-domain receipts for bulk Libxc qualification.

This module turns a complete, externally produced numerical matrix into the
stage-evidence envelope consumed by :mod:`libxc_bulk_capabilities`. It does not
evaluate XC mathematics and cannot promote partial coverage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vibeqc_compiler.common.evidence import canonical_hash

from .libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    BulkFunctionalCapability,
    functional_capability,
)

RESULT_SCHEMA = "vibeqc.libxc-production-domain-result.v1"
CASE_STATUSES = ("pass", "fail", "not-run")


def required_matrix(
    capability: BulkFunctionalCapability,
) -> tuple[tuple[str, str], ...]:
    """Return the exact spin x case coverage required by the current profile."""
    profile = capability.production_domain_profile
    return tuple(
        (spin, case_id)
        for spin in profile.spin_layouts
        for case_id in profile.case_ids_for_spin(spin)
    )


def _normalize_case(
    value: Mapping[str, Any],
    capability: BulkFunctionalCapability,
) -> dict[str, Any]:
    profile = capability.production_domain_profile
    if not isinstance(value, Mapping):
        raise TypeError("production-domain case result must be a mapping")
    spin = value.get("spin")
    case_id = value.get("case_id")
    status = value.get("status")
    reason = value.get("reason")
    outputs = value.get("outputs")
    if spin not in profile.spin_layouts:
        raise ValueError(f"production-domain case has unsupported spin {spin!r}")
    if case_id not in profile.case_ids_for_spin(spin):
        raise ValueError(
            "production-domain case is not valid for spin layout: "
            f"{spin!r}:{case_id!r}"
        )
    if status not in CASE_STATUSES:
        raise ValueError("production-domain case status must be pass, fail, or not-run")
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes))
        or list(outputs) != list(profile.outputs)
    ):
        raise ValueError("production-domain case must cover the exact profile outputs")
    if status == "pass":
        if reason is not None:
            raise ValueError("passing production-domain case cannot carry a reason")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError("failed/not-run production-domain case requires a reason")
    return {
        "spin": spin,
        "case_id": case_id,
        "status": status,
        "outputs": list(profile.outputs),
        "reason": reason,
    }


def _canonical_payload(
    capability: BulkFunctionalCapability,
    cases: Sequence[Mapping[str, Any]],
    *,
    evidence: str,
) -> dict[str, Any]:
    profile = capability.production_domain_profile
    if not profile.eligible:
        raise ValueError(
            "production-domain result is structurally blocked: " + str(profile.blocker)
        )
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("production-domain result requires an evidence reference")

    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in cases:
        row = _normalize_case(raw, capability)
        key = (row["spin"], row["case_id"])
        if key in normalized:
            raise ValueError(f"duplicate production-domain case result: {key!r}")
        normalized[key] = row

    required = required_matrix(capability)
    missing = [key for key in required if key not in normalized]
    extra = [key for key in normalized if key not in required]
    if missing or extra:
        raise ValueError(
            "production-domain result does not cover the exact profile matrix: "
            f"missing={missing!r}, extra={extra!r}"
        )
    ordered = [normalized[key] for key in required]
    return {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "profile_identity": profile.identity,
        "evidence": evidence.strip(),
        "cases": ordered,
    }


def build_result(
    name: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    evidence: str,
) -> dict[str, Any]:
    """Build a deterministic receipt for one complete production-domain run."""
    capability = functional_capability(name)
    payload = _canonical_payload(capability, cases, evidence=evidence)
    return {**payload, "identity": canonical_hash(payload)}


def validate_result(
    name: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an externally stored production-domain receipt."""
    if not isinstance(value, Mapping) or value.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported production-domain result schema")
    capability = functional_capability(name)
    if value.get("subject_identity") != capability.identity:
        raise ValueError("production-domain result subject identity mismatch")
    if value.get("profile_identity") != capability.production_domain_profile.identity:
        raise ValueError("production-domain result profile identity mismatch")
    raw_cases = value.get("cases")
    if (
        not isinstance(raw_cases, Sequence)
        or isinstance(raw_cases, (str, bytes))
        or not raw_cases
    ):
        raise TypeError("production-domain result cases must be a nonempty sequence")
    evidence = value.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("production-domain result requires an evidence reference")
    payload = _canonical_payload(
        capability,
        raw_cases,
        evidence=evidence,
    )
    if value.get("identity") != canonical_hash(payload):
        raise ValueError("production-domain result identity mismatch")
    return {**payload, "identity": value["identity"]}


def stage_evidence(
    name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one exact matrix receipt into fail-closed capability evidence."""
    normalized = validate_result(name, result)
    capability = functional_capability(name)
    failed = [row for row in normalized["cases"] if row["status"] == "fail"]
    not_run = [row for row in normalized["cases"] if row["status"] == "not-run"]
    if failed:
        status = "fail"
        first = failed[0]
        reason = (
            f"{first['spin']}:{first['case_id']}: {first['reason']}"
            f" ({len(failed)} failed case(s))"
        )
    elif not_run:
        status = "not-run"
        first = not_run[0]
        reason = (
            f"{first['spin']}:{first['case_id']}: {first['reason']}"
            f" ({len(not_run)} not-run case(s))"
        )
    else:
        status = "pass"
        reason = None

    evidence = f"{normalized['evidence']}#sha256={normalized['identity']}"
    payload: dict[str, Any] = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": "production-domain",
        "status": status,
        "reason": reason,
        "evidence": evidence,
    }
    if status == "pass":
        payload["qualification"] = capability.production_domain_profile.to_payload()
    return payload
