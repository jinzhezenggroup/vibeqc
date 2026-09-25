"""Identity-bound production-domain receipts for bulk Libxc qualification.

This module turns a complete, externally produced numerical matrix into the
stage-evidence envelope consumed by :mod:`libxc_bulk_capabilities`. It does not
evaluate XC mathematics and cannot promote partial coverage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from vibeqc_compiler.common.evidence import canonical_hash

from .libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    BulkFunctionalCapability,
    functional_capability,
)

if TYPE_CHECKING:
    from .bulk_runtime import BulkRuntimeProgram

RESULT_SCHEMA = "vibeqc.libxc-production-domain-result.v3"
EXECUTION_SCHEMA = "vibeqc.libxc-production-domain-execution/v2"
EXECUTOR = "bulk-runtime-array-graph/v1"
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


def _expected_outputs(size: int) -> tuple[tuple[int, ...], ...]:
    return ((), *((index,) for index in range(size)))


def build_execution_binding(
    name: str,
    programs: Mapping[str, BulkRuntimeProgram],
) -> dict[str, Any]:
    """Bind the exact two-spin first-order programs used by a campaign."""
    capability = functional_capability(name)
    profile = capability.production_domain_profile
    if not isinstance(programs, Mapping):
        raise TypeError("production-domain execution programs must be a mapping")
    if set(programs) != set(profile.spin_layouts):
        raise ValueError(
            "production-domain execution binding requires every exact spin layout"
        )

    records = []
    for spin in profile.spin_layouts:
        program = programs[spin]
        if program.spec.identifier != capability.name:
            raise ValueError("production-domain execution functional mismatch")
        if program.spec.capability_identity != capability.identity:
            raise ValueError("production-domain execution capability identity mismatch")
        if program.spec.spin != spin:
            raise ValueError("production-domain execution spin mismatch")
        if program.order != 1 or program.outputs != _expected_outputs(
            len(program.spec.features)
        ):
            raise ValueError(
                "production-domain execution requires complete E/vxc outputs"
            )
        records.append(
            {
                "spin": spin,
                "executor": EXECUTOR,
                "domain": program.spec.domain,
                "source_identity": program.spec.source_identity,
                "expression_identity": program.expression_hash,
                "optimization": program.optimization,
                "features": list(program.spec.features),
                "outputs": [list(output) for output in program.outputs],
            }
        )
    payload = {
        "schema": EXECUTION_SCHEMA,
        "subject_identity": capability.identity,
        "programs": records,
    }
    return {**payload, "identity": canonical_hash(payload)}


def _normalize_execution(
    value: Mapping[str, Any],
    capability: BulkFunctionalCapability,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("unsupported production-domain execution schema")
    if value.get("subject_identity") != capability.identity:
        raise ValueError("production-domain execution subject identity mismatch")
    raw_programs = value.get("programs")
    if (
        not isinstance(raw_programs, Sequence)
        or isinstance(raw_programs, (str, bytes))
        or not raw_programs
    ):
        raise TypeError("production-domain execution programs must be a sequence")

    expected_spins = capability.production_domain_profile.spin_layouts
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_programs:
        if not isinstance(raw, Mapping):
            raise TypeError("production-domain execution record must be a mapping")
        spin = raw.get("spin")
        if spin not in expected_spins:
            raise ValueError(
                f"production-domain execution has unsupported spin {spin!r}"
            )
        if spin in records:
            raise ValueError(f"duplicate production-domain execution spin {spin!r}")
        if raw.get("executor") != EXECUTOR:
            raise ValueError("production-domain execution has unsupported executor")
        for field in (
            "domain",
            "source_identity",
            "expression_identity",
            "optimization",
        ):
            field_value = raw.get(field)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"production-domain execution {field} must be nonempty"
                )
        features = raw.get("features")
        if (
            not isinstance(features, Sequence)
            or isinstance(features, (str, bytes))
            or not features
            or not all(isinstance(item, str) and item for item in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("production-domain execution features are invalid")
        outputs = raw.get("outputs")
        expected_outputs = [list(output) for output in _expected_outputs(len(features))]
        if outputs != expected_outputs:
            raise ValueError(
                "production-domain execution must cover complete E/vxc outputs"
            )
        records[spin] = {
            "spin": spin,
            "executor": EXECUTOR,
            "domain": raw["domain"],
            "source_identity": raw["source_identity"],
            "expression_identity": raw["expression_identity"],
            "optimization": raw["optimization"],
            "features": list(features),
            "outputs": expected_outputs,
        }

    if tuple(records) != expected_spins:
        missing = [spin for spin in expected_spins if spin not in records]
        raise ValueError(
            "production-domain execution does not cover exact spin layouts: "
            f"missing={missing!r}"
        )
    payload = {
        "schema": EXECUTION_SCHEMA,
        "subject_identity": capability.identity,
        "programs": [records[spin] for spin in expected_spins],
    }
    if value.get("identity") != canonical_hash(payload):
        raise ValueError("production-domain execution identity mismatch")
    return {**payload, "identity": value["identity"]}


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
            f"production-domain case is not valid for spin layout: {spin!r}:{case_id!r}"
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
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    profile = capability.production_domain_profile
    if not profile.eligible:
        raise ValueError(
            "production-domain result is structurally blocked: " + str(profile.blocker)
        )
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("production-domain result requires an evidence reference")
    normalized_execution = _normalize_execution(execution, capability)

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
        "execution": normalized_execution,
        "evidence": evidence.strip(),
        "cases": ordered,
    }


def build_result(
    name: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    evidence: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic receipt for one complete production-domain run."""
    capability = functional_capability(name)
    payload = _canonical_payload(
        capability,
        cases,
        evidence=evidence,
        execution=execution,
    )
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
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise TypeError("production-domain result requires execution binding")
    payload = _canonical_payload(
        capability,
        raw_cases,
        evidence=evidence,
        execution=execution,
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
