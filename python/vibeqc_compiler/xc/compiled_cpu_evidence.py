"""Exact compiled-CPU evidence for automatic Libxc semilocal point programs.

This module validates content-addressed compilation/runtime receipts and converts
them into the generic capability-stage envelope. It does not invoke a compiler
or create production-domain/public-method capability by itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from vibeqc_compiler.common.evidence import canonical_hash

from .bulk_point_program import SemilocalPointBinding
from .bulk_runtime import PRODUCTION_CANDIDATE_DOMAIN
from .libxc_bulk_capabilities import STAGE_EVIDENCE_SCHEMA, functional_capability

RESULT_SCHEMA = "vibeqc.libxc-compiled-cpu-result/v1"
QUALIFICATION_SCHEMA = "vibeqc.libxc-compiled-cpu-qualification/v1"
_STATUSES = ("pass", "fail", "not-run")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _compiler(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("compiled-CPU compiler identity must be a mapping")
    executable = value.get("executable_sha256")
    version = value.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("compiled-CPU compiler version must be nonempty")
    return {
        "executable_sha256": _sha(executable, "compiler executable"),
        "version": version.strip(),
    }


def _smoke(value: Any, *, required: bool) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ValueError("passing compiled-CPU result requires runtime smoke")
        return None
    if not isinstance(value, Mapping):
        raise TypeError("compiled-CPU smoke result must be a mapping")
    status = value.get("status")
    if status != "pass":
        if required:
            raise ValueError("passing compiled-CPU result requires passing smoke")
        if status not in ("fail", "not-run"):
            raise ValueError("compiled-CPU smoke has invalid status")
    expected = value.get("expected")
    observed = value.get("observed")
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        raise TypeError("compiled-CPU smoke expected values must be a sequence")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        raise TypeError("compiled-CPU smoke observed values must be a sequence")
    if not expected:
        raise ValueError("compiled-CPU smoke expected vector must be nonempty")
    if required and len(expected) != len(observed):
        raise ValueError("passing compiled-CPU smoke vectors must be aligned")
    expected_values = [float(item) for item in expected]
    observed_values = [float(item) for item in observed]
    tolerance = value.get("absolute_tolerance")
    max_error = value.get("maximum_absolute_error")
    if not isinstance(tolerance, (int, float)) or float(tolerance) < 0.0:
        raise ValueError("compiled-CPU smoke tolerance must be nonnegative")
    if not isinstance(max_error, (int, float)) or float(max_error) < 0.0:
        raise ValueError("compiled-CPU smoke error must be nonnegative")
    if status == "pass" and float(max_error) > float(tolerance):
        raise ValueError("passing compiled-CPU smoke exceeds its tolerance")
    return {
        "schema": "vibeqc.libxc-compiled-cpu-smoke/v1",
        "status": status,
        "input_identity": _sha(value.get("input_identity"), "smoke input identity"),
        "expected": expected_values,
        "observed": observed_values,
        "absolute_tolerance": float(tolerance),
        "maximum_absolute_error": float(max_error),
    }


def build_result(
    name: str,
    binding: SemilocalPointBinding,
    outcome: Mapping[str, Any],
    *,
    evidence: str,
) -> dict[str, Any]:
    """Build one exact compilation/execution receipt for a point binding."""
    capability = functional_capability(name)
    if not isinstance(binding, SemilocalPointBinding):
        raise TypeError("compiled-CPU result requires SemilocalPointBinding")
    if binding.variant.name != capability.name:
        raise ValueError("compiled-CPU binding functional mismatch")
    if binding.capability_identity != capability.identity:
        raise ValueError("compiled-CPU binding capability identity mismatch")
    if binding.variant.domain != PRODUCTION_CANDIDATE_DOMAIN:
        raise ValueError("compiled-CPU evidence requires production candidate domain")
    if not isinstance(outcome, Mapping):
        raise TypeError("compiled-CPU outcome must be a mapping")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("compiled-CPU result requires an evidence reference")

    status = outcome.get("status")
    reason = outcome.get("reason")
    if status not in _STATUSES:
        raise ValueError("compiled-CPU status must be pass, fail, or not-run")
    if status == "pass":
        if reason is not None:
            raise ValueError("passing compiled-CPU result cannot carry a reason")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError("failed/not-run compiled-CPU result requires a reason")

    compiler = _compiler(outcome.get("compiler"))
    translation_sha = outcome.get("translation_unit_sha256")
    executable_sha = outcome.get("executable_sha256")
    required = status == "pass"
    if required:
        if compiler is None:
            raise ValueError("passing compiled-CPU result requires compiler identity")
        translation_sha = _sha(translation_sha, "translation unit")
        executable_sha = _sha(executable_sha, "compiled executable")
    else:
        translation_sha = (
            _sha(translation_sha, "translation unit")
            if translation_sha is not None
            else None
        )
        executable_sha = (
            _sha(executable_sha, "compiled executable")
            if executable_sha is not None
            else None
        )
    smoke = _smoke(outcome.get("smoke"), required=required)

    payload = {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "binding_identity": binding.identity,
        "binding": binding.to_payload(),
        "evidence": evidence.strip(),
        "status": status,
        "reason": reason,
        "compiler": compiler,
        "translation_unit_sha256": translation_sha,
        "executable_sha256": executable_sha,
        "smoke": smoke,
    }
    return {**payload, "identity": canonical_hash(payload)}


def validate_result(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one stored result against current capability/binding semantics."""
    if not isinstance(value, Mapping) or value.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported compiled-CPU result schema")
    binding_payload = value.get("binding")
    if not isinstance(binding_payload, Mapping):
        raise TypeError("compiled-CPU result requires binding payload")

    capability = functional_capability(name)
    if value.get("subject_identity") != capability.identity:
        raise ValueError("compiled-CPU result subject identity mismatch")
    if binding_payload.get("capability_identity") != capability.identity:
        raise ValueError("compiled-CPU result binding capability mismatch")
    if binding_payload.get("domain") != PRODUCTION_CANDIDATE_DOMAIN:
        raise ValueError("compiled-CPU result binding domain mismatch")
    if value.get("binding_identity") != canonical_hash(dict(binding_payload)):
        raise ValueError("compiled-CPU result binding identity mismatch")

    status = value.get("status")
    reason = value.get("reason")
    outcome = {
        "status": status,
        "reason": reason,
        "compiler": value.get("compiler"),
        "translation_unit_sha256": value.get("translation_unit_sha256"),
        "executable_sha256": value.get("executable_sha256"),
        "smoke": value.get("smoke"),
    }

    # Reconstruct only the canonical detached payload. SemilocalPointBinding
    # construction is intentionally unnecessary for stored evidence validation.
    if status not in _STATUSES:
        raise ValueError("compiled-CPU status must be pass, fail, or not-run")
    if status == "pass":
        if reason is not None:
            raise ValueError("passing compiled-CPU result cannot carry a reason")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError("failed/not-run compiled-CPU result requires a reason")

    compiler = _compiler(outcome["compiler"])
    required = status == "pass"
    translation_sha = outcome["translation_unit_sha256"]
    executable_sha = outcome["executable_sha256"]
    if required:
        if compiler is None:
            raise ValueError("passing compiled-CPU result requires compiler identity")
        translation_sha = _sha(translation_sha, "translation unit")
        executable_sha = _sha(executable_sha, "compiled executable")
    else:
        translation_sha = (
            _sha(translation_sha, "translation unit")
            if translation_sha is not None
            else None
        )
        executable_sha = (
            _sha(executable_sha, "compiled executable")
            if executable_sha is not None
            else None
        )
    smoke = _smoke(outcome["smoke"], required=required)

    payload = {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "binding_identity": value["binding_identity"],
        "binding": dict(binding_payload),
        "evidence": value.get("evidence"),
        "status": status,
        "reason": reason,
        "compiler": compiler,
        "translation_unit_sha256": translation_sha,
        "executable_sha256": executable_sha,
        "smoke": smoke,
    }
    if not isinstance(payload["evidence"], str) or not payload["evidence"].strip():
        raise ValueError("compiled-CPU result requires an evidence reference")
    payload["evidence"] = payload["evidence"].strip()
    if value.get("identity") != canonical_hash(payload):
        raise ValueError("compiled-CPU result identity mismatch")
    return {**payload, "identity": value["identity"]}


def stage_evidence(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one exact result into the generic compiled-cpu stage envelope."""
    normalized = validate_result(name, result)
    qualification = {
        "schema": QUALIFICATION_SCHEMA,
        "result_identity": normalized["identity"],
        "binding_identity": normalized["binding_identity"],
        "binding": normalized["binding"],
        "translation_unit_sha256": normalized["translation_unit_sha256"],
        "executable_sha256": normalized["executable_sha256"],
        "compiler": normalized["compiler"],
        "smoke": normalized["smoke"],
    }
    return {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": normalized["subject_identity"],
        "stage": "compiled-cpu",
        "status": normalized["status"],
        "reason": normalized["reason"],
        "evidence": f"{normalized['evidence']}#sha256={normalized['identity']}",
        "qualification": qualification,
    }


__all__ = [
    "QUALIFICATION_SCHEMA",
    "RESULT_SCHEMA",
    "build_result",
    "stage_evidence",
    "validate_result",
]
