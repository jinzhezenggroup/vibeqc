"""Exact compiled-CPU evidence for automatic Libxc semilocal point programs.

This module validates content-addressed compilation/runtime receipts and converts
them into the generic capability-stage envelope. It does not invoke a compiler
or create production-domain/public-method capability by itself.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from vibeqc_compiler.common.evidence import canonical_hash

from . import libxc_bulk
from .bulk_point_program import (
    POINT_PROGRAM_BINDING_SCHEMA,
    SemilocalPointBinding,
    native_domain_version,
)
from .bulk_runtime import PRODUCTION_DENSITY_CANDIDATE_DOMAIN
from .libxc_bulk_capabilities import STAGE_EVIDENCE_SCHEMA, functional_capability

RESULT_SCHEMA = "vibeqc.libxc-compiled-cpu-result/v2"
QUALIFICATION_SCHEMA = "vibeqc.libxc-compiled-cpu-qualification/v2"
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
    case_labels = value.get("case_labels")
    if case_labels != ["interior", "vacuum"]:
        raise ValueError(
            "compiled-CPU smoke must cover exact interior and vacuum cases"
        )
    expected = value.get("expected")
    observed = value.get("observed")
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        raise TypeError("compiled-CPU smoke expected values must be a sequence")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        raise TypeError("compiled-CPU smoke observed values must be a sequence")
    if not expected:
        raise ValueError("compiled-CPU smoke expected vector must be nonempty")
    if status == "pass" and len(expected) != len(observed):
        raise ValueError("passing compiled-CPU smoke vectors must be aligned")
    expected_values = [float(item) for item in expected]
    observed_values = [float(item) for item in observed]
    if not all(math.isfinite(item) for item in (*expected_values, *observed_values)):
        raise ValueError("compiled-CPU smoke vectors must be finite")
    tolerance = value.get("absolute_tolerance")
    max_error = value.get("maximum_absolute_error")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise ValueError("compiled-CPU smoke tolerance must be finite and nonnegative")
    if (
        isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or not math.isfinite(float(max_error))
        or float(max_error) < 0.0
    ):
        raise ValueError("compiled-CPU smoke error must be finite and nonnegative")
    if status == "pass" and float(max_error) > float(tolerance):
        raise ValueError("passing compiled-CPU smoke exceeds its tolerance")
    if status == "pass":
        # Receipt producers do not get to certify their own error bound. Recheck
        # the retained native values before a stored result can promote a stage.
        actual_error = max(
            abs(left - right)
            for left, right in zip(expected_values, observed_values, strict=True)
        )
        if not math.isclose(float(max_error), actual_error, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("compiled-CPU smoke reported error is inconsistent")
        if actual_error > float(tolerance):
            raise ValueError("passing compiled-CPU smoke exceeds its tolerance")
    if required and len(expected_values) != 22:
        raise ValueError("passing compiled-CPU smoke must contain two native vectors")
    return {
        "schema": "vibeqc.libxc-compiled-cpu-smoke/v2",
        "status": status,
        "case_labels": ["interior", "vacuum"],
        "input_identity": _sha(value.get("input_identity"), "smoke input identity"),
        "expected": expected_values,
        "observed": observed_values,
        "absolute_tolerance": float(tolerance),
        "maximum_absolute_error": float(max_error),
    }


def _pinned_density_threshold(name: str) -> float:
    """Return the exact Libxc threshold bound into the current capability source."""
    record = next(
        (
            item
            for item in libxc_bulk.read_catalog()["registrations"]
            if item["name"] == name
        ),
        None,
    )
    if record is None:
        raise ValueError("compiled-CPU binding registration is unavailable")
    try:
        threshold = float(record["bindings"]["p_a_dens_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "compiled-CPU binding density threshold is unavailable"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("compiled-CPU pinned density threshold is invalid")
    return threshold


def _binding_payload(
    value: Mapping[str, Any],
    *,
    capability_name: str,
    capability_identity: str,
) -> dict[str, Any]:
    if value.get("schema") != POINT_PROGRAM_BINDING_SCHEMA:
        raise ValueError("compiled-CPU binding has unsupported schema")
    if value.get("name") != capability_name:
        raise ValueError("compiled-CPU binding functional mismatch")
    if value.get("capability_identity") != capability_identity:
        raise ValueError("compiled-CPU binding capability identity mismatch")
    domain = value.get("domain")
    if domain != PRODUCTION_DENSITY_CANDIDATE_DOMAIN:
        raise ValueError("compiled-CPU binding domain mismatch")
    if value.get("domain_version") != native_domain_version(domain):
        raise ValueError("compiled-CPU binding native domain version mismatch")

    features = value.get("features")
    layouts = {
        ("rho_a", "rho_b"): 1,
        ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"): 7,
        (
            "rho_a",
            "rho_b",
            "sigma_aa",
            "sigma_ab",
            "sigma_bb",
            "tau_a",
            "tau_b",
        ): 15,
    }
    if (
        not isinstance(features, Sequence)
        or isinstance(features, (str, bytes))
        or tuple(features) not in layouts
    ):
        raise ValueError("compiled-CPU binding has unsupported feature layout")
    feature_tuple = tuple(features)
    if value.get("ingredient_mask") != layouts[feature_tuple]:
        raise ValueError("compiled-CPU binding ingredient mask mismatch")
    density_threshold = value.get("density_threshold")
    if (
        isinstance(density_threshold, bool)
        or not isinstance(density_threshold, (int, float))
        or not math.isfinite(float(density_threshold))
        or float(density_threshold) < 0.0
    ):
        raise ValueError(
            "compiled-CPU binding requires finite nonnegative density threshold"
        )
    if float(density_threshold) != _pinned_density_threshold(capability_name):
        raise ValueError("compiled-CPU binding pinned density threshold mismatch")

    payload = {
        "schema": POINT_PROGRAM_BINDING_SCHEMA,
        "name": capability_name,
        "capability_identity": capability_identity,
        "point_expression_identity": _sha(
            value.get("point_expression_identity"), "point expression identity"
        ),
        "artifact_emission_identity": _sha(
            value.get("artifact_emission_identity"), "artifact emission identity"
        ),
        "artifact_source_sha256": _sha(
            value.get("artifact_source_sha256"), "artifact source"
        ),
        "import_identity": _sha(value.get("import_identity"), "import identity"),
        "domain": domain,
        "domain_version": value["domain_version"],
        "density_threshold": float(density_threshold),
        "features": list(feature_tuple),
        "ingredient_mask": value["ingredient_mask"],
    }
    return payload


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
    if binding.variant.domain != PRODUCTION_DENSITY_CANDIDATE_DOMAIN:
        raise ValueError(
            "compiled-CPU evidence requires density-screened production candidate domain"
        )
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
    raw_binding = value.get("binding")
    if not isinstance(raw_binding, Mapping):
        raise TypeError("compiled-CPU result requires binding payload")

    capability = functional_capability(name)
    if value.get("subject_identity") != capability.identity:
        raise ValueError("compiled-CPU result subject identity mismatch")
    binding_payload = _binding_payload(
        raw_binding,
        capability_name=capability.name,
        capability_identity=capability.identity,
    )
    if value.get("binding_identity") != canonical_hash(binding_payload):
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
        "binding": binding_payload,
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
