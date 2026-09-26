"""Exact public-method admission evidence for automatic bulk Libxc endpoints.

This module is the final evidence producer in the automatic semilocal capability
ladder. It never infers public support from representation alone: a public CPU
energy method is emitted only after exact endpoint resolution succeeds for both
spin layouts using already-retained compiled-CPU, production-domain and
molecular-SCF evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vibeqc_compiler.common.evidence import canonical_hash

from .endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    resolve_endpoint_capability,
)
from .libxc_bulk_capabilities import (
    SPIN_LAYOUTS,
    STAGE_EVIDENCE_SCHEMA,
    functional_capability,
)

RESULT_SCHEMA = "vibeqc.libxc-public-method-result/v1"
QUALIFICATION_SCHEMA = "vibeqc.libxc-public-method-qualification/v1"


def _resolved_cpu_energy_endpoints(
    name: str,
    *,
    prerequisite_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve the exact prerequisite endpoint for every supported spin layout."""
    resolved = []
    for spin in SPIN_LAYOUTS:
        endpoint = resolve_endpoint_capability(
            name,
            backend="cpu",
            product="energy",
            spin=spin,
            require_public=False,
            evidence=prerequisite_evidence,
        )
        resolved.append(endpoint.to_payload())
    return tuple(resolved)


def build_result(
    name: str,
    *,
    prerequisite_evidence: Mapping[str, Any],
    evidence: str,
) -> dict[str, Any]:
    """Build one public CPU-energy admission receipt.

    The prerequisite map must already qualify the exact CPU energy endpoint for
    both RKS and UKS. Public-method evidence is deliberately not accepted as a
    prerequisite; this function is the producer for that final stage.
    """
    if not isinstance(prerequisite_evidence, Mapping):
        raise TypeError("public-method prerequisite evidence must be a mapping")
    if "public-method" in prerequisite_evidence:
        raise ValueError("public-method evidence cannot authorize its own production")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("public-method result requires an evidence reference")

    capability = functional_capability(name, evidence=prerequisite_evidence)
    endpoints = _resolved_cpu_energy_endpoints(
        capability.name,
        prerequisite_evidence=prerequisite_evidence,
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "backend": "cpu",
        "product": "energy",
        "spins": list(SPIN_LAYOUTS),
        "endpoint_resolutions": list(endpoints),
        "evidence": evidence.strip(),
    }
    return {**payload, "identity": canonical_hash(payload)}


def validate_result(
    name: str,
    value: Mapping[str, Any],
    *,
    prerequisite_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stored public receipt against current prerequisite evidence."""
    if not isinstance(value, Mapping) or value.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported public-method result schema")
    rebuilt = build_result(
        name,
        prerequisite_evidence=prerequisite_evidence,
        evidence=value.get("evidence"),
    )
    if value.get("identity") != rebuilt["identity"]:
        raise ValueError("public-method result identity mismatch")
    if dict(value) != rebuilt:
        raise ValueError("public-method result is not canonical")
    return rebuilt


def stage_evidence(
    name: str,
    result: Mapping[str, Any],
    *,
    prerequisite_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one exact public result into the generic capability envelope."""
    normalized = validate_result(
        name,
        result,
        prerequisite_evidence=prerequisite_evidence,
    )
    coverage = [
        {"backend": "cpu", "spin": spin, "products": ["energy"]}
        for spin in SPIN_LAYOUTS
    ]
    qualification = {
        "schema": ENDPOINT_COVERAGE_SCHEMA,
        "coverage": coverage,
        "admission": {
            "schema": QUALIFICATION_SCHEMA,
            "result_identity": normalized["identity"],
            "endpoint_resolutions": normalized["endpoint_resolutions"],
        },
    }
    return {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": normalized["subject_identity"],
        "stage": "public-method",
        "status": "pass",
        "reason": None,
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
