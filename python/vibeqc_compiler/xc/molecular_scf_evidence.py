"""Exact molecular-SCF receipts for automatic bulk Libxc CPU endpoints.

This module defines what evidence is sufficient to promote the generic
molecular-scf stage. It consumes already resolved bulk-KS candidates and
externally produced numerical endpoint rows; it does not execute SCF itself.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from vibeqc_compiler.common.evidence import canonical_hash

from .endpoint_capability import ENDPOINT_COVERAGE_SCHEMA
from .libxc_bulk_capabilities import (
    SPIN_LAYOUTS,
    STAGE_EVIDENCE_SCHEMA,
    functional_capability,
)

RESULT_SCHEMA = "vibeqc.libxc-molecular-scf-result/v1"
QUALIFICATION_SCHEMA = "vibeqc.libxc-molecular-scf-qualification/v1"
PHASES = ("cold", "warm-replay", "changed-geometry")
STATUSES = ("pass", "fail", "not-run")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MolecularScfResolution(Protocol):
    """Detached execution provenance consumed without importing method policy.

    Bulk KS resolutions implement this interface. The XC evidence owner validates
    their serialized contract, keeping the compiler dependency direction intact.
    """

    def to_payload(self) -> dict[str, Any]:
        """Return the exact versioned CPU resolution provenance."""
        ...


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def required_matrix() -> tuple[tuple[str, str], ...]:
    """Return exact CPU molecular lifecycle coverage required for promotion."""
    return tuple((spin, phase) for spin in SPIN_LAYOUTS for phase in PHASES)


def _validate_resolution_contract(
    payload: Mapping[str, Any], *, name: str, identity: str, spin: str
) -> None:
    """Check v2 semantics even when a malformed payload has been rehashed.

    Content hashes bind evidence bytes; they do not establish that those bytes
    describe the required CPU semilocal execution contract.
    """
    if payload.get("schema") != "vibeqc.bulk-libxc-ks-resolution.v2":
        raise ValueError("unsupported molecular-SCF resolution schema")
    capability = payload.get("capability")
    if (
        not isinstance(capability, Mapping)
        or capability.get("name") != name
        or capability.get("identity") != identity
    ):
        raise ValueError("molecular-SCF stored resolution capability mismatch")
    if payload.get("spin") != spin or payload.get("backend") != "cpu":
        raise ValueError("molecular-SCF stored resolution endpoint mismatch")
    reference = "restricted" if spin == "unpolarized" else "unrestricted"
    if payload.get("reference") != reference:
        raise ValueError("molecular-SCF resolution reference mismatch")
    if payload.get("required_ingredients") != list(
        functional_capability(name).required_ingredients
    ) or payload.get("required_lowerers") != ["semilocal-xc"]:
        raise ValueError("molecular-SCF resolution semilocal contract mismatch")
    identifier = payload.get("method_identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("molecular-SCF resolution requires a method identifier")
    for field in (
        "method_identity",
        "plan_identity",
        "compiled_cpu_binding_identity",
        "compiled_cpu_result_identity",
    ):
        _sha(payload.get(field), field)


def _resolution_payload(
    resolution: MolecularScfResolution,
    *,
    capability_name: str,
    capability_identity: str,
) -> dict[str, Any]:
    payload = resolution.to_payload()
    if not isinstance(payload, Mapping):
        raise TypeError("molecular-SCF resolution payload must be a mapping")
    spin = payload.get("spin")
    if spin not in SPIN_LAYOUTS:
        raise ValueError("molecular-SCF KS resolution has unsupported spin")
    _validate_resolution_contract(
        payload,
        name=capability_name,
        identity=capability_identity,
        spin=spin,
    )
    return {
        "spin": spin,
        "identity": canonical_hash(payload),
        "payload": payload,
    }


def _normalize_row(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("molecular-SCF result row must be a mapping")
    spin = value.get("spin")
    phase = value.get("phase")
    status = value.get("status")
    reason = value.get("reason")
    if spin not in SPIN_LAYOUTS:
        raise ValueError(f"unsupported molecular-SCF spin {spin!r}")
    if phase not in PHASES:
        raise ValueError(f"unsupported molecular-SCF phase {phase!r}")
    if status not in STATUSES:
        raise ValueError("molecular-SCF status must be pass, fail, or not-run")

    normalized: dict[str, Any] = {
        "spin": spin,
        "phase": phase,
        "status": status,
        "reason": reason,
        "fixture_identity": _sha(value.get("fixture_identity"), "fixture identity"),
        "geometry_identity": _sha(value.get("geometry_identity"), "geometry identity"),
        "reference_identity": _sha(
            value.get("reference_identity"), "reference identity"
        ),
    }
    if status != "pass":
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failed/not-run molecular-SCF row requires a reason")
        normalized.update(
            {
                "converged": False,
                "iterations": None,
                "energy_hartree": None,
                "reference_energy_hartree": None,
                "absolute_energy_error_hartree": None,
                "energy_tolerance_hartree": None,
                "physical_residual_rms": None,
                "residual_tolerance": None,
            }
        )
        return normalized

    if reason is not None:
        raise ValueError("passing molecular-SCF row cannot carry a reason")
    if value.get("converged") is not True:
        raise ValueError("passing molecular-SCF row must be converged")
    iterations = value.get("iterations")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("passing molecular-SCF row requires positive iterations")
    energy = _finite(value.get("energy_hartree"), "molecular-SCF energy")
    reference = _finite(
        value.get("reference_energy_hartree"), "molecular-SCF reference energy"
    )
    error = _finite(
        value.get("absolute_energy_error_hartree"), "molecular-SCF energy error"
    )
    energy_tolerance = _finite(
        value.get("energy_tolerance_hartree"), "molecular-SCF energy tolerance"
    )
    residual = _finite(
        value.get("physical_residual_rms"), "molecular-SCF physical residual"
    )
    residual_tolerance = _finite(
        value.get("residual_tolerance"), "molecular-SCF residual tolerance"
    )
    if error < 0.0 or energy_tolerance < 0.0:
        raise ValueError("molecular-SCF energy error/tolerance must be nonnegative")
    if residual < 0.0 or residual_tolerance < 0.0:
        raise ValueError("molecular-SCF residual/tolerance must be nonnegative")
    recomputed_error = abs(energy - reference)
    if not math.isclose(error, recomputed_error, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("molecular-SCF reported energy error is inconsistent")
    if error > energy_tolerance or recomputed_error > energy_tolerance:
        raise ValueError("passing molecular-SCF row exceeds energy tolerance")
    if residual > residual_tolerance:
        raise ValueError("passing molecular-SCF row exceeds residual tolerance")

    normalized.update(
        {
            "converged": True,
            "iterations": iterations,
            "energy_hartree": energy,
            "reference_energy_hartree": reference,
            "absolute_energy_error_hartree": error,
            "energy_tolerance_hartree": energy_tolerance,
            "physical_residual_rms": residual,
            "residual_tolerance": residual_tolerance,
        }
    )
    return normalized


def _canonical_payload(
    name: str,
    resolutions: Mapping[str, MolecularScfResolution],
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence: str,
) -> dict[str, Any]:
    capability = functional_capability(name)
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("molecular-SCF result requires an evidence reference")
    if set(resolutions) != set(SPIN_LAYOUTS):
        raise ValueError("molecular-SCF result requires exact RKS/UKS resolutions")

    resolution_rows = {
        spin: _resolution_payload(
            resolutions[spin],
            capability_name=capability.name,
            capability_identity=capability.identity,
        )
        for spin in SPIN_LAYOUTS
    }
    if any(resolution_rows[spin]["spin"] != spin for spin in SPIN_LAYOUTS):
        raise ValueError("molecular-SCF resolution spin key mismatch")

    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = _normalize_row(raw)
        key = (row["spin"], row["phase"])
        if key in normalized:
            raise ValueError(f"duplicate molecular-SCF result row: {key!r}")
        normalized[key] = row

    required = required_matrix()
    missing = [key for key in required if key not in normalized]
    extra = [key for key in normalized if key not in required]
    if missing or extra:
        raise ValueError(
            "molecular-SCF result does not cover the exact lifecycle matrix: "
            f"missing={missing!r}, extra={extra!r}"
        )

    for spin in SPIN_LAYOUTS:
        cold = normalized[(spin, "cold")]
        warm = normalized[(spin, "warm-replay")]
        changed = normalized[(spin, "changed-geometry")]
        if warm["fixture_identity"] != cold["fixture_identity"]:
            raise ValueError("warm molecular-SCF row changed fixture identity")
        if warm["geometry_identity"] != cold["geometry_identity"]:
            raise ValueError("warm molecular-SCF row changed geometry identity")
        if changed["fixture_identity"] != cold["fixture_identity"]:
            raise ValueError("changed molecular-SCF row changed fixture identity")
        if changed["geometry_identity"] == cold["geometry_identity"]:
            raise ValueError("changed-geometry molecular-SCF row reused cold geometry")

    return {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "evidence": evidence.strip(),
        "resolutions": resolution_rows,
        "rows": [normalized[key] for key in required],
    }


def build_result(
    name: str,
    resolutions: Mapping[str, MolecularScfResolution],
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence: str,
) -> dict[str, Any]:
    """Build one exact dual-spin molecular-SCF lifecycle receipt."""
    payload = _canonical_payload(name, resolutions, rows, evidence=evidence)
    return {**payload, "identity": canonical_hash(payload)}


def validate_result(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a stored receipt against current capability and lifecycle rules."""
    if not isinstance(result, Mapping) or result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported molecular-SCF result schema")
    capability = functional_capability(name)
    if result.get("subject_identity") != capability.identity:
        raise ValueError("molecular-SCF result subject identity mismatch")
    evidence = result.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("molecular-SCF result requires an evidence reference")

    resolutions = result.get("resolutions")
    if not isinstance(resolutions, Mapping) or set(resolutions) != set(SPIN_LAYOUTS):
        raise ValueError(
            "molecular-SCF result requires exact stored RKS/UKS resolutions"
        )
    normalized_resolutions: dict[str, Any] = {}
    for spin in SPIN_LAYOUTS:
        item = resolutions[spin]
        if not isinstance(item, Mapping) or item.get("spin") != spin:
            raise ValueError("molecular-SCF stored resolution spin mismatch")
        payload = item.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("molecular-SCF stored resolution payload must be a mapping")
        _validate_resolution_contract(
            payload, name=capability.name, identity=capability.identity, spin=spin
        )
        if item.get("identity") != canonical_hash(dict(payload)):
            raise ValueError("molecular-SCF stored resolution identity mismatch")
        normalized_resolutions[spin] = {
            "spin": spin,
            "identity": item["identity"],
            "payload": dict(payload),
        }

    raw_rows = result.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise TypeError("molecular-SCF result rows must be a sequence")
    normalized_rows = [_normalize_row(row) for row in raw_rows]
    expected = list(required_matrix())
    if [(row["spin"], row["phase"]) for row in normalized_rows] != expected:
        raise ValueError("molecular-SCF result row order/coverage mismatch")
    for spin in SPIN_LAYOUTS:
        by_phase = {row["phase"]: row for row in normalized_rows if row["spin"] == spin}
        cold = by_phase["cold"]
        warm = by_phase["warm-replay"]
        changed = by_phase["changed-geometry"]
        if warm["fixture_identity"] != cold["fixture_identity"]:
            raise ValueError("warm molecular-SCF row changed fixture identity")
        if warm["geometry_identity"] != cold["geometry_identity"]:
            raise ValueError("warm molecular-SCF row changed geometry identity")
        if changed["fixture_identity"] != cold["fixture_identity"]:
            raise ValueError("changed molecular-SCF row changed fixture identity")
        if changed["geometry_identity"] == cold["geometry_identity"]:
            raise ValueError("changed-geometry molecular-SCF row reused cold geometry")

    payload = {
        "schema": RESULT_SCHEMA,
        "subject_identity": capability.identity,
        "evidence": evidence.strip(),
        "resolutions": normalized_resolutions,
        "rows": normalized_rows,
    }
    if result.get("identity") != canonical_hash(payload):
        raise ValueError("molecular-SCF result identity mismatch")
    return {**payload, "identity": result["identity"]}


def stage_evidence(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one receipt into fail-closed molecular-SCF stage evidence."""
    normalized_result = validate_result(name, result)
    capability = functional_capability(name)
    identity = normalized_result["identity"]
    normalized = normalized_result["rows"]
    expected = list(required_matrix())
    if [(row["spin"], row["phase"]) for row in normalized] != expected:
        raise ValueError("molecular-SCF result row order/coverage mismatch")

    failures = [row for row in normalized if row["status"] == "fail"]
    not_run = [row for row in normalized if row["status"] == "not-run"]
    if failures:
        status = "fail"
        first = failures[0]
        reason = (
            f"{first['spin']}:{first['phase']}: {first['reason']} "
            f"({len(failures)} failed row(s))"
        )
    elif not_run:
        status = "not-run"
        first = not_run[0]
        reason = (
            f"{first['spin']}:{first['phase']}: {first['reason']} "
            f"({len(not_run)} not-run row(s))"
        )
    else:
        status = "pass"
        reason = None

    qualification = None
    if status == "pass":
        qualification = {
            "schema": ENDPOINT_COVERAGE_SCHEMA,
            "coverage": [
                {
                    "backend": "cpu",
                    "spin": spin,
                    "products": ["energy"],
                }
                for spin in SPIN_LAYOUTS
            ],
            "result_schema": RESULT_SCHEMA,
            "result_identity": identity,
            "qualification_schema": QUALIFICATION_SCHEMA,
        }
    evidence_ref = normalized_result["evidence"]
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        raise ValueError("molecular-SCF result requires an evidence reference")
    envelope: dict[str, Any] = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": "molecular-scf",
        "status": status,
        "reason": reason,
        "evidence": f"{evidence_ref.strip()}#sha256={identity}",
    }
    if qualification is not None:
        envelope["qualification"] = qualification
    return envelope


__all__ = [
    "PHASES",
    "QUALIFICATION_SCHEMA",
    "RESULT_SCHEMA",
    "build_result",
    "required_matrix",
    "stage_evidence",
    "validate_result",
]
