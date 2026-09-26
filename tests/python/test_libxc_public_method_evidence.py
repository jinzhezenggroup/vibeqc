"""Regression coverage for automatic Libxc public-method admission."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.capability_resolution import CapabilityNotQualified
from vibeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.public_method_evidence import (
    RESULT_SCHEMA,
    build_result,
    stage_evidence,
    validate_result,
)

NAME = "GGA_X_PBE_SOL"


def _coverage(*spins: str) -> dict:
    return {
        "schema": ENDPOINT_COVERAGE_SCHEMA,
        "coverage": [
            {"backend": "cpu", "spin": spin, "products": ["energy"]} for spin in spins
        ],
    }


def _stage(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    qualification: dict | None = None,
) -> dict:
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": "pass",
        "reason": None,
        "evidence": f"test://{capability.name}/{stage}",
    }
    if stage == "production-domain" and qualification is None:
        qualification = capability.production_domain_profile.to_payload()
    if qualification is not None:
        payload["qualification"] = qualification
    return payload


def _prerequisites(*, spins: tuple[str, ...] = ("polarized", "unpolarized")) -> dict:
    capability = libxc_bulk_capabilities.functional_capability(NAME)
    return {
        "compiled-cpu": _stage(capability, "compiled-cpu"),
        "production-domain": _stage(capability, "production-domain"),
        "molecular-scf": _stage(
            capability,
            "molecular-scf",
            qualification=_coverage(*spins),
        ),
    }


def test_dual_spin_cpu_energy_endpoints_produce_public_method_evidence() -> None:
    prerequisites = _prerequisites()
    result = build_result(
        NAME,
        prerequisite_evidence=prerequisites,
        evidence="artifact://libxc-public/GGA_X_PBE_SOL.json",
    )
    envelope = stage_evidence(
        NAME,
        result,
        prerequisite_evidence=prerequisites,
    )

    assert result["schema"] == RESULT_SCHEMA
    assert (
        validate_result(
            NAME,
            result,
            prerequisite_evidence=prerequisites,
        )
        == result
    )
    assert envelope["stage"] == "public-method"
    assert envelope["status"] == "pass"
    assert envelope["qualification"]["schema"] == ENDPOINT_COVERAGE_SCHEMA
    assert envelope["qualification"]["coverage"] == [
        {"backend": "cpu", "spin": "polarized", "products": ["energy"]},
        {"backend": "cpu", "spin": "unpolarized", "products": ["energy"]},
    ]

    evidence = {**prerequisites, "public-method": envelope}
    for spin in ("polarized", "unpolarized"):
        resolved = resolve_endpoint_capability(
            NAME,
            backend="cpu",
            product="energy",
            spin=spin,
            require_public=True,
            evidence=evidence,
        )
        assert resolved.public_dft is True


def test_public_admission_requires_both_spin_endpoints() -> None:
    prerequisites = _prerequisites(spins=("unpolarized",))
    with pytest.raises(CapabilityNotQualified) as caught:
        build_result(
            NAME,
            prerequisite_evidence=prerequisites,
            evidence="test://missing-uks",
        )
    assert caught.value.blockers == (
        (
            "molecular-scf",
            "pass evidence does not cover backend=cpu spin=polarized product=energy",
        ),
    )


def test_public_evidence_cannot_authorize_itself() -> None:
    prerequisites = _prerequisites()
    capability = libxc_bulk_capabilities.functional_capability(NAME)
    prerequisites["public-method"] = _stage(
        capability,
        "public-method",
        qualification=_coverage("polarized", "unpolarized"),
    )
    with pytest.raises(ValueError, match="cannot authorize its own"):
        build_result(
            NAME,
            prerequisite_evidence=prerequisites,
            evidence="test://self-authorizing",
        )


def test_stored_public_receipt_is_bound_to_exact_endpoint_resolutions() -> None:
    prerequisites = _prerequisites()
    result = build_result(
        NAME,
        prerequisite_evidence=prerequisites,
        evidence="test://valid",
    )
    tampered = deepcopy(result)
    tampered["endpoint_resolutions"][0]["spin"] = "unpolarized"

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_result(
            NAME,
            tampered,
            prerequisite_evidence=prerequisites,
        )


def test_public_admission_fails_when_prerequisite_stage_disappears() -> None:
    prerequisites = _prerequisites()
    result = build_result(
        NAME,
        prerequisite_evidence=prerequisites,
        evidence="test://valid",
    )
    stale = dict(prerequisites)
    stale.pop("compiled-cpu")

    with pytest.raises(CapabilityNotQualified):
        validate_result(
            NAME,
            result,
            prerequisite_evidence=stale,
        )
