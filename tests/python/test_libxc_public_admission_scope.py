"""Public admission must remain bound to the exact producer's endpoint scope."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.capability_resolution import CapabilityNotQualified
from vibeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    EndpointCapabilityResolution,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.public_method_evidence import build_result, stage_evidence

NAME = "GGA_X_PBE_SOL"


def _evidence() -> dict:
    capability = libxc_bulk_capabilities.functional_capability(NAME)
    prerequisites = {}
    for stage in ("compiled-cpu", "production-domain", "molecular-scf"):
        payload = {
            "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
            "subject_identity": capability.identity,
            "stage": stage,
            "status": "pass",
            "reason": None,
            "evidence": f"test://public-admission-scope/{stage}",
        }
        if stage == "production-domain":
            payload["qualification"] = capability.production_domain_profile.to_payload()
        elif stage == "molecular-scf":
            payload["qualification"] = {
                "schema": ENDPOINT_COVERAGE_SCHEMA,
                "coverage": [
                    {"backend": "cpu", "spin": spin, "products": ["energy"]}
                    for spin in libxc_bulk_capabilities.SPIN_LAYOUTS
                ],
            }
        prerequisites[stage] = payload
    result = build_result(
        NAME,
        prerequisite_evidence=prerequisites,
        evidence="test://public-admission-scope/result",
    )
    public = stage_evidence(NAME, result, prerequisite_evidence=prerequisites)
    return {**prerequisites, "public-method": public}


def _resolve(
    evidence: dict, *, require_public: bool
) -> EndpointCapabilityResolution:
    return resolve_endpoint_capability(
        NAME,
        backend="cpu",
        product="energy",
        spin="unpolarized",
        require_public=require_public,
        evidence=evidence,
    )


@pytest.mark.parametrize("extra_product", ("forces", "response"))
def test_public_receipt_rejects_unsigned_product_expansion(extra_product: str) -> None:
    evidence = _evidence()
    qualification = evidence["public-method"]["qualification"]
    qualification["coverage"][0]["products"].append(extra_product)

    with pytest.raises(CapabilityNotQualified) as caught:
        _resolve(evidence, require_public=True)
    assert caught.value.blockers == (
        (
            "public-method",
            "public admission coverage does not match its exact CPU energy receipt",
        ),
    )
    assert _resolve(evidence, require_public=False).public_dft is False


def test_public_receipt_rejects_unsigned_cuda_expansion() -> None:
    evidence = _evidence()
    evidence["public-method"]["qualification"]["coverage"].append(
        {"backend": "cuda", "spin": "unpolarized", "products": ["energy"]}
    )
    with pytest.raises(CapabilityNotQualified):
        _resolve(evidence, require_public=True)
    assert _resolve(evidence, require_public=False).public_dft is False


@pytest.mark.parametrize("mutation", ("missing-receipt", "bad-hash", "stale-spin"))
def test_private_resolution_does_not_report_unvalidated_public_support(
    mutation: str,
) -> None:
    evidence = deepcopy(_evidence())
    qualification = evidence["public-method"]["qualification"]
    if mutation == "missing-receipt":
        qualification.pop("admission")
    elif mutation == "bad-hash":
        qualification["admission"]["result_identity"] = "0" * 64
    else:
        # The requested unpolarized private endpoint remains qualified, while
        # the other spin required by the public receipt has become stale.
        evidence["molecular-scf"]["qualification"]["coverage"] = [
            {"backend": "cpu", "spin": "unpolarized", "products": ["energy"]}
        ]
    assert _resolve(evidence, require_public=False).public_dft is False


def test_genuine_receipt_reports_public_support_on_private_resolution() -> None:
    evidence = _evidence()
    assert _resolve(evidence, require_public=False).public_dft is True
    assert _resolve(evidence, require_public=True).public_dft is True
