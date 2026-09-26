"""Exact production-domain receipt coverage and promotion gates."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import production_domain_catalog
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    available_capabilities,
    functional_capability,
)
from vibeqc_compiler.xc.production_domain_evidence import (
    build_result,
    required_matrix,
    stage_evidence,
    validate_result,
)

NAME = "GGA_X_PBE_SOL"


def _cases(name: str = NAME) -> list[dict]:
    capability = functional_capability(name)
    outputs = list(capability.production_domain_profile.outputs)
    return [
        {
            "spin": spin,
            "case_id": case_id,
            "status": "pass",
            "outputs": outputs,
            "reason": None,
        }
        for spin, case_id in required_matrix(capability)
    ]


def test_required_matrix_is_spin_aware() -> None:
    capability = functional_capability(NAME)
    matrix = set(required_matrix(capability))

    assert ("polarized", "spin/zero-a") in matrix
    assert ("polarized", "spin/full-b") in matrix
    assert ("unpolarized", "spin/zero-a") not in matrix
    assert ("unpolarized", "spin/full-b") not in matrix

    forged = _cases()
    forged.append(
        {
            "spin": "unpolarized",
            "case_id": "spin/zero-a",
            "status": "pass",
            "outputs": list(capability.production_domain_profile.outputs),
            "reason": None,
        }
    )
    with pytest.raises(ValueError, match="not valid for spin layout"):
        build_result(NAME, forged, evidence="test://forged-spin-case")


def test_complete_matrix_can_produce_exact_stage_evidence() -> None:
    capability = functional_capability(NAME)
    result = build_result(NAME, _cases(), evidence="test://pbe-sol/domain-matrix")
    envelope = stage_evidence(NAME, result)

    assert validate_result(NAME, result) == result
    assert envelope["status"] == "pass"
    assert envelope["reason"] is None
    assert envelope["qualification"] == (
        capability.production_domain_profile.to_payload()
    )
    assert result["identity"] in envelope["evidence"]

    promoted = functional_capability(
        NAME,
        evidence={"production-domain": envelope},
    )
    assert "production-domain" in promoted.qualified_stages


def test_partial_or_tampered_matrix_cannot_be_promoted() -> None:
    cases = _cases()
    with pytest.raises(ValueError, match="exact profile matrix"):
        build_result(NAME, cases[:-1], evidence="test://partial")

    result = build_result(NAME, cases, evidence="test://complete")
    tampered = deepcopy(result)
    tampered["cases"][0]["status"] = "fail"
    tampered["cases"][0]["reason"] = "changed after receipt"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_result(NAME, tampered)

    wrong_outputs = _cases()
    wrong_outputs[0]["outputs"] = ["energy", "vxc"]
    with pytest.raises(ValueError, match="exact profile outputs"):
        build_result(NAME, wrong_outputs, evidence="test://wrong-outputs")


def test_failed_case_remains_an_actionable_blocker() -> None:
    cases = _cases()
    cases[0]["status"] = "fail"
    cases[0]["reason"] = "independent oracle mismatch"
    result = build_result(NAME, cases, evidence="test://failed-domain-matrix")
    envelope = stage_evidence(NAME, result)

    assert envelope["status"] == "fail"
    assert "independent oracle mismatch" in envelope["reason"]
    assert "qualification" not in envelope

    capability = functional_capability(
        NAME,
        evidence={"production-domain": envelope},
    )
    assert "production-domain" not in capability.qualified_stages

    summary = production_domain_catalog.production_domain_summary(
        {NAME: {"production-domain": envelope}}
    )
    row = summary["functionals"][NAME]
    assert row["status"] == "blocked"
    assert "independent oracle mismatch" in row["blocker"]


def test_not_run_case_never_becomes_a_pass() -> None:
    cases = _cases()
    cases[-1]["status"] = "not-run"
    cases[-1]["reason"] = "reference unavailable"
    result = build_result(NAME, cases, evidence="test://incomplete-run")
    envelope = stage_evidence(NAME, result)

    assert envelope["status"] == "not-run"
    assert "reference unavailable" in envelope["reason"]
    assert "qualification" not in envelope


def test_structural_blocker_cannot_receive_a_matrix_receipt() -> None:
    blocked = next(
        capability
        for capability in available_capabilities()
        if not capability.production_domain_profile.eligible
    )
    with pytest.raises(ValueError, match="structurally blocked"):
        build_result(
            blocked.name,
            [],
            evidence="test://blocked-registration",
        )
