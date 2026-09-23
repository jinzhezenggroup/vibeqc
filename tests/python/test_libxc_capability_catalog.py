"""Regression coverage for bulk Libxc qualification summaries."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import capability_catalog, libxc_bulk_capabilities


def _stage_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
) -> dict:
    return {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": "pass",
        "reason": None,
        "evidence": f"test://{capability.name}/{stage}",
    }


def test_capability_summary_counts_exact_evidence_state() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    summary = capability_catalog.capability_summary(
        {base.name: {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}}
    )

    total = summary["total_functionals"]
    assert total == len(summary["functionals"])
    assert summary["stage_order"] == list(libxc_bulk_capabilities.CAPABILITY_STAGES)
    assert summary["qualified_counts"]["graph-imported"] == total
    assert summary["qualified_counts"]["pointwise-validated"] == total
    assert summary["qualified_counts"]["compiled-cpu"] == 1
    assert summary["qualified_counts"]["compiled-cuda"] == 0

    record = summary["functionals"][base.name]
    assert "compiled-cpu" in record["qualified_stages"]
    assert "production-domain" in record["ready_stages"]
    assert record["blocked_stages"] == {}
    assert record["public_dft"] is False


def test_capability_changes_reports_promotion_and_demotion() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    initial = capability_catalog.capability_summary()
    promoted = capability_catalog.capability_summary(
        {base.name: {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}}
    )

    forward = capability_catalog.capability_changes(initial, promoted)
    assert forward["promotions"] == {"compiled-cpu": [base.name]}
    assert forward["demotions"] == {}
    assert forward["identity_changes"] == []

    reverse = capability_catalog.capability_changes(promoted, initial)
    assert reverse["promotions"] == {}
    assert reverse["demotions"] == {"compiled-cpu": [base.name]}


def test_capability_changes_separates_identity_and_inventory_changes() -> None:
    initial = capability_catalog.capability_summary()
    changed = deepcopy(initial)
    names = sorted(changed["functionals"])
    subject = names[0]
    removed = names[-1]
    changed["functionals"][subject]["identity"] = "changed-scientific-identity"
    del changed["functionals"][removed]
    changed["total_functionals"] -= 1
    changed["functionals"]["TEST_ADDED"] = {
        "identity": "test-added-identity",
        "qualified_stages": ["graph-imported", "pointwise-validated"],
        "ready_stages": ["compiled-cpu", "compiled-cuda", "production-domain"],
        "blocked_stages": {},
        "public_dft": False,
    }
    changed["total_functionals"] += 1

    delta = capability_catalog.capability_changes(initial, changed)
    assert delta["identity_changes"] == [subject]
    assert delta["removed_functionals"] == [removed]
    assert delta["added_functionals"] == ["TEST_ADDED"]


def test_capability_changes_rejects_stale_or_malformed_snapshot() -> None:
    current = capability_catalog.capability_summary()

    wrong_order = deepcopy(current)
    wrong_order["stage_order"] = list(reversed(wrong_order["stage_order"]))
    with pytest.raises(ValueError, match="stage order mismatch"):
        capability_catalog.capability_changes(wrong_order, current)

    wrong_total = deepcopy(current)
    wrong_total["total_functionals"] += 1
    with pytest.raises(ValueError, match="total does not match inventory"):
        capability_catalog.capability_changes(wrong_total, current)

    unknown_stage = deepcopy(current)
    name = next(iter(unknown_stage["functionals"]))
    unknown_stage["functionals"][name]["ready_stages"].append("invented-stage")
    with pytest.raises(ValueError, match="unknown stage"):
        capability_catalog.capability_changes(unknown_stage, current)
