"""Regression coverage for bulk Libxc production-domain reporting."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import (
    libxc_bulk_capabilities,
    production_domain_catalog,
)


def _stage_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    *,
    status: str,
    reason: str | None,
) -> dict:
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": "production-domain",
        "status": status,
        "reason": reason,
        "evidence": f"test://{capability.name}/production-domain",
    }
    if status == "pass":
        payload["qualification"] = capability.production_domain_profile.to_payload()
    return payload


def test_production_domain_summary_matches_canonical_inventory() -> None:
    capabilities = libxc_bulk_capabilities.available_capabilities()
    summary = production_domain_catalog.production_domain_summary()

    assert summary["status_order"] == list(production_domain_catalog.STATUS_ORDER)
    assert summary["total_functionals"] == len(capabilities)
    assert summary["total_functionals"] == len(summary["functionals"])
    assert sum(summary["status_counts"].values()) == len(capabilities)
    assert list(summary["functionals"]) == sorted(summary["functionals"])

    expected_blockers: dict[str, int] = {}
    for capability in capabilities:
        blocker = capability.production_domain_profile.blocker
        if blocker is not None:
            expected_blockers[blocker] = expected_blockers.get(blocker, 0) + 1
    assert summary["blocker_reasons"] == {
        reason: expected_blockers[reason] for reason in sorted(expected_blockers)
    }

    for capability in capabilities:
        row = summary["functionals"][capability.name]
        assert row["profile_identity"] == capability.production_domain_profile.identity
        assert row["required_ingredients"] == list(capability.required_ingredients)
        if capability.production_domain_profile.blocker is not None:
            assert row["status"] == "blocked"
            assert row["blocker"] == capability.production_domain_profile.blocker


def test_production_domain_pass_is_qualified_not_merely_ready() -> None:
    capability = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    assert capability.production_domain_profile.eligible

    summary = production_domain_catalog.production_domain_summary(
        {
            capability.name: {
                "production-domain": _stage_evidence(
                    capability,
                    status="pass",
                    reason=None,
                )
            }
        }
    )

    row = summary["functionals"][capability.name]
    assert row["status"] == "qualified"
    assert row["blocker"] is None
    assert summary["status_counts"]["qualified"] == 1
    assert summary["by_family"][capability.family]["status_counts"]["qualified"] == 1
    assert summary["by_ingredients"]["rho+sigma"]["status_counts"]["qualified"] == 1


def test_production_domain_failure_retains_actionable_blocker() -> None:
    capability = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    reason = "boundary fixture mismatch: spin/zero-b"
    summary = production_domain_catalog.production_domain_summary(
        {
            capability.name: {
                "production-domain": _stage_evidence(
                    capability,
                    status="fail",
                    reason=reason,
                )
            }
        }
    )

    row = summary["functionals"][capability.name]
    assert row["status"] == "blocked"
    assert row["blocker"] == reason
    assert summary["blocker_reasons"][reason] == 1
    assert summary["status_counts"]["qualified"] == 0


def test_render_production_domain_summary_is_deterministic() -> None:
    summary = production_domain_catalog.production_domain_summary()
    report = production_domain_catalog.render_production_domain_summary(summary)

    assert report == production_domain_catalog.render_production_domain_summary(summary)
    assert report.startswith(
        f"Libxc production-domain summary ({summary['total_functionals']} functionals)"
    )
    assert "\nfamily:\n" in report
    assert "\ningredients:\n" in report
    assert report.splitlines()[1].startswith("status: qualified=")


def test_render_production_domain_summary_rejects_stale_counts() -> None:
    summary = production_domain_catalog.production_domain_summary()

    wrong_order = deepcopy(summary)
    wrong_order["status_order"] = list(reversed(wrong_order["status_order"]))
    with pytest.raises(ValueError, match="status order mismatch"):
        production_domain_catalog.render_production_domain_summary(wrong_order)

    wrong_total = deepcopy(summary)
    wrong_total["total_functionals"] += 1
    with pytest.raises(ValueError, match="total does not match status counts"):
        production_domain_catalog.render_production_domain_summary(wrong_total)
