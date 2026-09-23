"""A rendered domain report must agree with every retained functional row."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc import production_domain_catalog as catalog


@pytest.fixture
def summary(monkeypatch: pytest.MonkeyPatch) -> dict:
    ready_profile = SimpleNamespace(eligible=True, blocker=None, identity="profile-r")
    blocked_profile = SimpleNamespace(
        eligible=False, blocker="unsupported ingredient", identity="profile-b"
    )
    capabilities = (
        SimpleNamespace(
            name="GGA_READY",
            family="GGA",
            required_ingredients=("rho", "sigma"),
            production_domain_profile=ready_profile,
            qualified_stages=(),
            ready_stages=("production-domain",),
            stage_evidence=(),
        ),
        SimpleNamespace(
            name="MGGA_BLOCKED",
            family="MGGA",
            required_ingredients=("rho", "sigma", "laplacian"),
            production_domain_profile=blocked_profile,
            qualified_stages=(),
            ready_stages=(),
            stage_evidence=(),
        ),
    )
    monkeypatch.setattr(catalog, "available_capabilities", lambda evidence: capabilities)
    return catalog.production_domain_summary()


@pytest.mark.parametrize(
    "mutation",
    (
        "status-counts",
        "family-counts",
        "ingredient-counts",
        "lost-blocker",
        "lost-row",
        "renamed-row",
        "changed-status",
        "changed-ingredients",
        "stale-blocker",
        "missing-blocker",
    ),
)
def test_balanced_but_stale_report_is_rejected(summary: dict, mutation: str) -> None:
    if mutation == "status-counts":
        summary["status_counts"].update(qualified=1, ready=0)
    elif mutation == "family-counts":
        summary["by_family"]["GGA"]["status_counts"].update(qualified=1, ready=0)
    elif mutation == "ingredient-counts":
        summary["by_ingredients"]["rho+sigma"]["status_counts"].update(
            qualified=1, ready=0
        )
    elif mutation == "lost-blocker":
        summary["blocker_reasons"].clear()
    elif mutation == "lost-row":
        del summary["functionals"]["MGGA_BLOCKED"]
    elif mutation == "renamed-row":
        summary["functionals"]["GGA_READY"]["name"] = "OTHER"
    elif mutation == "changed-status":
        summary["functionals"]["GGA_READY"]["status"] = "qualified"
    elif mutation == "changed-ingredients":
        summary["functionals"]["GGA_READY"]["required_ingredients"] = ["rho"]
    elif mutation == "stale-blocker":
        summary["functionals"]["GGA_READY"]["blocker"] = "old exception"
    else:
        summary["functionals"]["MGGA_BLOCKED"]["blocker"] = None
    with pytest.raises(ValueError):
        catalog.render_production_domain_summary(summary)


@pytest.mark.parametrize("scope", ("global", "family", "ingredients"))
@pytest.mark.parametrize("counter", (True, 1.0, -1))
def test_counters_are_nonnegative_integers(
    summary: dict, scope: str, counter: object
) -> None:
    if scope == "global":
        counts = summary["status_counts"]
    elif scope == "family":
        counts = summary["by_family"]["GGA"]["status_counts"]
    else:
        counts = summary["by_ingredients"]["rho+sigma"]["status_counts"]
    counts["ready"] = counter
    with pytest.raises(ValueError):
        catalog.render_production_domain_summary(summary)


def test_valid_render_preserves_rows_and_is_order_independent(summary: dict) -> None:
    original = deepcopy(summary)
    rendered = catalog.render_production_domain_summary(summary)
    assert "status: qualified=0  ready=1  blocked=1  pending=0" in rendered
    assert "  unsupported ingredient: 1" in rendered
    assert summary == original
    for field in ("functionals", "by_family", "by_ingredients", "status_counts"):
        summary[field] = dict(reversed(tuple(summary[field].items())))
    assert catalog.render_production_domain_summary(summary) == rendered


def test_empty_generated_inventory_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "available_capabilities", lambda evidence: ())
    report = catalog.render_production_domain_summary(catalog.production_domain_summary())
    assert report.startswith("Libxc production-domain summary (0 functionals)")
    assert report.endswith("blockers: none")
