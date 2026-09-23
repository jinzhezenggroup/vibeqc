"""Regression checks for fail-closed Libxc capability transitions."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import capability_catalog, capability_transition
from vibeqc_compiler.xc.libxc_bulk_capabilities import CAPABILITY_STAGES


def _snapshot(
    *,
    name: str = "TEST",
    identity: str = "identity-v1",
    qualified: tuple[str, ...] = ("graph-imported", "pointwise-validated"),
) -> dict:
    ready = [
        stage
        for stage in CAPABILITY_STAGES
        if stage not in qualified
        and stage in ("compiled-cpu", "compiled-cuda", "production-domain")
    ]
    summary = {
        "schema": capability_catalog.SUMMARY_SCHEMA,
        "stage_order": list(CAPABILITY_STAGES),
        "total_functionals": 1,
        "qualified_counts": {
            stage: int(stage in qualified) for stage in CAPABILITY_STAGES
        },
        "ready_counts": {stage: int(stage in ready) for stage in CAPABILITY_STAGES},
        "blocked_counts": dict.fromkeys(CAPABILITY_STAGES, 0),
        "functionals": {
            name: {
                "identity": identity,
                "qualified_stages": list(qualified),
                "ready_stages": ready,
                "blocked_stages": {},
                "public_dft": "public-method" in qualified,
            }
        },
    }
    return summary


def test_promotion_only_transition_needs_no_acknowledgement() -> None:
    before = _snapshot()
    after = _snapshot(
        qualified=("graph-imported", "pointwise-validated", "compiled-cpu")
    )

    assert capability_transition.capability_regressions(before, after) == ()
    assert capability_transition.require_acknowledged_capability_regressions(
        before, after
    ) == ()


def test_stage_demotion_requires_exact_acknowledgement() -> None:
    before = _snapshot(
        qualified=("graph-imported", "pointwise-validated", "compiled-cpu")
    )
    after = _snapshot()

    regressions = capability_transition.capability_regressions(before, after)
    assert [item.token for item in regressions] == [
        "stage-demotion:compiled-cpu:TEST"
    ]
    with pytest.raises(ValueError, match="unacknowledged capability regression"):
        capability_transition.require_acknowledged_capability_regressions(before, after)
    assert capability_transition.require_acknowledged_capability_regressions(
        before,
        after,
        acknowledged=("stage-demotion:compiled-cpu:TEST",),
    ) == regressions


def test_identity_change_and_removal_are_independent_regressions() -> None:
    before = _snapshot()
    changed = _snapshot(identity="identity-v2")
    changed_regressions = capability_transition.capability_regressions(before, changed)
    assert [item.token for item in changed_regressions] == ["identity-change:TEST"]

    removed = deepcopy(before)
    removed["functionals"] = {}
    removed["total_functionals"] = 0
    for counts in ("qualified_counts", "ready_counts", "blocked_counts"):
        removed[counts] = dict.fromkeys(CAPABILITY_STAGES, 0)
    removed_regressions = capability_transition.capability_regressions(before, removed)
    assert [item.token for item in removed_regressions] == ["removed-functional:TEST"]


def test_stale_or_duplicate_acknowledgement_fails_closed() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="stale or unknown"):
        capability_transition.require_acknowledged_capability_regressions(
            snapshot,
            snapshot,
            acknowledged=("identity-change:TEST",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        capability_transition.require_acknowledged_capability_regressions(
            snapshot,
            snapshot,
            acknowledged=("identity-change:TEST", "identity-change:TEST"),
        )


def test_transition_reuses_catalog_validation() -> None:
    valid = _snapshot()
    malformed = deepcopy(valid)
    malformed["qualified_counts"]["compiled-cpu"] = 1

    with pytest.raises(ValueError, match="inconsistent qualified_counts"):
        capability_transition.capability_regressions(valid, malformed)
