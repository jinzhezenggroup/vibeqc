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
    assert (
        capability_transition.require_acknowledged_capability_regressions(before, after)
        == ()
    )


def test_stage_demotion_requires_exact_acknowledgement() -> None:
    before = _snapshot(
        qualified=("graph-imported", "pointwise-validated", "compiled-cpu")
    )
    after = _snapshot()

    regressions = capability_transition.capability_regressions(before, after)
    assert [(item.kind, item.functional, item.stage) for item in regressions] == [
        ("stage-demotion", "TEST", "compiled-cpu")
    ]
    token = regressions[0].token
    assert token.startswith("stage-demotion:compiled-cpu:TEST:")
    assert len(token.rsplit(":", 1)[1]) == 64
    with pytest.raises(ValueError, match="unacknowledged capability regression"):
        capability_transition.require_acknowledged_capability_regressions(before, after)
    assert (
        capability_transition.require_acknowledged_capability_regressions(
            before,
            after,
            acknowledged=(token,),
        )
        == regressions
    )


def test_identity_change_and_removal_are_independent_regressions() -> None:
    before = _snapshot()
    changed = _snapshot(identity="identity-v2")
    changed_regressions = capability_transition.capability_regressions(before, changed)
    assert [(item.kind, item.functional) for item in changed_regressions] == [
        ("identity-change", "TEST")
    ]

    removed = deepcopy(before)
    removed["functionals"] = {}
    removed["total_functionals"] = 0
    for counts in ("qualified_counts", "ready_counts", "blocked_counts"):
        removed[counts] = dict.fromkeys(CAPABILITY_STAGES, 0)
    removed_regressions = capability_transition.capability_regressions(before, removed)
    assert [(item.kind, item.functional) for item in removed_regressions] == [
        ("removed-functional", "TEST")
    ]


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


def _empty_snapshot() -> dict:
    snapshot = _snapshot()
    snapshot["functionals"] = {}
    snapshot["total_functionals"] = 0
    for field in ("qualified_counts", "ready_counts", "blocked_counts"):
        snapshot[field] = dict.fromkeys(CAPABILITY_STAGES, 0)
    return snapshot


@pytest.mark.parametrize(
    "scenario", ("next-identity", "different-target", "reverse", "removal", "demotion")
)
def test_acknowledgement_cannot_authorize_a_different_transition(scenario: str) -> None:
    first_before, first_after = _snapshot(), _snapshot(identity="identity-v2")
    second_before, second_after = first_after, _snapshot(identity="identity-v3")
    if scenario == "different-target":
        second_before = first_before
    elif scenario == "reverse":
        second_after = first_before
    elif scenario == "removal":
        first_after = second_after = _empty_snapshot()
        second_before = _snapshot(identity="identity-v2")
    elif scenario == "demotion":
        stages = ("graph-imported", "pointwise-validated", "compiled-cpu")
        first_before = _snapshot(qualified=stages)
        first_after = _snapshot()
        second_before = _snapshot(identity="identity-v2", qualified=stages)
        second_after = _snapshot(identity="identity-v2")
    approved = tuple(
        item.token
        for item in capability_transition.capability_regressions(
            first_before, first_after
        )
    )
    assert approved
    with pytest.raises(ValueError, match="stale or unknown"):
        capability_transition.require_acknowledged_capability_regressions(
            second_before, second_after, acknowledged=approved
        )


def test_token_ignores_unrelated_functionals_and_stage_order() -> None:
    before, after = _snapshot(), _snapshot(identity="identity-v2")
    expected = capability_transition.capability_regressions(before, after)
    for snapshot in (before, after):
        record = snapshot["functionals"]["TEST"]
        record["qualified_stages"].reverse()
        record["ready_stages"].reverse()
        other = _snapshot(name="OTHER")["functionals"]["OTHER"]
        snapshot["functionals"]["OTHER"] = other
        snapshot["total_functionals"] += 1
        for field, inventory in (
            ("qualified_counts", "qualified_stages"),
            ("ready_counts", "ready_stages"),
            ("blocked_counts", "blocked_stages"),
        ):
            for stage in other[inventory]:
                snapshot[field][stage] += 1
    assert capability_transition.capability_regressions(before, after) == expected
    assert (
        capability_transition.require_acknowledged_capability_regressions(
            before, after, acknowledged=(item.token for item in expected)
        )
        == expected
    )


def test_regression_identity_is_detached_from_input_snapshots() -> None:
    before, after = _snapshot(), _snapshot(identity="identity-v2")
    regressions = capability_transition.capability_regressions(before, after)
    tokens = tuple(item.token for item in regressions)
    after["functionals"]["TEST"]["identity"] = "identity-v3"
    assert tuple(item.token for item in regressions) == tokens
    assert (
        tuple(
            item.token
            for item in capability_transition.capability_regressions(before, after)
        )
        != tokens
    )
