"""Persisted summary counters and public flags must agree with their inventory."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc import capability_catalog as catalog


def snapshot() -> dict:
    """A small valid reporting snapshot, with no runtime admission implied."""
    stages = catalog.CAPABILITY_STAGES
    return {
        "schema": catalog.SUMMARY_SCHEMA,
        "stage_order": list(stages),
        "total_functionals": 1,
        "qualified_counts": {s: int(s == "graph-imported") for s in stages},
        "ready_counts": {s: int(s == "pointwise-validated") for s in stages},
        "blocked_counts": {s: 0 for s in stages},
        "functionals": {
            "TEST": {
                "identity": "test-identity",
                "qualified_stages": ["graph-imported"],
                "ready_stages": ["pointwise-validated"],
                "blocked_stages": {},
                "public_dft": False,
            }
        },
    }


@pytest.mark.parametrize("field", ("qualified_counts", "ready_counts", "blocked_counts"))
@pytest.mark.parametrize("damage", ("value", "bool", "missing", "extra"))
def test_inconsistent_counters_are_rejected(field: str, damage: str) -> None:
    original = snapshot()
    broken = deepcopy(original)
    if damage == "value":
        broken[field]["public-method"] = 1
    elif damage == "bool":
        broken[field]["public-method"] = False
    elif damage == "missing":
        del broken[field]["public-method"]
    else:
        broken[field]["invented-stage"] = 0
    with pytest.raises((ValueError, TypeError)):
        catalog.capability_changes(original, broken)
    with pytest.raises((ValueError, TypeError)):
        catalog.capability_changes(broken, original)


@pytest.mark.parametrize("field", ("qualified_stages", "ready_stages"))
def test_duplicate_stage_entries_are_rejected(field: str) -> None:
    original = snapshot()
    broken = deepcopy(original)
    entries = broken["functionals"]["TEST"][field]
    entries.append(entries[0])
    with pytest.raises(ValueError, match="duplicate"):
        catalog.capability_changes(original, broken)


@pytest.mark.parametrize("flag", (True, 0, "false", None))
def test_public_flag_cannot_contradict_qualified_stages(flag: object) -> None:
    original = snapshot()
    broken = deepcopy(original)
    broken["functionals"]["TEST"]["public_dft"] = flag
    with pytest.raises(ValueError, match="public"):
        catalog.capability_changes(original, broken)


def test_boolean_total_is_not_an_inventory_count() -> None:
    original = snapshot()
    broken = deepcopy(original)
    broken["total_functionals"] = True
    with pytest.raises(ValueError, match="total"):
        catalog.capability_changes(original, broken)


def test_explicit_blocker_cannot_also_be_qualified() -> None:
    original = snapshot()
    broken = deepcopy(original)
    broken["functionals"]["TEST"]["blocked_stages"]["graph-imported"] = "failed"
    broken["blocked_counts"]["graph-imported"] = 1
    with pytest.raises(ValueError, match="qualified/blocked"):
        catalog.capability_changes(original, broken)


def test_valid_snapshot_is_detached_and_does_not_imply_promotion() -> None:
    original = snapshot()
    before = deepcopy(original)
    delta = catalog.capability_changes(original, original)
    assert original == before
    assert delta["promotions"] == delta["demotions"] == {}
