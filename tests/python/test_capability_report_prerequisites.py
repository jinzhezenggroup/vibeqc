"""Human reports must reject internally impossible qualification snapshots."""

from __future__ import annotations

from copy import deepcopy

import pytest
from vibeqc_compiler.xc.capability_catalog import (
    SUMMARY_SCHEMA,
    render_capability_changes,
    render_capability_summary,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import CAPABILITY_STAGES


def snapshot(qualified: tuple[str, ...] = (), ready: tuple[str, ...] = ()) -> dict:
    """Keep redundant counts valid so tests isolate prerequisite validation."""
    return {
        "schema": SUMMARY_SCHEMA,
        "stage_order": list(CAPABILITY_STAGES),
        "total_functionals": 1,
        "qualified_counts": {
            stage: int(stage in qualified) for stage in CAPABILITY_STAGES
        },
        "ready_counts": {stage: int(stage in ready) for stage in CAPABILITY_STAGES},
        "blocked_counts": dict.fromkeys(CAPABILITY_STAGES, 0),
        "functionals": {
            "TEST": {
                "identity": "test-identity",
                "qualified_stages": list(qualified),
                "ready_stages": list(ready),
                "blocked_stages": {},
                "public_dft": "public-method" in qualified,
            }
        },
    }


@pytest.mark.parametrize("stage", CAPABILITY_STAGES[1:])
@pytest.mark.parametrize("field", ("qualified", "ready"))
def test_report_rejects_stage_without_qualified_prerequisites(
    stage: str, field: str
) -> None:
    malformed = snapshot(**{field: (stage,)})
    with pytest.raises(ValueError, match="prerequisites"):
        render_capability_summary(malformed)


@pytest.mark.parametrize("bad_previous", (False, True))
def test_delta_rejects_false_public_promotion(bad_previous: bool) -> None:
    previous = snapshot(("graph-imported", "pointwise-validated"))
    current = snapshot(("graph-imported", "pointwise-validated", "public-method"))
    if bad_previous:
        previous, current = current, previous
    with pytest.raises(ValueError, match="prerequisites"):
        render_capability_changes(previous, current)


@pytest.mark.parametrize(
    "backend", (("compiled-cpu",), ("compiled-cuda", "gpu-runtime"))
)
def test_valid_cpu_and_gpu_alternatives_remain_accepted(
    backend: tuple[str, ...],
) -> None:
    qualified = ("graph-imported", "pointwise-validated", *backend, "production-domain")
    before = snapshot(qualified, ("molecular-scf",))
    after = snapshot(
        (*qualified, "molecular-scf", "public-method"), ("forces", "response")
    )
    detached = deepcopy(after)
    report = render_capability_summary(after)
    assert "public DFT: 1/1" in report
    delta = render_capability_changes(before, after)
    assert "  molecular-scf: TEST" in delta
    assert "  public-method: TEST" in delta
    assert after == detached


def test_ready_prerequisite_is_not_a_qualified_prerequisite() -> None:
    malformed = snapshot(
        ("graph-imported", "pointwise-validated"), ("compiled-cuda", "gpu-runtime")
    )
    with pytest.raises(ValueError, match="prerequisites"):
        render_capability_summary(malformed)


def test_explicit_failure_may_remain_ready_for_new_evidence() -> None:
    valid = snapshot(("graph-imported", "pointwise-validated"), ("compiled-cpu",))
    valid["functionals"]["TEST"]["blocked_stages"] = {"compiled-cpu": "compiler failed"}
    valid["blocked_counts"]["compiled-cpu"] = 1
    assert "explicit blockers: 1 across 1 functionals" in render_capability_summary(valid)
