"""#192 adaptive refinement must be conditioned, hysteretic and fail closed."""

from dataclasses import replace

import pytest
from vibeqc.accuracy import (
    ErrorEvidence,
    EvidenceKind,
    ObservableTarget,
    ResolvedModel,
    TargetAccuracy,
)
from vibeqc.progressive_controller import (
    ArithmeticPolicy,
    HFConvergence,
    StagePlan,
    StageRole,
    TargetProblem,
    TransferOperation,
)
from vibeqc.progressive_guard import (
    RefinementGuard,
    RefinementIdentity,
    decide_guarded_refinement,
)

TARGET = ResolvedModel("rhf", "g" * 64, "b" * 64, 2)
SOURCE = ResolvedModel("rhf", "g" * 64, "s" * 64, 2)
PROBLEM = TargetProblem(
    TARGET,
    "a" * 64,
    (("orbital.mathematical_identity", "c" * 64),),
    TargetAccuracy((ObservableTarget("energy", "absolute", "Eh", absolute=1e-5),)),
    HFConvergence(1e-10, 1e-8, 100, 1e-8),
    2,
)
GUARD = RefinementGuard(
    enter_ratio=1.2,
    release_ratio=0.8,
    maximum_condition_estimate=10.0,
)
IDENTITY = RefinementIdentity("basis", "energy", "absolute", "Eh")


def _evidence(
    value: float,
    *,
    kind: EvidenceKind = EvidenceKind.OBSERVED,
    condition_estimate: float | None = None,
) -> ErrorEvidence:
    estimated = kind is not EvidenceKind.OBSERVED
    return ErrorEvidence(
        kind,
        TARGET.identity,
        SOURCE.identity,
        TARGET.identity,
        "energy",
        "absolute",
        "Eh",
        value,
        1.0,
        "basis",
        "relaxed_target",
        (("reference", "independent-comparison"),),
        assumptions=("validated estimator domain",) if estimated else (),
        calibration_id="basis-estimator-v1" if estimated else None,
        condition_estimate=condition_estimate,
        actual_reference_error=None if estimated else value,
    )


def _stage(item: ErrorEvidence) -> StagePlan:
    return StagePlan(
        "initial-hf",
        StageRole.INITIALIZATION,
        SOURCE,
        "d" * 64,
        (("orbital.mathematical_identity", "e" * 64),),
        ArithmeticPolicy("fp64"),
        HFConvergence(1e-6, 1e-5, 20),
        TransferOperation.METRIC_PROJECTED_DENSITY,
        error_evidence=(item,),
        allowed_next_stages=("target-hf",),
    )


def test_inactive_source_requires_enter_threshold() -> None:
    below = decide_guarded_refinement(PROBLEM, _stage(_evidence(1.1e-5)), GUARD)
    above = decide_guarded_refinement(PROBLEM, _stage(_evidence(1.3e-5)), GUARD)

    assert not below.requires_refinement
    assert below.candidate is None
    assert above.action == "start_refinement"
    assert above.requires_refinement
    assert above.candidate is not None
    assert above.candidate.source == "basis"


def test_active_source_uses_lower_release_threshold() -> None:
    decision = decide_guarded_refinement(
        PROBLEM,
        _stage(_evidence(0.9e-5)),
        GUARD,
        active=IDENTITY,
    )

    assert decision.action == "continue_refinement"
    assert decision.requires_refinement


def test_active_source_releases_below_hysteresis_band() -> None:
    decision = decide_guarded_refinement(
        PROBLEM,
        _stage(_evidence(0.7e-5)),
        GUARD,
        active=IDENTITY,
    )

    assert decision.action == "no_refinement"
    assert decision.candidate is None


@pytest.mark.parametrize("condition", (None, 10.1))
def test_untrusted_estimator_falls_back_to_exact_target(
    condition: float | None,
) -> None:
    decision = decide_guarded_refinement(
        PROBLEM,
        _stage(
            _evidence(
                1.3e-5,
                kind=EvidenceKind.EMPIRICAL,
                condition_estimate=condition,
            )
        ),
        GUARD,
    )

    assert decision.action == "target_baseline"
    assert decision.requires_target_baseline
    assert not decision.requires_refinement


def test_conditioned_estimator_can_request_refinement() -> None:
    decision = decide_guarded_refinement(
        PROBLEM,
        _stage(
            _evidence(
                1.3e-5,
                kind=EvidenceKind.ASYMPTOTIC,
                condition_estimate=10.0,
            )
        ),
        GUARD,
    )

    assert decision.action == "start_refinement"
    assert decision.requires_refinement


def test_observed_evidence_does_not_require_estimator_conditioning() -> None:
    decision = decide_guarded_refinement(PROBLEM, _stage(_evidence(1.3e-5)), GUARD)

    assert decision.action == "start_refinement"
    assert "observed evidence" in decision.reason


def test_stale_active_identity_fails_closed() -> None:
    stale = replace(IDENTITY, source="grid")
    with pytest.raises(ValueError, match="stale"):
        decide_guarded_refinement(
            PROBLEM,
            _stage(_evidence(1.3e-5)),
            GUARD,
            active=stale,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"enter_ratio": 1.0}, "enter_ratio"),
        ({"release_ratio": 1.0}, "release_ratio"),
        ({"maximum_condition_estimate": float("inf")}, "maximum_condition_estimate"),
    ),
)
def test_invalid_guard_limits_are_rejected(
    kwargs: dict[str, float], message: str
) -> None:
    values = {
        "enter_ratio": 1.2,
        "release_ratio": 0.8,
        "maximum_condition_estimate": 10.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RefinementGuard(**values)


@pytest.mark.parametrize("active", (None, IDENTITY))
@pytest.mark.parametrize("scope", ("relaxed_target", "fixed_density"))
@pytest.mark.parametrize("reverse", (False, True))
def test_other_scope_cannot_shadow_selected_observation(
    active: RefinementIdentity | None, scope: str, reverse: bool
) -> None:
    selected = replace(_evidence(1.3e-5), scope=scope)
    other = replace(
        _evidence(1e10, kind=EvidenceKind.EMPIRICAL, condition_estimate=None),
        scope="fixed_density" if scope == "relaxed_target" else "relaxed_target",
    )
    evidence = (selected, other) if not reverse else (other, selected)
    problem = replace(PROBLEM, accuracy=replace(PROBLEM.accuracy, scope=scope))
    stage = replace(_stage(selected), error_evidence=evidence)
    actual = decide_guarded_refinement(problem, stage, GUARD, active=active)
    expected = decide_guarded_refinement(
        problem, _stage(selected), GUARD, active=active
    )
    assert actual == expected
    assert actual.requires_refinement
    assert not actual.requires_target_baseline


@pytest.mark.parametrize("reverse", (False, True))
def test_other_scope_observation_cannot_bypass_conditioning(reverse: bool) -> None:
    selected = _evidence(1.3e-5, kind=EvidenceKind.ASYMPTOTIC, condition_estimate=None)
    other = replace(_evidence(1.3e-5), scope="fixed_density")
    evidence = (selected, other) if not reverse else (other, selected)
    stage = replace(_stage(selected), error_evidence=evidence)
    actual = decide_guarded_refinement(PROBLEM, stage, GUARD)
    assert actual.requires_target_baseline
    assert not actual.requires_refinement


def test_duplicate_evidence_in_selected_scope_still_rejected() -> None:
    item = _evidence(1.3e-5)
    with pytest.raises(ValueError, match="duplicate actionable"):
        decide_guarded_refinement(
            PROBLEM, replace(_stage(item), error_evidence=(item, item)), GUARD
        )
