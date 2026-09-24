"""Exercise legal zero allowances, extreme ratios and comparator identity."""

import math
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
from vibeqc.progressive_policy import decide_refinement, rank_refinement_candidates

TARGET = ResolvedModel("rhf", "g" * 64, "b" * 64, 2)
SOURCE = ResolvedModel("rhf", "g" * 64, "s" * 64, 2)


def _problem(target: ObservableTarget) -> TargetProblem:
    return TargetProblem(
        TARGET,
        "a" * 64,
        (("orbital.mathematical_identity", "c" * 64),),
        TargetAccuracy((target,)),
        HFConvergence(1e-10, 1e-8, 100, 1e-8),
        2,
    )


def _evidence(source: str, value: float, reference_norm: float = 0.0) -> ErrorEvidence:
    return ErrorEvidence(
        EvidenceKind.OBSERVED,
        TARGET.identity,
        SOURCE.identity,
        TARGET.identity,
        "energy",
        "absolute",
        "Eh",
        value,
        reference_norm,
        source,
        "relaxed_target",
        (("reference", "independent-comparison"),),
        actual_reference_error=value,
    )


def _stage(*items: ErrorEvidence) -> StagePlan:
    return StagePlan(
        "initial-hf",
        StageRole.INITIALIZATION,
        SOURCE,
        "d" * 64,
        (("orbital.mathematical_identity", "e" * 64),),
        ArithmeticPolicy("fp64"),
        HFConvergence(1e-6, 1e-5, 20),
        TransferOperation.METRIC_PROJECTED_DENSITY,
        error_evidence=items,
        allowed_next_stages=("target-hf",),
    )


@pytest.mark.parametrize("value", (0.0, 1e-5))
@pytest.mark.parametrize("reference_norm", (0.0, 5e-324))
def test_zero_relative_allowance_has_defined_refinement_semantics(
    value: float, reference_norm: float
) -> None:
    # Exact zero and finite multiplication underflow are valid contract inputs.
    problem = _problem(ObservableTarget("energy", "absolute", "Eh", relative=1e-3))
    selected = _stage(_evidence("basis", value, reference_norm))
    ranked = rank_refinement_candidates(problem, selected)
    assert ranked[0].allowance == 0
    assert ranked[0].normalized_error == (0.0 if value == 0 else math.inf)
    assert decide_refinement(problem, selected).requires_refinement is (value > 0)


def test_comparator_must_be_bound_to_requested_target() -> None:
    item = replace(_evidence("basis", 1e-5), reference_model_id="wrong-reference")
    with pytest.raises(ValueError, match="reference model"):
        rank_refinement_candidates(
            _problem(ObservableTarget("energy", "absolute", "Eh", absolute=1e-6)),
            _stage(item),
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_finite_ratios_that_overflow_do_not_turn_into_alphabetic_ties(
    reverse: bool,
) -> None:
    problem = _problem(ObservableTarget("energy", "absolute", "Eh", absolute=5e-324))
    items = [_evidence("alpha", 1.0), _evidence("zeta", 2.0)]
    if reverse:
        items.reverse()
    ranked = rank_refinement_candidates(problem, _stage(*items))
    assert [item.source for item in ranked] == ["zeta", "alpha"]
    assert all(math.isinf(item.normalized_error) for item in ranked)


def test_underflowed_display_ratios_retain_actual_order() -> None:
    problem = _problem(ObservableTarget("energy", "absolute", "Eh", absolute=1e308))
    ranked = rank_refinement_candidates(
        problem, _stage(_evidence("alpha", 5e-324), _evidence("zeta", 1e-323))
    )
    assert [item.source for item in ranked] == ["zeta", "alpha"]
    assert all(item.normalized_error == 0 for item in ranked)
    decision = decide_refinement(problem, _stage(_evidence("basis", 1e-323)))
    assert not decision.requires_refinement


def test_positive_error_at_zero_allowance_dominates_finite_overflowed_ratio() -> None:
    problem = _problem(ObservableTarget("energy", "absolute", "Eh", relative=1.0))
    ranked = rank_refinement_candidates(
        problem,
        _stage(_evidence("alpha", 2.0, 5e-324), _evidence("zeta", 1e-6, 0)),
    )
    assert [item.source for item in ranked] == ["zeta", "alpha"]


@pytest.mark.parametrize("value,refine", ((0.5e-5, False), (1e-5, False), (2e-5, True)))
def test_regular_threshold_semantics_are_unchanged(value: float, refine: bool) -> None:
    problem = _problem(ObservableTarget("energy", "absolute", "Eh", absolute=1e-5))
    decision = decide_refinement(problem, _stage(_evidence("basis", value)))
    assert decision.requires_refinement is refine
