"""#192 refinement choices compare error evidence only through typed allowances."""

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

TARGET_MODEL = ResolvedModel("rhf", "g" * 64, "b" * 64, 2)
SOURCE_MODEL = ResolvedModel("rhf", "g" * 64, "s" * 64, 2)
ACCURACY = TargetAccuracy(
    (
        ObservableTarget("energy", "absolute", "Eh", absolute=1.0e-5),
        ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1.0e-3),
    )
)
PROBLEM = TargetProblem(
    TARGET_MODEL,
    "a" * 64,
    (("orbital.mathematical_identity", "c" * 64),),
    ACCURACY,
    HFConvergence(1.0e-10, 1.0e-8, 100, 1.0e-8),
    2,
)


def evidence(
    source: str,
    observable: str,
    norm: str,
    unit: str,
    value: float,
    *,
    scope: str = "relaxed_target",
    target_model_id: str | None = None,
    evaluated_model_id: str | None = None,
) -> ErrorEvidence:
    return ErrorEvidence(
        EvidenceKind.OBSERVED,
        TARGET_MODEL.identity if target_model_id is None else target_model_id,
        SOURCE_MODEL.identity if evaluated_model_id is None else evaluated_model_id,
        TARGET_MODEL.identity,
        observable,
        norm,
        unit,
        value,
        1.0,
        source,
        scope,
        (("reference", "independent-comparison"),),
        actual_reference_error=value,
    )


def stage(*items: ErrorEvidence) -> StagePlan:
    return StagePlan(
        "initial-hf",
        StageRole.INITIALIZATION,
        SOURCE_MODEL,
        "d" * 64,
        (("orbital.mathematical_identity", "e" * 64),),
        ArithmeticPolicy("fp64"),
        HFConvergence(1.0e-6, 1.0e-5, 20),
        TransferOperation.METRIC_PROJECTED_DENSITY,
        error_evidence=items,
        allowed_next_stages=("target-hf",),
    )


def test_ranking_uses_dimensionless_allowances_not_raw_magnitudes() -> None:
    selected = stage(
        evidence("basis", "energy", "absolute", "Eh", 2.0e-5),
        evidence("grid", "forces", "max_abs", "Eh/bohr", 5.0e-4),
    )

    ranked = rank_refinement_candidates(PROBLEM, selected)
    decision = decide_refinement(PROBLEM, selected)

    assert [item.source for item in ranked] == ["basis", "grid"]
    assert ranked[0].normalized_error == pytest.approx(2.0)
    assert ranked[1].normalized_error == pytest.approx(0.5)
    assert decision.requires_refinement
    assert decision.candidate == ranked[0]


def test_total_error_and_wrong_scope_cannot_name_a_refinement_source() -> None:
    selected = stage(
        evidence("total_numerical", "energy", "absolute", "Eh", 2.0e-5),
        evidence(
            "basis",
            "energy",
            "absolute",
            "Eh",
            3.0e-5,
            scope="fixed_density",
        ),
    )

    assert rank_refinement_candidates(PROBLEM, selected) == ()
    decision = decide_refinement(PROBLEM, selected)
    assert not decision.requires_refinement
    assert decision.candidate is None
    assert decision.candidate_count == 0


def test_known_sources_within_allowance_do_not_trigger_adapter_refinement() -> None:
    selected = stage(
        evidence("basis", "energy", "absolute", "Eh", 9.0e-6),
        evidence("grid", "forces", "max_abs", "Eh/bohr", 8.0e-4),
    )

    decision = decide_refinement(PROBLEM, selected)

    assert not decision.requires_refinement
    assert decision.candidate is None
    assert decision.candidate_count == 2


def test_ties_are_deterministic_without_combining_independent_sources() -> None:
    selected = stage(
        evidence("zeta", "energy", "absolute", "Eh", 2.0e-5),
        evidence("alpha", "forces", "max_abs", "Eh/bohr", 2.0e-3),
    )

    ranked = rank_refinement_candidates(PROBLEM, selected)

    assert [item.source for item in ranked] == ["alpha", "zeta"]
    assert all(item.normalized_error == pytest.approx(2.0) for item in ranked)


def test_duplicate_actionable_evidence_fails_closed() -> None:
    duplicate = evidence("basis", "energy", "absolute", "Eh", 2.0e-5)
    with pytest.raises(ValueError, match="duplicate actionable"):
        rank_refinement_candidates(PROBLEM, stage(duplicate, duplicate))


def test_mismatched_target_or_evaluated_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="requested target"):
        rank_refinement_candidates(
            PROBLEM,
            stage(
                evidence(
                    "basis",
                    "energy",
                    "absolute",
                    "Eh",
                    2.0e-5,
                    target_model_id="x" * 64,
                )
            ),
        )
    with pytest.raises(ValueError, match="evaluated stage"):
        rank_refinement_candidates(
            PROBLEM,
            stage(
                evidence(
                    "basis",
                    "energy",
                    "absolute",
                    "Eh",
                    2.0e-5,
                    evaluated_model_id=TARGET_MODEL.identity,
                )
            ),
        )


def test_incompatible_or_target_stage_is_rejected() -> None:
    selected = stage(evidence("basis", "energy", "absolute", "Eh", 2.0e-5))
    target_stage = replace(selected, role=StageRole.TARGET)
    with pytest.raises(ValueError, match="initialization stages"):
        decide_refinement(PROBLEM, target_stage)

    incompatible = replace(
        selected,
        model=ResolvedModel("rhf", "z" * 64, "s" * 64, 2),
        error_evidence=(),
    )
    with pytest.raises(ValueError, match="geometry_hash"):
        decide_refinement(PROBLEM, incompatible)
