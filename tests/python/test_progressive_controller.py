"""#192's controller must never promote an initialization into a target result."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from vibeqc.accuracy import (
    AccuracyAssessment,
    ErrorEvidence,
    EvidenceKind,
    ObservableTarget,
    ResolvedModel,
    TargetAccuracy,
)
from vibeqc.progressive_controller import (
    ArithmeticPolicy,
    DeterministicHFPlan,
    FinalVerification,
    HFConvergence,
    ProgressiveBudget,
    StageExecution,
    StagePlan,
    StageRole,
    TargetProblem,
    TransferOperation,
    finalize_hf_verification,
)


def model(basis: str = "b" * 64, geometry: str = "g" * 64) -> ResolvedModel:
    return ResolvedModel("rhf", geometry, basis, 2)


TARGET_MODEL = model()
SOURCE_MODEL = model("s" * 64)
TARGET_ACCURACY = TargetAccuracy(
    (ObservableTarget("energy", "absolute", "Eh", absolute=1.0e-8),)
)
TARGET_HASHES = (("orbital.mathematical_identity", "c" * 64),)
SOURCE_HASHES = (("orbital.mathematical_identity", "d" * 64),)
TARGET_CONVERGENCE = HFConvergence(1.0e-10, 1.0e-8, 100, 1.0e-8)
PROBLEM = TargetProblem(
    TARGET_MODEL,
    "a" * 64,
    TARGET_HASHES,
    TARGET_ACCURACY,
    TARGET_CONVERGENCE,
)


def stages(
    *, arithmetic: ArithmeticPolicy | None = None
) -> tuple[StagePlan, StagePlan]:
    source = StagePlan(
        "initial-hf",
        StageRole.INITIALIZATION,
        SOURCE_MODEL,
        "b" * 64,
        SOURCE_HASHES,
        ArithmeticPolicy("fp64"),
        HFConvergence(1.0e-6, 1.0e-5, 20),
        TransferOperation.METRIC_PROJECTED_DENSITY,
        1.0,
        allowed_next_stages=("target-hf",),
    )
    target = StagePlan(
        "target-hf",
        StageRole.TARGET,
        TARGET_MODEL,
        "a" * 64,
        TARGET_HASHES,
        ArithmeticPolicy("fp64") if arithmetic is None else arithmetic,
        TARGET_CONVERGENCE,
        TransferOperation.NONE,
        4.0,
    )
    return source, target


def plan(
    *,
    arithmetic: ArithmeticPolicy | None = None,
    budget: ProgressiveBudget | None = None,
) -> DeterministicHFPlan:
    return DeterministicHFPlan(
        PROBLEM,
        stages(arithmetic=arithmetic),
        ProgressiveBudget(maximum_source_iterations=20, maximum_total_iterations=120)
        if budget is None
        else budget,
    )


def execution(
    role: StageRole, *, failed: bool = False, fock_builds: int | None = 2
) -> StageExecution:
    return StageExecution(
        "initial-hf" if role is StageRole.INITIALIZATION else "target-hf",
        role,
        "failed" if failed else "succeeded",
        SOURCE_MODEL.identity
        if role is StageRole.INITIALIZATION
        else TARGET_MODEL.identity,
        "b" * 64 if role is StageRole.INITIALIZATION else "a" * 64,
        3,
        fock_builds,
        0.1,
        1.0e-12 if role is StageRole.TARGET else None,
        "skipped_failed_source" if failed else "accepted",
        "cold" if failed else "basis_projection",
        "failed" if failed else "success",
    )


def assessment(*, observed: bool) -> AccuracyAssessment:
    evidence = ()
    if observed:
        evidence = (
            ErrorEvidence(
                EvidenceKind.OBSERVED,
                TARGET_MODEL.identity,
                TARGET_MODEL.identity,
                TARGET_MODEL.identity,
                "energy",
                "absolute",
                "Eh",
                1.0e-12,
                1.0,
                "total_numerical",
                "relaxed_target",
                (("reference", "strict-independent-audit"),),
                actual_reference_error=1.0e-12,
            ),
        )
    return AccuracyAssessment(TARGET_MODEL, TARGET_ACCURACY, evidence)


def result(
    *, accuracy: AccuracyAssessment | None, precision: dict | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        succeeded=True,
        converged=True,
        energy=-1.0,
        energy_change=1.0e-12,
        density_rms=1.0e-12,
        physical_residual_rms=1.0e-12,
        forces=None,
        precision={
            "requested_mode": "fp64",
            "effective_bits": 64,
            "strict_refinement_applied": False,
        }
        if precision is None
        else precision,
        basis_metadata={"model_identity": "a" * 64},
        accuracy=accuracy,
    )


def verify(
    output: SimpleNamespace,
    *,
    selected_plan: DeterministicHFPlan | None = None,
    actual_model: ResolvedModel = TARGET_MODEL,
    hashes: tuple[tuple[str, str], ...] = TARGET_HASHES,
    executions: tuple[StageExecution, ...] | None = None,
) -> FinalVerification:
    selected = plan() if selected_plan is None else selected_plan
    history = (
        (execution(StageRole.INITIALIZATION), execution(StageRole.TARGET))
        if executions is None
        else executions
    )
    return finalize_hf_verification(
        PROBLEM,
        selected.stages[1],
        output,
        actual_model,
        hashes,
        selected.budget,
        history,
    )


def test_target_problem_and_stage_plan_are_immutable_and_identity_stable() -> None:
    before = PROBLEM.identity
    with pytest.raises(FrozenInstanceError):
        PROBLEM.provider_identity = "x" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stages()[0].allowed_next_stages = ()  # type: ignore[misc]
    assert PROBLEM.identity == before


def test_plan_rejects_substituted_target_and_unbounded_work() -> None:
    source, target = stages()
    with pytest.raises(ValueError, match="exact TargetProblem"):
        DeterministicHFPlan(PROBLEM, (source, replace(target, model=SOURCE_MODEL)))
    with pytest.raises(ValueError, match="planned iterations"):
        DeterministicHFPlan(
            PROBLEM,
            (source, target),
            ProgressiveBudget(
                maximum_source_iterations=20,
                maximum_total_iterations=119,
            ),
        )
    with pytest.raises(ValueError, match="stage cost"):
        DeterministicHFPlan(
            PROBLEM,
            (source, target),
            ProgressiveBudget(
                maximum_source_iterations=20,
                maximum_total_iterations=120,
                maximum_estimated_cost_units=4.9,
            ),
        )


def test_plan_rejects_incompatible_source_and_non_strict_target() -> None:
    source, target = stages()
    incompatible = replace(source, model=model("s" * 64, "z" * 64))
    with pytest.raises(ValueError, match="geometry_hash"):
        DeterministicHFPlan(PROBLEM, (incompatible, target))
    with pytest.raises(ValueError, match="strict target refinement"):
        ArithmeticPolicy("auto")


def test_exact_target_requires_independent_accuracy_before_verified_status() -> None:
    final = verify(result(accuracy=assessment(observed=False)))
    assert final.status == "unverified"
    assert final.target_established
    assert final.accuracy_status == "unverified"
    assert not final.succeeded


def test_observed_accuracy_and_exact_target_produce_verified_record() -> None:
    final = verify(result(accuracy=assessment(observed=True)))
    assert final.status == "verified"
    assert final.succeeded and final.target_established
    assert final.actual_model_identity == final.requested_model_identity
    assert final.actual_provider_hashes == final.requested_provider_hashes


def test_failed_source_is_charged_but_cannot_invalidate_verified_cold_target() -> None:
    history = (
        execution(StageRole.INITIALIZATION, failed=True),
        execution(StageRole.TARGET),
    )
    final = verify(result(accuracy=assessment(observed=True)), executions=history)
    assert final.status == "verified"
    assert final.total_iterations == 6


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"actual_model": model("x" * 64)}, "scientific model"),
        ({"hashes": (("orbital.mathematical_identity", "e" * 64),)}, "provider hashes"),
    ],
)
def test_target_identity_substitution_is_unmet(change: dict, reason: str) -> None:
    final = verify(result(accuracy=assessment(observed=True)), **change)
    assert final.status == "unmet" and not final.target_established
    assert any(reason in item for item in final.reasons)


def test_missing_physical_residual_and_auto_cleanup_are_unmet() -> None:
    missing = result(accuracy=assessment(observed=True))
    missing.physical_residual_rms = None
    assert verify(missing).status == "unmet"
    auto_plan = plan(
        arithmetic=ArithmeticPolicy("auto", require_strict_refinement=True)
    )
    no_cleanup = result(
        accuracy=assessment(observed=True),
        precision={
            "requested_mode": "auto",
            "effective_bits": 64,
            "strict_refinement_applied": False,
        },
    )
    final = verify(no_cleanup, selected_plan=auto_plan)
    assert final.status == "unmet" and final.strict_cleanup == "missing"


def test_unverifiable_fock_budget_fails_closed() -> None:
    selected = plan(
        budget=ProgressiveBudget(
            maximum_source_iterations=20,
            maximum_total_iterations=120,
            maximum_total_fock_builds=20,
        )
    )
    history = (
        execution(StageRole.INITIALIZATION, fock_builds=None),
        execution(StageRole.TARGET),
    )
    final = verify(
        result(accuracy=assessment(observed=True)),
        selected_plan=selected,
        executions=history,
    )
    assert final.status == "budget_exhausted"
    assert any("could not be verified" in item for item in final.reasons)
