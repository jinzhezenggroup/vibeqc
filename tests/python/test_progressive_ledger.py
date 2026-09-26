"""Runtime stage-count regression for #192."""

from vibeqc.accuracy import ObservableTarget, ResolvedModel, TargetAccuracy
from vibeqc.progressive_controller import (
    ArithmeticPolicy,
    HFConvergence,
    ProgressiveBudget,
    StageExecution,
    StagePlan,
    StageRole,
    TargetProblem,
    TransferOperation,
)
from vibeqc.progressive_ledger import audit_runtime_ledger


def test_runtime_stage_count_is_charged() -> None:
    target_model = ResolvedModel("rhf", "g" * 64, "b" * 64, 2)
    source_model = ResolvedModel("rhf", "g" * 64, "s" * 64, 2)
    problem = TargetProblem(
        target_model,
        "a" * 64,
        (("orbital.mathematical_identity", "c" * 64),),
        TargetAccuracy((ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),)),
        HFConvergence(1e-10, 1e-8, 100, 1e-8),
        2,
    )
    target_stage = StagePlan(
        "target-hf",
        StageRole.TARGET,
        target_model,
        problem.provider_identity,
        problem.provider_hashes,
        ArithmeticPolicy("fp64"),
        problem.convergence,
        TransferOperation.NONE,
    )
    source = StageExecution(
        "initial-hf",
        StageRole.INITIALIZATION,
        "succeeded",
        source_model.identity,
        "d" * 64,
        3,
        2,
        0.1,
        None,
        "accepted",
        "cold",
        "success",
    )
    target = StageExecution(
        "target-hf",
        StageRole.TARGET,
        "succeeded",
        target_model.identity,
        problem.provider_identity,
        4,
        3,
        0.2,
        1e-12,
        "none",
        "basis_projection",
        "success",
    )
    budget = ProgressiveBudget(
        maximum_stages=1,
        maximum_source_iterations=20,
        maximum_total_iterations=20,
    )
    audit = audit_runtime_ledger(problem, target_stage, budget, (source, target))
    assert audit.status == "budget_exhausted"
    assert audit.stage_count == 2
    assert audit.total_iterations == 7
