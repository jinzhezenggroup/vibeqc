"""Fail-closed runtime accounting for progressive solver work.

The planner owns prospective bounds. This module independently audits the work
that actually ran so failed or speculative stages cannot disappear from the
runtime budget ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .progressive_controller import (
    ProgressiveBudget,
    StageExecution,
    StagePlan,
    StageRole,
    TargetProblem,
)


@dataclass(frozen=True)
class RuntimeLedgerAudit:
    """Accounting and provenance disposition for one progressive run."""

    status: str
    stage_count: int
    total_iterations: int
    total_fock_builds: int | None
    total_seconds: float
    reasons: tuple[str, ...]

    @property
    def within_budget(self) -> bool:
        return self.status == "within_budget"


def audit_runtime_ledger(
    problem: TargetProblem,
    target_stage: StagePlan,
    budget: ProgressiveBudget,
    records: tuple[StageExecution, ...],
) -> RuntimeLedgerAudit:
    """Recompute actual work and target provenance from immutable stage records."""

    if not isinstance(problem, TargetProblem):
        raise TypeError("problem must use the TargetProblem contract")
    if not isinstance(target_stage, StagePlan) or target_stage.role is not StageRole.TARGET:
        raise TypeError("target_stage must be the typed target StagePlan")
    if not isinstance(budget, ProgressiveBudget):
        raise TypeError("budget must use the ProgressiveBudget contract")

    history = tuple(records)
    if any(not isinstance(item, StageExecution) for item in history):
        raise TypeError("runtime ledger must contain StageExecution records")

    budget_reasons: list[str] = []
    integrity_reasons: list[str] = []
    if len(history) > budget.maximum_stages:
        budget_reasons.append("actual stage count exceeded the progressive budget")

    total_iterations = 0
    total_seconds = 0.0
    for item in history:
        if type(item.iterations) is not int or item.iterations < 0:
            raise ValueError("stage iterations must be a nonnegative integer")
        if item.fock_builds is not None and (
            type(item.fock_builds) is not int or item.fock_builds < 0
        ):
            raise ValueError("stage Fock builds must be a nonnegative integer")
        if isinstance(item.seconds, bool):
            raise TypeError("stage seconds must be finite and nonnegative")
        seconds = float(item.seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("stage seconds must be finite and nonnegative")
        total_iterations += item.iterations
        total_seconds += seconds

    if total_iterations > budget.maximum_total_iterations:
        budget_reasons.append("actual iterations exceeded the progressive budget")

    fock_values = tuple(item.fock_builds for item in history)
    total_fock_builds = (
        sum(value for value in fock_values if value is not None)
        if all(value is not None for value in fock_values)
        else None
    )
    if budget.maximum_total_fock_builds is not None:
        if total_fock_builds is None:
            budget_reasons.append("Fock-build budget could not be verified")
        elif total_fock_builds > budget.maximum_total_fock_builds:
            budget_reasons.append("actual Fock builds exceeded the progressive budget")

    targets = tuple(
        (index, item)
        for index, item in enumerate(history)
        if item.role is StageRole.TARGET
    )
    if len(targets) != 1:
        integrity_reasons.append("runtime ledger must contain exactly one target stage")
    else:
        index, target = targets[0]
        if index != len(history) - 1:
            integrity_reasons.append("target stage must be final in the runtime ledger")
        if target.stage_id != target_stage.stage_id:
            integrity_reasons.append("target stage id differs from the StagePlan")
        if target.model_identity != problem.model.identity:
            integrity_reasons.append("target model differs from the TargetProblem")
        if target.provider_identity != problem.provider_identity:
            integrity_reasons.append("target provider differs from the TargetProblem")
        if target.status != "succeeded":
            integrity_reasons.append("target stage did not record successful completion")

    if integrity_reasons:
        status = "invalid"
    elif budget_reasons:
        status = "budget_exhausted"
    else:
        status = "within_budget"
    return RuntimeLedgerAudit(
        status=status,
        stage_count=len(history),
        total_iterations=total_iterations,
        total_fock_builds=total_fock_builds,
        total_seconds=total_seconds,
        reasons=tuple(integrity_reasons + budget_reasons),
    )
