"""Conservative evidence ranking for progressive-model refinement decisions.

This module never establishes target accuracy and never authorizes skipping the
exact target stage.  It only compares compatible, per-source error evidence in
the dimensionless units of the requested observable allowances so a future
adapter controller can refine the largest known contributor without comparing
raw energies, forces, thresholds, or residuals as if they shared units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .accuracy import ErrorEvidence, EvidenceKind
from .progressive_controller import StagePlan, StageRole, TargetProblem


@dataclass(frozen=True)
class RefinementCandidate:
    """One actionable error component normalized by its requested allowance."""

    source: str
    observable: str
    norm: str
    unit: str
    error_value: float
    allowance: float
    # Display ratio: zero allowance with positive error is infinity. Sorting
    # uses the exact finite input ratio, not this potentially saturated float.
    normalized_error: float
    evidence_kind: EvidenceKind
    calibration_id: str | None


@dataclass(frozen=True)
class RefinementDecision:
    """Disposition for optional adapter refinement before the exact target solve."""

    action: str
    candidate: RefinementCandidate | None
    candidate_count: int
    reason: str

    @property
    def requires_refinement(self) -> bool:
        return self.action == "refine_known_source"


def _validate_stage_identity(problem: TargetProblem, stage: StagePlan) -> None:
    if not isinstance(problem, TargetProblem) or not isinstance(stage, StagePlan):
        raise TypeError("refinement policy requires typed target and stage contracts")
    if stage.role is not StageRole.INITIALIZATION:
        raise ValueError("refinement policy only accepts initialization stages")
    for name in (
        "method",
        "geometry_hash",
        "electron_count",
        "multiplicity",
        "charge",
        "hamiltonian",
    ):
        if getattr(stage.model, name) != getattr(problem.model, name):
            raise ValueError(f"initialization stage changed target-compatible {name}")


def rank_refinement_candidates(
    problem: TargetProblem, stage: StagePlan
) -> tuple[RefinementCandidate, ...]:
    """Rank compatible per-source evidence by requested-observable error ratio.

    Evidence named ``total_numerical`` is intentionally excluded: it can assess
    overall error but does not identify an adapter or error source to refine.
    Evidence for another scope or an unrequested observable is retained by the
    owning stage but is not actionable for this target request.
    """

    _validate_stage_identity(problem, stage)
    targets = {
        (target.observable, target.norm, target.unit): target
        for target in problem.accuracy.observables
    }
    candidates: list[RefinementCandidate] = []
    identities: set[tuple[str, str, str, str, str]] = set()
    for evidence in stage.error_evidence:
        if not isinstance(evidence, ErrorEvidence):
            raise TypeError("stage error evidence must use the typed accuracy contract")
        if evidence.model_id != problem.model.identity:
            raise ValueError("error evidence changed the requested target")
        if evidence.evaluated_model_id != stage.model.identity:
            raise ValueError("error evidence names a different evaluated stage")
        if evidence.reference_model_id != problem.model.identity:
            raise ValueError("error evidence names a different reference model")
        if (
            evidence.scope != problem.accuracy.scope
            or evidence.source == "total_numerical"
        ):
            continue
        target = targets.get((evidence.observable, evidence.norm, evidence.unit))
        if target is None:
            continue
        identity = (
            evidence.source,
            evidence.observable,
            evidence.norm,
            evidence.unit,
            evidence.scope,
        )
        if identity in identities:
            raise ValueError("duplicate actionable refinement evidence")
        identities.add(identity)
        allowance = target.allowance(evidence.reference_norm)
        candidates.append(
            RefinementCandidate(
                source=evidence.source,
                observable=evidence.observable,
                norm=evidence.norm,
                unit=evidence.unit,
                error_value=evidence.value,
                allowance=allowance,
                normalized_error=(
                    evidence.value / allowance
                    if allowance
                    else (math.inf if evidence.value else 0.0)
                ),
                evidence_kind=evidence.kind,
                calibration_id=evidence.calibration_id,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                # A relative-only target may resolve to zero. Positive error
                # then outranks every finite ratio; zero error stays in budget.
                -int(item.allowance == 0 and item.error_value > 0),
                # Finite ratios can overflow/underflow binary64 and otherwise
                # become false ties. Preserve their order without mixing units.
                -(Fraction(item.error_value) / Fraction(item.allowance))
                if item.allowance
                else Fraction(0),
                item.source,
                item.observable,
                item.norm,
                item.unit,
                item.evidence_kind.value,
                item.calibration_id or "",
            ),
        )
    )


def decide_refinement(problem: TargetProblem, stage: StagePlan) -> RefinementDecision:
    """Select the largest known over-budget source without making accuracy claims."""

    candidates = rank_refinement_candidates(problem, stage)
    if not candidates:
        return RefinementDecision(
            "no_refinement",
            None,
            0,
            "no compatible per-source evidence identifies an adapter to refine",
        )
    dominant = candidates[0]
    if dominant.normalized_error <= 1.0:
        return RefinementDecision(
            "no_refinement",
            None,
            len(candidates),
            (
                "all known per-source evidence is within its requested "
                "observable allowance"
            ),
        )
    return RefinementDecision(
        "refine_known_source",
        dominant,
        len(candidates),
        "largest known normalized error exceeds its requested observable allowance",
    )
