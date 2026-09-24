"""Conditioning and hysteresis guards for progressive refinement decisions.

The guard consumes the dimensionless ranking from :mod:`progressive_policy`.
It never certifies target accuracy and never replaces the exact target stage.
Its only job is to avoid chattering around an allowance boundary and to refuse
estimator-driven speculation when the estimator's reported conditioning is not
within an explicit caller-owned limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from .accuracy import ErrorEvidence, EvidenceKind
from .progressive_policy import RefinementCandidate, rank_refinement_candidates

if TYPE_CHECKING:
    from .progressive_controller import StagePlan, TargetProblem


@dataclass(frozen=True, order=True)
class RefinementIdentity:
    """Stable identity of one actionable source/observable refinement."""

    source: str
    observable: str
    norm: str
    unit: str

    @classmethod
    def from_candidate(cls, candidate: RefinementCandidate) -> RefinementIdentity:
        return cls(
            candidate.source,
            candidate.observable,
            candidate.norm,
            candidate.unit,
        )


@dataclass(frozen=True)
class RefinementGuard:
    """Caller-owned thresholds for bounded adaptive refinement.

    ``enter_ratio`` must lie above one and ``release_ratio`` below one.  The
    gap is deliberate hysteresis around the requested observable allowance.
    ``maximum_condition_estimate`` is not inferred from dimensions or method;
    callers must supply a validated limit for the estimator they choose to use.
    """

    enter_ratio: float
    release_ratio: float
    maximum_condition_estimate: float

    def __post_init__(self) -> None:
        for name in (
            "enter_ratio",
            "release_ratio",
            "maximum_condition_estimate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a finite positive number")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, value)
        if self.enter_ratio <= 1.0:
            raise ValueError("enter_ratio must exceed the requested allowance")
        if self.release_ratio >= 1.0:
            raise ValueError("release_ratio must be below the requested allowance")


@dataclass(frozen=True)
class GuardedRefinementDecision:
    """One bounded policy decision; the exact target stage remains mandatory."""

    action: str
    candidate: RefinementCandidate | None
    reason: str

    @property
    def requires_refinement(self) -> bool:
        return self.action in ("start_refinement", "continue_refinement")

    @property
    def requires_target_baseline(self) -> bool:
        return self.action == "target_baseline"


def _ratio_exceeds(candidate: RefinementCandidate, threshold: float) -> bool:
    if candidate.allowance == 0:
        return candidate.error_value > 0
    return Fraction(candidate.error_value) > (
        Fraction(candidate.allowance) * Fraction(threshold)
    )


def _evidence_for(stage: StagePlan, candidate: RefinementCandidate) -> ErrorEvidence:
    matches = tuple(
        evidence
        for evidence in stage.error_evidence
        if (
            evidence.source == candidate.source
            and evidence.observable == candidate.observable
            and evidence.norm == candidate.norm
            and evidence.unit == candidate.unit
        )
    )
    if len(matches) != 1:
        raise ValueError("refinement candidate does not resolve to one evidence record")
    return matches[0]


def _conditioning_is_usable(
    evidence: ErrorEvidence, guard: RefinementGuard
) -> tuple[bool, str]:
    if evidence.kind is EvidenceKind.OBSERVED:
        return True, "observed evidence does not require an estimator conditioning gate"
    if evidence.condition_estimate is None:
        return False, "estimated evidence has no conditioning estimate"
    if evidence.condition_estimate > guard.maximum_condition_estimate:
        return False, "estimated evidence exceeds the allowed conditioning limit"
    return True, "estimated evidence satisfies the explicit conditioning limit"


def decide_guarded_refinement(
    problem: TargetProblem,
    stage: StagePlan,
    guard: RefinementGuard,
    *,
    active: RefinementIdentity | None = None,
) -> GuardedRefinementDecision:
    """Apply hysteresis and estimator conditioning to ranked source evidence.

    With no active refinement, a source starts only above ``enter_ratio``.  An
    active source continues until it reaches ``release_ratio`` or below.  A
    stale active identity is rejected rather than silently mapped by shape or
    source name alone.  Estimated evidence with missing/excessive conditioning
    sends execution to the unchanged exact-target baseline instead of starting
    speculative work.
    """

    if not isinstance(guard, RefinementGuard):
        raise TypeError("guard must use the typed RefinementGuard contract")
    if active is not None and not isinstance(active, RefinementIdentity):
        raise TypeError("active refinement must use RefinementIdentity")

    candidates = rank_refinement_candidates(problem, stage)
    indexed = {RefinementIdentity.from_candidate(item): item for item in candidates}

    selected: RefinementCandidate | None = None
    action = "no_refinement"
    if active is not None:
        selected = indexed.get(active)
        if selected is None:
            raise ValueError(
                "active refinement is stale for the current stage evidence"
            )
        if _ratio_exceeds(selected, guard.release_ratio):
            action = "continue_refinement"
        else:
            selected = None

    if selected is None and candidates:
        dominant = candidates[0]
        if _ratio_exceeds(dominant, guard.enter_ratio):
            selected = dominant
            action = "start_refinement"

    if selected is None:
        return GuardedRefinementDecision(
            "no_refinement",
            None,
            "no source crossed the hysteretic refinement boundary",
        )

    evidence = _evidence_for(stage, selected)
    usable, reason = _conditioning_is_usable(evidence, guard)
    if not usable:
        return GuardedRefinementDecision("target_baseline", selected, reason)
    return GuardedRefinementDecision(action, selected, reason)
