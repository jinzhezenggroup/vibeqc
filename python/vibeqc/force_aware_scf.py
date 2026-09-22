"""Opt-in force-aware SCF effort allocation for geometry-optimization research.

This module owns only the NUM03 error-allocation policy.  It does not provide a
geometry optimizer and it does not replace the progressive orchestration owned
by PROG04.  A consumer supplies the current geometry/force signal and executes
the returned, already-defined SCF convergence level.

The calibration is empirical and limited-domain.  In particular, a density
update RMS is a solver-state diagnostic; it is not silently relabeled as a
physical commutator residual or as a rigorous force-error bound.
"""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

from .accuracy import _identity, _number
from .force_aware_numerics import ObservableDelta, TargetErrorBudget
from .profiles import canonical_hash

_DIAGNOSTIC_KINDS = ("density_rms", "physical_residual_rms")


@dataclass(frozen=True)
class ScfEffortLevel:
    """One SCF convergence setting, ordered from loose to strict.

    Arithmetic precision, grid and screening are deliberately absent.  NUM02
    and the existing numerical policy own those controls independently.
    """

    name: str
    rank: int
    energy_tolerance: float
    density_tolerance: float
    max_iterations: int
    strict: bool = False

    def __post_init__(self) -> None:
        _identity(self.name, "SCF effort level name")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("SCF effort level rank must be a nonnegative integer")
        object.__setattr__(
            self,
            "energy_tolerance",
            _number(self.energy_tolerance, "SCF energy tolerance", positive=True),
        )
        object.__setattr__(
            self,
            "density_tolerance",
            _number(self.density_tolerance, "SCF density tolerance", positive=True),
        )
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("SCF max_iterations must be a positive integer")
        if type(self.strict) is not bool:
            raise TypeError("SCF strict marker must be boolean")

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class ScfForceCalibrationSample:
    """Measured incomplete-SCF error against an independently strict endpoint."""

    family: str
    sample_id: str
    method: str
    basis_id: str
    diagnostic_kind: str
    diagnostic_value: float
    strict_error: ObservableDelta

    def __post_init__(self) -> None:
        for name in ("family", "sample_id", "method", "basis_id"):
            _identity(getattr(self, name), name)
        if self.diagnostic_kind not in _DIAGNOSTIC_KINDS:
            raise ValueError("unsupported SCF force-calibration diagnostic")
        value = _number(self.diagnostic_value, "SCF diagnostic value")
        object.__setattr__(self, "diagnostic_value", value)
        if not isinstance(self.strict_error, ObservableDelta):
            raise TypeError("strict_error must be an ObservableDelta")


@dataclass(frozen=True)
class ScfForceErrorEstimate:
    """Empirical error envelope predicted from one SCF state diagnostic."""

    energy_abs: float
    force_max_abs: float
    force_rms: float
    calibration_id: str
    method: str
    basis_id: str
    diagnostic_kind: str
    diagnostic_value: float
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("energy_abs", "force_max_abs", "force_rms", "diagnostic_value"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("calibration_id", "method", "basis_id"):
            _identity(getattr(self, name), name)
        if self.diagnostic_kind not in _DIAGNOSTIC_KINDS:
            raise ValueError("unsupported SCF error-estimate diagnostic")
        assumptions = tuple(self.assumptions)
        if not assumptions:
            raise ValueError("SCF empirical estimates require explicit assumptions")
        for value in assumptions:
            _identity(value, "SCF estimate assumption")
        object.__setattr__(self, "assumptions", assumptions)

    def ratios(self, budget: TargetErrorBudget) -> tuple[float, ...]:
        values = (
            self.energy_abs / budget.energy_abs,
            self.force_max_abs / budget.force_max_abs,
        )
        if budget.force_rms is not None:
            values += (self.force_rms / budget.force_rms,)
        return values

    def accepted(self, budget: TargetErrorBudget) -> bool:
        return max(self.ratios(budget)) <= 1.0


@dataclass(frozen=True)
class ScfForceErrorEstimator:
    """Limited-domain linear envelope from an SCF diagnostic to observable error.

    The envelope is calibrated as max(error / diagnostic) times an explicit
    safety factor.  That makes it monotone in the chosen diagnostic, but it is
    still empirical: no universal force-error theorem is claimed.
    """

    methods: tuple[str, ...]
    basis_ids: tuple[str, ...]
    diagnostic_kind: str
    energy_scale: float
    force_max_scale: float
    force_rms_scale: float
    training_families: tuple[str, ...]
    training_data_hash: str
    safety_factor: float = 1.25
    diagnostic_floor: float = 1e-14
    schema_version: int = 2
    method_basis_domains: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("unsupported SCF force-estimator schema")
        if self.diagnostic_kind not in _DIAGNOSTIC_KINDS:
            raise ValueError("unsupported SCF force-estimator diagnostic")
        methods = tuple(sorted(set(self.methods)))
        bases = tuple(sorted(set(self.basis_ids)))
        families = tuple(sorted(set(self.training_families)))
        if not methods or not bases:
            raise ValueError("SCF force estimator requires method and basis coverage")
        if len(families) < 2:
            raise ValueError("SCF force calibration requires two molecular families")
        domains = tuple(sorted({tuple(pair) for pair in self.method_basis_domains}))
        if not domains or any(len(pair) != 2 for pair in domains):
            raise ValueError("SCF estimator requires explicit method/basis domains")
        if {pair[0] for pair in domains} != set(methods) or {
            pair[1] for pair in domains
        } != set(bases):
            raise ValueError("SCF joint domains disagree with method/basis inventories")
        object.__setattr__(self, "method_basis_domains", domains)
        for value in (*methods, *bases, *families):
            _identity(value, "SCF calibration identity")
        for name in ("energy_scale", "force_max_scale", "force_rms_scale"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("safety_factor", "diagnostic_floor"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )
        if self.safety_factor < 1:
            raise ValueError("SCF force-estimator safety factor must be at least one")
        _identity(self.training_data_hash, "SCF training data hash")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "basis_ids", bases)
        object.__setattr__(self, "training_families", families)

    @classmethod
    def fit(
        cls,
        samples: typing.Iterable[ScfForceCalibrationSample],
        *,
        safety_factor: typing.Any = 1.25,
        diagnostic_floor: typing.Any = 1e-14,
    ) -> ScfForceErrorEstimator:
        samples = tuple(samples)
        if not samples or any(
            not isinstance(item, ScfForceCalibrationSample) for item in samples
        ):
            raise ValueError("SCF force-estimator fit requires typed samples")
        if len({item.sample_id for item in samples}) != len(samples):
            raise ValueError("duplicate SCF calibration sample identity")
        if len({item.family for item in samples}) < 2:
            raise ValueError("SCF force calibration requires two molecular families")
        diagnostics = {item.diagnostic_kind for item in samples}
        if len(diagnostics) != 1:
            raise ValueError("SCF force calibration cannot mix diagnostic kinds")
        safety_factor = _number(safety_factor, "SCF safety factor", positive=True)
        diagnostic_floor = _number(
            diagnostic_floor, "SCF diagnostic floor", positive=True
        )

        def scale(selector: typing.Callable[[ObservableDelta], float]) -> float:
            return safety_factor * max(
                selector(item.strict_error)
                / max(item.diagnostic_value, diagnostic_floor)
                for item in samples
            )

        return cls(
            tuple(item.method for item in samples),
            tuple(item.basis_id for item in samples),
            next(iter(diagnostics)),
            scale(lambda value: value.energy_abs),
            scale(lambda value: value.force_max_abs),
            scale(lambda value: value.force_rms),
            tuple(item.family for item in samples),
            canonical_hash(
                [
                    asdict(item)
                    for item in sorted(samples, key=lambda item: item.sample_id)
                ]
            ),
            safety_factor=safety_factor,
            diagnostic_floor=diagnostic_floor,
            method_basis_domains=tuple(
                (item.method, item.basis_id) for item in samples
            ),
        )

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))

    def predict(
        self,
        method: str,
        basis_id: str,
        diagnostic_value: typing.Any,
        *,
        allow_unseen_basis: bool = False,
    ) -> ScfForceErrorEstimate:
        if method not in self.methods:
            raise ValueError("method is outside SCF force-estimator calibration domain")
        if type(allow_unseen_basis) is not bool:
            raise TypeError("allow_unseen_basis must be boolean")
        known_domain = (method, basis_id) in self.method_basis_domains
        if not known_domain and not allow_unseen_basis:
            raise ValueError(
                "method/basis pair is outside SCF force-estimator calibration domain"
            )
        value = _number(diagnostic_value, "SCF diagnostic value")
        effective = max(value, self.diagnostic_floor)
        basis_scope = (
            "method/basis pair is in the recorded calibration domain"
            if known_domain
            else "method/basis pair is held out and this prediction is validation-only"
        )
        return ScfForceErrorEstimate(
            self.energy_scale * effective,
            self.force_max_scale * effective,
            self.force_rms_scale * effective,
            self.identity,
            method,
            basis_id,
            self.diagnostic_kind,
            value,
            (
                "SCF diagnostic-to-observable relation is empirical, not a rigorous bound",
                basis_scope,
                f"diagnostic kind is exactly {self.diagnostic_kind}",
            ),
        )

    def evaluate_holdout(
        self,
        samples: typing.Iterable[ScfForceCalibrationSample],
        budget: TargetErrorBudget,
    ) -> dict:
        """Report pass/fail reliability and force-envelope coverage.

        ``force_coverage`` measures whether the empirical max-force envelope
        covers each observed strict error; it does not promote the envelope to
        a certified bound.  The per-row underestimation factor is
        ``actual_force_max / predicted_force_max``.  The aggregate maximum is
        taken over underestimation rows only, or is ``None`` if any such row
        has no finite floating-point ratio.  Zero over zero is recorded as
        1.0 (no miss); a positive observation over a zero prediction remains
        an underestimation with a ``None`` factor.
        """
        rows = []
        for sample in samples:
            if sample.family in self.training_families:
                raise ValueError("molecular-family leakage in SCF holdout evaluation")
            if sample.diagnostic_kind != self.diagnostic_kind:
                raise ValueError(
                    "SCF holdout diagnostic kind does not match calibration"
                )
            estimate = self.predict(
                sample.method,
                sample.basis_id,
                sample.diagnostic_value,
                allow_unseen_basis=True,
            )
            predicted_pass = estimate.accepted(budget)
            actual_pass = budget.accepts(sample.strict_error)
            predicted_force_max = estimate.force_max_abs
            actual_force_max = sample.strict_error.force_max_abs
            force_covered = actual_force_max <= predicted_force_max
            if predicted_force_max > 0.0:
                force_underestimation_factor = actual_force_max / predicted_force_max
                if not math.isfinite(force_underestimation_factor):
                    # Keep strict JSON output valid even when finite errors
                    # have a ratio beyond the floating-point reporting range.
                    force_underestimation_factor = None
            elif actual_force_max == 0.0:
                force_underestimation_factor = 1.0
            else:
                force_underestimation_factor = None
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "family": sample.family,
                    "basis_seen_in_training": sample.basis_id in self.basis_ids,
                    "domain_seen_in_training": (sample.method, sample.basis_id)
                    in self.method_basis_domains,
                    "predicted_pass": predicted_pass,
                    "actual_pass": actual_pass,
                    "false_success": predicted_pass and not actual_pass,
                    "overconservative": actual_pass and not predicted_pass,
                    "force_covered": force_covered,
                    "force_underestimated": not force_covered,
                    "predicted_energy": estimate.energy_abs,
                    "actual_energy": sample.strict_error.energy_abs,
                    "predicted_force_max": predicted_force_max,
                    "actual_force_max": actual_force_max,
                    "force_underestimation_factor": force_underestimation_factor,
                    "predicted_force_rms": estimate.force_rms,
                    "actual_force_rms": sample.strict_error.force_rms,
                }
            )
        false_successes = sum(row["false_success"] for row in rows)
        force_coverage = sum(row["force_covered"] for row in rows)
        force_underestimation_rows = sum(row["force_underestimated"] for row in rows)
        finite_underestimation_factors = [
            row["force_underestimation_factor"]
            for row in rows
            if row["force_underestimated"]
            and row["force_underestimation_factor"] is not None
        ]
        force_underestimation_nonfinite_rows = sum(
            row["force_underestimation_factor"] is None for row in rows
        )
        return {
            "rows": rows,
            "samples": len(rows),
            "false_successes": false_successes,
            "false_success_rate": false_successes / len(rows) if rows else 0.0,
            "overconservative": sum(row["overconservative"] for row in rows),
            "force_coverage": force_coverage,
            "force_coverage_rate": force_coverage / len(rows) if rows else 0.0,
            "force_underestimation_rows": force_underestimation_rows,
            "max_force_underestimation_factor": (
                None
                if force_underestimation_nonfinite_rows
                else max(finite_underestimation_factors, default=None)
            ),
            "force_underestimation_nonfinite_rows": force_underestimation_nonfinite_rows,
            "certified": False,
        }


@dataclass(frozen=True)
class ScfEffortTransition:
    from_level: str
    to_level: str
    step_index: int
    geometry_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("from_level", "to_level", "geometry_id", "reason"):
            _identity(getattr(self, name), name)
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("geometry step index must be a nonnegative integer")


@dataclass(frozen=True)
class ScfEffortState:
    level_index: int
    safe_streak: int = 0
    transitions: tuple[ScfEffortTransition, ...] = ()


@dataclass(frozen=True)
class ScfEffortDecision:
    action: str
    level: ScfEffortLevel
    state: ScfEffortState
    reason: str
    allowed_energy_error: float
    allowed_force_error: float
    estimate: ScfForceErrorEstimate | None


@dataclass(frozen=True)
class OptimizationFinalVerification:
    status: str
    reasons: tuple[str, ...]
    target_model_id: str
    observed_model_id: str
    force_max_abs: float
    optimizer_force_tolerance: float
    energy_change: float
    density_rms: float


@dataclass(frozen=True)
class ForceAwareScfPolicy:
    """Opt-in next-step SCF effort proposal with strict final fallback.

    The policy starts strict because there is no trustworthy previous-force
    signal yet.  Loosening is hysteretic.  Tightening and near-stationary strict
    cleanup are immediate.  Only SCF convergence controls are selected here.
    """

    levels: tuple[ScfEffortLevel, ...]
    estimator: ScfForceErrorEstimator
    optimizer_force_tolerance: float
    intermediate_energy_error: float
    intermediate_force_fraction: float
    max_intermediate_force_error: float
    near_stationary_factor: float = 5.0
    relax_after: int = 2
    strict_reproducible: bool = False

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels or any(not isinstance(item, ScfEffortLevel) for item in levels):
            raise ValueError("force-aware SCF policy requires typed effort levels")
        if [item.rank for item in levels] != sorted(item.rank for item in levels):
            raise ValueError("SCF effort levels must be ordered loose to strict")
        if len({item.rank for item in levels}) != len(levels) or len(
            {item.name for item in levels}
        ) != len(levels):
            raise ValueError("SCF effort level names/ranks must be unique")
        if any(
            later.energy_tolerance > earlier.energy_tolerance
            or later.density_tolerance > earlier.density_tolerance
            for earlier, later in pairwise(levels)
        ):
            raise ValueError("SCF effort tolerances must tighten with level rank")
        if not levels[-1].strict or any(item.strict for item in levels[:-1]):
            raise ValueError("exactly the final SCF effort level must be strict")
        if not isinstance(self.estimator, ScfForceErrorEstimator):
            raise TypeError("force-aware SCF policy requires an SCF estimator")
        if self.estimator.diagnostic_kind != "density_rms":
            raise ValueError(
                "automatic SCF effort proposal currently requires density_rms calibration"
            )
        for name in (
            "optimizer_force_tolerance",
            "intermediate_energy_error",
            "intermediate_force_fraction",
            "max_intermediate_force_error",
            "near_stationary_factor",
        ):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )
        if self.intermediate_force_fraction >= 1:
            raise ValueError("intermediate force fraction must be below one")
        if self.near_stationary_factor < 1:
            raise ValueError("near-stationary factor must be at least one")
        if type(self.relax_after) is not int or self.relax_after < 1:
            raise ValueError("SCF effort relax_after must be a positive integer")
        if type(self.strict_reproducible) is not bool:
            raise TypeError("strict_reproducible must be boolean")
        object.__setattr__(self, "levels", levels)

    @property
    def strict_index(self) -> int:
        return len(self.levels) - 1

    @property
    def strict_level(self) -> ScfEffortLevel:
        return self.levels[-1]

    def initial_state(self) -> ScfEffortState:
        return ScfEffortState(self.strict_index)

    def _transition(
        self,
        state: ScfEffortState,
        target: int,
        *,
        step_index: int,
        geometry_id: str,
        reason: str,
        safe_streak: int = 0,
    ) -> ScfEffortState:
        if target == state.level_index:
            return replace(state, safe_streak=safe_streak)
        transition = ScfEffortTransition(
            self.levels[state.level_index].name,
            self.levels[target].name,
            step_index,
            geometry_id,
            reason,
        )
        return ScfEffortState(target, safe_streak, state.transitions + (transition,))

    def propose_next(
        self,
        state: ScfEffortState,
        *,
        method: str,
        basis_id: str,
        geometry_id: str,
        step_index: int,
        current_force_max: typing.Any | None,
    ) -> ScfEffortDecision:
        _identity(method, "method")
        _identity(basis_id, "basis identity")
        _identity(geometry_id, "geometry identity")
        if type(step_index) is not int or step_index < 0:
            raise ValueError("geometry step index must be a nonnegative integer")
        if not 0 <= state.level_index < len(self.levels):
            raise ValueError("SCF effort state belongs to another policy")
        allowed_energy = self.intermediate_energy_error
        if self.strict_reproducible or current_force_max is None:
            reason = (
                "strict reproducible mode"
                if self.strict_reproducible
                else "no trustworthy previous-force signal"
            )
            moved = self._transition(
                state,
                self.strict_index,
                step_index=step_index,
                geometry_id=geometry_id,
                reason=reason,
            )
            return ScfEffortDecision(
                "strict", self.strict_level, moved, reason, allowed_energy, 0.0, None
            )
        force = _number(current_force_max, "current force maximum")
        allowed_force = min(
            self.max_intermediate_force_error,
            self.intermediate_force_fraction
            * max(force, self.optimizer_force_tolerance),
        )
        if force <= self.near_stationary_factor * self.optimizer_force_tolerance:
            moved = self._transition(
                state,
                self.strict_index,
                step_index=step_index,
                geometry_id=geometry_id,
                reason="near-stationary strict cleanup",
            )
            return ScfEffortDecision(
                "strict",
                self.strict_level,
                moved,
                "near-stationary strict cleanup",
                allowed_energy,
                allowed_force,
                None,
            )

        candidates: list[tuple[int, ScfForceErrorEstimate]] = []
        try:
            for index, level in enumerate(self.levels):
                estimate = self.estimator.predict(
                    method, basis_id, level.density_tolerance
                )
                if (
                    estimate.energy_abs <= allowed_energy
                    and estimate.force_max_abs <= allowed_force
                ):
                    candidates.append((index, estimate))
        except ValueError as error:
            moved = self._transition(
                state,
                self.strict_index,
                step_index=step_index,
                geometry_id=geometry_id,
                reason="calibration-domain fallback",
            )
            return ScfEffortDecision(
                "strict_fallback",
                self.strict_level,
                moved,
                str(error),
                allowed_energy,
                allowed_force,
                None,
            )
        if not candidates:
            moved = self._transition(
                state,
                self.strict_index,
                step_index=step_index,
                geometry_id=geometry_id,
                reason="no calibrated level meets observable allowance",
            )
            return ScfEffortDecision(
                "strict",
                self.strict_level,
                moved,
                "no calibrated level meets observable allowance",
                allowed_energy,
                allowed_force,
                None,
            )
        target, estimate = candidates[0]
        if target < state.level_index:
            streak = state.safe_streak + 1
            if streak < self.relax_after:
                current = self.levels[state.level_index]
                current_estimate = self.estimator.predict(
                    method, basis_id, current.density_tolerance
                )
                return ScfEffortDecision(
                    "hold",
                    current,
                    replace(state, safe_streak=streak),
                    "waiting for SCF-effort relaxation hysteresis",
                    allowed_energy,
                    allowed_force,
                    current_estimate,
                )
            moved = self._transition(
                state,
                target,
                step_index=step_index,
                geometry_id=geometry_id,
                reason="calibrated force margin permits looser next step",
            )
            return ScfEffortDecision(
                "relax",
                self.levels[target],
                moved,
                "sustained calibrated force margin",
                allowed_energy,
                allowed_force,
                estimate,
            )
        if target > state.level_index:
            moved = self._transition(
                state,
                target,
                step_index=step_index,
                geometry_id=geometry_id,
                reason="observable allowance requires tighter SCF",
            )
            return ScfEffortDecision(
                "tighten",
                self.levels[target],
                moved,
                "observable allowance requires tighter SCF",
                allowed_energy,
                allowed_force,
                estimate,
            )
        return ScfEffortDecision(
            "hold",
            self.levels[target],
            replace(state, safe_streak=0),
            "current SCF effort matches calibrated allowance",
            allowed_energy,
            allowed_force,
            estimate,
        )

    def verify_final(
        self,
        *,
        target_model_id: str,
        observed_model_id: str,
        level: ScfEffortLevel,
        converged: bool,
        energy: typing.Any,
        energy_change: typing.Any,
        density_rms: typing.Any,
        force_max_abs: typing.Any,
    ) -> OptimizationFinalVerification:
        if type(converged) is not bool:
            raise TypeError("final converged must be boolean")
        for value, name in (
            (target_model_id, "target model identity"),
            (observed_model_id, "observed model identity"),
        ):
            _identity(value, name)
        force = _number(force_max_abs, "final force maximum")
        change = _number(energy_change, "final energy change")
        density = _number(density_rms, "final density RMS")
        _finite = float(energy)
        if not math.isfinite(_finite):
            raise ValueError("final energy must be finite")
        reasons = []
        if observed_model_id != target_model_id:
            reasons.append("final model identity does not match target")
        if level != self.strict_level:
            reasons.append("final evaluation did not use strict SCF effort")
        if not converged:
            reasons.append("final SCF did not converge")
        if change > self.strict_level.energy_tolerance:
            reasons.append("final energy-change gate is unmet")
        if density > self.strict_level.density_tolerance:
            reasons.append("final density gate is unmet")
        if force > self.optimizer_force_tolerance:
            reasons.append("optimizer force termination gate is unmet")
        status = (
            "unverified"
            if any("identity" in reason for reason in reasons)
            else "observed_unmet"
            if reasons
            else "observed_met"
        )
        return OptimizationFinalVerification(
            status,
            tuple(reasons),
            target_model_id,
            observed_model_id,
            force,
            self.optimizer_force_tolerance,
            change,
            density,
        )
