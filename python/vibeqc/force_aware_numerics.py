"""Force-aware numerical error accounting and adaptive effort policy.

This module deliberately owns policy and evidence, not a DFT execution
schedule. A caller supplies results from already-defined numerical levels
(screening thresholds, grids, fitting cutoffs, and so on), and the policy
decides whether observable evidence justifies holding, tightening or relaxing.

Paired differences are empirical estimators. They are never promoted to
rigorous bounds merely because a calibration envelope covered training data.
"""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, replace
from numbers import Real

import numpy as np

from .accuracy import ErrorEvidence, EvidenceKind, ResolvedModel, _identity, _number
from .profiles import canonical_hash


def _finite_real(value: typing.Any, name: str) -> float:
    """Accept any finite real scalar, including negative physical energies."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _forces(value: typing.Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not array.shape[0]:
        raise ValueError(f"{name} must have shape (natom, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ObservableDelta:
    """Absolute energy and Cartesian-force differences for one comparison.

    force_components retains per-atom/per-component absolute values.
    force_max_abs is the maximum over all 3N components, force_rms is the RMS
    over all 3N components, and per-atom helpers aggregate only within an atom.
    No cancellation across components is used.
    """

    energy_abs: float
    force_components: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_abs", _number(self.energy_abs, "energy error"))
        values = _forces(self.force_components, "force-component errors")
        if np.any(values < 0):
            raise ValueError("force-component errors must be absolute/nonnegative")
        object.__setattr__(
            self,
            "force_components",
            tuple(tuple(float(x) for x in row) for row in values),
        )

    @classmethod
    def between(
        cls,
        candidate_energy: typing.Any,
        candidate_forces: typing.Any,
        reference_energy: typing.Any,
        reference_forces: typing.Any,
    ) -> ObservableDelta:
        """Measure candidate-reference differences without broadcasting atoms."""
        candidate = _forces(candidate_forces, "candidate forces")
        reference = _forces(reference_forces, "reference forces")
        if candidate.shape != reference.shape:
            raise ValueError("candidate/reference forces must have identical shape")
        energy = abs(
            _finite_real(candidate_energy, "candidate energy")
            - _finite_real(reference_energy, "reference energy")
        )
        return cls(energy, tuple(map(tuple, np.abs(candidate - reference))))

    @property
    def force_max_abs(self) -> float:
        return float(np.max(self.force_array))

    @property
    def force_rms(self) -> float:
        values = self.force_array
        scale = float(np.max(values))
        if scale == 0.0:
            return 0.0
        # Scale before squaring so finite subnormal/very-large force errors do
        # not underflow to a false zero or overflow during RMS aggregation.
        return float(scale * np.sqrt(np.mean((values / scale) ** 2)))

    @property
    def per_atom_max_abs(self) -> tuple[float, ...]:
        return tuple(float(v) for v in np.max(self.force_array, axis=1))

    @property
    def per_atom_l2(self) -> tuple[float, ...]:
        # hypot.reduce scales internally, avoiding the square/accumulate
        # underflow/overflow of an ordinary Euclidean norm for finite extremes.
        return tuple(float(v) for v in np.hypot.reduce(self.force_array, axis=1))

    @property
    def force_array(self) -> np.ndarray:
        return np.asarray(self.force_components, dtype=float)

    def scaled(
        self, energy_factor: typing.Any, force_factor: typing.Any
    ) -> ObservableDelta:
        energy_factor = _number(energy_factor, "energy scale")
        force_factor = _number(force_factor, "force scale")
        return ObservableDelta(
            self.energy_abs * energy_factor,
            tuple(map(tuple, self.force_array * force_factor)),
        )

    def absolute_sum(self, other: ObservableDelta) -> ObservableDelta:
        """Compose estimates by componentwise absolute sum."""
        if self.force_array.shape != other.force_array.shape:
            raise ValueError(
                "cannot compose force estimates with different atom counts"
            )
        return ObservableDelta(
            self.energy_abs + other.energy_abs,
            tuple(map(tuple, self.force_array + other.force_array)),
        )


@dataclass(frozen=True)
class TargetErrorBudget:
    """Independent energy/force requirements used by an adaptive policy."""

    energy_abs: float
    force_max_abs: float
    force_rms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "energy_abs", _number(self.energy_abs, "energy target", positive=True)
        )
        object.__setattr__(
            self,
            "force_max_abs",
            _number(self.force_max_abs, "force max target", positive=True),
        )
        if self.force_rms is not None:
            object.__setattr__(
                self,
                "force_rms",
                _number(self.force_rms, "force RMS target", positive=True),
            )

    def ratios(self, delta: ObservableDelta) -> tuple[float, ...]:
        values = (
            delta.energy_abs / self.energy_abs,
            delta.force_max_abs / self.force_max_abs,
        )
        if self.force_rms is not None:
            values += (delta.force_rms / self.force_rms,)
        return values

    def accepts(self, delta: ObservableDelta) -> bool:
        return max(self.ratios(delta)) <= 1.0


@dataclass(frozen=True)
class NumericalTargetModel:
    """Resolved finite-basis/grid target for observable-error comparisons."""

    method: str
    geometry_id: str
    basis_id: str
    functional_id: str
    grid_id: str
    derivative_semantics: str = "analytic-moving-grid-with-partition-response"
    approximation: str = "conventional"
    auxiliary_basis_id: str | None = None
    metric_relative_threshold: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported numerical-target-model schema")
        for name in (
            "method",
            "geometry_id",
            "basis_id",
            "functional_id",
            "grid_id",
            "derivative_semantics",
            "approximation",
        ):
            _identity(getattr(self, name), name)
        if self.derivative_semantics not in (
            "analytic-moving-grid-with-partition-response",
            "fixed-density-moving-grid-with-partition-response",
        ):
            raise ValueError("unsupported derivative semantics")
        if self.approximation == "conventional":
            if (
                self.auxiliary_basis_id is not None
                or self.metric_relative_threshold is not None
            ):
                raise ValueError("conventional target cannot carry fitting metadata")
        elif self.approximation == "density_fitting":
            _identity(self.auxiliary_basis_id, "auxiliary basis identity")
            threshold = _number(
                self.metric_relative_threshold,
                "metric relative threshold",
                positive=True,
            )
            if threshold >= 1:
                raise ValueError("metric relative threshold must be below one")
            object.__setattr__(self, "metric_relative_threshold", threshold)
        else:
            raise ValueError("unsupported target approximation")

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class NumericalContribution:
    """One shell/grid/auxiliary contribution-ledger entry.

    estimate and actual are intentionally separate. actual must be produced by
    an audit comparison; it is never inferred from the estimator.
    """

    source: str
    block_id: str
    scope: str
    estimator_kind: str
    estimate: ObservableDelta | None
    actual: ObservableDelta | None = None
    grid_identity: str | None = None
    mask_identity: str | None = None
    motion_response_included: bool = False
    estimator_seconds: float = 0.0
    metadata: tuple[tuple[str, str], ...] = ()
    block_kind: str = "aggregate"

    def __post_init__(self) -> None:
        for name in ("source", "block_id", "estimator_kind"):
            _identity(getattr(self, name), name)
        if self.scope not in ("fixed_density", "relaxed_target"):
            raise ValueError(
                "contribution scope must be fixed_density or relaxed_target"
            )
        if self.estimator_kind not in ("observed", "empirical", "asymptotic"):
            raise ValueError("unsupported contribution estimator kind")
        if self.block_kind not in (
            "shell_block",
            "grid_region",
            "auxiliary_rank",
            "aggregate",
        ):
            raise ValueError("unsupported contribution block kind")
        if self.estimate is None and self.actual is None:
            raise ValueError("ledger entry requires an estimate or an actual audit")
        if self.estimate is not None and not isinstance(self.estimate, ObservableDelta):
            raise TypeError("estimate must be an ObservableDelta")
        if self.actual is not None and not isinstance(self.actual, ObservableDelta):
            raise TypeError("actual must be an ObservableDelta")
        object.__setattr__(
            self,
            "estimator_seconds",
            _number(self.estimator_seconds, "estimator seconds"),
        )
        for name in ("grid_identity", "mask_identity"):
            value = getattr(self, name)
            if value is not None:
                _identity(value, name)
        metadata = tuple(tuple(pair) for pair in self.metadata)
        if any(len(pair) != 2 for pair in metadata):
            raise ValueError("metadata entries must be key/value pairs")
        if len({key for key, _ in metadata}) != len(metadata):
            raise ValueError("duplicate contribution metadata keys")
        for key, value in metadata:
            _identity(key, "metadata key")
            _identity(value, "metadata value")
        object.__setattr__(self, "metadata", tuple(sorted(metadata)))


@dataclass(frozen=True)
class ContributionLedger:
    """Auditable collection of omitted-work/error contributions.

    estimated_absolute_envelope uses an explicit conservative composition rule:
    componentwise absolute sum. This does not make empirical inputs rigorous
    bounds. Actual errors remain individual audit observations.
    """

    target_id: str
    geometry_id: str
    entries: tuple[NumericalContribution, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.target_id, "target identity")
        _identity(self.geometry_id, "geometry identity")
        entries = tuple(self.entries)
        if any(not isinstance(item, NumericalContribution) for item in entries):
            raise TypeError("ledger entries must be NumericalContribution values")
        keys = [(x.source, x.block_id, x.scope) for x in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate source/block/scope contribution")
        object.__setattr__(self, "entries", entries)

    def estimated_absolute_envelope(
        self, *, scope: str, sources: typing.Iterable[str] | None = None
    ) -> ObservableDelta:
        selected_sources = None if sources is None else set(sources)
        selected = [
            item.estimate
            for item in self.entries
            if item.scope == scope
            and item.estimate is not None
            and (selected_sources is None or item.source in selected_sources)
        ]
        if not selected:
            raise ValueError("no estimates match the requested ledger selection")
        total = selected[0]
        for item in selected[1:]:
            total = total.absolute_sum(item)
        return total

    def block_coverage(self) -> dict[str, int]:
        """Return auditable counts for shell/grid/auxiliary contribution classes."""
        kinds = ("shell_block", "grid_region", "auxiliary_rank", "aggregate")
        return {
            kind: sum(item.block_kind == kind for item in self.entries)
            for kind in kinds
        }

    @property
    def estimator_seconds(self) -> float:
        return sum(item.estimator_seconds for item in self.entries)


@dataclass(frozen=True)
class PairedCalibrationSample:
    """Paired-level estimator input with an independent strict-reference error."""

    family: str
    sample_id: str
    method: str
    numerical_family_id: str
    paired_delta: ObservableDelta
    strict_error: ObservableDelta

    def __post_init__(self) -> None:
        for name in ("family", "sample_id", "method", "numerical_family_id"):
            _identity(getattr(self, name), name)
        if self.paired_delta.force_array.shape != self.strict_error.force_array.shape:
            raise ValueError("paired and strict errors require the same force shape")


@dataclass(frozen=True)
class NumericalEstimate:
    """One empirical observable estimate plus validity/provenance status."""

    delta: ObservableDelta
    calibration_id: str
    assumptions: tuple[str, ...]
    estimator_seconds: float = 0.0
    method: str | None = None

    def __post_init__(self) -> None:
        if self.method is not None:
            _identity(self.method, "estimate method")
        if not isinstance(self.delta, ObservableDelta):
            raise TypeError("numerical estimate requires an ObservableDelta")
        _identity(self.calibration_id, "calibration identity")
        assumptions = tuple(self.assumptions)
        if not assumptions:
            raise ValueError("empirical estimates require explicit assumptions")
        for assumption in assumptions:
            _identity(assumption, "assumption")
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(
            self,
            "estimator_seconds",
            _number(self.estimator_seconds, "estimator seconds"),
        )


@dataclass(frozen=True)
class PairedDifferenceEstimator:
    """Limited-domain empirical envelope for paired grid/screening differences."""

    methods: tuple[str, ...]
    numerical_family_id: str
    energy_scale: float
    force_scale: float
    training_families: tuple[str, ...]
    training_data_hash: str
    safety_factor: float = 1.25
    energy_floor: float = 1e-12
    force_floor: float = 1e-10
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("unsupported paired-estimator schema")
        methods = tuple(sorted(set(self.methods)))
        families = tuple(sorted(set(self.training_families)))
        _identity(self.numerical_family_id, "numerical family identity")
        if not methods:
            raise ValueError("paired estimator requires at least one method")
        if len(families) < 2:
            raise ValueError(
                "paired estimator calibration requires two molecular families"
            )
        for value in (*methods, *families):
            _identity(value, "paired-estimator identity")
        for name in (
            "energy_scale",
            "force_scale",
            "safety_factor",
            "energy_floor",
            "force_floor",
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name, positive=True),
            )
        if self.safety_factor < 1:
            raise ValueError("paired-estimator safety factor must be at least one")
        _identity(self.training_data_hash, "training data hash")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "training_families", families)

    @classmethod
    def fit(
        cls,
        samples: typing.Iterable[PairedCalibrationSample],
        *,
        safety_factor: typing.Any = 1.25,
        energy_floor: typing.Any = 1e-12,
        force_floor: typing.Any = 1e-10,
    ) -> PairedDifferenceEstimator:
        samples = tuple(samples)
        if not samples or any(
            not isinstance(x, PairedCalibrationSample) for x in samples
        ):
            raise ValueError("fit requires paired calibration samples")
        if len({x.sample_id for x in samples}) != len(samples):
            raise ValueError("duplicate paired calibration sample identity")
        if len({x.family for x in samples}) < 2:
            raise ValueError("paired calibration requires two molecular families")
        numerical_families = {x.numerical_family_id for x in samples}
        if len(numerical_families) != 1:
            raise ValueError("paired calibration cannot mix numerical-level families")
        safety_factor = _number(safety_factor, "safety factor", positive=True)
        energy_floor = _number(energy_floor, "energy floor", positive=True)
        force_floor = _number(force_floor, "force floor", positive=True)
        energy_scale = safety_factor * max(
            x.strict_error.energy_abs / max(x.paired_delta.energy_abs, energy_floor)
            for x in samples
        )
        force_scale = safety_factor * max(
            x.strict_error.force_max_abs
            / max(x.paired_delta.force_max_abs, force_floor)
            for x in samples
        )
        return cls(
            tuple(x.method for x in samples),
            next(iter(numerical_families)),
            max(energy_scale, 1.0),
            max(force_scale, 1.0),
            tuple(x.family for x in samples),
            canonical_hash(
                [asdict(x) for x in sorted(samples, key=lambda x: x.sample_id)]
            ),
            safety_factor=safety_factor,
            energy_floor=energy_floor,
            force_floor=force_floor,
        )

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))

    def predict(
        self,
        method: str,
        paired_delta: ObservableDelta,
        *,
        numerical_family_id: str,
        estimator_seconds: typing.Any = 0.0,
    ) -> NumericalEstimate:
        if method not in self.methods:
            raise ValueError("method is outside paired-estimator calibration domain")
        if numerical_family_id != self.numerical_family_id:
            raise ValueError(
                "numerical-level family is outside paired-estimator calibration"
            )
        floored = ObservableDelta(
            max(paired_delta.energy_abs, self.energy_floor),
            tuple(
                tuple(max(value, self.force_floor) for value in row)
                for row in paired_delta.force_components
            ),
        )
        return NumericalEstimate(
            floored.scaled(self.energy_scale, self.force_scale),
            self.identity,
            (
                "paired grid/screening differences are empirical, not rigorous bounds",
                "same electronic branch and derivative semantics, including moving-grid response",
                f"calibration covers only numerical-level family {self.numerical_family_id}",
            ),
            estimator_seconds=_number(estimator_seconds, "estimator seconds"),
            method=method,
        )

    def evaluate_holdout(
        self,
        samples: typing.Iterable[PairedCalibrationSample],
        budget: TargetErrorBudget,
    ) -> dict:
        """Report coverage, false-success and conservatism on held-out families."""
        rows = []
        for sample in samples:
            if sample.family in self.training_families:
                raise ValueError(
                    "molecular-family leakage between calibration and holdout"
                )
            estimate = self.predict(
                sample.method,
                sample.paired_delta,
                numerical_family_id=sample.numerical_family_id,
            )
            predicted_pass = budget.accepts(estimate.delta)
            actual_pass = budget.accepts(sample.strict_error)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "family": sample.family,
                    "predicted_pass": predicted_pass,
                    "actual_pass": actual_pass,
                    "false_success": predicted_pass and not actual_pass,
                    "overconservative": actual_pass and not predicted_pass,
                    "energy_covered": sample.strict_error.energy_abs
                    <= estimate.delta.energy_abs,
                    "force_covered": sample.strict_error.force_max_abs
                    <= estimate.delta.force_max_abs,
                    "predicted_energy": estimate.delta.energy_abs,
                    "actual_energy": sample.strict_error.energy_abs,
                    "predicted_force_max": estimate.delta.force_max_abs,
                    "actual_force_max": sample.strict_error.force_max_abs,
                }
            )
        false_successes = sum(row["false_success"] for row in rows)
        return {
            "rows": rows,
            "samples": len(rows),
            "false_successes": false_successes,
            "false_success_rate": false_successes / len(rows) if rows else 0.0,
            "overconservative": sum(row["overconservative"] for row in rows),
            "energy_coverage": sum(row["energy_covered"] for row in rows),
            "force_coverage": sum(row["force_covered"] for row in rows),
            "certified": False,
        }

    def as_error_evidence(
        self,
        model: ResolvedModel,
        estimate: NumericalEstimate,
        *,
        energy_reference_norm: typing.Any,
        force_reference_norm: typing.Any,
        scope: str = "relaxed_target",
    ) -> tuple[ErrorEvidence, ErrorEvidence]:
        """Project a supported-model estimate into the existing empirical contract."""
        if not isinstance(model, ResolvedModel):
            raise TypeError(
                "ErrorEvidence projection requires a supported ResolvedModel"
            )
        if estimate.calibration_id != self.identity:
            raise ValueError("estimate/calibration identity mismatch")
        if model.method not in self.methods or estimate.method != model.method:
            raise ValueError("estimate method does not match the target model method")
        common = {
            "kind": EvidenceKind.EMPIRICAL,
            "model_id": model.identity,
            "evaluated_model_id": model.identity,
            "reference_model_id": model.identity,
            "source": "total_numerical",
            "scope": scope,
            "provenance": (
                ("paired_estimator", self.identity),
                ("numerical_family", self.numerical_family_id),
                ("training_data", self.training_data_hash),
            ),
            "assumptions": estimate.assumptions,
            "calibration_id": self.identity,
        }
        return (
            ErrorEvidence(
                observable="energy",
                norm="absolute",
                unit="Eh",
                value=estimate.delta.energy_abs,
                reference_norm=_number(energy_reference_norm, "energy reference norm"),
                **common,
            ),
            ErrorEvidence(
                observable="forces",
                norm="max_abs",
                unit="Eh/bohr",
                value=estimate.delta.force_max_abs,
                reference_norm=_number(force_reference_norm, "force reference norm"),
                **common,
            ),
        )


@dataclass(frozen=True)
class NumericalLevel:
    """One already-defined numerical execution level, ordered loose to strict."""

    name: str
    rank: int
    screening_tolerance: float
    grid_identity: str
    strict: bool = False

    def __post_init__(self) -> None:
        _identity(self.name, "level name")
        _identity(self.grid_identity, "grid identity")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("level rank must be a nonnegative integer")
        object.__setattr__(
            self,
            "screening_tolerance",
            _number(self.screening_tolerance, "screening tolerance"),
        )
        if type(self.strict) is not bool:
            raise TypeError("strict level marker must be boolean")


@dataclass(frozen=True)
class DiscreteTransition:
    """Recorded change of grid/screening/mask policy identity."""

    from_level: str
    to_level: str
    reason: str
    geometry_id: str
    from_mask: str | None = None
    to_mask: str | None = None

    def __post_init__(self) -> None:
        for name in ("from_level", "to_level", "reason", "geometry_id"):
            _identity(getattr(self, name), name)
        for name in ("from_mask", "to_mask"):
            value = getattr(self, name)
            if value is not None:
                _identity(value, name)


@dataclass(frozen=True)
class AdaptiveNumericsState:
    level_index: int
    safe_streak: int = 0
    transitions: tuple[DiscreteTransition, ...] = ()
    mask_identity: str | None = None


@dataclass(frozen=True)
class NumericalDecision:
    action: str
    state: AdaptiveNumericsState
    reason: str
    worst_ratio: float | None


@dataclass(frozen=True)
class AdaptiveNumericsPolicy:
    """Hysteretic force-aware controller over pre-existing execution levels."""

    levels: tuple[NumericalLevel, ...]
    budget: TargetErrorBudget
    relax_ratio: float = 0.25
    relax_after: int = 2
    strict_reproducible: bool = False

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels or any(not isinstance(x, NumericalLevel) for x in levels):
            raise ValueError("policy requires typed numerical levels")
        if [x.rank for x in levels] != sorted(x.rank for x in levels):
            raise ValueError("numerical levels must be ordered from loose to strict")
        if len({x.rank for x in levels}) != len(levels) or len(
            {x.name for x in levels}
        ) != len(levels):
            raise ValueError("numerical level ranks/names must be unique")
        if not levels[-1].strict or any(x.strict for x in levels[:-1]):
            raise ValueError("exactly the final numerical level must be strict")
        if not isinstance(self.budget, TargetErrorBudget):
            raise TypeError("policy requires a TargetErrorBudget")
        object.__setattr__(
            self,
            "relax_ratio",
            _number(self.relax_ratio, "relax ratio", positive=True),
        )
        if self.relax_ratio >= 1:
            raise ValueError("relax ratio must be below one")
        if type(self.relax_after) is not int or self.relax_after < 1:
            raise ValueError("relax_after must be a positive integer")
        if type(self.strict_reproducible) is not bool:
            raise TypeError("strict_reproducible must be boolean")
        object.__setattr__(self, "levels", levels)

    def initial_state(
        self, *, start_index: int = 0, mask_identity: str | None = None
    ) -> AdaptiveNumericsState:
        if type(start_index) is not int or not 0 <= start_index < len(self.levels):
            raise ValueError("invalid initial numerical level")
        if mask_identity is not None:
            _identity(mask_identity, "mask identity")
        if self.strict_reproducible:
            start_index = len(self.levels) - 1
        return AdaptiveNumericsState(start_index, mask_identity=mask_identity)

    def _move(
        self,
        state: AdaptiveNumericsState,
        target: int,
        *,
        reason: str,
        geometry_id: str,
        mask_identity: str | None,
    ) -> AdaptiveNumericsState:
        current = self.levels[state.level_index]
        future = self.levels[target]
        if target == state.level_index and mask_identity == state.mask_identity:
            return replace(state, safe_streak=0)
        transition = DiscreteTransition(
            current.name,
            future.name,
            reason,
            geometry_id,
            state.mask_identity,
            mask_identity,
        )
        return AdaptiveNumericsState(
            target,
            0,
            state.transitions + (transition,),
            mask_identity,
        )

    def decide(
        self,
        state: AdaptiveNumericsState,
        estimate: NumericalEstimate | None,
        *,
        geometry_id: str,
        mask_identity: str | None = None,
        uncertainty_reason: str | None = None,
        switching_observed: bool = False,
    ) -> NumericalDecision:
        """Choose effort; uncertainty and missed targets always tighten/fail closed."""
        _identity(geometry_id, "geometry identity")
        if not 0 <= state.level_index < len(self.levels):
            raise ValueError("controller state level is outside this policy")
        if self.strict_reproducible:
            strict = len(self.levels) - 1
            moved = self._move(
                state,
                strict,
                reason="strict_reproducible",
                geometry_id=geometry_id,
                mask_identity=mask_identity,
            )
            return NumericalDecision("strict", moved, "strict reproducible mode", None)
        if uncertainty_reason is not None or estimate is None:
            reason = uncertainty_reason or "missing estimator evidence"
            target = min(state.level_index + 1, len(self.levels) - 1)
            moved = self._move(
                state,
                target,
                reason=reason,
                geometry_id=geometry_id,
                mask_identity=mask_identity,
            )
            action = (
                "strict_fallback" if self.levels[target].strict else "tighten_retry"
            )
            return NumericalDecision(action, moved, reason, None)
        if not isinstance(estimate, NumericalEstimate):
            raise TypeError("estimate must be NumericalEstimate or None")
        worst = max(self.budget.ratios(estimate.delta))
        if worst > 1:
            if self.levels[state.level_index].strict:
                return NumericalDecision(
                    "fail_unattainable",
                    replace(state, safe_streak=0),
                    "strict level still exceeds requested observable target",
                    worst,
                )
            target = state.level_index + 1
            moved = self._move(
                state,
                target,
                reason="estimated observable error exceeds target",
                geometry_id=geometry_id,
                mask_identity=mask_identity,
            )
            return NumericalDecision(
                "tighten_retry", moved, "estimated target miss", worst
            )
        if switching_observed:
            held = self._move(
                state,
                state.level_index,
                reason="discrete grid/mask switch observed",
                geometry_id=geometry_id,
                mask_identity=mask_identity,
            )
            return NumericalDecision(
                "hold_switch",
                held,
                "switching kept separate from smooth branch",
                worst,
            )
        if worst <= self.relax_ratio and state.level_index > 0:
            streak = state.safe_streak + 1
            if streak >= self.relax_after:
                moved = self._move(
                    state,
                    state.level_index - 1,
                    reason="hysteretic low-error relaxation",
                    geometry_id=geometry_id,
                    mask_identity=mask_identity,
                )
                return NumericalDecision(
                    "relax", moved, "sustained conservative margin", worst
                )
            held = self._move(
                state,
                state.level_index,
                reason="hysteresis hold",
                geometry_id=geometry_id,
                mask_identity=mask_identity,
            )
            return NumericalDecision(
                "hold",
                replace(held, safe_streak=streak),
                "waiting for hysteresis streak",
                worst,
            )
        held = self._move(
            state,
            state.level_index,
            reason="observable estimate within target",
            geometry_id=geometry_id,
            mask_identity=mask_identity,
        )
        return NumericalDecision(
            "hold",
            held,
            "observable estimate within target",
            worst,
        )

    def verify_final(
        self,
        *,
        reference_level: NumericalLevel,
        actual_error: ObservableDelta,
        uncertainty_reason: str | None = None,
    ) -> str:
        """Require an actual comparison against the declared strict target level."""
        if reference_level != self.levels[-1] or not reference_level.strict:
            raise ValueError(
                "final verification requires the policy's strict target level"
            )
        if uncertainty_reason is not None:
            return "unverified"
        return "observed_met" if self.budget.accepts(actual_error) else "observed_unmet"
