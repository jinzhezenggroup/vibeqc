"""Observable accuracy contracts, independent of SCF convergence and arithmetic.

Evidence describes a comparison with one fixed scientific model. Neither a
small iteration change nor an empirical predictor certifies a relaxed energy
or force. This first contract deliberately has no public ``certified`` status;
validated observable-bound implementations must precede such a claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Real

import numpy as np

from .elements import checked_integer
from .profiles import canonical_hash

SCHEMA_VERSION = 1


def _number(value, name, *, positive=False):
    """Accept real finite values without silently coercing strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")


@dataclass(frozen=True)
class ResolvedModel:
    """Backend-independent all-electron HF model at a specified geometry.

    Hashes identify actual ordered nuclei, coordinates, normalized basis data and
    AO conventions, not basis aliases or array dimensions. The metric threshold
    belongs to the fitted model: changing it changes the operator. Screening and
    iteration/arithmetic settings belong to the numerical experiment instead.
    DF's effective rank is recorded in evidence because it can vary by geometry.
    DFT and correlated models require additional identities before this schema
    can represent them and are rejected for now.
    """

    method: str
    geometry_hash: str
    basis_hash: str
    electron_count: int
    multiplicity: int = 1
    charge: int = 0
    representation: str = "cartesian"
    hamiltonian: str = "all-electron-nonrelativistic-coulomb"
    approximation: str = "conventional"
    auxiliary_basis_hash: str | None = None
    metric_relative_threshold: float | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported resolved-model schema")
        if self.method not in ("rhf", "uhf"):
            raise ValueError("accuracy models currently support RHF/UHF only")
        if self.hamiltonian != "all-electron-nonrelativistic-coulomb":
            raise ValueError("unsupported Hamiltonian/core treatment")
        if self.representation not in ("cartesian", "real_spherical"):
            raise ValueError("unsupported AO convention")
        for name in ("geometry_hash", "basis_hash"):
            _identity(getattr(self, name), name)
        for name, low in (
            ("electron_count", 1),
            ("multiplicity", 1),
            ("charge", -(2**31)),
        ):
            object.__setattr__(
                self, name, checked_integer(getattr(self, name), name, low=low)
            )
        unpaired = self.multiplicity - 1
        if unpaired > self.electron_count or (self.electron_count - unpaired) % 2:
            raise ValueError("inconsistent electron count and spin populations")
        if self.method == "rhf" and self.multiplicity != 1:
            raise ValueError("RHF requires a closed-shell singlet")
        if self.approximation == "conventional":
            if (
                self.auxiliary_basis_hash is not None
                or self.metric_relative_threshold is not None
            ):
                raise ValueError("conventional HF cannot carry a fitted operator")
        elif self.approximation == "density_fitting":
            _identity(self.auxiliary_basis_hash, "auxiliary_basis_hash")
            threshold = _number(
                self.metric_relative_threshold, "metric threshold", positive=True
            )
            if threshold >= 1:
                raise ValueError("metric relative threshold must be less than one")
            object.__setattr__(self, "metric_relative_threshold", threshold)
        else:
            raise ValueError("unsupported approximation identity")

    @property
    def identity(self) -> str:
        """Reuse the canonical JSON hash used by profiles and reference snapshots."""
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict:
        """Return a detached, portable record with its scientific identity."""
        return {**asdict(self), "identity": self.identity}

    @classmethod
    def from_dict(cls, record: dict) -> ResolvedModel:
        """Validate a portable model and detect an altered identity payload."""
        payload = dict(record)
        identity = payload.pop("identity")
        model = cls(**payload)
        if model.identity != identity:
            raise ValueError("resolved-model identity checksum mismatch")
        return model


@dataclass(frozen=True)
class ObservableTarget:
    """Require ``error_norm <= absolute + relative * reference_norm``.

    Energy uses scalar absolute magnitude in Eh. Forces use either maximum
    absolute Cartesian component or RMS over all 3N components in Eh/bohr.
    Relative tolerances are dimensionless; a zero reference norm contributes
    zero relative allowance. Units are explicit and never converted implicitly.
    """

    observable: str
    norm: str
    unit: str
    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self):
        if (self.observable, self.norm, self.unit) not in (
            ("energy", "absolute", "Eh"),
            ("forces", "max_abs", "Eh/bohr"),
            ("forces", "rms", "Eh/bohr"),
        ):
            raise ValueError("unsupported observable, norm, or unit")
        for name in ("absolute", "relative"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.absolute == self.relative == 0:
            raise ValueError("an accuracy target requires a positive tolerance")

    def allowance(self, reference_norm: float) -> float:
        """Resolve the tolerance using the declared norm of the reference value."""
        value = self.absolute + self.relative * _number(
            reference_norm, "reference norm"
        )
        if not math.isfinite(value):
            raise ValueError("resolved observable tolerance overflowed")
        return value


@dataclass(frozen=True)
class TargetAccuracy:
    """Independent observable requirements relative to a fixed resolved model.

    Requirements are combined with AND, never traded against each other. A
    fixed-density audit cannot satisfy the default relaxed-target scope. This
    object is an evidence request and does not alter native solver defaults.
    """

    observables: tuple[ObservableTarget, ...]
    scope: str = "relaxed_target"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        values = tuple(self.observables)
        if not values or any(not isinstance(v, ObservableTarget) for v in values):
            raise ValueError("observables must contain typed accuracy requirements")
        keys = [(v.observable, v.norm) for v in values]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate observable/norm requirement")
        if self.scope not in ("fixed_density", "relaxed_target"):
            raise ValueError("unknown accuracy scope")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported target-accuracy schema")
        object.__setattr__(
            self,
            "observables",
            tuple(sorted(values, key=lambda v: (v.observable, v.norm))),
        )

    def to_dict(self) -> dict:
        """Return explicit units, norms and scope for result/benchmark records."""
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> TargetAccuracy:
        """Reconstruct typed requirements while rejecting unsupported fields."""
        payload = dict(record)
        payload["observables"] = tuple(
            ObservableTarget(**item) for item in payload["observables"]
        )
        return cls(**payload)


class EvidenceKind(str, Enum):
    """Numerical comparisons are observations; extrapolations remain estimates."""

    OBSERVED = "reference_difference"
    EMPIRICAL = "empirical_predictor"
    ASYMPTOTIC = "asymptotic_estimator"
    BOUND = "proven_bound"


@dataclass(frozen=True)
class ErrorEvidence:
    """One scoped error observation/estimate, with immutable provenance.

    ``model_id`` is the declared target, ``evaluated_model_id`` is the model
    actually evaluated, and ``reference_model_id`` identifies the comparator.
    All three are checked at assessment time. An experiment may record a changed
    model, but that record cannot establish numerical accuracy for the target.

    Only ``source='total_numerical'`` evidence covers the entire requested error.
    Per-source estimates are retained for experiments and are never added without
    a validated composition rule. Public proven-bound claims are rejected until
    an observable-specific proof implementation exists.
    """

    kind: EvidenceKind
    model_id: str
    evaluated_model_id: str
    reference_model_id: str
    observable: str
    norm: str
    unit: str
    value: float
    reference_norm: float
    source: str
    scope: str
    provenance: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...] = ()
    calibration_id: str | None = None
    condition_estimate: float | None = None
    actual_reference_error: float | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("unsupported error-evidence schema")
        kind = EvidenceKind(self.kind)
        if kind is EvidenceKind.BOUND:
            raise ValueError(
                "proven observable bounds are not implemented; a label is not a proof"
            )
        object.__setattr__(self, "kind", kind)
        ObservableTarget(self.observable, self.norm, self.unit, absolute=1.0)
        for name in ("model_id", "evaluated_model_id", "reference_model_id", "source"):
            _identity(getattr(self, name), name)
        if self.scope not in ("fixed_density", "relaxed_target"):
            raise ValueError("unknown evidence scope")
        for name in (
            "value",
            "reference_norm",
            "condition_estimate",
            "actual_reference_error",
        ):
            value = getattr(self, name)
            if value is not None or name in ("value", "reference_norm"):
                object.__setattr__(self, name, _number(value, name))
        if isinstance(self.assumptions, str):
            raise TypeError("assumptions must be a sequence of statements")
        assumptions = tuple(self.assumptions)
        for assumption in assumptions:
            _identity(assumption, "assumption")
        if kind is not EvidenceKind.OBSERVED:
            _identity(self.calibration_id, "calibration or derivation identity")
            if not assumptions:
                raise ValueError("estimates require explicit validity assumptions")
        if (
            kind is EvidenceKind.OBSERVED
            and self.actual_reference_error is not None
            and self.actual_reference_error != self.value
        ):
            raise ValueError("observed and actual reference errors disagree")
        provenance = tuple(tuple(pair) for pair in self.provenance)
        if not provenance or any(len(pair) != 2 for pair in provenance):
            raise ValueError("provenance requires key/value identity pairs")
        for key, value in provenance:
            _identity(key, "provenance key")
            _identity(value, "provenance value")
        if len({key for key, _ in provenance}) != len(provenance):
            raise ValueError("duplicate provenance keys")
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "provenance", tuple(sorted(provenance)))

    def to_dict(self) -> dict:
        """Return evidence with stable, JSON-compatible enum values."""
        return {**asdict(self), "kind": self.kind.value}


@dataclass(frozen=True)
class AccuracyAssessment:
    """Derived outcome; observed agreement is not a certified error guarantee."""

    model: ResolvedModel
    target: TargetAccuracy
    evidence: tuple[ErrorEvidence, ...] = ()
    converged: bool = True

    def __post_init__(self):
        if not isinstance(self.model, ResolvedModel) or not isinstance(
            self.target, TargetAccuracy
        ):
            raise TypeError("assessment requires typed model and target identities")
        if type(self.converged) is not bool:
            raise ValueError("converged must be a boolean")
        evidence = tuple(self.evidence)
        keys = []
        for item in evidence:
            if not isinstance(item, ErrorEvidence):
                raise TypeError("assessment requires typed error evidence")
            if {item.model_id, item.evaluated_model_id, item.reference_model_id} != {
                self.model.identity
            }:
                raise ValueError(
                    "evidence model mismatch: keep the original target fixed"
                )
            keys.append((item.observable, item.norm, item.source, item.scope))
        if len(set(keys)) != len(keys):
            raise ValueError("ambiguous duplicate evidence for one error source")
        object.__setattr__(self, "evidence", evidence)

    @property
    def outcomes(self) -> tuple[str, ...]:
        """Evaluate each requirement without combining independent error sources."""
        if not self.converged:
            return tuple("unconverged" for _ in self.target.observables)
        values = []
        for requirement in self.target.observables:
            item = next(
                (
                    e
                    for e in self.evidence
                    if (
                        e.observable == requirement.observable
                        and e.norm == requirement.norm
                        and e.scope == self.target.scope
                        and e.source == "total_numerical"
                    )
                ),
                None,
            )
            if item is None:
                values.append("unverified")
                continue
            actual = item.actual_reference_error
            observed = item.kind is EvidenceKind.OBSERVED or actual is not None
            error = item.value if actual is None else actual
            passed = error <= requirement.allowance(item.reference_norm)
            # Validity-domain membership belongs to the calibrated estimator.
            # A raw estimate cannot establish success merely by being small.
            values.append(
                ("observed_met" if passed else "observed_unmet")
                if observed
                else ("estimated_below_target" if passed else "estimated_above_target")
            )
        return tuple(values)

    @property
    def status(self) -> str:
        """Report incomplete coverage even when another observable passes."""
        outcomes = self.outcomes
        for status in (
            "unconverged",
            "observed_unmet",
            "unverified",
            "estimated_above_target",
            "estimated_below_target",
        ):
            if status in outcomes:
                return status
        return "observed_met"

    def to_dict(self) -> dict:
        """Return a self-contained record suitable for local result export."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model.to_dict(),
            "target": self.target.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "converged": self.converged,
            "outcomes": self.outcomes,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, record: dict) -> AccuracyAssessment:
        """Recompute outcomes on load; serialized claims are never authoritative."""
        expected = {
            "schema_version",
            "model",
            "target",
            "evidence",
            "converged",
            "outcomes",
            "status",
        }
        if (
            set(record) != expected
            or type(record["schema_version"]) is not int
            or record["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported accuracy-assessment schema")
        result = cls(
            ResolvedModel.from_dict(record["model"]),
            TargetAccuracy.from_dict(record["target"]),
            tuple(ErrorEvidence(**item) for item in record["evidence"]),
            record["converged"],
        )
        if result.status != record["status"] or result.outcomes != tuple(
            record["outcomes"]
        ):
            raise ValueError("serialized accuracy outcome disagrees with its evidence")
        return result


def compare_observables(
    model: ResolvedModel,
    evaluated_model: ResolvedModel,
    reference_model: ResolvedModel,
    target: TargetAccuracy,
    values: dict,
    reference_values: dict,
    *,
    scope: str,
    provenance: tuple[tuple[str, str], ...],
    converged: bool,
) -> AccuracyAssessment:
    """Measure energy/force differences against a separately computed reference.

    This helper performs no solve and makes no stationarity or electronic-root
    inference. Callers must supply the actual audit scope and convergence status.
    The resulting status describes observed agreement only, even for a strict
    reference: its residual/reference uncertainty still needs separate evidence.
    """
    evidence = []
    for requirement in target.observables:
        name = requirement.observable
        arrays = []
        for mapping in (values, reference_values):
            raw = np.asarray(mapping[name])
            if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
                raise ValueError("observables must be finite real numeric values")
            value = np.asarray(raw, dtype=np.float64)
            if name == "energy" and value.shape != ():
                raise ValueError("energy must be a scalar")
            if name == "forces" and (
                value.ndim != 2 or value.shape[1] != 3 or not len(value)
            ):
                raise ValueError("forces must have shape (Natom, 3)")
            arrays.append(value)
        value, reference = arrays
        if value.shape != reference.shape:
            raise ValueError("observable dimensions differ")

        def norm(array, norm_name=requirement.norm):
            # Scaling avoids overflow when squaring large finite values.
            maximum = float(np.max(np.abs(array)))
            if not math.isfinite(maximum):
                raise ValueError("observable difference overflowed")
            if norm_name == "rms" and maximum:
                return maximum * float(np.sqrt(np.mean((array / maximum) ** 2)))
            return maximum

        with np.errstate(over="ignore", invalid="ignore"):
            difference = value - reference
        evidence.append(
            ErrorEvidence(
                EvidenceKind.OBSERVED,
                model.identity,
                evaluated_model.identity,
                reference_model.identity,
                name,
                requirement.norm,
                requirement.unit,
                norm(difference),
                norm(reference),
                "total_numerical",
                scope,
                provenance,
            )
        )
    return AccuracyAssessment(model, target, tuple(evidence), converged)
