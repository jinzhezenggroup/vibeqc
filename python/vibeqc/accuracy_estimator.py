"""Limited-domain empirical HF error envelopes with explicit rejection.

The envelope relates measured physical residuals to observed energy/force
errors. It is a calibration result, not an inverse-Jacobian bound. Whole
molecular families must be held out when assessing its reliability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .accuracy import ErrorEvidence, EvidenceKind, ResolvedModel, _identity, _number
from .elements import checked_integer
from .profiles import canonical_hash


@dataclass(frozen=True)
class HFErrorFeatures:
    """Physical operator diagnostics; iteration density changes are excluded.

    ``basis_family_id`` identifies a verified complete basis-data family, not
    a caller's basis alias. The feature extractor must compare the actual
    atom-bound shells before assigning that identity.
    """

    model_id: str
    basis_family_id: str
    atomic_numbers: tuple[int, ...]
    nao: int
    physical_residual: float
    overlap_condition: float
    minimum_gap: float | None
    electron_trace_error: float
    idempotency_error: float
    minimum_nuclear_distance: float | None
    backend: str = "cpu"
    precision: str = "fp64"

    def __post_init__(self):
        for name in ("model_id", "basis_family_id", "backend", "precision"):
            _identity(getattr(self, name), name)
        numbers = tuple(
            checked_integer(z, "atomic number", low=1, high=118)
            for z in self.atomic_numbers
        )
        if not numbers:
            raise ValueError("error features require nuclei")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "nao", checked_integer(self.nao, "AO count", low=1))
        for name in (
            "physical_residual",
            "overlap_condition",
            "minimum_gap",
            "electron_trace_error",
            "idempotency_error",
            "minimum_nuclear_distance",
        ):
            value = getattr(self, name)
            if value is None and name in ("minimum_gap", "minimum_nuclear_distance"):
                continue
            object.__setattr__(self, name, _number(value, name))


@dataclass(frozen=True)
class HFCalibrationDomain:
    """A deliberately narrow real, conventional, minimum-basis RHF domain.

    Gap/metric cutoffs are validity filters chosen before holdout evaluation.
    They are not observable-error bounds and do not establish HF stability.
    """

    basis_family_id: str
    atomic_numbers: tuple[int, ...] = (1, 6, 7, 8, 9)
    max_nao: int = 12
    max_electrons: int = 10
    maximum_residual: float = 1e-2
    maximum_overlap_condition: float = 30.0
    minimum_gap: float = 0.1
    maximum_trace_error: float = 1e-8
    maximum_idempotency_error: float = 1e-8
    minimum_nuclear_distance: float = 0.5
    backend: str = "cpu"
    schema_version: int = 1

    def __post_init__(self):
        _identity(self.basis_family_id, "basis family")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported calibration domain schema")
        if self.backend != "cpu":
            raise ValueError("this first empirical domain is CPU FP64 only")
        numbers = tuple(
            sorted(
                {
                    checked_integer(z, "atomic number", low=1, high=118)
                    for z in self.atomic_numbers
                }
            )
        )
        if not numbers:
            raise ValueError("calibration domain requires elements")
        object.__setattr__(self, "atomic_numbers", numbers)
        for name in ("max_nao", "max_electrons"):
            object.__setattr__(
                self, name, checked_integer(getattr(self, name), name, low=1)
            )
        for name in (
            "maximum_residual",
            "maximum_overlap_condition",
            "minimum_gap",
            "maximum_trace_error",
            "maximum_idempotency_error",
            "minimum_nuclear_distance",
        ):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=True)
            )

    def rejection_reasons(
        self, model: ResolvedModel, features: HFErrorFeatures
    ) -> tuple[str, ...]:
        """Reject unsupported chemistry, arithmetic, rank or conditioning explicitly."""
        reasons = []
        if model.identity != features.model_id:
            reasons.append("model/diagnostic identity mismatch")
        if (
            model.method != "rhf"
            or model.approximation != "conventional"
            or model.representation != "cartesian"
        ):
            reasons.append("only conventional Cartesian RHF is calibrated")
        if features.basis_family_id != self.basis_family_id:
            reasons.append("basis family is outside calibration")
        if not set(features.atomic_numbers) <= set(self.atomic_numbers):
            reasons.append("elements are outside calibration")
        if features.backend != self.backend or features.precision != "fp64":
            reasons.append("backend/arithmetic is outside calibration")
        if features.nao > self.max_nao or model.electron_count > self.max_electrons:
            reasons.append("system size is outside calibration")
        if features.physical_residual > self.maximum_residual:
            reasons.append("physical residual is outside calibration")
        if not 1 <= features.overlap_condition <= self.maximum_overlap_condition:
            reasons.append("ill-conditioned AO metric")
        if features.minimum_gap is None or features.minimum_gap < self.minimum_gap:
            reasons.append("missing or small occupied-virtual gap")
        if features.electron_trace_error > self.maximum_trace_error:
            reasons.append("electron populations are not validated")
        if features.idempotency_error > self.maximum_idempotency_error:
            reasons.append("density idempotency is not validated")
        # A missing separation is meaningful only for an isolated atom. For a
        # molecule it is absent evidence, not permission to skip a domain gate.
        if (
            len(features.atomic_numbers) > 1
            and features.minimum_nuclear_distance is None
        ):
            reasons.append(
                "missing nuclear separation diagnostic for a multi-atom system"
            )
        elif (
            features.minimum_nuclear_distance is not None
            and features.minimum_nuclear_distance < self.minimum_nuclear_distance
        ):
            reasons.append("nuclear separation is outside calibration")
        return tuple(reasons)


@dataclass(frozen=True)
class HFCalibrationSample:
    """One whole-family-labelled observation against a relaxed strict reference."""

    family: str
    sample_id: str
    model: ResolvedModel
    features: HFErrorFeatures
    energy_error: float
    force_max_error: float
    reference_id: str

    def __post_init__(self):
        for name in ("family", "sample_id", "reference_id"):
            _identity(getattr(self, name), name)
        if not isinstance(self.model, ResolvedModel) or not isinstance(
            self.features, HFErrorFeatures
        ):
            raise TypeError("calibration requires typed scientific state")
        if self.model.identity != self.features.model_id:
            raise ValueError("calibration sample model mismatch")
        for name in ("energy_error", "force_max_error"):
            object.__setattr__(self, name, _number(getattr(self, name), name))


@dataclass(frozen=True)
class EmpiricalHFEstimator:
    """Versioned conservative training envelope with honest empirical status.

    For observable q the prediction is ``floor_q + slope_q*max(residual, r0)``.
    A safety factor multiplies the largest training error/residual ratio.
    This intentionally simple first estimator exposes over-conservatism and
    holdout misses instead of fitting a high-capacity model to a tiny dataset.
    """

    domain: HFCalibrationDomain
    energy_slope: float
    force_slope: float
    training_families: tuple[str, ...]
    training_data_hash: str
    safety_factor: float = 4.0
    residual_floor: float = 1e-12
    energy_floor: float = 1e-12
    force_floor: float = 1e-10
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.domain, HFCalibrationDomain):
            raise TypeError("estimator requires a typed calibration domain")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported estimator schema")
        _identity(self.training_data_hash, "training data hash")
        families = tuple(sorted(set(self.training_families)))
        if len(families) < 2:
            raise ValueError("calibration requires at least two molecular families")
        for family in families:
            _identity(family, "training family")
        object.__setattr__(self, "training_families", families)
        for name in (
            "energy_slope",
            "force_slope",
            "safety_factor",
            "residual_floor",
            "energy_floor",
            "force_floor",
        ):
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name,
                    positive=name not in ("energy_slope", "force_slope"),
                ),
            )
        if self.safety_factor < 1:
            raise ValueError("training safety factor must be at least one")

    @classmethod
    def fit(cls, domain: HFCalibrationDomain, samples, *, safety_factor=4.0):
        """Fit only declared training families; never ingest held-out observations."""
        samples = tuple(samples)
        if not samples or any(not isinstance(s, HFCalibrationSample) for s in samples):
            raise ValueError("fit requires typed calibration samples")
        if len({s.sample_id for s in samples}) != len(samples):
            raise ValueError("duplicate calibration sample identities")
        safety_factor = _number(safety_factor, "safety factor", positive=True)
        if safety_factor < 1:
            raise ValueError("training safety factor must be at least one")
        for sample in samples:
            if reasons := domain.rejection_reasons(sample.model, sample.features):
                raise ValueError(
                    "training sample outside domain: " + "; ".join(reasons)
                )
        residual_floor = 1e-12
        slopes = [
            safety_factor
            * max(
                getattr(s, error) / max(s.features.physical_residual, residual_floor)
                for s in samples
            )
            for error in ("energy_error", "force_max_error")
        ]
        return cls(
            domain,
            *slopes,
            tuple(s.family for s in samples),
            canonical_hash(
                [asdict(s) for s in sorted(samples, key=lambda s: s.sample_id)]
            ),
            safety_factor=safety_factor,
            residual_floor=residual_floor,
        )

    @property
    def identity(self) -> str:
        """Tie predictions to exact training evidence and domain/version choices."""
        return canonical_hash(asdict(self))

    def predict(
        self,
        model: ResolvedModel,
        features: HFErrorFeatures,
        *,
        energy_reference_norm: float,
        force_reference_norm: float,
    ) -> tuple[ErrorEvidence, ErrorEvidence]:
        """Return estimates, or reject; never turn small residuals into certification."""
        if reasons := self.domain.rejection_reasons(model, features):
            raise ValueError("outside calibration domain: " + "; ".join(reasons))
        residual = max(features.physical_residual, self.residual_floor)
        rows = (
            (
                "energy",
                "absolute",
                "Eh",
                self.energy_floor + self.energy_slope * residual,
                energy_reference_norm,
            ),
            (
                "forces",
                "max_abs",
                "Eh/bohr",
                self.force_floor + self.force_slope * residual,
                force_reference_norm,
            ),
        )
        return tuple(
            ErrorEvidence(
                EvidenceKind.EMPIRICAL,
                model.identity,
                model.identity,
                model.identity,
                observable,
                norm,
                unit,
                value,
                reference_norm,
                "total_numerical",
                "relaxed_target",
                (
                    ("training_data", self.training_data_hash),
                    ("estimator", self.identity),
                ),
                assumptions=(
                    "same intended stationary HF branch; stability is not established by the gap",
                    "finite training envelope only; no inverse-Jacobian/adjoint bound",
                    "CPU FP64 conventional HF within the recorded basis/size/conditioning domain",
                ),
                calibration_id=self.identity,
                condition_estimate=features.overlap_condition,
            )
            for observable, norm, unit, value, reference_norm in rows
        )

    def to_dict(self) -> dict:
        """Serialize a non-executable, checksum-identified calibration artifact."""
        return {**asdict(self), "identity": self.identity, "kind": "empirical_envelope"}

    @classmethod
    def from_dict(cls, record: dict) -> EmpiricalHFEstimator:
        """Validate a saved envelope; a status label cannot upgrade its evidence."""
        payload = dict(record)
        identity = payload.pop("identity")
        if payload.pop("kind") != "empirical_envelope":
            raise ValueError("unsupported estimator evidence kind")
        payload["domain"] = HFCalibrationDomain(**payload["domain"])
        result = cls(**payload)
        if result.identity != identity:
            raise ValueError("estimator checksum mismatch")
        return result

    def evaluate_holdout(
        self, samples, *, energy_tolerance=1e-6, force_tolerance=1e-6
    ) -> dict:
        """Report whole-family coverage, missed tolerances and excess rejection."""
        energy_tolerance = _number(energy_tolerance, "energy tolerance", positive=True)
        force_tolerance = _number(force_tolerance, "force tolerance", positive=True)
        rows = []
        for sample in samples:
            if sample.family in self.training_families:
                raise ValueError(
                    "molecular-family leakage between training and holdout"
                )
            reasons = self.domain.rejection_reasons(sample.model, sample.features)
            if reasons:
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "family": sample.family,
                        "status": "outside_domain",
                        "reasons": reasons,
                    }
                )
                continue
            estimates = self.predict(
                sample.model,
                sample.features,
                energy_reference_norm=0,
                force_reference_norm=0,
            )
            observable_rows = []
            for estimate, actual, tolerance in zip(
                estimates,
                (sample.energy_error, sample.force_max_error),
                (energy_tolerance, force_tolerance),
                strict=True,
            ):
                observable_rows.append(
                    {
                        "observable": estimate.observable,
                        "predicted": estimate.value,
                        "actual": actual,
                        "tolerance": tolerance,
                        "covered": actual <= estimate.value,
                        "missed_tolerance": estimate.value <= tolerance < actual,
                        "overconservative": actual <= tolerance < estimate.value,
                    }
                )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "family": sample.family,
                    "status": "evaluated",
                    "observables": observable_rows,
                }
            )
        if not rows:
            raise ValueError("holdout analysis requires at least one observation")
        observed = [o for row in rows for o in row.get("observables", ())]
        return {
            "estimator_id": self.identity,
            "split": "whole_molecular_family",
            "rows": rows,
            "observables_evaluated": len(observed),
            "outside_domain": sum(r["status"] == "outside_domain" for r in rows),
            "covered": sum(o["covered"] for o in observed),
            "missed_tolerances": sum(o["missed_tolerance"] for o in observed),
            "overconservative": sum(o["overconservative"] for o in observed),
            "certified": False,
        }
