"""Typed, bounded orchestration for deterministic low-cost-to-target HF solves.

The source stage is only an initialization proposal.  It never substitutes for
the requested target model, its convergence, or its observable-accuracy audit.
"""

from __future__ import annotations

import math
import time
import typing
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256

import numpy as np

from . import _native
from .accuracy import AccuracyAssessment, ErrorEvidence, ResolvedModel, TargetAccuracy
from .calculator import Atom
from .fock import FockBuildSpec, FockPlan
from .overlap import cross_overlap
from .profiles import canonical_hash
from .progressive import _retained_density, initialize_from
from .projection import ProjectionPolicy, ProjectionRejected

if typing.TYPE_CHECKING:
    from .batch import BatchItemResult

_SCHEMA_VERSION = 1


def _positive(value: typing.Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _identity(value: typing.Any, name: str, *, digest: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    if digest and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _provider_hashes(
    metadata: typing.Mapping[str, typing.Any],
) -> tuple[tuple[str, str], ...]:
    """Detach mathematical and source identities for every final basis provider."""
    values: list[tuple[str, str]] = []
    for role in ("orbital", "auxiliary"):
        record = metadata.get(role)
        if record is None:
            continue
        for name in ("mathematical_identity", "basis_identity"):
            values.append(
                (f"{role}.{name}", _identity(record[name], name, digest=True))
            )
        provenance = record.get("provenance", {})
        values.append(
            (
                f"{role}.source_checksum",
                _identity(
                    provenance.get("checksum"), "basis source checksum", digest=True
                ),
            )
        )
    if not values:
        raise ValueError("target provider metadata has no orbital basis identity")
    return tuple(values)


@dataclass(frozen=True)
class HFConvergence:
    """One stage's solver tolerances, with a target-only physical residual gate."""

    energy_change: float
    density_rms: float
    maximum_iterations: int
    physical_residual_rms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "energy_change", _positive(self.energy_change, "energy tolerance")
        )
        object.__setattr__(
            self, "density_rms", _positive(self.density_rms, "density tolerance")
        )
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be a positive integer")
        if self.physical_residual_rms is not None:
            object.__setattr__(
                self,
                "physical_residual_rms",
                _positive(self.physical_residual_rms, "physical residual tolerance"),
            )


@dataclass(frozen=True)
class ArithmeticPolicy:
    """Requested arithmetic and the strict final-state contract from #174."""

    requested_precision: str
    required_final_bits: int = 64
    require_strict_refinement: bool = False

    def __post_init__(self) -> None:
        if self.requested_precision not in ("fp64", "auto"):
            raise ValueError("HF arithmetic must request fp64 or auto")
        if type(self.required_final_bits) is not int or self.required_final_bits != 64:
            raise ValueError("the first HF controller requires a final FP64 state")
        if type(self.require_strict_refinement) is not bool:
            raise TypeError("require_strict_refinement must be boolean")
        if self.requested_precision == "auto" and not self.require_strict_refinement:
            raise ValueError("auto arithmetic must require strict target refinement")
        if self.requested_precision == "fp64" and self.require_strict_refinement:
            raise ValueError("an all-FP64 stage does not have a mixed-state refinement")

    @property
    def is_strict_target(self) -> bool:
        return self.required_final_bits == 64 and (
            self.requested_precision == "fp64" or self.require_strict_refinement
        )


class StageRole(str, Enum):
    INITIALIZATION = "initialization"
    TARGET = "target"


class TransferOperation(str, Enum):
    METRIC_PROJECTED_DENSITY = "metric_projected_density"
    NONE = "none"


@dataclass(frozen=True)
class TargetProblem:
    """Immutable requested HF physics, provider, observables and final gates.

    Grid and local-policy identities are explicit even though this first HF slice
    requires both to be absent.  Future adapters therefore cannot silently add
    or redefine either part of the target.
    """

    model: ResolvedModel
    provider_identity: str
    provider_hashes: tuple[tuple[str, str], ...]
    accuracy: TargetAccuracy
    convergence: HFConvergence
    atom_count: int
    grid_policy_identity: str | None = None
    local_policy_identity: str | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.model, ResolvedModel) or self.model.method not in (
            "rhf",
            "uhf",
        ):
            raise TypeError(
                "the first progressive target must be a resolved RHF/UHF model"
            )
        if not isinstance(self.accuracy, TargetAccuracy):
            raise TypeError("target accuracy must use the #173 typed contract")
        if not isinstance(self.convergence, HFConvergence):
            raise TypeError("target convergence must be typed")
        if self.convergence.physical_residual_rms is None:
            raise ValueError("target convergence requires a physical residual gate")
        if type(self.atom_count) is not int or self.atom_count < 1:
            raise ValueError("target atom_count must be a positive integer")
        _identity(self.provider_identity, "target provider identity", digest=True)
        hashes = tuple(tuple(pair) for pair in self.provider_hashes)
        if not hashes or any(len(pair) != 2 for pair in hashes):
            raise ValueError("target provider hashes require named hash pairs")
        for name, value in hashes:
            _identity(name, "provider hash name")
            _identity(value, "provider hash", digest=True)
        if len({name for name, _ in hashes}) != len(hashes):
            raise ValueError("duplicate target provider hash names")
        object.__setattr__(self, "provider_hashes", tuple(sorted(hashes)))
        if (
            self.grid_policy_identity is not None
            or self.local_policy_identity is not None
        ):
            raise ValueError(
                "grid and local-policy adapters are outside the first HF slice"
            )
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported TargetProblem schema")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "provider_identity": self.provider_identity,
            "provider_hashes": self.provider_hashes,
            "accuracy": self.accuracy.to_dict(),
            "convergence": asdict(self.convergence),
            "atom_count": self.atom_count,
            "grid_policy_identity": self.grid_policy_identity,
            "local_policy_identity": self.local_policy_identity,
        }

    @classmethod
    def from_calculator(
        cls,
        calculator: typing.Any,
        atoms: typing.Iterable[typing.Any],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        accuracy: TargetAccuracy | None = None,
        physical_residual_tolerance: float | None = None,
    ) -> TargetProblem:
        """Snapshot an HF calculator without performing integral or SCF work."""
        atoms = tuple(Atom.from_value(atom) for atom in atoms)
        selected_accuracy = (
            calculator._target_accuracy if accuracy is None else accuracy
        )
        if not isinstance(selected_accuracy, TargetAccuracy):
            raise TypeError("TargetProblem requires an explicit TargetAccuracy request")
        metadata = calculator.basis_metadata(
            atoms, charge=charge, multiplicity=multiplicity
        )
        residual = (
            min(1.0e-8, calculator._density_tolerance)
            if physical_residual_tolerance is None
            else physical_residual_tolerance
        )
        return cls(
            model=calculator.resolved_model(
                atoms, charge=charge, multiplicity=multiplicity
            ),
            provider_identity=metadata["model_identity"],
            provider_hashes=_provider_hashes(metadata),
            accuracy=selected_accuracy,
            convergence=HFConvergence(
                calculator._energy_tolerance,
                calculator._density_tolerance,
                calculator._max_iterations,
                residual,
            ),
            atom_count=len(atoms),
        )


@dataclass(frozen=True)
class StagePlan:
    """One resolved model, numerical policy, transfer edge and typed estimates."""

    stage_id: str
    role: StageRole
    model: ResolvedModel
    provider_identity: str
    provider_hashes: tuple[tuple[str, str], ...]
    arithmetic: ArithmeticPolicy
    convergence: HFConvergence
    transfer: TransferOperation
    estimated_cost_units: float | None = None
    error_evidence: tuple[ErrorEvidence, ...] = ()
    allowed_next_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.stage_id, "stage id")
        object.__setattr__(self, "role", StageRole(self.role))
        object.__setattr__(self, "transfer", TransferOperation(self.transfer))
        if not isinstance(self.model, ResolvedModel) or self.model.method not in (
            "rhf",
            "uhf",
        ):
            raise TypeError("HF StagePlan requires a resolved RHF/UHF model")
        _identity(self.provider_identity, "stage provider identity", digest=True)
        hashes = tuple(tuple(pair) for pair in self.provider_hashes)
        if not hashes or any(len(pair) != 2 for pair in hashes):
            raise ValueError("stage provider hashes require named hash pairs")
        for name, value in hashes:
            _identity(name, "provider hash name")
            _identity(value, "provider hash", digest=True)
        if len({name for name, _ in hashes}) != len(hashes):
            raise ValueError("duplicate stage provider hash names")
        object.__setattr__(self, "provider_hashes", tuple(sorted(hashes)))
        if not isinstance(self.arithmetic, ArithmeticPolicy) or not isinstance(
            self.convergence, HFConvergence
        ):
            raise TypeError("StagePlan requires typed arithmetic and convergence")
        if self.estimated_cost_units is not None:
            object.__setattr__(
                self,
                "estimated_cost_units",
                _positive(self.estimated_cost_units, "estimated stage cost"),
            )
        evidence = tuple(self.error_evidence)
        if any(not isinstance(item, ErrorEvidence) for item in evidence):
            raise TypeError("stage error estimates must use #173 ErrorEvidence")
        object.__setattr__(self, "error_evidence", evidence)
        next_stages = tuple(self.allowed_next_stages)
        for name in next_stages:
            _identity(name, "allowed next stage")
        if len(set(next_stages)) != len(next_stages):
            raise ValueError("duplicate allowed next stages")
        object.__setattr__(self, "allowed_next_stages", next_stages)


@dataclass(frozen=True)
class ProgressiveBudget:
    """Hard planning and execution bounds; failed source work remains charged."""

    maximum_stages: int = 2
    maximum_source_iterations: int = 100
    maximum_total_iterations: int = 200
    maximum_host_bytes: int = 256 << 20
    maximum_verification_host_bytes: int = 256 << 20
    maximum_total_fock_builds: int | None = None
    maximum_estimated_cost_units: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "maximum_stages",
            "maximum_source_iterations",
            "maximum_total_iterations",
            "maximum_host_bytes",
            "maximum_verification_host_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_total_fock_builds is not None and (
            type(self.maximum_total_fock_builds) is not int
            or self.maximum_total_fock_builds < 1
        ):
            raise ValueError("maximum_total_fock_builds must be a positive integer")
        if self.maximum_estimated_cost_units is not None:
            object.__setattr__(
                self,
                "maximum_estimated_cost_units",
                _positive(
                    self.maximum_estimated_cost_units,
                    "maximum estimated cost",
                ),
            )


@dataclass(frozen=True)
class DeterministicHFPlan:
    """Exactly one optional initialization stage followed by the exact target."""

    problem: TargetProblem
    stages: tuple[StagePlan, ...]
    budget: ProgressiveBudget = ProgressiveBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.problem, TargetProblem) or not isinstance(
            self.budget, ProgressiveBudget
        ):
            raise TypeError("plan requires typed target and budget")
        stages = tuple(self.stages)
        object.__setattr__(self, "stages", stages)
        if len(stages) != 2 or len(stages) > self.budget.maximum_stages:
            raise ValueError(
                "the first HF plan requires exactly source and target stages"
            )
        source, target = stages
        if (
            source.role is not StageRole.INITIALIZATION
            or target.role is not StageRole.TARGET
        ):
            raise ValueError("HF stages must be ordered initialization then target")
        if source.transfer is not TransferOperation.METRIC_PROJECTED_DENSITY:
            raise ValueError("the source stage must use #189 metric density projection")
        if target.transfer is not TransferOperation.NONE:
            raise ValueError("the final target stage cannot transfer onward")
        if (
            source.allowed_next_stages != (target.stage_id,)
            or target.allowed_next_stages
        ):
            raise ValueError(
                "the deterministic plan must have one source-to-target edge"
            )
        if source.stage_id == target.stage_id:
            raise ValueError("stage ids must be unique")
        for name in (
            "method",
            "geometry_hash",
            "electron_count",
            "multiplicity",
            "charge",
            "hamiltonian",
        ):
            if getattr(source.model, name) != getattr(self.problem.model, name):
                raise ValueError(f"source stage changed target-compatible {name}")
        if (
            target.model != self.problem.model
            or target.provider_identity != self.problem.provider_identity
            or target.provider_hashes != self.problem.provider_hashes
        ):
            raise ValueError(
                "final stage must preserve the exact TargetProblem identity"
            )
        if target.convergence != self.problem.convergence:
            raise ValueError("final stage changed target convergence requirements")
        if not target.arithmetic.is_strict_target:
            raise ValueError("final stage must establish a strict FP64 target state")
        if source.convergence.energy_change < target.convergence.energy_change or (
            source.convergence.density_rms < target.convergence.density_rms
        ):
            raise ValueError(
                "the initialization solve cannot be stricter than the target"
            )
        if (
            source.convergence.maximum_iterations
            > self.budget.maximum_source_iterations
        ):
            raise ValueError("source iteration limit exceeds the progressive budget")
        if sum(stage.convergence.maximum_iterations for stage in stages) > (
            self.budget.maximum_total_iterations
        ):
            raise ValueError("planned iterations exceed the progressive budget")
        for stage in stages:
            for evidence in stage.error_evidence:
                if evidence.model_id != self.problem.model.identity:
                    raise ValueError(
                        "stage error evidence changed the requested target"
                    )
                if evidence.evaluated_model_id != stage.model.identity:
                    raise ValueError(
                        "stage error evidence names a different evaluated model"
                    )
        if self.budget.maximum_estimated_cost_units is not None:
            costs = tuple(stage.estimated_cost_units for stage in stages)
            if any(value is None for value in costs):
                raise ValueError(
                    "a cost-limited plan requires every stage cost estimate"
                )
            if sum(typing.cast("float", value) for value in costs) > (
                self.budget.maximum_estimated_cost_units
            ):
                raise ValueError("estimated stage cost exceeds the progressive budget")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "problem": self.problem.to_dict(),
                "stages": [
                    {
                        **asdict(stage),
                        "role": stage.role.value,
                        "transfer": stage.transfer.value,
                        "error_evidence": [
                            item.to_dict() for item in stage.error_evidence
                        ],
                    }
                    for stage in self.stages
                ],
                "budget": asdict(self.budget),
            }
        )


def _convergence(
    calculator: typing.Any, physical: float | None = None
) -> HFConvergence:
    return HFConvergence(
        calculator._energy_tolerance,
        calculator._density_tolerance,
        calculator._max_iterations,
        physical,
    )


def _arithmetic(calculator: typing.Any) -> ArithmeticPolicy:
    requested = {
        _native.PRECISION_FP64: "fp64",
        _native.PRECISION_AUTO: "auto",
    }.get(calculator._precision_mode)
    if requested is None:
        raise ValueError("unsupported progressive HF precision policy")
    return ArithmeticPolicy(
        requested,
        require_strict_refinement=requested == "auto",
    )


def make_deterministic_hf_plan(
    problem: TargetProblem,
    source_calculator: typing.Any,
    target_calculator: typing.Any,
    atoms: typing.Iterable[typing.Any],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    budget: ProgressiveBudget | None = None,
    source_estimated_cost_units: float | None = None,
    target_estimated_cost_units: float | None = None,
    source_error_evidence: tuple[ErrorEvidence, ...] = (),
    target_error_evidence: tuple[ErrorEvidence, ...] = (),
) -> DeterministicHFPlan:
    """Resolve the fixed two-stage baseline against the current calculators."""
    if not isinstance(problem, TargetProblem):
        raise TypeError("problem must be a TargetProblem")
    atoms = tuple(Atom.from_value(atom) for atom in atoms)
    if target_calculator._target_accuracy != problem.accuracy:
        raise ValueError(
            "target calculator does not carry the TargetProblem accuracy request"
        )
    current_target = TargetProblem.from_calculator(
        target_calculator,
        atoms,
        charge=charge,
        multiplicity=multiplicity,
        accuracy=problem.accuracy,
        physical_residual_tolerance=problem.convergence.physical_residual_rms,
    )
    if current_target != problem:
        raise ValueError(
            "target calculator no longer matches the TargetProblem snapshot"
        )
    source_metadata = source_calculator.basis_metadata(
        atoms, charge=charge, multiplicity=multiplicity
    )
    source = StagePlan(
        "initial-hf",
        StageRole.INITIALIZATION,
        source_calculator.resolved_model(
            atoms, charge=charge, multiplicity=multiplicity
        ),
        source_metadata["model_identity"],
        _provider_hashes(source_metadata),
        _arithmetic(source_calculator),
        _convergence(source_calculator),
        TransferOperation.METRIC_PROJECTED_DENSITY,
        source_estimated_cost_units,
        source_error_evidence,
        ("target-hf",),
    )
    target = StagePlan(
        "target-hf",
        StageRole.TARGET,
        problem.model,
        problem.provider_identity,
        problem.provider_hashes,
        _arithmetic(target_calculator),
        problem.convergence,
        TransferOperation.NONE,
        target_estimated_cost_units,
        target_error_evidence,
    )
    return DeterministicHFPlan(
        problem,
        (source, target),
        ProgressiveBudget() if budget is None else budget,
    )


@dataclass(frozen=True)
class StageExecution:
    stage_id: str
    role: StageRole
    status: str
    model_identity: str
    provider_identity: str
    iterations: int
    fock_builds: int | None
    seconds: float
    physical_residual_rms: float | None
    transfer_status: str
    restart_origin: str
    status_message: str


@dataclass(frozen=True)
class HFPhysicalResidualAudit:
    """Independent fixed-density target Fock/overlap residual and provenance."""

    model_identity: str
    provider_identity: str
    operator_identity: str
    execution_identity: str
    density_sha256: str
    overlap_sha256: str
    maximum_commutator: float
    rms_commutator: float
    fixed_density_energy: float
    energy_difference: float
    seconds: float

    def __post_init__(self) -> None:
        for name in (
            "model_identity",
            "provider_identity",
            "operator_identity",
            "execution_identity",
            "density_sha256",
            "overlap_sha256",
        ):
            _identity(getattr(self, name), name, digest=True)
        for name in (
            "maximum_commutator",
            "rms_commutator",
            "energy_difference",
            "seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if not math.isfinite(self.fixed_density_energy):
            raise ValueError("fixed-density audit energy must be finite")


@dataclass(frozen=True)
class FinalVerification:
    """Exact-target, state, arithmetic, budget and accuracy-audit disposition."""

    status: str
    target_established: bool
    problem_identity: str
    requested_model_identity: str
    actual_model_identity: str
    requested_provider_identity: str
    actual_provider_identity: str
    requested_provider_hashes: tuple[tuple[str, str], ...]
    actual_provider_hashes: tuple[tuple[str, str], ...]
    physical_residual_rms: float | None
    physical_residual_max: float | None
    physical_residual_tolerance: float
    physical_residual_source: str
    physical_audit_identity: str | None
    fixed_density_energy_difference: float | None
    accuracy_status: str
    capabilities: tuple[str, ...]
    total_iterations: int
    total_fock_builds: int | None
    strict_cleanup: str
    reasons: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.status == "verified"


def finalize_hf_verification(
    problem: TargetProblem,
    target_stage: StagePlan,
    result: typing.Any,
    actual_model: ResolvedModel,
    actual_provider_hashes: tuple[tuple[str, str], ...],
    budget: ProgressiveBudget,
    executions: tuple[StageExecution, ...],
    *,
    accuracy: AccuracyAssessment | None = None,
    physical_audit: HFPhysicalResidualAudit | None = None,
    physical_audit_error: str | None = None,
    physical_audit_budget_exhausted: bool = False,
) -> FinalVerification:
    """Derive a fail-closed final record; serialized status is never trusted."""
    if (
        not isinstance(problem, TargetProblem)
        or target_stage.role is not StageRole.TARGET
    ):
        raise TypeError("final verification requires the typed target stage")
    actual_provider_identity = result.basis_metadata.get("model_identity", "")
    target_reasons: list[str] = []
    budget_reasons: list[str] = []
    if physical_audit_budget_exhausted:
        budget_reasons.append("physical verification host budget was exhausted")
    if actual_model.identity != problem.model.identity:
        target_reasons.append("executed scientific model differs from TargetProblem")
    if actual_provider_identity != problem.provider_identity:
        target_reasons.append("executed provider identity differs from TargetProblem")
    actual_hashes = tuple(sorted(tuple(pair) for pair in actual_provider_hashes))
    if actual_hashes != problem.provider_hashes:
        target_reasons.append("executed provider hashes differ from TargetProblem")
    if not bool(getattr(result, "succeeded", False)) or not bool(result.converged):
        target_reasons.append("target HF did not converge successfully")
    if not math.isfinite(float(result.energy)):
        target_reasons.append("target energy is nonfinite")
    if not math.isfinite(float(result.energy_change)) or abs(result.energy_change) > (
        problem.convergence.energy_change
    ):
        target_reasons.append("target energy-change tolerance is unmet")
    if not math.isfinite(float(result.density_rms)) or result.density_rms > (
        problem.convergence.density_rms
    ):
        target_reasons.append("target density tolerance is unmet")
    residual = result.physical_residual_rms
    residual_max = None
    residual_source = "native_result" if residual is not None else "unavailable"
    audit_identity = None
    energy_difference = None
    if physical_audit is not None:
        if not isinstance(physical_audit, HFPhysicalResidualAudit):
            raise TypeError("physical audit must be an HFPhysicalResidualAudit")
        if (
            physical_audit.model_identity != problem.model.identity
            or physical_audit.provider_identity != problem.provider_identity
        ):
            target_reasons.append(
                "physical residual audit changed the requested target"
            )
        residual = physical_audit.rms_commutator
        residual_max = physical_audit.maximum_commutator
        residual_source = "fixed_density_target_audit"
        audit_identity = physical_audit.operator_identity
        energy_difference = physical_audit.energy_difference
        if energy_difference > max(1.0e-10, problem.convergence.energy_change):
            target_reasons.append(
                "fixed-density audit energy differs from target result"
            )
    elif physical_audit_error is not None:
        target_reasons.append(
            f"target physical residual audit failed: {physical_audit_error}"
        )
    if residual is None or not math.isfinite(float(residual)):
        target_reasons.append("target physical residual is unavailable")
    elif (residual_max if residual_max is not None else residual) > typing.cast(
        "float", problem.convergence.physical_residual_rms
    ):
        target_reasons.append("target physical residual tolerance is unmet")
    requested_observables = {item.observable for item in problem.accuracy.observables}
    capabilities = {"energy"}
    if result.forces is not None:
        forces = np.asarray(result.forces)
        real_numeric = np.issubdtype(forces.dtype, np.number) and not np.issubdtype(
            forces.dtype, np.complexfloating
        )
        if (
            forces.shape == (problem.atom_count, 3)
            and real_numeric
            and np.isfinite(forces).all()
        ):
            capabilities.add("forces")
    missing = requested_observables - capabilities
    if missing:
        target_reasons.append(
            "target result omitted required observables: " + ", ".join(sorted(missing))
        )
    precision = result.precision
    strict_cleanup = "unavailable"
    if not isinstance(precision, dict) or precision.get("requested_mode") != (
        target_stage.arithmetic.requested_precision
    ):
        target_reasons.append("executed arithmetic policy differs from StagePlan")
    elif precision.get("effective_bits") != 64:
        target_reasons.append("strict final arithmetic was not established")
    elif target_stage.arithmetic.requested_precision == "auto":
        if not precision.get("strict_refinement_applied"):
            target_reasons.append("auto arithmetic omitted strict target cleanup")
            strict_cleanup = "missing"
        else:
            strict_cleanup = "applied"
    else:
        strict_cleanup = "not_required_fp64"
    total_iterations = sum(item.iterations for item in executions)
    if total_iterations > budget.maximum_total_iterations:
        budget_reasons.append("actual iterations exceeded the progressive budget")
    fock_values = tuple(item.fock_builds for item in executions)
    total_fock_builds = (
        sum(typing.cast("int", value) for value in fock_values)
        if all(value is not None for value in fock_values)
        else None
    )
    if budget.maximum_total_fock_builds is not None:
        if total_fock_builds is None:
            budget_reasons.append("Fock-build budget could not be verified")
        elif total_fock_builds > budget.maximum_total_fock_builds:
            budget_reasons.append("actual Fock builds exceeded the progressive budget")
    selected_accuracy = result.accuracy if accuracy is None else accuracy
    accuracy_status = "unverified"
    if selected_accuracy is not None:
        if not isinstance(selected_accuracy, AccuracyAssessment):
            raise TypeError("final accuracy must be an AccuracyAssessment")
        if (
            selected_accuracy.model != problem.model
            or selected_accuracy.target != problem.accuracy
        ):
            target_reasons.append("accuracy assessment changed the requested target")
        else:
            accuracy_status = selected_accuracy.status
    if budget_reasons:
        status = "budget_exhausted"
    elif target_reasons or accuracy_status in (
        "unconverged",
        "observed_unmet",
        "estimated_above_target",
    ):
        status = "unmet"
    elif accuracy_status == "observed_met":
        status = "verified"
    else:
        status = "unverified"
    reasons = target_reasons + budget_reasons
    if status == "unverified":
        reasons.append(f"observable accuracy status is {accuracy_status}")
    return FinalVerification(
        status=status,
        target_established=not target_reasons,
        problem_identity=problem.identity,
        requested_model_identity=problem.model.identity,
        actual_model_identity=actual_model.identity,
        requested_provider_identity=problem.provider_identity,
        actual_provider_identity=actual_provider_identity,
        requested_provider_hashes=problem.provider_hashes,
        actual_provider_hashes=actual_hashes,
        physical_residual_rms=residual,
        physical_residual_max=residual_max,
        physical_residual_tolerance=typing.cast(
            "float", problem.convergence.physical_residual_rms
        ),
        physical_residual_source=residual_source,
        physical_audit_identity=audit_identity,
        fixed_density_energy_difference=energy_difference,
        accuracy_status=accuracy_status,
        capabilities=tuple(sorted(capabilities)),
        total_iterations=total_iterations,
        total_fock_builds=total_fock_builds,
        strict_cleanup=strict_cleanup,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class ProgressiveHFResult:
    target: BatchItemResult
    source: BatchItemResult
    plan: DeterministicHFPlan
    executions: tuple[StageExecution, ...]
    verification: FinalVerification
    diagnostics: dict
    target_density: np.ndarray | None

    @property
    def succeeded(self) -> bool:
        return self.verification.succeeded


def _audit_target_physical_residual(
    problem: TargetProblem,
    calculator: typing.Any,
    atoms: tuple[Atom, ...],
    density: np.ndarray,
    target_energy: float,
    *,
    charge: int,
    multiplicity: int,
    maximum_host_bytes: int,
) -> HFPhysicalResidualAudit:
    """Rebuild the exact target Fock once and audit FDS-SDF at its final density."""
    from vibeqc_compiler.common.resources import ResourceBudget
    from vibeqc_compiler.dft import NativeAO

    started = time.perf_counter()
    unrestricted = problem.model.method == "uhf"
    if type(maximum_host_bytes) is not int or maximum_host_bytes < 1:
        raise ValueError("maximum verification host bytes must be positive")
    # Reserve the controller's density/Fock snapshots, overlap, commutator
    # products and immutable/hash copies in addition to the provider envelope.
    # The full HF inventory is deliberately conservative for a fixed-D audit;
    # do not create NativeAO/Fock owners first and check only the final overlap.
    n = density.shape[-1]
    spin_channels = 2 if unrestricted else 1
    workspace_bytes = 8 * n * n * (12 * spin_channels + 8)
    if workspace_bytes > maximum_host_bytes:
        raise MemoryError("physical verification workspace exceeds host budget")
    calculator.estimate_resources(
        [atoms],
        charges=[charge],
        multiplicities=[multiplicity],
        budget=ResourceBudget(
            host_bytes=maximum_host_bytes, host_reserve_bytes=workspace_bytes
        ),
    ).require_feasible()
    fitted = calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
    approximation = "density_fitted" if fitted else "exact"
    with ExitStack() as stack:
        basis = stack.enter_context(
            NativeAO(
                atoms,
                calculator._basis,
                representation=calculator._representation_name,
                charge=charge,
                multiplicity=multiplicity,
            )
        )
        auxiliary = None
        if fitted:
            auxiliary_basis = (
                calculator._basis
                if calculator._auxiliary_basis is None
                else calculator._auxiliary_basis
            )
            auxiliary = stack.enter_context(
                NativeAO(
                    atoms,
                    auxiliary_basis,
                    representation=calculator._representation_name,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )
        fock_plan = stack.enter_context(
            FockPlan(
                basis,
                FockBuildSpec.hf(
                    "unrestricted" if unrestricted else "restricted",
                    coulomb=approximation,
                    exchange=approximation,
                    derivative_order=0,
                ),
                auxiliary=auxiliary,
                device=calculator._device_name,
                device_id=calculator._device_id,
                screening_tolerance=calculator._screening_tolerance,
                metric_relative_threshold=(
                    calculator._density_fitting_relative_threshold
                ),
                device_budget_bytes=calculator._density_fitting_memory_budget_bytes,
            )
        )
        evaluated_density = density if unrestricted else density[0]
        evaluation = fock_plan.evaluate(evaluated_density)
        operator_identity = fock_plan.identity
        execution_identity = fock_plan.execution_identity
    overlap = cross_overlap(
        calculator,
        calculator,
        atoms,
        charge=charge,
        multiplicity=multiplicity,
        maximum_bytes=maximum_host_bytes,
    )
    densities = density if unrestricted else density[:1]
    focks = evaluation.fock if unrestricted else evaluation.fock[None, :, :]
    residuals = np.asarray(
        [
            fock @ state @ overlap - overlap @ state @ fock
            for fock, state in zip(focks, densities, strict=True)
        ]
    )
    maximum = float(np.max(np.abs(residuals)))
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    return HFPhysicalResidualAudit(
        model_identity=problem.model.identity,
        provider_identity=problem.provider_identity,
        operator_identity=operator_identity,
        execution_identity=execution_identity,
        density_sha256=sha256(density.tobytes()).hexdigest(),
        overlap_sha256=sha256(overlap.tobytes()).hexdigest(),
        maximum_commutator=maximum,
        rms_commutator=rms,
        fixed_density_energy=evaluation.energy,
        energy_difference=abs(evaluation.energy - target_energy),
        seconds=time.perf_counter() - started,
    )


def run_progressive_hf(
    plan: DeterministicHFPlan,
    source_calculator: typing.Any,
    target_calculator: typing.Any,
    atoms: typing.Iterable[typing.Any],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    projection_policy: ProjectionPolicy | None = None,
    accuracy: AccuracyAssessment | None = None,
) -> ProgressiveHFResult:
    """Run the source proposal and always finish with the exact target calculator."""
    if not isinstance(plan, DeterministicHFPlan):
        raise TypeError("plan must be a DeterministicHFPlan")
    atoms = tuple(Atom.from_value(atom) for atom in atoms)
    current = make_deterministic_hf_plan(
        plan.problem,
        source_calculator,
        target_calculator,
        atoms,
        charge=charge,
        multiplicity=multiplicity,
        budget=plan.budget,
        source_estimated_cost_units=plan.stages[0].estimated_cost_units,
        target_estimated_cost_units=plan.stages[1].estimated_cost_units,
        source_error_evidence=plan.stages[0].error_evidence,
        target_error_evidence=plan.stages[1].error_evidence,
    )
    if current != plan:
        raise ValueError(
            "calculator state changed after the progressive plan was built"
        )
    started = time.perf_counter()
    executions: list[StageExecution] = []
    projection: dict = {
        "schema": "vibeqc.progressive_hf.transfer",
        "status": "not_attempted",
        "reason": None,
    }
    target_density = None
    physical_audit = None
    physical_audit_error = None
    physical_audit_budget_exhausted = False
    source_setup_started = time.perf_counter()
    with source_calculator.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity]
    ) as source_batch:
        source_setup_seconds = time.perf_counter() - source_setup_started
        source_started = time.perf_counter()
        source_result = source_batch.execute(
            strict=False, properties=("energy",)
        ).items[0]
        source_execution_seconds = time.perf_counter() - source_started
        target_setup_started = time.perf_counter()
        with target_calculator.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as target_batch:
            target_setup_seconds = time.perf_counter() - target_setup_started
            projection_seconds = 0.0
            if source_result.succeeded:
                projection_started = time.perf_counter()
                try:
                    projection = initialize_from(
                        target_batch,
                        source_batch,
                        policy=projection_policy,
                        strict=False,
                        maximum_host_bytes=plan.budget.maximum_host_bytes,
                    )
                    item = projection["items"][0]
                    projection["status"] = (
                        "accepted" if item["accepted"] else "rejected"
                    )
                    projection["reason"] = item["reason"]
                except (MemoryError, ProjectionRejected) as error:
                    projection = {
                        "schema": "vibeqc.progressive_hf.transfer",
                        "status": "rejected",
                        "reason": str(error),
                    }
                projection_seconds = time.perf_counter() - projection_started
            else:
                projection["status"] = "skipped_failed_source"
                projection["reason"] = source_result.status_message
            executions.append(
                StageExecution(
                    plan.stages[0].stage_id,
                    StageRole.INITIALIZATION,
                    "succeeded" if source_result.succeeded else "failed",
                    source_calculator.resolved_model(
                        atoms, charge=charge, multiplicity=multiplicity
                    ).identity,
                    source_result.basis_metadata["model_identity"],
                    source_result.iterations,
                    source_result.fock_builds,
                    source_setup_seconds + source_execution_seconds,
                    source_result.physical_residual_rms,
                    projection["status"],
                    source_result.restart_origin,
                    source_result.status_message,
                )
            )
            requested_properties = tuple(
                sorted({item.observable for item in plan.problem.accuracy.observables})
            )
            target_started = time.perf_counter()
            target_result = target_batch.execute(
                strict=False, properties=requested_properties
            ).items[0]
            target_seconds = time.perf_counter() - target_started
            if target_result.succeeded and target_result.converged:
                target_density = _retained_density(target_batch)
            target_cleanup_started = time.perf_counter()
        target_cleanup_seconds = time.perf_counter() - target_cleanup_started
        source_cleanup_started = time.perf_counter()
    source_cleanup_seconds = time.perf_counter() - source_cleanup_started
    cleanup_seconds = target_cleanup_seconds + source_cleanup_seconds
    verification_started = time.perf_counter()
    if target_density is not None:
        try:
            physical_audit = _audit_target_physical_residual(
                plan.problem,
                target_calculator,
                atoms,
                target_density,
                target_result.energy,
                charge=charge,
                multiplicity=multiplicity,
                maximum_host_bytes=plan.budget.maximum_verification_host_bytes,
            )
        except MemoryError as error:
            physical_audit_error = (
                f"physical verification host budget exhausted: {error}"
            )
            physical_audit_budget_exhausted = True
        except (
            ArithmeticError,
            NotImplementedError,
            RuntimeError,
            ValueError,
        ) as error:
            physical_audit_error = str(error)
    verification_seconds = time.perf_counter() - verification_started
    target_fock_builds = target_result.fock_builds
    if physical_audit is not None and target_fock_builds is not None:
        target_fock_builds += 1
    elif physical_audit_error is not None:
        target_fock_builds = None
    target_residual = (
        physical_audit.rms_commutator
        if physical_audit is not None
        else target_result.physical_residual_rms
    )
    executions.append(
        StageExecution(
            plan.stages[1].stage_id,
            StageRole.TARGET,
            "succeeded" if target_result.succeeded else "failed",
            target_calculator.resolved_model(
                atoms, charge=charge, multiplicity=multiplicity
            ).identity,
            target_result.basis_metadata["model_identity"],
            target_result.iterations,
            target_fock_builds,
            target_setup_seconds + target_seconds + verification_seconds,
            target_residual,
            "none",
            target_result.restart_origin,
            target_result.status_message,
        )
    )
    actual_model = target_calculator.resolved_model(
        atoms, charge=charge, multiplicity=multiplicity
    )
    actual_hashes = _provider_hashes(target_result.basis_metadata)
    execution_tuple = tuple(executions)
    verification = finalize_hf_verification(
        plan.problem,
        plan.stages[1],
        target_result,
        actual_model,
        actual_hashes,
        plan.budget,
        execution_tuple,
        accuracy=accuracy,
        physical_audit=physical_audit,
        physical_audit_error=physical_audit_error,
        physical_audit_budget_exhausted=physical_audit_budget_exhausted,
    )
    diagnostics = {
        "schema": "vibeqc.progressive_hf",
        "version": 1,
        "plan_identity": plan.identity,
        "source_seconds": executions[0].seconds,
        "source_setup_seconds": source_setup_seconds,
        "source_execution_seconds": source_execution_seconds,
        "projection_seconds": projection_seconds,
        "target_setup_seconds": target_setup_seconds,
        "target_execution_seconds": target_seconds,
        "final_verification_seconds": verification_seconds,
        "cleanup_seconds": cleanup_seconds,
        "target_seconds": executions[1].seconds,
        "total_seconds": time.perf_counter() - started,
        "projection": deepcopy(projection),
        "physical_residual_audit": (
            asdict(physical_audit)
            if physical_audit is not None
            else {"status": "unavailable", "reason": physical_audit_error}
        ),
        "verification_status": verification.status,
    }
    return ProgressiveHFResult(
        target_result,
        source_result,
        plan,
        execution_tuple,
        verification,
        diagnostics,
        target_density,
    )
