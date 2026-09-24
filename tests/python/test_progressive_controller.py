"""#192's controller must never promote an initialization into a target result."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
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
    HFPhysicalResidualAudit,
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
FORCE_ACCURACY = TargetAccuracy(
    (ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1.0e-7),)
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
    2,
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


def force_assessment() -> AccuracyAssessment:
    evidence = (
        ErrorEvidence(
            EvidenceKind.OBSERVED,
            TARGET_MODEL.identity,
            TARGET_MODEL.identity,
            TARGET_MODEL.identity,
            "forces",
            "max_abs",
            "Eh/bohr",
            1.0e-12,
            1.0,
            "total_numerical",
            "relaxed_target",
            (("reference", "strict-independent-audit"),),
            actual_reference_error=1.0e-12,
        ),
    )
    return AccuracyAssessment(TARGET_MODEL, FORCE_ACCURACY, evidence)


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
    problem: TargetProblem = PROBLEM,
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
        problem,
        selected.stages[1],
        output,
        actual_model,
        hashes,
        selected.budget,
        history,
    )


def test_target_problem_and_stage_plan_are_immutable_and_identity_stable() -> None:
    before = PROBLEM.identity
    for owner, attribute, value in (
        (PROBLEM, "provider_identity", "x" * 64),
        (stages()[0], "allowed_next_stages", ()),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(owner, attribute, value)
    assert PROBLEM.identity == before


def test_target_problem_identity_binds_atom_count() -> None:
    assert PROBLEM.atom_count == 2
    assert replace(PROBLEM, atom_count=3).identity != PROBLEM.identity
    with pytest.raises(ValueError, match="atom_count"):
        replace(PROBLEM, atom_count=0)
    with pytest.raises(ValueError, match="atom_count"):
        replace(PROBLEM, atom_count=True)


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


@pytest.mark.parametrize(
    "forces",
    [
        np.empty((0, 3)),
        np.zeros((1, 3)),
        np.zeros((3, 3)),
        np.zeros((2, 3), dtype=np.complex128),
    ],
    ids=("empty", "missing-atom", "extra-atom", "complex"),
)
def test_force_capability_rejects_incomplete_or_nonreal_target_array(
    forces: np.ndarray,
) -> None:
    force_problem = replace(PROBLEM, accuracy=FORCE_ACCURACY)
    output = result(accuracy=force_assessment())
    output.forces = forces

    final = verify(output, problem=force_problem)

    assert final.status == "unmet"
    assert not final.target_established
    assert final.accuracy_status == "observed_met"
    assert "forces" not in final.capabilities
    assert any("omitted required observables: forces" in item for item in final.reasons)


def test_force_capability_accepts_complete_finite_real_target_array() -> None:
    force_problem = replace(PROBLEM, accuracy=FORCE_ACCURACY)
    output = result(accuracy=force_assessment())
    output.forces = np.zeros((2, 3), dtype=np.float64)

    final = verify(output, problem=force_problem)

    assert final.status == "verified"
    assert final.target_established
    assert "forces" in final.capabilities


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


def test_fixed_density_target_audit_supplies_missing_public_hf_residual() -> None:
    output = result(accuracy=assessment(observed=True))
    output.physical_residual_rms = None
    selected = plan()
    audit = HFPhysicalResidualAudit(
        model_identity=TARGET_MODEL.identity,
        provider_identity=PROBLEM.provider_identity,
        operator_identity="f" * 64,
        execution_identity="e" * 64,
        density_sha256="d" * 64,
        overlap_sha256="c" * 64,
        maximum_commutator=2.0e-12,
        rms_commutator=1.0e-12,
        fixed_density_energy=-1.0,
        energy_difference=0.0,
        seconds=0.01,
    )
    final = finalize_hf_verification(
        PROBLEM,
        selected.stages[1],
        output,
        TARGET_MODEL,
        TARGET_HASHES,
        selected.budget,
        (execution(StageRole.INITIALIZATION), execution(StageRole.TARGET)),
        physical_audit=audit,
    )
    assert final.status == "verified"
    assert final.target_established
    assert final.physical_residual_source == "fixed_density_target_audit"
    assert final.physical_residual_max == 2.0e-12


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


@pytest.mark.parametrize("maximum_bytes", [1, 32, 128])
def test_physical_audit_rejects_workspace_before_native_owners(
    monkeypatch: pytest.MonkeyPatch, maximum_bytes: int
) -> None:
    import vibeqc.progressive_controller as controller
    from vibeqc_compiler import dft

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("native Fock/basis work started before verification admission")

    monkeypatch.setattr(dft, "NativeAO", forbidden)
    monkeypatch.setattr(controller, "FockPlan", forbidden)
    calculator = SimpleNamespace(
        _density_fitting_mode=0, _basis="sto-3g", _representation_name="cartesian"
    )
    with pytest.raises(MemoryError, match="verification"):
        controller._audit_target_physical_residual(
            PROBLEM,
            calculator,
            (),
            np.eye(2)[None, :, :],
            -1.0,
            charge=0,
            multiplicity=1,
            maximum_host_bytes=maximum_bytes,
        )


def test_physical_audit_admits_provider_and_workspace_before_native_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.progressive_controller as controller
    from vibeqc_compiler import dft

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("native Fock/basis work started before provider admission")

    def estimate(systems: object, **kwargs: object) -> SimpleNamespace:
        from vibeqc_compiler.common.resources import ResourceBudget

        budget = kwargs["budget"]
        assert isinstance(budget, ResourceBudget)
        assert budget.host_bytes == 4096
        assert budget.host_reserve_bytes > 8 * 2 * 2

        def reject() -> None:
            raise MemoryError("verification provider allocation exceeds budget")

        return SimpleNamespace(require_feasible=reject)

    monkeypatch.setattr(dft, "NativeAO", forbidden)
    monkeypatch.setattr(controller, "FockPlan", forbidden)
    calculator = SimpleNamespace(
        _density_fitting_mode=0,
        _basis="sto-3g",
        _representation_name="cartesian",
        estimate_resources=estimate,
    )
    with pytest.raises(MemoryError, match="verification provider"):
        controller._audit_target_physical_residual(
            PROBLEM,
            calculator,
            (),
            np.eye(2)[None, :, :],
            -1.0,
            charge=0,
            multiplicity=1,
            maximum_host_bytes=4096,
        )
