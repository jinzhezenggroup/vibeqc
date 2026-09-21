"""Force-aware policy must fail closed and keep empirical/actual errors distinct."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import (
    AccuracyAssessment,
    AdaptiveNumericsPolicy,
    ContributionLedger,
    NumericalContribution,
    NumericalLevel,
    NumericalTargetModel,
    ObservableDelta,
    PairedCalibrationSample,
    PairedDifferenceEstimator,
    ResolvedModel,
    TargetAccuracy,
    TargetErrorBudget,
)
from vibeqc.accuracy import ObservableTarget


def delta(energy: float, force: float, *, natom: int = 2) -> ObservableDelta:
    return ObservableDelta(energy, tuple((force, force / 2, 0.0) for _ in range(natom)))


def sample(
    family: str,
    *,
    paired_energy: float = 1e-7,
    paired_force: float = 1e-5,
    actual_energy: float = 2e-7,
    actual_force: float = 2e-5,
    method: str = "pbe-rks",
    numerical_family_id: str = "coarse-to-standard-v1",
) -> PairedCalibrationSample:
    return PairedCalibrationSample(
        family,
        family,
        method,
        numerical_family_id,
        delta(paired_energy, paired_force),
        delta(actual_energy, actual_force),
    )


def estimator() -> PairedDifferenceEstimator:
    return PairedDifferenceEstimator.fit((sample("h2"), sample("water")))


def levels() -> tuple[NumericalLevel, ...]:
    return (
        NumericalLevel("coarse", 0, 1e-8, "grid-coarse"),
        NumericalLevel("standard", 1, 1e-10, "grid-standard"),
        NumericalLevel("strict", 2, 1e-12, "grid-tight", strict=True),
    )


def test_numerical_target_model_separates_grid_physics_from_execution_policy() -> None:
    target = NumericalTargetModel(
        "pbe-rks", "geom-a", "def2-svp-hash", "pbe-functional-hash", "tight-grid-hash"
    )
    assert target.identity != replace(target, grid_id="standard-grid-hash").identity
    assert (
        target.identity
        != replace(
            target,
            derivative_semantics="fixed-density-moving-grid-with-partition-response",
        ).identity
    )
    assert not hasattr(target, "screening_tolerance")
    fitted = replace(
        target,
        approximation="density_fitting",
        auxiliary_basis_id="def2-universal-jkfit",
        metric_relative_threshold=1e-10,
    )
    assert fitted.identity != target.identity
    with pytest.raises(ValueError, match="fitting metadata"):
        replace(target, auxiliary_basis_id="aux")


def test_force_error_norms_retain_atom_and_component_semantics() -> None:
    measured = ObservableDelta.between(
        -1.0,
        [[1.0, -2.0, 3.0], [0.0, 1.0, -1.0]],
        -1.25,
        [[0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
    )
    assert measured.energy_abs == pytest.approx(0.25)
    assert measured.force_components == ((1.0, 1.0, 2.0), (0.0, 1.0, 1.0))
    assert measured.force_max_abs == 2.0
    assert measured.force_rms == pytest.approx(np.sqrt(8 / 6))
    assert measured.per_atom_max_abs == (2.0, 1.0)
    assert measured.per_atom_l2 == pytest.approx((np.sqrt(6), np.sqrt(2)))
    with pytest.raises(ValueError, match="identical shape"):
        ObservableDelta.between(0, [[0, 0, 0]], 0, [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="real values"):
        ObservableDelta(0.0, ((1.0 + 1.0j, 0.0, 0.0),))

    tiny = ObservableDelta(0.0, ((1e-300, 0.0, 0.0),))
    assert tiny.force_rms > 0.0
    assert tiny.force_rms / (1e-300 / np.sqrt(3.0)) == pytest.approx(1.0)
    huge = ObservableDelta(0.0, ((1e300, 0.0, 0.0),))
    assert np.isfinite(huge.force_rms)
    assert huge.force_rms / (1e300 / np.sqrt(3.0)) == pytest.approx(1.0)
    vector_tiny = ObservableDelta(0.0, ((1e-200, 1e-200, 1e-200),))
    assert vector_tiny.per_atom_l2[0] > 0.0
    assert vector_tiny.per_atom_l2[0] / (np.sqrt(3.0) * 1e-200) == pytest.approx(1.0)
    vector_huge = ObservableDelta(0.0, ((1e200, 1e200, 1e200),))
    assert np.isfinite(vector_huge.per_atom_l2[0])
    assert vector_huge.per_atom_l2[0] / (np.sqrt(3.0) * 1e200) == pytest.approx(1.0)


def test_contribution_ledger_separates_estimator_from_actual_and_motion_semantics() -> (
    None
):
    first = NumericalContribution(
        "quadrature",
        "atom:0/grid:coarse",
        "fixed_density",
        "empirical",
        delta(2e-8, 3e-6),
        actual=delta(1e-8, 2e-6),
        grid_identity="grid-coarse",
        mask_identity="mask-a",
        motion_response_included=True,
        estimator_seconds=0.01,
        block_kind="grid_region",
    )
    second = NumericalContribution(
        "screening",
        "shell:1-2",
        "fixed_density",
        "asymptotic",
        delta(1e-8, 2e-6),
        grid_identity="grid-coarse",
        mask_identity="mask-a",
        motion_response_included=True,
        estimator_seconds=0.02,
        block_kind="shell_block",
    )
    ledger = ContributionLedger("pbe/grid-target", "geom-a", (first, second))
    envelope = ledger.estimated_absolute_envelope(scope="fixed_density")
    assert envelope.energy_abs == pytest.approx(3e-8)
    assert envelope.force_max_abs == pytest.approx(5e-6)
    assert first.actual.force_max_abs == pytest.approx(2e-6)
    assert ledger.estimator_seconds == pytest.approx(0.03)
    assert ledger.block_coverage() == {
        "shell_block": 1,
        "grid_region": 1,
        "auxiliary_rank": 0,
        "aggregate": 0,
    }
    auxiliary = NumericalContribution(
        "density_fitting",
        "aux-rank:17",
        "relaxed_target",
        "empirical",
        delta(1e-9, 1e-7),
        block_kind="auxiliary_rank",
    )
    assert (
        ContributionLedger("df-target", "geom-a", (auxiliary,)).block_coverage()[
            "auxiliary_rank"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="block kind"):
        replace(first, block_kind="mystery")


def test_small_energy_but_large_force_error_is_a_deliberate_tightening_case() -> None:
    policy = AdaptiveNumericsPolicy(
        levels(), TargetErrorBudget(energy_abs=1e-6, force_max_abs=1e-5)
    )
    estimate = estimator().predict(
        "pbe-rks",
        delta(1e-9, 1e-4),
        numerical_family_id="coarse-to-standard-v1",
    )
    decision = policy.decide(
        policy.initial_state(),
        estimate,
        geometry_id="asymmetric-water",
        mask_identity="mask-1",
    )
    assert estimate.delta.energy_abs < policy.budget.energy_abs
    assert estimate.delta.force_max_abs > policy.budget.force_max_abs
    assert decision.action == "tighten_retry"
    assert decision.state.level_index == 1


def test_paired_estimator_holdout_reports_false_success_and_conservatism() -> None:
    fit = estimator()
    budget = TargetErrorBudget(energy_abs=5e-7, force_max_abs=5e-5)
    optimistic = sample(
        "heldout-miss",
        paired_energy=1e-9,
        paired_force=1e-8,
        actual_energy=1e-4,
        actual_force=1e-2,
    )
    conservative = sample(
        "heldout-safe",
        paired_energy=1e-4,
        paired_force=1e-2,
        actual_energy=1e-8,
        actual_force=1e-7,
    )
    report = fit.evaluate_holdout((optimistic, conservative), budget)
    assert report["false_successes"] == 1
    assert report["false_success_rate"] == pytest.approx(0.5)
    assert report["overconservative"] == 1
    assert report["certified"] is False
    with pytest.raises(ValueError, match="leakage"):
        fit.evaluate_holdout((sample("h2"),), budget)
    with pytest.raises(ValueError, match="numerical-level family"):
        fit.predict(
            "pbe-rks",
            delta(1e-9, 1e-8),
            numerical_family_id="standard-to-strict-v1",
        )
    with pytest.raises(ValueError, match="mix numerical-level families"):
        PairedDifferenceEstimator.fit(
            (
                sample("h2"),
                sample("water", numerical_family_id="standard-to-strict-v1"),
            )
        )


def test_uncertainty_hysteresis_switching_and_strict_reproducibility_fail_closed() -> (
    None
):
    policy = AdaptiveNumericsPolicy(
        levels(),
        TargetErrorBudget(energy_abs=1e-5, force_max_abs=1e-3),
        relax_after=2,
    )
    state = policy.initial_state(start_index=1, mask_identity="mask-a")
    uncertain = policy.decide(
        state,
        None,
        geometry_id="g0",
        mask_identity="mask-b",
        uncertainty_reason="paired-grid branch mismatch",
    )
    assert uncertain.action == "strict_fallback"
    assert uncertain.state.level_index == 2
    assert uncertain.state.transitions[-1].from_mask == "mask-a"
    assert uncertain.state.transitions[-1].to_mask == "mask-b"

    tiny = estimator().predict(
        "pbe-rks",
        delta(1e-12, 1e-9),
        numerical_family_id="coarse-to-standard-v1",
    )
    state = policy.initial_state(start_index=1)
    first = policy.decide(state, tiny, geometry_id="g1")
    assert first.action == "hold"
    second = policy.decide(first.state, tiny, geometry_id="g1")
    assert second.action == "relax"
    changed_mask = policy.decide(
        policy.initial_state(start_index=1, mask_identity="mask-old"),
        tiny,
        geometry_id="g-mask",
        mask_identity="mask-new",
    )
    assert changed_mask.action == "hold"
    assert changed_mask.state.safe_streak == 1
    assert changed_mask.state.mask_identity == "mask-new"
    assert changed_mask.state.transitions[-1].from_mask == "mask-old"
    assert changed_mask.state.transitions[-1].to_mask == "mask-new"
    ordinary = estimator().predict(
        "pbe-rks",
        delta(1e-6, 1e-4),
        numerical_family_id="coarse-to-standard-v1",
    )
    within = policy.decide(
        policy.initial_state(start_index=0, mask_identity="mask-a"),
        ordinary,
        geometry_id="g-within",
        mask_identity="mask-b",
    )
    assert within.action == "hold"
    assert within.state.mask_identity == "mask-b"
    assert within.state.transitions[-1].from_mask == "mask-a"
    assert within.state.transitions[-1].to_mask == "mask-b"
    switched = policy.decide(
        policy.initial_state(start_index=1),
        tiny,
        geometry_id="g2",
        switching_observed=True,
        mask_identity="mask-c",
    )
    assert switched.action == "hold_switch"

    strict = replace(policy, strict_reproducible=True)
    state = strict.initial_state()
    assert state.level_index == 2
    assert strict.decide(state, tiny, geometry_id="g3").action == "strict"


def test_final_verification_requires_strict_reference_and_actual_error() -> None:
    policy = AdaptiveNumericsPolicy(
        levels(), TargetErrorBudget(energy_abs=1e-6, force_max_abs=1e-5)
    )
    assert (
        policy.verify_final(
            reference_level=levels()[-1], actual_error=delta(1e-7, 1e-6)
        )
        == "observed_met"
    )
    assert (
        policy.verify_final(
            reference_level=levels()[-1], actual_error=delta(1e-4, 1e-3)
        )
        == "observed_unmet"
    )
    assert (
        policy.verify_final(
            reference_level=levels()[-1],
            actual_error=delta(1e-7, 1e-6),
            uncertainty_reason="state changed",
        )
        == "unverified"
    )
    with pytest.raises(ValueError, match="strict target"):
        policy.verify_final(reference_level=levels()[1], actual_error=delta(0, 0))


def test_empirical_estimate_projects_to_accuracy_contract_without_certification() -> (
    None
):
    fit = PairedDifferenceEstimator.fit(
        (sample("h2", method="rhf"), sample("water", method="rhf"))
    )
    estimate = fit.predict(
        "rhf",
        delta(1e-8, 1e-6),
        numerical_family_id="coarse-to-standard-v1",
    )
    model = ResolvedModel("rhf", "geometry", "basis", 2)
    evidence = fit.as_error_evidence(
        model, estimate, energy_reference_norm=1.0, force_reference_norm=1.0
    )
    target = TargetAccuracy(
        (
            ObservableTarget("energy", "absolute", "Eh", 1e-4),
            ObservableTarget("forces", "max_abs", "Eh/bohr", 1e-3),
        )
    )
    assessment = AccuracyAssessment(model, target, evidence)
    assert assessment.status == "estimated_below_target"
    assert all(item.kind.value == "empirical_predictor" for item in evidence)
    assert not hasattr(assessment, "certified")
