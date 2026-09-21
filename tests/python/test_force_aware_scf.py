"""NUM03 geometry-step SCF effort policy remains empirical and fail-closed."""

from dataclasses import replace

import pytest
from vibeqc import ObservableDelta, TargetErrorBudget
from vibeqc.force_aware_scf import (
    ForceAwareScfPolicy,
    ScfEffortLevel,
    ScfForceCalibrationSample,
    ScfForceErrorEstimator,
)


def error(energy: float, force: float) -> ObservableDelta:
    return ObservableDelta(energy, ((force, force / 2, 0.0),))


def sample(
    family: str,
    *,
    basis: str = "sto-3g",
    diagnostic: float = 1e-5,
    energy: float = 1e-6,
    force: float = 1e-4,
    kind: str = "density_rms",
) -> ScfForceCalibrationSample:
    return ScfForceCalibrationSample(
        family,
        family,
        "rhf",
        basis,
        kind,
        diagnostic,
        error(energy, force),
    )


def estimator() -> ScfForceErrorEstimator:
    return ScfForceErrorEstimator.fit(
        (
            sample("h2", diagnostic=1e-4, energy=2e-6, force=2e-4),
            sample("water", diagnostic=1e-5, energy=3e-7, force=3e-5),
        )
    )


def levels() -> tuple[ScfEffortLevel, ...]:
    return (
        ScfEffortLevel("loose", 0, 1e-6, 1e-4, 50),
        ScfEffortLevel("medium", 1, 1e-8, 1e-6, 80),
        ScfEffortLevel("strict", 2, 1e-10, 1e-8, 120, strict=True),
    )


def policy(**kwargs: object) -> ForceAwareScfPolicy:
    return ForceAwareScfPolicy(
        levels(),
        estimator(),
        optimizer_force_tolerance=1e-4,
        intermediate_energy_error=1e-5,
        intermediate_force_fraction=0.2,
        max_intermediate_force_error=5e-3,
        **kwargs,
    )


def test_calibration_is_diagnostic_specific_and_heldout_basis_is_evidence_only() -> (
    None
):
    fit = estimator()
    with pytest.raises(ValueError, match="diagnostic kinds"):
        ScfForceErrorEstimator.fit(
            (sample("h2"), sample("water", kind="physical_residual_rms"))
        )
    with pytest.raises(ValueError, match="basis"):
        fit.predict("rhf", "def2-svp", 1e-6)

    report = fit.evaluate_holdout(
        (
            sample(
                "lih",
                basis="def2-svp",
                diagnostic=1e-6,
                energy=1e-8,
                force=1e-6,
            ),
        ),
        TargetErrorBudget(energy_abs=1e-5, force_max_abs=1e-3),
    )
    assert report["rows"][0]["basis_seen_in_training"] is False
    assert report["certified"] is False


def test_holdout_reports_false_success_without_promoting_calibration_domain() -> None:
    fit = estimator()
    report = fit.evaluate_holdout(
        (
            sample(
                "miss",
                diagnostic=1e-8,
                energy=1e-3,
                force=1e-2,
            ),
        ),
        TargetErrorBudget(energy_abs=1e-5, force_max_abs=1e-3),
    )
    assert report["false_successes"] == 1
    assert report["false_success_rate"] == 1.0


def test_policy_starts_strict_and_relaxes_only_after_hysteresis() -> None:
    controller = policy(relax_after=2)
    state = controller.initial_state()
    assert state.level_index == 2
    first = controller.propose_next(
        state,
        method="rhf",
        basis_id="sto-3g",
        geometry_id="g0",
        step_index=0,
        current_force_max=None,
    )
    assert first.action == "strict"

    # Large current force permits an intermediate error budget; the first
    # request only starts the hysteresis streak.
    second = controller.propose_next(
        first.state,
        method="rhf",
        basis_id="sto-3g",
        geometry_id="g1",
        step_index=1,
        current_force_max=1e-1,
    )
    assert second.action == "hold"
    assert second.level.name == "strict"
    third = controller.propose_next(
        second.state,
        method="rhf",
        basis_id="sto-3g",
        geometry_id="g2",
        step_index=2,
        current_force_max=1e-1,
    )
    assert third.action == "relax"
    assert third.level.name in ("loose", "medium")
    assert third.state.transitions[-1].to_level == third.level.name


def test_near_stationary_and_domain_uncertainty_fail_closed() -> None:
    controller = policy()
    loose = replace(controller.initial_state(), level_index=0)
    near = controller.propose_next(
        loose,
        method="rhf",
        basis_id="sto-3g",
        geometry_id="near",
        step_index=5,
        current_force_max=4e-4,
    )
    assert near.action == "strict"
    assert near.level.strict
    unseen = controller.propose_next(
        loose,
        method="rhf",
        basis_id="def2-svp",
        geometry_id="unknown",
        step_index=6,
        current_force_max=1e-1,
    )
    assert unseen.action == "strict_fallback"
    assert unseen.level.strict


def test_energy_and_force_allowances_are_independent() -> None:
    controller = ForceAwareScfPolicy(
        levels(),
        estimator(),
        optimizer_force_tolerance=1e-4,
        # Force budget alone would admit a loose solve, but the explicit energy
        # allowance forces the policy to stay tighter.
        intermediate_energy_error=1e-8,
        intermediate_force_fraction=0.5,
        max_intermediate_force_error=1e-2,
        relax_after=1,
    )
    decision = controller.propose_next(
        controller.initial_state(),
        method="rhf",
        basis_id="sto-3g",
        geometry_id="g",
        step_index=1,
        current_force_max=1.0,
    )
    assert decision.level.name == "strict"


def test_final_verification_requires_original_model_strict_scf_and_force_gate() -> None:
    controller = policy()
    passed = controller.verify_final(
        target_model_id="target",
        observed_model_id="target",
        level=levels()[-1],
        converged=True,
        energy=-1.0,
        energy_change=1e-12,
        density_rms=1e-10,
        force_max_abs=5e-5,
    )
    assert passed.status == "observed_met"
    force_miss = controller.verify_final(
        target_model_id="target",
        observed_model_id="target",
        level=levels()[-1],
        converged=True,
        energy=-1.0,
        energy_change=1e-12,
        density_rms=1e-10,
        force_max_abs=5e-4,
    )
    assert force_miss.status == "observed_unmet"
    mismatch = controller.verify_final(
        target_model_id="target",
        observed_model_id="other",
        level=levels()[-1],
        converged=True,
        energy=-1.0,
        energy_change=1e-12,
        density_rms=1e-10,
        force_max_abs=5e-5,
    )
    assert mismatch.status == "unverified"
