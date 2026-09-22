"""Calibration coverage is joint, and final success requires a real boolean."""

from dataclasses import replace

import pytest
from test_force_aware_scf import policy, sample
from vibeqc.force_aware_scf import ScfForceErrorEstimator


def test_unmeasured_method_basis_cross_product_is_not_admitted() -> None:
    fit = ScfForceErrorEstimator.fit(
        (
            sample("h2", basis="sto-3g"),
            replace(sample("oh", basis="def2-svp"), method="uhf"),
        )
    )
    with pytest.raises(ValueError, match="domain"):
        fit.predict("rhf", "def2-svp", 1e-6)
    with pytest.raises(ValueError, match="domain"):
        fit.predict("uhf", "sto-3g", 1e-6)
    assert fit.predict("rhf", "sto-3g", 1e-6).method == "rhf"
    controller = replace(policy(relax_after=1), estimator=fit)
    decision = controller.propose_next(
        controller.initial_state(),
        method="rhf",
        basis_id="def2-svp",
        geometry_id="new-pair",
        step_index=2,
        current_force_max=1.0,
    )
    assert decision.action == "strict_fallback"
    assert decision.level.strict


@pytest.mark.parametrize("value", ("false", 1, float("nan"), object()))
def test_final_success_rejects_truthy_nonboolean_convergence(value: object) -> None:
    controller = policy()
    with pytest.raises((TypeError, ValueError), match="converged"):
        controller.verify_final(
            target_model_id="target",
            observed_model_id="target",
            level=controller.strict_level,
            converged=value,
            energy=-1.0,
            energy_change=1e-12,
            density_rms=1e-10,
            force_max_abs=1e-5,
        )
