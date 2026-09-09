"""Native diagnostic solves must agree with independent HF references."""

import os
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator

from tools.vibeqc_numerics.audit import ProbeControls, StrictHFAudit, probe_hf
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.fixtures import calculator_inputs, load_fixtures


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4", "hf-plus-uhf"])
def test_probe_and_physical_operator_against_pinned_pyscf(name):
    reference = next(r for r in load_fixtures() if r["inputs"]["name"] == name)
    inputs = reference["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    calc = Calculator(**calculator_inputs(inputs))
    model = calc.resolved_model(
        atoms, charge=inputs["charge"], multiplicity=inputs["multiplicity"]
    )
    with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
        probe = probe_hf(source, model)
        assert probe.converged
        assert abs(probe.energy - reference["data"]["energy"]) < 1e-9
        np.testing.assert_allclose(
            probe.forces, reference["data"]["forces"], atol=1e-8, rtol=0
        )
        audit = StrictHFAudit(source, model).evaluate(probe)
        assert audit["energy_operator_difference"] < 1e-10
        assert audit["orthonormal_commutator_max"] < 1e-8
        assert audit["electron_trace_error_max"] < 1e-10
        assert audit["density_idempotency_max"] < 1e-10
        assert audit["scope"] == "fixed_density"
        assert not audit["gap_is_error_certificate"]
        with pytest.raises(ValueError):
            probe.density.setflags(write=True)
        if source.nbf > 1:
            damaged = probe.density.copy()
            damaged[0, 0, 1] += 1e-3
            with pytest.raises(ValueError, match="Hermitian"):
                replace(probe, density=damaged)


def test_failed_probe_preserves_failure_without_publishing_reference_state():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    model = Calculator().resolved_model(atoms)
    with NativeSource(atoms) as source:
        failed = probe_hf(source, model, ProbeControls(max_iterations=1))
        assert failed.iterations == 1 and not failed.converged
        assert failed.energy_change is None
        assert failed.density is failed.forces is None
        with pytest.raises(ValueError, match="failed probe"):
            StrictHFAudit(source, model).evaluate(failed)
        changed = replace(model, geometry_hash="different-geometry")
        with pytest.raises(ValueError, match="identity"):
            probe_hf(source, changed)
        with pytest.raises(ValueError, match="identity"):
            StrictHFAudit(source, changed)


def test_df_operator_audit_uses_same_metric_and_spin_factors():
    atoms = [("He", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    for method, charge, multiplicity in (("rhf", 1, 1), ("uhf", 0, 2)):
        calc = Calculator(
            method=method, density_fitting="cpu", auxiliary_basis="sto-3g"
        )
        model = calc.resolved_model(atoms, charge=charge, multiplicity=multiplicity)
        with NativeSource(atoms, auxiliary_basis="sto-3g", charge=charge) as source:
            probe = probe_hf(source, model)
            audit = StrictHFAudit(source, model).evaluate(probe)
            assert probe.converged
            assert audit["energy_operator_difference"] < 1e-11
            assert audit["metric_rank"] == source.naux
            assert audit["orthonormal_commutator_max"] < 1e-9


def test_probe_rejects_mixed_override_before_gpu_execution(monkeypatch):
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    model = Calculator().resolved_model(atoms)
    monkeypatch.setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "1e-6")
    with (
        NativeSource(atoms) as source,
        pytest.raises(ValueError, match="mixed-precision"),
    ):
        probe_hf(source, model, backend="cuda")


def test_explicit_arithmetic_experiment_requires_matching_settings(monkeypatch):
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    model = Calculator().resolved_model(atoms)
    monkeypatch.setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "1e-6")
    with NativeSource(atoms) as source:
        with pytest.raises(ValueError, match="process settings"):
            probe_hf(
                source, model, backend="cuda", experimental_mixed_fock_threshold=1e-3
            )
        with pytest.raises(ValueError, match="experimental mixed"):
            probe_hf(
                source, model, backend="cpu", experimental_mixed_fock_threshold=1e-6
            )


def test_arithmetic_driver_restores_policy_after_failure(monkeypatch):
    from tools.vibeqc_numerics.precision_experiment import _mixed_override

    monkeypatch.setenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "0")
    with pytest.raises(RuntimeError), _mixed_override(1e-6):
        assert os.environ["VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"] == "1e-06"
        raise RuntimeError("failed native experiment")
    assert os.environ["VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"] == "0"


@pytest.mark.skipif(
    os.environ.get("VIBEQC_ACCURACY_CUDA_TEST") != "1",
    reason="requires an explicitly allocated GPU",
)
@pytest.mark.parametrize("name", ["h2o", "hf-plus-uhf"])
def test_cuda_probe_density_force_and_physical_residual(name):
    """Actual GPU execution must pass independent final-state numerical gates."""
    assert os.environ.get("SLURM_JOB_ID"), "this host's GPU tests require Slurm"
    reference = next(r for r in load_fixtures() if r["inputs"]["name"] == name)
    inputs = reference["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    calc = Calculator(**calculator_inputs(inputs))
    model = calc.resolved_model(
        atoms, charge=inputs["charge"], multiplicity=inputs["multiplicity"]
    )
    with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
        probe = probe_hf(source, model, backend="cuda")
        assert probe.converged
        assert abs(probe.energy - reference["data"]["energy"]) < 1e-9
        np.testing.assert_allclose(
            probe.forces, reference["data"]["forces"], atol=1e-8, rtol=0
        )
        assert (
            StrictHFAudit(source, model).evaluate(probe)["orthonormal_commutator_max"]
            < 1e-8
        )
