"""Accuracy contracts must reject evidence that changes the scientific target."""

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from vibeqc import (
    AccuracyAssessment,
    Calculator,
    ErrorEvidence,
    EvidenceKind,
    ObservableTarget,
    ResolvedModel,
    TargetAccuracy,
    compare_observables,
)

from tools.vibeqc_validation.fixtures import calculator_inputs, load_fixtures

MODEL = ResolvedModel("rhf", "geometry", "basis", 2)
ENERGY = ObservableTarget("energy", "absolute", "Eh", absolute=1e-6)
FORCE = ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-7)
TARGET = TargetAccuracy((ENERGY, FORCE))


def evidence(**changes):
    """One independently measured total energy error with explicit provenance."""
    return replace(
        ErrorEvidence(
            EvidenceKind.OBSERVED,
            MODEL.identity,
            MODEL.identity,
            MODEL.identity,
            "energy",
            "absolute",
            "Eh",
            1e-8,
            1.0,
            "total_numerical",
            "relaxed_target",
            (("reference", "test-reference-v1"),),
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "pbe"},
        {"hamiltonian": "ecp"},
        {"electron_count": 3},
        {"multiplicity": 2},
        {"electron_count": True},
        {"schema_version": True},
        {"schema_version": 2},
        {"approximation": "unknown"},
        {"auxiliary_basis_hash": "aux"},
        {"basis_hash": ""},
        {"representation": "unspecified-spherical"},
    ],
)
def test_invalid_or_unsupported_models(changes):
    with pytest.raises(ValueError):
        replace(MODEL, **changes)


def test_fitted_model_cannot_alias_conventional_or_different_threshold():
    fitted = replace(
        MODEL,
        approximation="density_fitting",
        auxiliary_basis_hash="aux",
        metric_relative_threshold=1e-10,
    )
    assert (
        len(
            {
                MODEL.identity,
                fitted.identity,
                replace(fitted, metric_relative_threshold=1e-8).identity,
            }
        )
        == 3
    )
    assert replace(MODEL, electron_count=np.int64(2)).identity == MODEL.identity
    with pytest.raises(ValueError):
        replace(fitted, auxiliary_basis_hash=None)
    with pytest.raises(ValueError):
        replace(fitted, metric_relative_threshold=1.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"unit": "eV"},
        {"norm": "rms"},
        {"observable": "hessian"},
        {"absolute": float("nan")},
        {"absolute": float("inf")},
        {"absolute": -1},
        {"absolute": True},
        {"absolute": "1e-6"},
        {"absolute": 0.0},
    ],
)
def test_units_norms_and_nonfinite_targets(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(ENERGY, **changes)


def test_requirement_ownership_duplicates_and_relative_zero():
    requirements = [ENERGY]
    target = TargetAccuracy(requirements)
    requirements.append(FORCE)
    assert target.observables == (ENERGY,)
    with pytest.raises(FrozenInstanceError):
        target.scope = "fixed_density"
    with pytest.raises(ValueError):
        TargetAccuracy((ENERGY, ENERGY))
    with pytest.raises(ValueError):
        TargetAccuracy(())
    relative = replace(ENERGY, absolute=0, relative=1e-6)
    assert relative.allowance(0) == 0
    assert relative.allowance(2) == 2e-6


def test_convergence_and_partial_or_fixed_density_evidence_cannot_pass():
    assert AccuracyAssessment(MODEL, TARGET).status == "unverified"
    assert AccuracyAssessment(MODEL, TARGET, (evidence(),)).status == "unverified"
    energy_target = TargetAccuracy((ENERGY,))
    for item in (evidence(scope="fixed_density"), evidence(source="scf_residual")):
        assert AccuracyAssessment(MODEL, energy_target, (item,)).status == "unverified"
    assert (
        AccuracyAssessment(MODEL, energy_target, (evidence(),), False).status
        == "unconverged"
    )
    assert (
        AccuracyAssessment(MODEL, energy_target, (evidence(),)).status == "observed_met"
    )
    assert (
        AccuracyAssessment(MODEL, TARGET, (evidence(value=1e-3),)).status
        == "observed_unmet"
    )


def test_small_empirical_estimate_with_bad_conditioning_is_never_success():
    item = evidence(
        kind=EvidenceKind.EMPIRICAL,
        value=1e-20,
        condition_estimate=1e16,
        calibration_id="limited-test-fit",
        assumptions=("only valid on the calibrated molecular family",),
    )
    assessment = AccuracyAssessment(MODEL, TargetAccuracy((ENERGY,)), (item,))
    assert assessment.status == "estimated_below_target"
    # Held-out misses override even a highly optimistic calibrated estimate.
    miss = replace(item, actual_reference_error=1e-2)
    assert replace(assessment, evidence=(miss,)).status == "observed_unmet"
    with pytest.raises(ValueError, match="proof"):
        evidence(kind=EvidenceKind.BOUND)
    with pytest.raises(ValueError):
        evidence(kind=EvidenceKind.EMPIRICAL)
    with pytest.raises(TypeError):
        ErrorEvidence(**{**item.to_dict(), "status": "certified"})


@pytest.mark.parametrize(
    "field", ["model_id", "evaluated_model_id", "reference_model_id"]
)
def test_changed_models_are_not_numerical_error_evidence(field):
    item = evidence(
        **{field: replace(MODEL, geometry_hash="another-geometry").identity}
    )
    with pytest.raises(ValueError, match="model mismatch"):
        AccuracyAssessment(MODEL, TargetAccuracy((ENERGY,)), (item,))


def test_different_source_errors_are_not_summed_and_duplicate_total_is_ambiguous():
    items = (
        evidence(source="screening", value=0.4e-6),
        evidence(source="arithmetic", value=0.4e-6),
    )
    assert (
        AccuracyAssessment(MODEL, TargetAccuracy((ENERGY,)), items).status
        == "unverified"
    )
    with pytest.raises(ValueError, match="duplicate"):
        AccuracyAssessment(MODEL, TARGET, (evidence(), evidence(value=2e-8)))


def test_portable_evidence_owns_mutable_inputs():
    provenance = [["reference", "v1"]]
    item = evidence(provenance=provenance)
    provenance[0][1] = "corrupted"
    record = AccuracyAssessment(MODEL, TargetAccuracy((ENERGY,)), (item,)).to_dict()
    assert json.loads(json.dumps(record))["evidence"][0]["provenance"] == [
        ["reference", "v1"]
    ]
    assert ErrorEvidence(**item.to_dict()) == item
    with pytest.raises(ValueError):
        evidence(provenance=(("same", "a"), ("same", "b")))


def test_loaded_status_is_derived_and_identity_changes_are_detected():
    assessment = AccuracyAssessment(MODEL, TARGET, (evidence(),))
    record = json.loads(json.dumps(assessment.to_dict()))
    assert AccuracyAssessment.from_dict(record) == assessment
    record["status"] = "certified"
    with pytest.raises(ValueError, match="outcome"):
        AccuracyAssessment.from_dict(record)
    record = json.loads(json.dumps(assessment.to_dict()))
    record["model"]["basis_hash"] = "altered"
    with pytest.raises(ValueError, match="checksum"):
        AccuracyAssessment.from_dict(record)


def test_comparison_measures_different_force_norms_without_broadcasting():
    target = TargetAccuracy(
        (
            ObservableTarget("forces", "max_abs", "Eh/bohr", 0.6),
            ObservableTarget("forces", "rms", "Eh/bohr", 0.5),
        )
    )
    result = compare_observables(
        MODEL,
        MODEL,
        MODEL,
        target,
        {"forces": [[1.0, 0, 0], [0, 0, 0]]},
        {"forces": np.zeros((2, 3))},
        scope="relaxed_target",
        provenance=(("reference", "oracle"),),
        converged=True,
    )
    assert result.status == "observed_unmet"
    assert result.outcomes == ("observed_unmet", "observed_met")
    np.testing.assert_allclose([e.value for e in result.evidence], [1, 1 / np.sqrt(6)])
    for forces in ([0, 0, 0], [[0, 0, np.nan]], [[0j, 0, 0]], [[0, 0, 0]]):
        with pytest.raises(ValueError):
            compare_observables(
                MODEL,
                MODEL,
                MODEL,
                target,
                {"forces": forces},
                {"forces": np.zeros((2, 3))},
                scope="relaxed_target",
                provenance=(("reference", "oracle"),),
                converged=True,
            )


def test_resolved_calculator_identity_separates_numerics_from_model():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    baseline = Calculator().resolved_model(atoms)
    loose = Calculator(energy_tolerance=1e-6, screening_tolerance=1e-8).resolved_model(
        atoms
    )
    assert baseline == loose
    assert (
        Calculator(basis="def2-svp").resolved_model(atoms).identity != baseline.identity
    )
    fitted = Calculator(density_fitting="cpu").resolved_model(atoms)
    assert fitted.auxiliary_basis_hash == fitted.basis_hash
    assert fitted.identity != baseline.identity
    assert (
        Calculator().resolved_model([(z, (-0.0, y, x)) for z, (_, y, x) in atoms])
        == baseline
    )


@pytest.mark.parametrize("name", ["h2", "hf-plus-uhf"])
def test_native_hf_energy_and_forces_against_independent_pinned_reference(name):
    reference = next(r for r in load_fixtures() if r["inputs"]["name"] == name)
    inputs = reference["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    calc = Calculator(**calculator_inputs(inputs))
    state = {"charge": inputs["charge"], "multiplicity": inputs["multiplicity"]}
    result = calc.singlepoint(atoms, **state)
    model = calc.resolved_model(atoms, **state)
    report = compare_observables(
        model,
        model,
        model,
        TARGET,
        {"energy": result.energy, "forces": result.forces},
        reference["data"],
        scope="relaxed_target",
        provenance=(("reference", reference["inputs_hash"]),),
        converged=result.converged,
    )
    assert report.status == "observed_met"
    assert all(item.value < 1e-9 for item in report.evidence)


def test_requested_accuracy_is_independent_of_iteration_convergence():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    ordinary = Calculator(energy_tolerance=1e-6).singlepoint(atoms)
    requested = Calculator(energy_tolerance=1e-6, target_accuracy=TARGET).singlepoint(
        atoms
    )
    assert ordinary.accuracy is None
    assert requested.accuracy.status == "unverified"
    assert requested.accuracy.target == TARGET
    assert ordinary.energy == requested.energy
    np.testing.assert_array_equal(ordinary.forces, requested.forces)


def test_batch_accuracy_preserves_item_geometry_failure_isolation_and_policy_identity():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(target_accuracy=TARGET)
    with calc.prepare_batch([atoms, atoms]) as batch:
        result = batch.execute(
            [[[0, 0, -0.8], [0, 0, 0.8]], [[np.nan, 0, 0], [0, 0, 0.7]]]
        )
        assert result.failure_indices == (1,)
        assert result.items[0].accuracy.status == "unverified"
        assert result.items[1].accuracy is None
        assert result.items[0].accuracy.model != calc.resolved_model(atoms)
        assert batch.execute().items[0].accuracy.model == calc.resolved_model(atoms)
        calc._energy_tolerance = 1e-7
        with pytest.raises(RuntimeError, match="identity changed"):
            batch.execute()
