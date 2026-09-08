"""Empirical predictors must remain scoped estimates and expose holdout misses."""

import json
from dataclasses import replace

import pytest
from vibeqc import AccuracyAssessment, ObservableTarget, ResolvedModel, TargetAccuracy
from vibeqc.accuracy_estimator import (
    EmpiricalHFEstimator,
    HFCalibrationDomain,
    HFCalibrationSample,
    HFErrorFeatures,
)

MODEL = ResolvedModel("rhf", "geometry", "basis", 2)
FEATURES = HFErrorFeatures(
    MODEL.identity,
    "verified-basis-family",
    (1, 1),
    2,
    1e-6,
    4.0,
    1.0,
    1e-14,
    1e-14,
    1.4,
)
DOMAIN = HFCalibrationDomain(FEATURES.basis_family_id)


def sample(family, **kwargs):
    """Synthetic observations exercise failure logic independently of fitting."""
    return HFCalibrationSample(
        family,
        family,
        MODEL,
        FEATURES,
        kwargs.get("energy_error", 1e-8),
        kwargs.get("force_error", 1e-7),
        "strict-reference",
    )


def estimator():
    return EmpiricalHFEstimator.fit(
        DOMAIN, (sample("molecule-a"), sample("molecule-b"))
    )


def test_empirical_prediction_never_becomes_a_certified_or_observed_pass():
    predictor = estimator()
    evidence = predictor.predict(
        MODEL, FEATURES, energy_reference_norm=1, force_reference_norm=1
    )
    target = TargetAccuracy((ObservableTarget("energy", "absolute", "Eh", 1e-6),))
    assessment = AccuracyAssessment(MODEL, target, evidence)
    assert assessment.status == "estimated_below_target"
    assert evidence[0].calibration_id == predictor.identity
    assert evidence[0].value >= 4e-8
    assert not hasattr(assessment, "certified")


@pytest.mark.parametrize(
    "changes",
    [
        {"basis_family_id": "another-basis"},
        {"atomic_numbers": (3, 1)},
        {"nao": 100},
        {"physical_residual": 1.0},
        {"overlap_condition": 1e16},
        {"minimum_gap": 1e-10},
        {"minimum_gap": None},
        {"electron_trace_error": 0.1},
        {"idempotency_error": 0.1},
        {"minimum_nuclear_distance": 0.01},
        {"minimum_nuclear_distance": None},
        {"precision": "fp32"},
        {"backend": "cuda"},
    ],
)
def test_out_of_domain_inputs_are_rejected_even_for_tiny_residuals(changes):
    features = (
        replace(FEATURES, physical_residual=1e-30, **changes)
        if "physical_residual" not in changes
        else replace(FEATURES, **changes)
    )
    with pytest.raises(ValueError, match="outside calibration"):
        estimator().predict(
            MODEL, features, energy_reference_norm=1, force_reference_norm=1
        )


def test_holdout_reports_failed_coverage_missed_tolerances_and_overconservatism():
    predictor = estimator()
    rows = (
        sample("heldout-good", energy_error=1e-9, force_error=1e-9),
        sample("heldout-miss", energy_error=1e-3, force_error=1e-3),
    )
    report = predictor.evaluate_holdout(rows)
    assert report["missed_tolerances"] == 2
    assert report["covered"] == 2
    assert report["observables_evaluated"] == 4
    assert report["certified"] is False
    conservative = predictor.evaluate_holdout(
        rows[:1], energy_tolerance=1e-8, force_tolerance=1e-8
    )
    assert conservative["overconservative"] == 2
    with pytest.raises(ValueError, match="leakage"):
        predictor.evaluate_holdout((sample("molecule-a"),))


def test_missing_separation_is_only_allowed_for_a_single_atom():
    # Synthetic atomic features isolate the domain gate: an isolated carbon
    # atom has no internuclear distance, unlike the two-centre fixture above.
    atomic_model = replace(MODEL, electron_count=6)
    atomic_features = replace(
        FEATURES,
        model_id=atomic_model.identity,
        atomic_numbers=(6,),
        nao=5,
        minimum_nuclear_distance=None,
    )
    assert not DOMAIN.rejection_reasons(atomic_model, atomic_features)
    missing = replace(
        sample("molecule-b"), features=replace(FEATURES, minimum_nuclear_distance=None)
    )
    with pytest.raises(ValueError, match="missing nuclear separation"):
        EmpiricalHFEstimator.fit(DOMAIN, (sample("molecule-a"), missing))


def test_training_rejects_single_family_duplicates_and_unsupported_observations():
    with pytest.raises(ValueError, match="two molecular families"):
        EmpiricalHFEstimator.fit(DOMAIN, (sample("one"),))
    with pytest.raises(ValueError, match="duplicate"):
        EmpiricalHFEstimator.fit(DOMAIN, (sample("one"), sample("one")))
    invalid = replace(sample("two"), features=replace(FEATURES, minimum_gap=0))
    with pytest.raises(ValueError, match="outside domain"):
        EmpiricalHFEstimator.fit(DOMAIN, (sample("one"), invalid))


def test_saved_model_cannot_change_domain_or_evidence_kind_without_detection():
    predictor = estimator()
    record = json.loads(json.dumps(predictor.to_dict()))
    assert EmpiricalHFEstimator.from_dict(record) == predictor
    record["domain"]["minimum_gap"] = 1e-30
    with pytest.raises(ValueError, match="checksum"):
        EmpiricalHFEstimator.from_dict(record)
    record = predictor.to_dict()
    record["kind"] = "proven_bound"
    with pytest.raises(ValueError, match="kind"):
        EmpiricalHFEstimator.from_dict(record)
