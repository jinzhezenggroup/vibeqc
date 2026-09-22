"""Empirical status does not permit relabeling the evaluated scientific model."""

import pytest
from vibeqc import (
    ObservableDelta,
    PairedCalibrationSample,
    PairedDifferenceEstimator,
    ResolvedModel,
)


@pytest.mark.parametrize(
    "methods, predicted_method, target_method",
    [(("pbe-rks", "pbe-rks"), "pbe-rks", "rhf"), (("rhf", "uhf"), "rhf", "uhf")],
)
def test_paired_estimate_cannot_be_published_for_another_method(
    methods: tuple[str, str], predicted_method: str, target_method: str
) -> None:
    delta = ObservableDelta(1e-7, ((1e-5, 0.0, 0.0),))
    fit = PairedDifferenceEstimator.fit(
        tuple(
            PairedCalibrationSample(
                f"family-{i}", f"sample-{i}", method, "same-numerics", delta, delta
            )
            for i, method in enumerate(methods)
        )
    )
    estimate = fit.predict(predicted_method, delta, numerical_family_id="same-numerics")
    model = ResolvedModel(target_method, "geometry", "basis", 2)
    with pytest.raises(ValueError, match="method"):
        fit.as_error_evidence(
            model, estimate, energy_reference_norm=1.0, force_reference_norm=1.0
        )
