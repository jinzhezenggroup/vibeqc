"""Publication gates include cold/priming failures as well as every repeat."""

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.readme_method_endpoints import validate_record


def test_large_export_tiles_preserve_reference_grid_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing export tile size must not change the common discrete energy."""
    import numpy as np
    from vibeqc import GridSpec
    from vibeqc_compiler.dft.grid import MolecularGrid

    from benchmarks import readme_method_endpoints as runner
    from benchmarks.readme_hf_scaling import scaling_cases

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    molecular = MolecularGrid(
        scaling_cases()["water-3"].atoms,
        spec=GridSpec(radial_points=4, angular_polar=8, angular_azimuth=16),
    )
    reference = molecular.explicit()
    for expected_hit in (False, True):
        points, weights, _, export = runner.comparison_grid(molecular)
        np.testing.assert_array_equal(points, reference.points)
        np.testing.assert_array_equal(weights, reference.weights)
        assert export["cache_hit"] is expected_hit


def accepted_record() -> dict:
    sample = {
        "seconds": 1.0,
        "energies_hartree": [-1.0],
        "convergence": [{"converged": True}],
    }
    return {
        "reference_cold": deepcopy(sample),
        "native_cold": deepcopy(sample),
        "reference_samples": [deepcopy(sample), deepcopy(sample)],
        "native_samples": [deepcopy(sample), deepcopy(sample)],
        "priming": {"native": deepcopy(sample), "reference": deepcopy(sample)},
        "accuracy": {"maximum_energy_error_hartree": 1e-10},
        "gates": {"maximum_energy_error_hartree": 1e-8},
    }


@pytest.mark.parametrize(
    "phase",
    ("native_cold", "reference_cold", "priming", "native_samples", "reference_samples"),
)
def test_no_unconverged_phase_can_be_published(phase: str) -> None:
    record = accepted_record()
    sample = record[phase]
    if phase == "priming":
        sample = sample["native"]
    elif phase.endswith("samples"):
        sample = sample[0]
    sample["convergence"][0]["converged"] = False
    with pytest.raises(RuntimeError, match="convergence"):
        validate_record(record)


def test_rejects_error_outlier_and_nonfinite_endpoint() -> None:
    record = accepted_record()
    record["accuracy"]["maximum_energy_error_hartree"] = 1e-5
    with pytest.raises(RuntimeError, match="acceptance"):
        validate_record(record)
    record = accepted_record()
    record["native_samples"][0]["energies_hartree"] = [float("nan")]
    with pytest.raises(RuntimeError, match="finite"):
        validate_record(record)


def test_accepts_converged_numerically_qualified_endpoints() -> None:
    record = accepted_record()
    validate_record(record)
    assert record["gates_passed"] is True


def test_reference_only_does_not_claim_numerical_parity() -> None:
    record = accepted_record()
    del record["native_cold"], record["accuracy"], record["priming"]["native"]
    record["native_samples"] = []
    validate_record(record)
    assert record["convergence_gate_passed"] is True
    assert record["gates_passed"] is None
