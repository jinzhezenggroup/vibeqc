from __future__ import annotations

import argparse

import numpy as np
import pytest

from tools.benchmark_d3_alchemi import (
    Workload,
    _cutoff_method,
    _error_summary,
    make_system,
    parse_workload,
)


def test_workload_parser_and_geometry_are_deterministic():
    workload = parse_workload("32x64")
    assert workload == Workload(32, 64)
    assert workload.total_atoms == 2048
    numbers_a, positions_a = make_system(32)
    numbers_b, positions_b = make_system(32)
    np.testing.assert_array_equal(numbers_a, numbers_b)
    np.testing.assert_array_equal(positions_a, positions_b)
    assert positions_a.shape == (32, 3)
    assert np.unique(positions_a, axis=0).shape[0] == 32


@pytest.mark.parametrize("value", ["bad", "0x1", "1x0", "4097x1"])
def test_workload_parser_rejects_invalid_cases(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_workload(value)


def test_cutoff_method_changes_semantic_identity_without_mutating_catalog():
    first = _cutoff_method("PBE-D3(BJ)", 20.0)
    second = _cutoff_method("PBE-D3(BJ)", 30.0)
    assert first.dispersion.cn_cutoff == 20.0
    assert first.dispersion.pair_cutoff == 20.0
    assert second.dispersion.cn_cutoff == 30.0
    assert first.dispersion.identity != second.dispersion.identity


def test_error_summary_uses_force_to_gradient_normalized_outputs():
    vibeqc = {
        "energies_hartree": [-1.0],
        "gradients_hartree_per_bohr": [[[1.0, 2.0, 3.0]]],
    }
    alchemi = {
        "energies_hartree": [-0.75],
        "gradients_hartree_per_bohr": [[[1.5, 1.0, 3.0]]],
    }
    summary = _error_summary(vibeqc, alchemi)
    assert summary["max_abs_energy_hartree"] == pytest.approx(0.25)
    assert summary["max_abs_gradient_hartree_per_bohr"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("energies_hartree", [-1.0, -1.0]),
        ("energies_hartree", [float("nan")]),
        ("gradients_hartree_per_bohr", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        ("gradients_hartree_per_bohr", [[float("inf"), 2.0, 3.0]]),
    ],
)
def test_error_summary_rejects_broadcasting_and_nonfinite_outputs(field, value):
    baseline = {
        "energies_hartree": [-1.0],
        "gradients_hartree_per_bohr": [[1.0, 2.0, 3.0]],
    }
    candidate = {**baseline, field: value}
    with pytest.raises(ValueError, match="finite.*matching"):
        _error_summary(baseline, candidate)


@pytest.mark.parametrize("gradient", [[1.0, 2.0, 3.0], [[1.0], [2.0], [3.0]]])
def test_error_summary_rejects_same_size_wrong_cartesian_layout(gradient):
    baseline = {
        "energies_hartree": [-1.0],
        "gradients_hartree_per_bohr": [[1.0, 2.0, 3.0]],
    }
    with pytest.raises(ValueError, match="Cartesian"):
        _error_summary(baseline, {**baseline, "gradients_hartree_per_bohr": gradient})
