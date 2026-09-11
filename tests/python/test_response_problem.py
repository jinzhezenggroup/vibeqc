"""Response problem snapshots, rotation layouts and fail-closed boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    ResponseCompatibilityError,
    ResponseProblem,
    ResponseUnsupported,
    RHFResponseOperator,
    RotationLayout,
)


def test_rotation_layout_pack_density_generator_and_invalid_spaces():
    layout = RotationLayout((0, 1), (2, 3, 4))
    values = np.arange(layout.dimension, dtype=float).reshape(layout.nocc, layout.nvirt)
    packed = layout.pack(values)
    np.testing.assert_array_equal(layout.as_ia(packed), values)
    density = layout.density_matrix(packed)
    assert density.shape == (5, 5)
    np.testing.assert_allclose(density, density.T)
    generator = layout.generator_matrix(packed)
    np.testing.assert_allclose(generator, -generator.T)
    with pytest.raises(ValueError, match="contiguous"):
        RotationLayout((0, 2), (3,))
    with pytest.raises(ResponseUnsupported, match="restricted"):
        RotationLayout((0,), (1,), ("alpha", "beta"))


def test_problem_invalidates_same_dimension_reference_change_and_operator_change():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(
        reference, backend, perturbation_labels=("x", "y")
    )
    changed_reference = replace(reference, coefficients=-reference.coefficients)
    assert changed_reference.nmo == reference.nmo
    with pytest.raises(ResponseCompatibilityError, match="reference_identity"):
        problem.assert_compatible(replace(problem, reference=changed_reference))
    with pytest.raises(ResponseCompatibilityError, match="operator_identity"):
        problem.assert_compatible(replace(problem, operator_identity="different"))
    assert (
        problem.compatibility_identity
        == replace(problem, perturbation_labels=("z",)).compatibility_identity
    )


def test_problem_rhs_layout_validation_and_label_count():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(
        reference, backend, perturbation_labels=("one",)
    )
    assert problem.validate_rhs(np.ones(problem.dimension)).shape == (
        problem.dimension,
        1,
    )
    with pytest.raises(ValueError, match="shape"):
        problem.validate_rhs(np.ones((problem.dimension - 1, 1)))
    with pytest.raises(ValueError, match="labels"):
        problem.validate_rhs(np.ones((problem.dimension, 2)))
    with pytest.raises(ResponseUnsupported, match="RHS layout"):
        replace(problem, rhs_layout="unsupported")
    with pytest.raises(ResponseUnsupported, match="gauge"):
        replace(problem, gauge="arbitrary")
    with pytest.raises(ResponseUnsupported, match="overlap metric"):
        replace(problem, overlap_metric="ao-overlap")


def test_problem_rejects_rotation_space_membership_mismatch():
    meta, arrays = load_fixture("water")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(reference, backend)
    bad = RotationLayout((0,), tuple(range(1, reference.nmo)))
    with pytest.raises(ValueError, match="occupations"):
        replace(problem, layout=bad)


def test_cpks_requires_converged_ks_reference_and_explicit_model_identity():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    with pytest.raises(ResponseUnsupported, match="converged KS reference"):
        ResponseProblem.from_reference(
            reference, method="cpks", operator_identity="cpks-test"
        )
    ks = replace(
        reference,
        algorithm="KS",
        functional_identity="functional-test",
        grid_identity="grid-test",
        hf_backend="test-ks",
    )
    problem = ResponseProblem.from_reference(
        ks, method="cpks", operator_identity="cpks-test"
    )
    assert problem.reference.algorithm == "KS"
    with pytest.raises(ResponseUnsupported, match="KS reference"):
        ResponseProblem.from_reference(
            ks, method="rhf", operator_identity=backend.identity
        )
    with pytest.raises(ValueError, match="functional and grid"):
        replace(ks, grid_identity=None)


def test_same_occupancy_degeneracy_is_not_a_response_singularity():
    """Occupied-occupied degeneracy must not trip the response stability gate.

    The nonredundant layout solves only occupied-virtual rotations, so a
    symmetry-degenerate occupied pair with a healthy occupied-virtual gap is a
    valid response reference and ``require_stable`` must accept it.
    """
    meta, arrays = load_fixture("water")
    reference = fixture_snapshot(meta, arrays)
    energies = reference.orbital_energies.copy()
    energies[1] = energies[0]
    fock = (
        reference.overlap
        @ reference.coefficients
        @ np.diag(energies)
        @ reference.coefficients.T
        @ reference.overlap
    )
    degenerate = replace(reference, orbital_energies=energies, fock=fock)
    problem = ResponseProblem.from_reference(
        degenerate, method="rhf", operator_identity="synthetic"
    )
    occupied = np.asarray(problem.layout.occupied)
    virtual = np.asarray(problem.layout.virtual)
    expected_gap = float(
        np.min(np.abs(energies[virtual][:, None] - energies[occupied][None, :]))
    )
    assert expected_gap > 1e-8
    assert problem.diagnostics["minimum_ov_gap"] == pytest.approx(expected_gap)
    assert not problem.diagnostics["near_degenerate"]
    assert problem.require_stable() is problem
