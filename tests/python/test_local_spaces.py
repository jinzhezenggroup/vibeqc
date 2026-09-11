"""Local-space gauges/ranks must preserve the audited restricted MP2 model."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc.profiles import canonical_hash

from tools.vibeqc_local_cc.localization import localize_occupied, population_operators
from tools.vibeqc_local_cc.spaces import (
    make_pair_space,
    pair_density,
    projected_virtual_space,
    spectral_selection,
)
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture


def fixture(name="water"):
    metadata, arrays = load_fixture(name)
    snapshot = fixture_snapshot(metadata, arrays)
    ao_atoms = []
    for shell in metadata["inputs"]["shells"]:
        ell = shell["angular_momentum"]
        size = (
            (ell + 1) * (ell + 2) // 2
            if snapshot.representation == "cartesian"
            else 2 * ell + 1
        )
        ao_atoms.extend([shell["atom_index"]] * size)
    return snapshot, ao_atoms, metadata, arrays


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_localization_matches_independent_pyscf_objective_and_occupied_metric(name):
    snapshot, ao_atoms, metadata, _ = fixture(name)
    record = json.loads(
        (Path(__file__).parents[1] / "reference_data/local-spaces/pm.json").read_text()
    )
    digest = record.pop("record_hash")
    assert digest == canonical_hash(record)
    expected = next(r for r in record["rows"] if r["name"] == name)
    assert expected["parent_array_hash"] == metadata["array_hash"]
    local = localize_occupied(snapshot, ao_atoms)
    assert abs(local.objective - expected["objective"]) < 1e-9
    assert (
        local.gradient_max < 1e-10
        and local.objective >= local.initial_objective - 1e-12
    )
    co = snapshot.coefficients[:, : snapshot.nocc]
    np.testing.assert_allclose(
        local.coefficients @ local.coefficients.T, co @ co.T, atol=1e-11
    )
    np.testing.assert_allclose(
        local.coefficients.T @ snapshot.overlap @ local.coefficients,
        np.eye(snapshot.nocc),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        local.occupied_fock,
        local.rotation.T
        @ np.diag(snapshot.orbital_energies[: snapshot.nocc])
        @ local.rotation,
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        local.rotation.setflags(write=True)


def test_alternative_occupied_gauges_and_atom_permutations_preserve_pm_objective():
    s, atoms, _, _ = fixture()
    rng = np.random.default_rng(182)
    rotation, _ = np.linalg.qr(rng.normal(size=(s.nocc, s.nocc)))
    first = localize_occupied(s, atoms)
    second = localize_occupied(s, atoms, initial_rotation=rotation)
    permuted = localize_occupied(s, [2 - a for a in atoms])
    assert abs(first.objective - second.objective) < 1e-9
    assert abs(first.objective - permuted.objective) < 1e-9
    np.testing.assert_allclose(
        first.coefficients @ first.coefficients.T,
        second.coefficients @ second.coefficients.T,
        atol=1e-11,
    )


def test_localization_failures_do_not_publish_unconverged_or_invalid_spaces():
    s, atoms, _, _ = fixture()
    with pytest.raises(RuntimeError, match="localization failed"):
        localize_occupied(s, atoms, max_sweeps=1, tolerance=1e-14)
    with pytest.raises(ValueError, match="finite"):
        localize_occupied(s, atoms, initial_rotation=np.full((s.nocc, s.nocc), np.nan))
    with pytest.raises(ValueError, match="complex"):
        localize_occupied(s, atoms, initial_rotation=np.eye(s.nocc) + 1j * 0.1)
    with pytest.raises(ValueError, match="label every AO"):
        localize_occupied(s, atoms[:-1])
    with pytest.raises(MemoryError):
        localize_occupied(s, atoms, budget_bytes=1)
    with pytest.raises(ValueError, match="linearly dependent"):
        replace(s, overlap=np.zeros_like(s.overlap))
    with pytest.raises(ValueError, match="frozen"):
        replace(s, frozen_mask=(0,))
    with pytest.raises(ValueError, match="RHF"):
        replace(s, algorithm="UHF")


def test_projected_aos_remove_occupied_components_and_report_duplicate_rank_loss():
    s, _, _, _ = fixture()
    domain = projected_virtual_space(s)
    duplicate = projected_virtual_space(s, list(range(s.nmo)) * 2)
    cv = s.coefficients[:, s.nocc :]
    q = cv @ domain.columns
    assert domain.rank == duplicate.rank == s.nmo - s.nocc
    assert (
        len(duplicate.gram_eigenvalues) - duplicate.rank == 2 * s.nmo - duplicate.rank
    )
    np.testing.assert_allclose(q.T @ s.overlap @ q, np.eye(domain.rank), atol=1e-11)
    np.testing.assert_allclose(
        s.coefficients[:, : s.nocc].T @ s.overlap @ q, 0, atol=1e-11
    )
    np.testing.assert_allclose(domain.projector, duplicate.projector, atol=1e-10)
    with pytest.raises(ValueError, match="no supported virtual rank"):
        projected_virtual_space(s, absolute_threshold=1e8)
    with pytest.raises(MemoryError):
        projected_virtual_space(s, budget_bytes=1)


def test_pair_density_spin_factors_and_degenerate_cluster_policy():
    rng = np.random.default_rng(182)
    t = rng.normal(size=(4, 4))
    tilde = 2 * t - t.T
    for diagonal in (True, False):
        expected = (tilde.T @ t + tilde @ t.T) / (1 + diagonal)
        actual = pair_density(t, diagonal_pair=diagonal)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.linalg.eigvalsh(actual)[0] >= -1e-12
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        rotated = pair_density(rotation.T @ t @ rotation, diagonal_pair=diagonal)
        np.testing.assert_allclose(rotated, rotation.T @ actual @ rotation, atol=1e-11)
    selected, crossing = spectral_selection([0, 1 - 1e-14, 1 + 1e-14, 2], 1)
    assert selected.tolist() == [False, True, True, True] and crossing
    selected, _ = spectral_selection([0, 0, 0], 0, retain_all=True)
    assert selected.all()


def test_pair_projectors_overlaps_gauges_and_stale_parent_rejection():
    s, atoms, _, _ = fixture("lih")
    local = localize_occupied(s, atoms)
    domain = projected_virtual_space(s)
    t = np.eye(domain.rank)
    space = make_pair_space(s, local, domain, (0, 1), t, occupation_threshold=1e-7)
    assert space.rank == domain.rank
    rotation, _ = np.linalg.qr(
        np.random.default_rng(10).normal(size=(space.rank, space.rank))
    )
    alternative = replace(space, columns=space.columns @ rotation)
    np.testing.assert_allclose(alternative.projector, space.projector, atol=1e-12)
    np.testing.assert_allclose(space.overlap(alternative), rotation, atol=1e-12)
    assert space.gauge_identity != alternative.gauge_identity
    zero = make_pair_space(s, local, domain, (0, 1), t, occupation_threshold=1e8)
    assert zero.rank == 0 and zero.overlap(space).shape == (0, space.rank)
    with pytest.raises(ValueError, match="rank/branch"):
        replace(zero, keep_full_space=True)
    with pytest.raises(ValueError, match="rank/branch"):
        replace(domain, rank_crossing=not domain.rank_crossing)
    with pytest.raises(ValueError, match="stale"):
        make_pair_space(replace(s, generation_id="changed"), local, domain, (0, 1), t)
    with pytest.raises(ValueError, match="same reference"):
        space.overlap(replace(space, reference_id="changed"))


def test_mulliken_charge_jacobi_gradient_matches_independent_finite_difference():
    s, atoms, _, _ = fixture()
    operators = population_operators(s, atoms)
    p, q = 0, 1
    analytic = 4 * np.sum(
        operators[:, p, q] * (operators[:, p, p] - operators[:, q, q])
    )

    def objective(theta):
        u = np.eye(s.nocc)
        u[np.ix_([p, q], [p, q])] = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
        return np.sum(np.diagonal(u.T @ operators @ u, axis1=1, axis2=2) ** 2)

    for step in (1e-4, 1e-5):
        assert abs((objective(step) - objective(-step)) / (2 * step) - analytic) < 1e-8
