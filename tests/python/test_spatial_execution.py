"""Local columns, complete D[I,I], scatter, and prepared lifetime regressions."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.features import density_features
from vibeqc_compiler.dft.spatial import SpatialPolicy, build_spatial_tasks
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid


@pytest.fixture(params=["cartesian", "spherical"])
def local_case(request):
    shells = tuple(
        Shell(atom, l, (Primitive(1.0, 1.0), Primitive(2.0, -0.1)))
        for atom in (0, 1)
        for l in (0, 1, 2, 3)
    )
    rng = np.random.default_rng(2341)
    points = np.concatenate(
        (
            rng.normal(size=(9, 3)) * 0.12 + [-3, 0, 0],
            rng.normal(size=(7, 3)) * 0.12 + [3, 0, 0],
        )
    )
    grid = ExplicitGrid(points, rng.random(len(points)), (0,) * 9 + (1,) * 7, {})
    with NativeAO(
        [("H", (-3, 0, 0)), ("H", (3, 0, 0))],
        basis=shells,
        representation=request.param,
    ) as basis:
        orbitals = rng.normal(size=(2, basis.nao, 5))
        density = orbitals @ orbitals.swapaxes(1, 2)
        yield basis, grid, density


def test_selected_native_jets_match_dense_including_empty_and_noncontiguous(local_case):
    basis, grid, _ = local_case
    full = basis.evaluate(grid.points, order=3)
    for ids in (
        np.arange(0, basis.nao, 3),
        np.arange(basis.nao),
        np.array([], dtype=int),
    ):
        got = basis.evaluate(grid.points, order=3, ao_ids=ids)
        np.testing.assert_array_equal(got, full[:, :, ids])
    for ids in ([0, 0], [2, 1], [-1], [basis.nao], [0.5]):
        with pytest.raises(ValueError, match="AO IDs"):
            basis.evaluate(grid.points, ao_ids=ids)


@pytest.mark.parametrize("screening", ["off", "absolute_ao_jet"])
def test_local_features_preserve_all_cross_ao_terms_and_fixed_masks(
    local_case, screening
):
    basis, grid, density = local_case
    policy = SpatialPolicy(
        region_points=4, screening=screening, cutoff=0 if screening == "off" else 1e-8
    )
    dense = basis.evaluate(grid.points, order=1)
    masks = build_spatial_tasks(basis, grid, policy=policy)
    assert screening == "off" or any(len(t.ao_ids) < basis.nao for t in masks.tasks)
    for tile_points in (1, 3, 16):
        for budget in (2 << 20, 4 << 20):
            with PreparedSpatialGrid(
                basis,
                grid,
                policy=policy,
                tasks=masks,
                tile_points=tile_points,
                resource_budget=ResourceBudget(host_bytes=budget),
            ) as p:
                seen = []
                for tile in p.iter_features(density, include_jets=True):
                    seen.extend(tile.point_ids)
                    masked = dense[:, tile.point_ids].copy()
                    omitted = np.ones(basis.nao, dtype=bool)
                    omitted[tile.ao_ids] = False
                    masked[:, :, omitted] = 0
                    # Global-index baseline keeps complete D. A shell-diagonal
                    # gather would lose large cross terms in this random D.
                    reference = density_features(masked, density)
                    for key in reference:
                        np.testing.assert_allclose(
                            tile.features[key], reference[key], atol=1e-11, rtol=1e-10
                        )
                    np.testing.assert_array_equal(
                        tile.ao_jets, dense[:, tile.point_ids][:, :, tile.ao_ids]
                    )
                np.testing.assert_array_equal(
                    np.sort(seen), np.arange(len(grid.points))
                )


def test_prepared_replacement_density_and_iterator_lifetimes(local_case):
    basis, grid, density = local_case
    with PreparedSpatialGrid(basis, grid, tile_points=2) as owner:
        stale = owner.iter_features(density)
        first = next(stale)
        current = owner.iter_features(2 * density)
        np.testing.assert_allclose(
            next(current).features["rho"], 2 * first.features["rho"]
        )
        with pytest.raises(RuntimeError, match="stale"):
            next(stale)
        identity = owner.tasks.generation_id
        with pytest.raises(MemoryError):
            owner.reconfigure(basis, grid, resource_budget=ResourceBudget(host_bytes=1))
        assert owner.tasks.generation_id == identity
        changed = ExplicitGrid(grid.points + 0.01, grid.weights, grid.owners, {})
        owner.reconfigure(basis, changed)
        assert owner.tasks.generation_id != identity
        assert (
            owner.replacement_plan.peak_bytes["host"]
            > owner.resource_plan.peak_bytes["host"]
        )
        with pytest.raises(RuntimeError, match="stale"):
            next(current)
    # The native basis is caller-owned and survives closing its prepared task.
    assert np.isfinite(basis.evaluate(grid.points[:1])).all()


def test_prepared_rejects_domain_and_forged_map(local_case):
    basis, grid, _ = local_case
    with pytest.raises(ValueError, match="all first"):
        PreparedSpatialGrid(basis, grid, policy=SpatialPolicy(derivatives=((0, 0, 0),)))
    with pytest.raises(ValueError, match="complete"):
        PreparedSpatialGrid(
            basis, grid, policy=SpatialPolicy(derivatives=(*jet_indices(1), (2, 0, 0)))
        )
    tasks = build_spatial_tasks(basis, grid)
    with pytest.raises(ValueError, match="generation"):
        PreparedSpatialGrid(basis, grid, tasks=replace(tasks, generation_id="0" * 64))


@pytest.mark.parametrize("screening", ["off", "absolute_ao_jet"])
def test_fixed_density_xc_scatter_and_trace_variation(local_case, screening):
    from vibeqc_compiler.xc import FixedDensityXC, functional

    basis, grid, density = local_case
    policy = SpatialPolicy(
        region_points=3, screening=screening, cutoff=0 if screening == "off" else 1e-8
    )
    consumer = FixedDensityXC(functional("PBE", spin="polarized"))
    with PreparedSpatialGrid(basis, grid, policy=policy, tile_points=2) as prepared:
        result = consumer.integrate(basis, grid, density, spatial=prepared)
        baseline = consumer.integrate(basis, grid, density, tile_points=7)
        if screening == "off":
            np.testing.assert_allclose(
                result.energy, baseline.energy, atol=1e-11, rtol=1e-10
            )
            np.testing.assert_allclose(
                result.potential, baseline.potential, atol=1e-11, rtol=1e-10
            )
        assert result.approximation_identity == prepared.tasks.identity
        np.testing.assert_array_equal(result.potential, result.potential.swapaxes(1, 2))
        # Offdiagonal, cross-shell and cross-spin perturbations expose omitted
        # local matrix terms and spin/weight factors. The mask stays fixed.
        rng = np.random.default_rng(23)
        direction = rng.normal(size=density.shape)
        direction = 0.02 * (direction + direction.swapaxes(1, 2))
        expected = np.sum(result.potential * direction)
        step = 1e-4
        plus = consumer.integrate(
            basis, grid, density + step * direction, spatial=prepared
        )
        minus = consumer.integrate(
            basis, grid, density - step * direction, spatial=prepared
        )
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step), expected, atol=2e-8, rtol=2e-7
        )


def test_spatial_xc_preserves_independent_fixtures():
    from vibeqc_compiler.dft.fixtures import basis_arguments
    from vibeqc_compiler.xc import FixedDensityXC, functional
    from vibeqc_compiler.xc.integration_fixtures import CASES, load_integration_fixture

    consumer = FixedDensityXC(functional("PBE", spin="polarized"))
    for case in CASES:
        meta, data, grid = load_integration_fixture(case)
        with (
            NativeAO(**basis_arguments(meta)) as basis,
            PreparedSpatialGrid(
                basis, grid, policy=SpatialPolicy(region_points=5), tile_points=3
            ) as prepared,
        ):
            result = consumer.integrate(
                basis, grid, data["density_spin"], spatial=prepared
            )
            np.testing.assert_allclose(
                result.energy, data["PBE_spin_energy"][0], atol=1e-11, rtol=1e-10
            )
            np.testing.assert_allclose(
                result.potential, data["PBE_spin_potential"], atol=1e-11, rtol=1e-10
            )


def test_molecular_grid_materialization_and_stale_geometry():
    from vibeqc_compiler.dft import GridSpec, MolecularGrid

    with NativeAO([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]) as basis:
        grid = MolecularGrid(
            basis.atoms, GridSpec(radial_points=3, angular_polar=2, angular_azimuth=4)
        )
        expected = grid.explicit()
        with PreparedSpatialGrid(
            basis,
            grid,
            tile_points=7,
            resource_budget=ResourceBudget(host_bytes=2 << 20),
        ) as prepared:
            np.testing.assert_array_equal(prepared.grid.points, expected.points)
            np.testing.assert_array_equal(prepared.grid.weights, expected.weights)
            assert prepared.grid.owners == expected.owners
            assert (
                sum(len(t.weights) for t in prepared.iter_features(np.eye(basis.nao)))
                == grid.npoint
            )
        with (
            NativeAO([("H", (0, 0, -0.6)), ("H", (0, 0, 0.7))]) as moved,
            pytest.raises(ValueError, match="stale molecular"),
        ):
            PreparedSpatialGrid(moved, grid)
