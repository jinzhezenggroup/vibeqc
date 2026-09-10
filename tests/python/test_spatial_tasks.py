"""Spatial mask/reordering invariants against an unscreened identical grid."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.spatial import SpatialPolicy, build_spatial_tasks


@pytest.fixture
def fixture():
    atoms = [("H", (-2, 0, 0)), ("H", (2, 0, 0))]
    shells = tuple(
        Shell(atom, angular, (Primitive(2.0, 1.0), Primitive(4.0, -0.1)))
        for atom in (0, 1)
        for angular in (0, 1, 3)
    )
    rng = np.random.default_rng(234)
    points = np.concatenate(
        (
            rng.normal(size=(11, 3)) * 0.08 + [-2, 0, 0],
            rng.normal(size=(13, 3)) * 0.08 + [2, 0, 0],
            [[19, 18, 17], [20, 18, 17]],
        )
    )
    grid = ExplicitGrid(
        points,
        rng.random(len(points)),
        tuple(i % 2 for i in range(len(points))),
        {"name": "spatial-test"},
    )
    with NativeAO(atoms, basis=shells, representation="spherical") as basis:
        yield basis, grid


def test_screening_off_partition_is_only_an_equivalent_reordering(fixture):
    basis, grid = fixture
    expected = basis.evaluate(grid.points, order=1)
    generations = set()
    for region in (1, 3, 9, 128):
        tasks = build_spatial_tasks(
            basis, grid, policy=SpatialPolicy(region_points=region)
        )
        tasks.validate(basis, grid)
        ids = np.concatenate([t.point_ids for t in tasks.tasks])
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(grid.points)))
        actual = np.empty_like(expected)
        weights, owners = (
            np.empty_like(grid.weights),
            np.empty(len(grid.points), dtype=int),
        )
        for task in tasks.tasks:
            assert len(task.point_ids) <= region
            np.testing.assert_array_equal(task.ao_ids, np.arange(basis.nao))
            assert task.discarded_count == 0
            actual[:, task.point_ids] = basis.evaluate(
                grid.points[task.point_ids], order=1
            )
            weights[task.point_ids] = grid.weights[task.point_ids]
            owners[task.point_ids] = np.array(grid.owners)[task.point_ids]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(weights, grid.weights)
        np.testing.assert_array_equal(owners, grid.owners)
        assert tasks.numeric_bytes <= tasks.resource_plan.peak_bytes["host"]
        repeated = build_spatial_tasks(basis, grid, policy=tasks.policy)
        assert [t.identity for t in repeated.tasks] == [t.identity for t in tasks.tasks]
        generations.add(tasks.generation_id)
    assert len(generations) == 4


def test_fixed_mask_is_discovered_without_evaluating_the_dense_ao_grid(
    fixture, monkeypatch
):
    basis, grid = fixture
    original = basis.evaluate

    def forbidden(*args, **kwargs):
        raise AssertionError("mask discovery must not evaluate AO values")

    monkeypatch.setattr(basis, "evaluate", forbidden)
    policy = SpatialPolicy(
        region_points=2,
        derivatives=jet_indices(2),
        screening="absolute_ao_jet",
        cutoff=1e-8,
    )
    tasks = build_spatial_tasks(basis, grid, policy=policy)
    assert any(len(t.ao_ids) < basis.nao for t in tasks.tasks)
    assert any(len(t.ao_ids) == 0 for t in tasks.tasks)
    for task in tasks.tasks:
        actual = original(grid.points[task.point_ids], order=2)
        omitted = np.ones(basis.nao, dtype=bool)
        omitted[task.ao_ids] = False
        if omitted.any():
            assert np.max(np.abs(actual[:, :, omitted])) <= policy.cutoff
        assert np.all(task.discarded_max <= policy.cutoff)
        assert np.all(np.diff(task.ao_ids) > 0)
        with pytest.raises(ValueError):
            task.ao_ids.setflags(write=True)


def test_stale_grid_weights_point_order_and_basis_fail(fixture):
    basis, grid = fixture
    tasks = build_spatial_tasks(basis, grid)
    for changed in (
        ExplicitGrid(grid.points, grid.weights * 2, grid.owners, {}),
        ExplicitGrid(grid.points[::-1], grid.weights[::-1], grid.owners[::-1], {}),
    ):
        with pytest.raises(ValueError, match="stale"):
            tasks.validate(basis, changed)
    with (
        NativeAO(
            [("H", (-1.9, 0, 0)), ("H", (2, 0, 0))],
            basis=basis.shells,
            representation="spherical",
        ) as moved,
        pytest.raises(ValueError, match="stale"),
    ):
        tasks.validate(moved, grid)
    # Density does not participate in a purely geometric task generation.
    assert tasks.generation_id == build_spatial_tasks(basis, grid).generation_id


def test_grid_and_atom_permutations_preserve_geometric_masks(fixture):
    basis, grid = fixture
    policy = SpatialPolicy(region_points=3, screening="absolute_ao_jet", cutoff=1e-8)
    original = build_spatial_tasks(basis, grid, policy=policy)
    permutation = np.random.default_rng(31).permutation(len(grid.points))
    shuffled_grid = ExplicitGrid(
        grid.points[permutation],
        grid.weights[permutation],
        tuple(grid.owners[i] for i in permutation),
        {},
    )
    shuffled = build_spatial_tasks(basis, shuffled_grid, policy=policy)
    shuffled.validate(basis, shuffled_grid)
    assert shuffled.generation_id != original.generation_id
    for left, right in zip(original.tasks, shuffled.tasks, strict=True):
        np.testing.assert_array_equal(
            np.sort(left.point_ids), np.sort(permutation[right.point_ids])
        )
        np.testing.assert_array_equal(left.ao_ids, right.ao_ids)
        np.testing.assert_array_equal(left.bounds, right.bounds)
        np.testing.assert_array_equal(left.discarded_max, right.discarded_max)

    # Relabel atoms and owner IDs together while retaining the physical AO
    # order. Geometric masks must not depend on numeric atom labels.
    shells = tuple(replace(s, atom_index=1 - s.atom_index) for s in basis.shells)
    relabelled_grid = ExplicitGrid(
        grid.points, grid.weights, tuple(1 - i for i in grid.owners), {}
    )
    with NativeAO(
        basis.atoms[::-1], basis=shells, representation="spherical"
    ) as relabelled_basis:
        relabelled = build_spatial_tasks(
            relabelled_basis, relabelled_grid, policy=policy
        )
        relabelled.validate(relabelled_basis, relabelled_grid)
        assert relabelled.generation_id != original.generation_id
        for left, right in zip(original.tasks, relabelled.tasks, strict=True):
            np.testing.assert_array_equal(left.point_ids, right.point_ids)
            np.testing.assert_array_equal(left.ao_ids, right.ao_ids)
            np.testing.assert_array_equal(left.discarded_max, right.discarded_max)


def test_shared_budget_preflight_precedes_envelope_construction(fixture, monkeypatch):
    from vibeqc_compiler.dft import spatial

    basis, grid = fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must precede mask construction")

    monkeypatch.setattr(spatial, "ao_region_envelopes", forbidden)
    with pytest.raises(MemoryError):
        build_spatial_tasks(
            basis,
            grid,
            policy=SpatialPolicy(screening="absolute_ao_jet", cutoff=1e-8),
            budget=ResourceBudget(host_bytes=1),
        )
    for budget in (1 << 20, 2 << 20):
        result = build_spatial_tasks(
            basis, grid, budget=ResourceBudget(host_bytes=budget)
        )
        assert result.resource_plan.peak_bytes["host"] <= budget


def test_empty_grid_and_invalid_owner(fixture):
    basis, _ = fixture
    grid = ExplicitGrid(np.empty((0, 3)), np.empty(0), (), {})
    assert build_spatial_tasks(basis, grid).tasks == ()
    invalid = ExplicitGrid(np.zeros((1, 3)), np.ones(1), (basis.natom,), {})
    with pytest.raises(ValueError, match="owner"):
        build_spatial_tasks(basis, invalid)


def test_mutated_maps_and_relabelled_generations_fail(fixture):
    basis, grid = fixture
    tasks = build_spatial_tasks(basis, grid, policy=SpatialPolicy(region_points=4))
    first = tasks.tasks[0]
    with pytest.raises(ValueError, match="integers"):
        replace(first, ao_ids=np.array([0.5]))
    with pytest.raises(ValueError, match="duplicate"):
        replace(first, point_ids=np.array([0, 0]))
    with pytest.raises(ValueError, match="every grid point"):
        replace(tasks, tasks=(*tasks.tasks, first)).validate(basis, grid)
    wrong_shells = replace(first, active_shell_ids=np.empty(0, dtype=np.int64))
    with pytest.raises(ValueError, match="shell map"):
        replace(tasks, tasks=(wrong_shells, *tasks.tasks[1:])).validate(basis, grid)
    changed = ExplicitGrid(grid.points, grid.weights * 2, grid.owners, {})
    with pytest.raises(ValueError, match="generation"):
        replace(tasks, grid_identity=changed.identity).validate(basis, changed)


def test_forged_screening_certificate_cannot_drop_large_ao(fixture):
    basis, grid = fixture
    policy = SpatialPolicy(region_points=4, screening="absolute_ao_jet", cutoff=1e-8)
    tasks = build_spatial_tasks(basis, grid, policy=policy)
    first = tasks.tasks[0]
    forged = replace(
        first,
        ao_ids=np.array([], dtype=int),
        active_shell_ids=np.array([], dtype=int),
        discarded_count=basis.nao,
        discarded_max=np.zeros(len(first.derivatives)),
    )
    altered = replace(tasks, tasks=(forged, *tasks.tasks[1:]))
    assert altered.identity != tasks.identity
    with pytest.raises(ValueError, match="certified"):
        altered.validate(basis, grid)
