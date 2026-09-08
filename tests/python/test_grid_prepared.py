"""State, ragged-offset and allocation regressions for the internal grid plan."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_dft import GridSpec, NativeAO
from tools.vibeqc_dft.fixtures import basis_arguments, load_fixture
from tools.vibeqc_dft.plan import plan_tiles
from tools.vibeqc_dft.prepared import PreparedGrid, PreparedGridBatch


def arguments(name="h2"):
    meta, data = load_fixture(name)
    return {**basis_arguments(meta), "spec": GridSpec(3, 3, 6), "tile_points": 7}, data


def test_replay_and_invalidated_iterator_density_and_geometry():
    args, data = arguments()
    with PreparedGrid(**args) as plan:
        first = plan.integrate(data["density"])
        np.testing.assert_array_equal(
            plan.integrate(data["density"])["electrons"], first["electrons"]
        )
        iterator = plan.iter_features(data["density"])
        next(iterator)
        # The second request replaces resident D; the old iterator must fail,
        # rather than quietly returning a mixture of two density fields.
        other = plan.iter_features(data["density"] * 2)
        next(other)
        with pytest.raises(RuntimeError, match="stale"):
            next(iterator)
        plan.reconfigure(coordinates=[[0, 0, 0], [0.4, 0.2, 1.5]])
        with pytest.raises(RuntimeError, match="stale"):
            next(other)
        changed = plan.integrate(data["density"])
        assert changed["generation"] == 1 and changed["identity"] != first["identity"]
        with PreparedGrid(
            **{**args, "atoms": [(1, [0, 0, 0]), (1, [0.4, 0.2, 1.5])]}
        ) as fresh:
            np.testing.assert_allclose(
                changed["electrons"],
                fresh.integrate(data["density"])["electrons"],
                atol=1e-13,
            )
        before = plan.identity
        with pytest.raises((ValueError, RuntimeError)):
            plan.reconfigure(coordinates=[[np.nan, 0, 0], [1, 0, 0]])
        assert plan.identity == before
        np.testing.assert_array_equal(
            plan.integrate(data["density"])["electrons"], changed["electrons"]
        )


def test_replacement_overlap_is_budgeted_before_device_allocation():
    args, _ = arguments()
    with PreparedGrid(**args) as measured:
        peak = measured.plan.peak_bytes
    with PreparedGrid(**args, budget_bytes=peak) as plan:
        with pytest.raises(ValueError, match="budget"):
            plan.reconfigure(coordinates=[[0.01, 0, 0], [0.1, 0.2, 1.4]])
        assert plan.generation == 0
        plan.reconfigure(
            coordinates=[[0.01, 0, 0], [0.1, 0.2, 1.4]], budget_bytes=2 * peak
        )
        assert plan.diagnostics()["replacement_peak_bytes"] == 2 * peak
        assert plan.diagnostics()["peak_bytes"] == peak


def test_basis_charge_spin_and_rule_invalidation():
    args, _ = arguments()
    with PreparedGrid(**args) as plan:
        identities = [plan.identity]
        # Repeated same-topology preparation still has a fresh generation.
        plan.reconfigure()
        identities.append(plan.identity)
        plan.reconfigure(charge=1, multiplicity=2)
        identities.append(plan.identity)
        plan.reconfigure(spec=replace(plan.grid.spec, angular_polar=4))
        identities.append(plan.identity)
        shells = list(plan._settings["basis"])
        primitives = list(shells[0].primitives)
        primitives[0] = replace(
            primitives[0], coefficient=primitives[0].coefficient * 1.2
        )
        shells[0] = replace(shells[0], primitives=tuple(primitives))
        plan.reconfigure(basis=tuple(shells))
        identities.append(plan.identity)
        assert len(set(identities)) == len(identities)


def test_ragged_offsets_failure_isolation_and_update_budget():
    args, a = arguments("h2")
    other, b = arguments("water")
    with PreparedGridBatch([args, other]) as batch:
        records = batch.execute([a["density"], b["density"]])
        assert [x["status"] for x in records] == ["pass", "pass"]
        assert [x["point_begin"] for x in records] == [0, 108]
        assert [x["ao_begin"] for x in records] == [0, 2]
        failures = batch.execute([np.array([1]), b["density"]])
        assert [x["status"] for x in failures] == ["fail", "pass"]
        np.testing.assert_array_equal(
            failures[1]["result"]["electrons"], records[1]["result"]["electrons"]
        )
        invalid_type = batch.execute([{"invalid": 1}, b["density"]])
        assert [x["status"] for x in invalid_type] == ["fail", "pass"]
        batch.reconfigure(0, spec=GridSpec(4, 3, 6))
        assert batch.execute([a["density"], b["density"]])[1]["point_begin"] == 144
        assert batch.peak_bytes == sum(i.plan.peak_bytes for i in batch._items)
    with pytest.raises(ValueError, match="budget"):
        PreparedGridBatch([args, other], budget_bytes=1)


def test_budget_scales_with_tile_and_jets_not_molecular_grid_size():
    args, _ = arguments("f_spherical")
    with NativeAO(
        **{k: v for k, v in args.items() if k not in ("spec", "tile_points")}
    ) as basis:
        one = plan_tiles(basis, tile_points=1, order=1)
        many = plan_tiles(basis, tile_points=97, order=3)
        assert many.host_bytes > one.host_bytes and one.device_bytes == 0
        cuda = plan_tiles(basis, backend="cuda", tile_points=97, order=3)
        assert cuda.device_bytes == cuda.allocation_bytes + (96 << 20)
        with pytest.raises(ValueError, match="budget"):
            plan_tiles(
                basis, backend="cuda", tile_points=2**31 - 1, budget_bytes=256 << 20
            )


def test_unsupported_public_basis_and_invalid_native_arrays_fail():
    from vibeqc import Primitive, Shell

    with pytest.raises(ValueError, match="through f"):
        NativeAO([(1, (0, 0, 0))], (Shell(0, 4, (Primitive(1, 1),)),), multiplicity=2)
    args, _ = arguments()
    with NativeAO(
        **{k: v for k, v in args.items() if k not in ("spec", "tile_points")}
    ) as basis:
        for points in ([[np.nan, 0, 0]], [[np.inf, 0, 0]], [[1j, 0, 0]]):
            with pytest.raises(ValueError):
                basis.evaluate(points)
