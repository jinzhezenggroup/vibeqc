"""Current D/C sources through fixed spatial masks and native CPU XC consumers."""

# Imported pytest fixtures are intentionally reused as test arguments.
# ruff: noqa: F811

import os
from dataclasses import replace

import numpy as np
import pytest
from test_density_cuda import artifact, check, factors, program  # noqa: F401
from test_spatial_execution import local_case  # noqa: F401
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import DensitySource, density_features
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1", reason="finite Slurm CUDA gate"
)


def owner(basis, grid, artifact, **kwargs):
    """Use point and orbital tails with strict local through-f AO supports."""
    return PreparedSpatialGrid(
        basis,
        grid,
        backend="cuda",
        artifact=artifact,
        tile_points=3,
        policy=kwargs.pop(
            "policy",
            SpatialPolicy(region_points=4, screening="absolute_ao_jet", cutoff=1e-8),
        ),
        orbital_capacity=kwargs.pop("orbital_capacity", (basis.nao + 3, 5)),
        orbital_tile=4,
        **kwargs,
    )


def masked_reference(basis, grid, density, tasks, name, spin):
    """Independent global-AO CPU contraction with omitted jets set to zero.

    Keeping global D and output matrices detects incorrect local gathering or
    scatter, including lost cross-shell terms. This is the same discrete mask,
    not an accuracy claim relative to an unscreened molecular calculation.
    """
    evaluator = ContractionProgram(functional(name, spin=spin))
    jets = basis.evaluate(grid.points, 1)
    result = {
        "energy": 0.0,
        "potential": np.zeros((2 if spin == "polarized" else 1, basis.nao, basis.nao)),
    }
    for task in tasks.tasks:
        if not len(task.ao_ids):
            continue
        local = jets[:, task.point_ids].copy()
        omitted = np.ones(basis.nao, dtype=bool)
        omitted[task.ao_ids] = False
        local[:, :, omitted] = 0
        values = evaluator.evaluate(local, density, grid.weights[task.point_ids])
        result["energy"] += values["energy"]
        result["potential"] += values["potential"]
    return result


@pytest.mark.parametrize("screening", ["off", "absolute_ao_jet"])
@pytest.mark.parametrize("empty_spin", [False, True])
def test_spatial_current_features_match_masked_global_d(
    artifact,
    local_case,
    screening,
    empty_spin,
):
    basis, grid, _ = local_case
    counts = (basis.nao + 3, 0 if empty_spin else 5)
    source = factors(basis, counts)
    full = basis.evaluate(grid.points, 1)
    policy = SpatialPolicy(
        region_points=4, screening=screening, cutoff=0 if screening == "off" else 1e-8
    )
    with owner(basis, grid, artifact, policy=policy) as spatial:
        if screening != "off":
            assert any(0 < len(t.ao_ids) < basis.nao for t in spatial.tasks.tasks)
        for route in ("density_matrix", "orbitals"):
            seen = []
            for tile in spatial.iter_features(
                source, stamp=source.stamp, route=route, include_jets=True
            ):
                seen.extend(tile.point_ids)
                masked = full[:, tile.point_ids].copy()
                omitted = np.ones(basis.nao, dtype=bool)
                omitted[tile.ao_ids] = False
                masked[:, :, omitted] = 0
                for key, value in density_features(masked, source.density).items():
                    check(tile.features[key], value)
                check(tile.ao_jets, full[:, tile.point_ids][:, :, tile.ao_ids])
            np.testing.assert_array_equal(np.sort(seen), np.arange(len(grid.points)))
            assert spatial.source_statistics["source_kind"] == route
        assert spatial.source_statistics["occupied_counts"] == counts
        assert spatial.tile_plan == spatial._cuda.plan


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "spin,layout",
    [("polarized", "spin"), ("polarized", "total"), ("unpolarized", "total")],
)
@pytest.mark.parametrize("cap", [128 << 20, 256 << 20])
def test_spatial_xc_current_routes_and_two_budgets(
    artifact,
    local_case,
    name,
    spin,
    layout,
    cap,
):
    basis, grid, _ = local_case
    source = factors(basis, (basis.nao + 3, 5))
    if layout == "total":
        c, f = source.coefficients[0], source.occupations[0]
        source = DensitySource(2 * source.density[0], basis_identity=basis.identity)
        source = source.with_orbitals((c, c), (f, f), stamp=source.stamp)
    counts = tuple(map(len, source.occupations))
    ingredients = ("rho",) if name == "LDA_XC_PW" else ("rho", "gradient", "sigma")
    budget = ResourceBudget(host_bytes=32 << 20, device_bytes=cap)
    with owner(
        basis,
        grid,
        artifact,
        orbital_capacity=counts,
        ingredients=ingredients,
        resource_budget=budget,
    ) as spatial:
        native = program(name, spin)
        expected = masked_reference(
            basis, grid, source.density, spatial.tasks, name, spin
        )
        with PreparedXCContractions(
            native, basis, grid, spatial=spatial, resource_budget=budget
        ) as endpoint:
            for route in ("density_matrix", "orbitals"):
                actual = endpoint.execute(source, stamp=source.stamp, route=route)
                for key in expected:
                    check(actual[key], expected[key])
                stats = endpoint.statistics
                assert stats["source"]["source_kind"] == route
                assert stats["xc_backend"] == "native_cpu"
                assert stats["spatial"]["mask_identity"] == spatial.tasks.identity
                assert stats["spatial"]["potential_scatter_backend"] == "native_cpu"
                assert (
                    stats["native_metrics"]["owned_device_bytes"]
                    == spatial.tile_plan.allocation_bytes
                )
                # No duplicate CUDA allowance when borrowing a spatial owner.
                assert (
                    endpoint.resource_plan.peak_bytes["device"]
                    == spatial.resource_plan.peak_bytes["device"]
                    <= cap
                )
                assert endpoint.resource_plan.peak_bytes["host"] <= budget.host_bytes
            with pytest.raises(MemoryError):
                PreparedXCContractions(
                    native,
                    basis,
                    grid,
                    spatial=spatial,
                    resource_budget=ResourceBudget(host_bytes=1),
                )
            iterator = spatial.iter_features(source, stamp=source.stamp)
            next(iterator)
            endpoint.execute(source, stamp=source.stamp)
            with pytest.raises(RuntimeError, match="stale"):
                next(iterator)
            # Even an identical scientific mask cannot revive a replaced owner.
            spatial.reconfigure(
                basis,
                grid,
                resource_budget=ResourceBudget(
                    host_bytes=32 << 20, device_bytes=256 << 20
                ),
            )
            with pytest.raises(ValueError, match="stale"):
                endpoint.execute(source, stamp=source.stamp)


def test_spatial_source_lifetime_fallback_and_device_leases(
    artifact,
    local_case,
):
    basis, grid, _ = local_case
    source = factors(basis, (basis.nao + 3, 5))
    with owner(basis, grid, artifact) as spatial:
        for stamp in (None, replace(source.stamp, density_generation=7)):
            with pytest.raises(ValueError, match=r"stamp|stale"):
                next(spatial.iter_features(source, stamp=stamp))
        iterator = spatial.iter_features(source, stamp=source.stamp)
        next(iterator)
        newer = DensitySource(
            0.9 * source.density, basis_identity=basis.identity, density_generation=1
        )
        rejected = newer.with_orbitals(
            source.coefficients, source.occupations, stamp=newer.stamp
        )
        assert rejected.fallback_reason == "incompatible_orbitals"
        tiles = list(spatial.iter_features(rejected, stamp=newer.stamp))
        assert spatial.source_statistics["fallback_reason"] == "incompatible_orbitals"
        for tile in tiles:
            jets = basis.evaluate(tile.points, 1, ao_ids=tile.ao_ids)
            expected = density_features(
                jets, newer.density[:, tile.ao_ids[:, None], tile.ao_ids[None, :]]
            )
            for key in expected:
                check(tile.features[key], expected[key])
        with pytest.raises(RuntimeError, match="stale"):
            next(iterator)
        for route in ("density_matrix", "orbitals", "orbitals"):
            expected = np.zeros_like(source.density)
            with spatial.device_tasks(source, stamp=source.stamp, route=route) as tasks:
                for task, ids, lease in tasks:
                    local = np.ones((2, len(task.ao_ids), len(task.ao_ids))) * len(ids)
                    expected[:, task.ao_ids[:, None], task.ao_ids[None, :]] += local
                    check(lease.scatter(local, download=True), expected)
                    with pytest.raises(RuntimeError, match="lease"):
                        spatial.reconfigure(basis, grid)
            with pytest.raises(RuntimeError, match="expired"):
                _ = lease.view
        # Failed replacement budgets leave the original source owner usable.
        with pytest.raises(MemoryError):
            spatial.reconfigure(
                basis, grid, resource_budget=ResourceBudget(host_bytes=1)
            )
        assert len(list(spatial.iter_features(source, stamp=source.stamp))) > 0
        changed = replace(grid, points=grid.points + 0.01)
        spatial.reconfigure(basis, changed, basis_generation=1)
        with pytest.raises(ValueError, match="basis"):
            next(spatial.iter_features(source, stamp=source.stamp))


def test_spatial_empty_masks_and_pruned_abi(artifact, local_case):
    basis, grid, _ = local_case
    source = factors(basis, (basis.nao + 3, 0))
    far = replace(grid, points=grid.points + 100)
    with owner(basis, far, artifact, ingredients=("rho",)) as spatial:
        assert all(len(t.ao_ids) == 0 for t in spatial.tasks.tasks)
        for route in ("density_matrix", "orbitals"):
            tiles = list(
                spatial.iter_features(
                    source, stamp=source.stamp, route=route, include_jets=True, order=0
                )
            )
            assert all(t.ao_jets.shape == (1, len(t.point_ids), 0) for t in tiles)
            assert all(np.count_nonzero(t.features["rho"]) == 0 for t in tiles)
            with PreparedXCContractions(
                program("LDA_XC_PW", "polarized"), basis, far, spatial=spatial
            ) as endpoint:
                got = endpoint.execute(source, stamp=source.stamp, route=route)
                assert got["energy"] == 0 and not got["potential"].any()
                assert endpoint.statistics["scalar_calls"] == 0
                assert endpoint.statistics["total_matrix_products"] == 0
        with (
            pytest.raises(ValueError, match="full feature layout"),
            spatial.device_tasks(source, stamp=source.stamp),
        ):
            pytest.fail("pruned features cannot expose a full-layout device lease")
        with pytest.raises(ValueError, match="ingredients"):
            next(
                spatial.iter_features(source, stamp=source.stamp, ingredients=("tau",))
            )


def test_spatial_density_direction_tracks_current_factors(artifact, local_case):
    basis, grid, _ = local_case
    source = factors(basis, (basis.nao + 3, 5))
    rng = np.random.default_rng(299)
    directions = tuple(rng.normal(size=c.shape) / 10 for c in source.coefficients)
    direction = np.stack(
        [
            (dc * f) @ c.T + (c * f) @ dc.T
            for c, dc, f in zip(
                source.coefficients, directions, source.occupations, strict=True
            )
        ]
    )
    with (
        owner(
            basis, grid, artifact, ingredients=("rho", "gradient", "sigma")
        ) as spatial,
        PreparedXCContractions(
            program("PBE", "polarized"), basis, grid, spatial=spatial
        ) as endpoint,
    ):
        value = endpoint.execute(source, stamp=source.stamp, route="orbitals")
        analytic = np.sum(value["potential"] * direction)
        for step in (1e-3, 3e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                c = tuple(
                    c + sign * step * dc
                    for c, dc in zip(source.coefficients, directions, strict=True)
                )
                d = np.stack(
                    [
                        (cs * f) @ cs.T
                        for cs, f in zip(c, source.occupations, strict=True)
                    ]
                )
                current = DensitySource(
                    d, basis_identity=basis.identity, density_generation=2 + sign
                )
                current = current.with_orbitals(
                    c, source.occupations, stamp=current.stamp
                )
                energies.append(
                    endpoint.execute(current, stamp=current.stamp, route="orbitals")[
                        "energy"
                    ]
                )
            np.testing.assert_allclose(
                (energies[1] - energies[0]) / (2 * step), analytic, atol=3e-7, rtol=3e-6
            )


def test_spatial_capacity_fallback_and_failed_upload_expire_old_execution(
    artifact, local_case, monkeypatch
):
    basis, grid, _ = local_case
    source = factors(basis, (7, 5))
    with owner(basis, grid, artifact, orbital_capacity=(2, 2)) as spatial:
        iterator = spatial.iter_features(source, stamp=source.stamp)
        tile = next(iterator)
        assert (
            spatial.source_statistics["fallback_reason"] == "orbital_capacity_exceeded"
        )
        jets = basis.evaluate(tile.points, 1, ao_ids=tile.ao_ids)
        for key, expected in source.features(
            jets, stamp=source.stamp, route="density_matrix", ao_ids=tile.ao_ids
        ).items():
            check(tile.features[key], expected)
        with pytest.raises(ValueError, match="capacity"):
            next(spatial.iter_features(source, stamp=source.stamp, route="orbitals"))
        native_call = spatial._cuda._call

        def fail_upload(name, *args):
            if name == "grid_cuda_source_v1":
                raise RuntimeError("injected source transport failure")
            return native_call(name, *args)

        monkeypatch.setattr(spatial._cuda, "_call", fail_upload)
        with pytest.raises(RuntimeError, match="transport"):
            next(spatial.iter_features(source, stamp=source.stamp))
        assert spatial.source_statistics == {}
        with pytest.raises(RuntimeError, match="stale"):
            next(iterator)
        monkeypatch.setattr(spatial._cuda, "_call", native_call)
        assert list(spatial.iter_features(source, stamp=source.stamp))


def test_xc_rejects_borrowed_device_tasks_before_waiting_for_cuda_lock(
    artifact, local_case
):
    from concurrent.futures import ThreadPoolExecutor

    basis, grid, _ = local_case
    source = factors(basis, (7, 5))
    with (
        owner(basis, grid, artifact) as spatial,
        PreparedXCContractions(
            program("PBE", "polarized"), basis, grid, spatial=spatial
        ) as endpoint,
        ThreadPoolExecutor(1) as pool,
    ):
        with spatial.device_tasks(source, stamp=source.stamp) as tasks:
            next(tasks)
            future = pool.submit(endpoint.execute, source, stamp=source.stamp)
            try:
                with pytest.raises(RuntimeError, match="lease"):
                    future.result(timeout=3)
            finally:
                # Release CUDA before the outer spatial close even on a
                # timeout, so a regressed implementation fails without hanging
                # the test runner while it shuts down the competing thread.
                tasks.close()
        assert np.isfinite(endpoint.execute(source, stamp=source.stamp)["energy"])
