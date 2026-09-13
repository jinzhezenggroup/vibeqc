"""CPU-only preflight and adapter boundary gates for spatial density sources."""

# Imported pytest fixtures are intentionally reused as test arguments.
# ruff: noqa: F811

import pytest
from test_spatial_execution import local_case  # noqa: F401
from test_xc_contractions_native import native_factory  # noqa: F401
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import DensitySource
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid


def test_orbital_capacity_is_charged_before_cuda_allocation(local_case, monkeypatch):
    basis, grid, _ = local_case

    def forbidden(*args, **kwargs):
        pytest.fail("allocated CUDA before the combined orbital-capacity preflight")

    monkeypatch.setattr("vibeqc_compiler.dft.spatial_prepared.CudaGrid", forbidden)
    with pytest.raises(MemoryError):
        PreparedSpatialGrid(
            basis,
            grid,
            backend="cuda",
            artifact=object(),
            orbital_capacity=(1 << 20, 1 << 20),
            orbital_tile=3,
            resource_budget=ResourceBudget(host_bytes=32 << 20, device_bytes=128 << 20),
        )


def test_cpu_spatial_keeps_explicit_original_density_interface(local_case):
    basis, grid, density = local_case
    source = DensitySource(density, basis_identity=basis.identity)
    with PreparedSpatialGrid(basis, grid) as spatial:
        with pytest.raises(TypeError, match="requires CUDA"):
            next(spatial.iter_features(source, stamp=source.stamp))
        with pytest.raises(ValueError, match="requires DensitySource"):
            next(spatial.iter_features(density, route="orbitals"))
        assert len(list(spatial.iter_features(density))) > 0
    with pytest.raises(ValueError, match="CUDA orbital capacity"):
        PreparedSpatialGrid(basis, grid, orbital_capacity=(3, 2))


def test_prepared_output_contract_preserves_legacy_spatial_topology(local_case):
    basis, grid, _ = local_case
    policy = SpatialPolicy(region_points=4)
    with (
        PreparedSpatialGrid(basis, grid, policy=policy) as full,
        PreparedSpatialGrid(basis, grid, policy=policy, ingredients=("rho",)) as pruned,
    ):
        # Existing #234 publication identity remains reproducible. New output
        # contracts have their own resource identity without changing the mask.
        import json

        request = next(
            r for r in full.resource_plan.requests if r.name == "spatial_execution"
        )
        topology = json.loads(request.identity.topology)
        assert set(topology) == {"points", "ao", "active", "jets"}
        assert full.tasks.identity == pruned.tasks.identity
        assert full.resource_plan.identity != pruned.resource_plan.identity


def test_xc_construction_serializes_spatial_snapshot_and_replacement(
    local_case, native_factory, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from threading import Event

    from vibeqc_compiler.xc.contracts import DiscreteEnergyContract
    from vibeqc_compiler.xc.prepared import PreparedXCContractions

    basis, grid, density = local_case
    native = native_factory("PBE", "potential")
    entered, release = Event(), Event()
    ao_order = DiscreteEnergyContract.ao_order.fget

    def pause_after_geometry_validation(contract):
        entered.set()
        assert release.wait(3), "constructor did not release its snapshot probe"
        return ao_order(contract)

    with PreparedSpatialGrid(basis, grid) as spatial, ThreadPoolExecutor(2) as pool:
        monkeypatch.setattr(
            DiscreteEnergyContract,
            "ao_order",
            property(pause_after_geometry_validation),
        )
        construction = pool.submit(
            PreparedXCContractions, native, basis, grid, spatial=spatial
        )
        assert entered.wait(3)
        try:
            # Probe from a different thread at the old-grid/new-owner race
            # boundary. No sleep or successful reconfiguration is needed to
            # detect an unprotected snapshot.
            acquired = spatial._lock.acquire(blocking=False)
            if acquired:
                spatial._lock.release()
            assert not acquired, "spatial snapshot is exposed during construction"
            replacement = pool.submit(
                spatial.reconfigure, basis, replace(grid, points=grid.points + 0.01)
            )
        finally:
            release.set()
        endpoint = construction.result(timeout=3)
        replacement.result(timeout=3)
        try:
            with pytest.raises(ValueError, match="stale"):
                endpoint.execute(density)
        finally:
            endpoint.close()


def test_same_mask_cpu_replacement_invalidates_borrowed_resource_plan(
    local_case, native_factory
):
    from vibeqc_compiler.xc.prepared import PreparedXCContractions

    basis, grid, density = local_case
    with (
        PreparedSpatialGrid(basis, grid, tile_points=1) as spatial,
        PreparedXCContractions(
            native_factory("PBE", "potential"), basis, grid, spatial=spatial
        ) as endpoint,
    ):
        mask = spatial.tasks.identity
        spatial.reconfigure(basis, grid, tile_points=64)
        assert spatial.tasks.identity == mask
        with pytest.raises(ValueError, match="resource contract"):
            endpoint.execute(density)
