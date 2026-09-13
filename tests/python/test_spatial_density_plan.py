"""CPU-only preflight and adapter boundary gates for spatial density sources."""

import pytest
from test_spatial_execution import local_case  # noqa: F401
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import DensitySource
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid


def test_orbital_capacity_is_charged_before_cuda_allocation(local_case, monkeypatch):  # noqa: F811
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


def test_cpu_spatial_keeps_explicit_original_density_interface(local_case):  # noqa: F811
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


def test_prepared_output_contract_preserves_legacy_spatial_topology(local_case):  # noqa: F811
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
