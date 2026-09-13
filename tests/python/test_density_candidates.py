# ruff: noqa: F811
"""Executable registrations preserve current D and prepared consumer identity."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_density_cuda import artifact, factors  # noqa: F401
from test_spatial_execution import local_case  # noqa: F401
from test_xc_contractions_native import compare, native_factory  # noqa: F401
from vibeqc.autotune import dft_density_candidates
from vibeqc_compiler.common.resources import ResourceBudget, plan_resources
from vibeqc_compiler.dft import DensitySource, NativeAO
from vibeqc_compiler.dft.cuda import CudaGrid
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.prepared import PreparedXCContractions


def source_for(basis, data):
    """Reuse the independent fixed-density fixture; no production factorization."""
    source = DensitySource(data["density_spin"], basis_identity=basis.identity)
    factors = tuple(np.linalg.cholesky(d) for d in source.density)
    return source.with_orbitals(factors, (np.ones(basis.nao),) * 2, stamp=source.stamp)


@pytest.mark.parametrize("observable", ["energy", "potential", "response", "geometry"])
def test_cpu_registration_keeps_existing_d_consumer(native_factory, observable):
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = source_for(basis, data)
        with PreparedXCContractions(
            native_factory("PBE", observable), basis, grid, tile_points=7
        ) as endpoint:
            options = (
                {"delta_density": 0.01 * source.density}
                if observable == "response"
                else {}
            )
            d, c = dft_density_candidates(
                endpoint, source, stamp=source.stamp, **options
            )
            assert d.available and not c.available
            assert c.reason == (
                "unvalidated_orbital_derivative"
                if observable in ("response", "geometry")
                else "prepared_cpu_orbital_consumer_unavailable"
            )
            assert d.workload is c.workload
            assert d.workload.npoint == len(grid.points)
            assert sum(row[2] for row in d.workload.active_ao_distribution) == len(
                grid.points
            )
            assert d.workload.occupied_counts == (basis.nao,) * 2
            assert d.describe()["promotion"]["eligible"] is False
            expected = endpoint.execute(source.density, **options)
            value, record = d.execute(stamp=source.stamp)
            compare(value, expected)
            assert (
                record["requested_route"]
                == record["executed_route"]
                == "density_matrix"
            )
            saved = record["statistics"]["seconds"]
            endpoint.execute(0.9 * source.density, **options)
            assert record["statistics"]["seconds"] == saved
            with pytest.raises(ValueError, match="unavailable"):
                c.execute(stamp=source.stamp)
            stale = replace(
                source.stamp, density_generation=source.stamp.density_generation + 1
            )
            with pytest.raises(ValueError, match="stale"):
                d.execute(stamp=stale)
            with pytest.raises(ValueError, match="stale"):
                dft_density_candidates(endpoint, source, stamp=stale, **options)
        with pytest.raises(RuntimeError, match="closed"):
            d.execute(stamp=source.stamp)


def test_response_candidate_binds_immutable_direction(native_factory):
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = source_for(basis, data)
        with PreparedXCContractions(
            native_factory("PBE", "response"), basis, grid
        ) as endpoint:
            direction = 0.01 * source.density
            with pytest.raises(ValueError, match="requires a density direction"):
                dft_density_candidates(endpoint, source, stamp=source.stamp)
            original = dft_density_candidates(
                endpoint, source, stamp=source.stamp, delta_density=direction
            )[0]
            expected = endpoint.execute(source.density, delta_density=direction)
            direction *= (
                -2
            )  # Caller changes cannot rewrite the registered perturbation.
            changed = dft_density_candidates(
                endpoint, source, stamp=source.stamp, delta_density=direction
            )[0]
            assert original.workload.identity != changed.workload.identity
            assert (
                original.workload.density_direction
                != changed.workload.density_direction
            )
            value, execution = original.execute(stamp=source.stamp)
            compare(value, expected)
            assert execution["workload"] == original.workload.identity
            value, _ = changed.execute(stamp=source.stamp)
            np.testing.assert_allclose(
                value["response"], -2 * expected["response"], atol=1e-11, rtol=1e-10
            )
            with pytest.raises(ValueError):
                original._delta_density.setflags(write=True)


def test_registration_rejects_changed_resources_and_wrong_basis(native_factory):
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = source_for(basis, data)
        with PreparedXCContractions(
            native_factory("LDA_XC_PW", "potential"), basis, grid
        ) as endpoint:
            with pytest.raises(ValueError, match="requires a response request"):
                dft_density_candidates(
                    endpoint, source, stamp=source.stamp, delta_density=source.density
                )
            wrong = DensitySource(source.density, basis_identity="a" * 64)
            with pytest.raises(ValueError, match="AO basis"):
                dft_density_candidates(endpoint, wrong, stamp=wrong.stamp)
            candidate = dft_density_candidates(endpoint, source, stamp=source.stamp)[0]
            endpoint.resource_plan = plan_resources(
                endpoint.resource_plan.requests, ResourceBudget(host_bytes=256 << 20)
            )
            with pytest.raises(ValueError, match="resource identity"):
                candidate.execute(stamp=source.stamp)


GPU = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1", reason="finite Slurm GPU gate"
)


@GPU
@pytest.mark.parametrize("capacity", [None, (1, 1), (2, 2)])
def test_gpu_registered_execution_and_capacity_fallback(
    artifact, native_factory, capacity
):
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = source_for(basis, data)
        with (
            CudaGrid(
                basis,
                artifact,
                order=1,
                tile_points=7,
                orbital_tile=1,
                orbital_capacity=capacity,
                ingredients=("rho", "gradient", "sigma"),
            ) as cuda,
            PreparedXCContractions(
                native_factory("PBE", "potential"), basis, grid, density_grid=cuda
            ) as endpoint,
        ):
            d, c = dft_density_candidates(endpoint, source, stamp=source.stamp)
            actual_d, record_d = d.execute(stamp=source.stamp)
            if capacity == (2, 2):
                assert c.available
                actual_c, record_c = c.execute(stamp=source.stamp)
                compare(actual_c, actual_d)
                assert record_c["executed_route"] == "orbitals"
                assert record_c["statistics"]["source"]["factor_packing_seconds"] > 0
                assert record_c["statistics"]["orbital_matrix_products"] > 0
            else:
                assert not c.available and c.reason == "orbital_capacity_exceeded"
                with pytest.raises(ValueError, match="capacity"):
                    c.execute(stamp=source.stamp)
            assert record_d["executed_route"] == "density_matrix"
            assert (
                d.workload.device_bytes == endpoint.resource_plan.peak_bytes["device"]
            )
            original = DensitySource(
                source.density, basis_identity=basis.identity, density_generation=1
            )
            d2, c2 = dft_density_candidates(endpoint, original, stamp=original.stamp)
            assert not c2.available and c2.reason == "missing_orbitals"
            value, _ = d2.execute(stamp=original.stamp)
            compare(value, actual_d)
            response = DensitySource(
                source.density, basis_identity=basis.identity, role="response"
            )
            assert (
                dft_density_candidates(endpoint, response, stamp=response.stamp)[
                    1
                ].reason
                == "response_density"
            )
            with pytest.raises(ValueError, match="stale"):
                d.execute(stamp=original.stamp)
            # Interleaving other sources does not change the bound original.
            replay, _ = d.execute(stamp=source.stamp)
            compare(replay, actual_d)


@GPU
def test_gpu_candidate_propagates_failed_upload(artifact, native_factory, monkeypatch):
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        source = source_for(basis, data)
        with (
            CudaGrid(
                basis,
                artifact,
                order=0,
                tile_points=7,
                orbital_capacity=(2, 2),
                ingredients=("rho",),
            ) as cuda,
            PreparedXCContractions(
                native_factory("LDA_XC_PW", "potential"), basis, grid, density_grid=cuda
            ) as endpoint,
        ):
            candidate = dft_density_candidates(endpoint, source, stamp=source.stamp)[1]

            def failed(*args, **kwargs):
                raise RuntimeError("injected source upload failure")

            monkeypatch.setattr(cuda, "set_source", failed)
            with pytest.raises(RuntimeError, match="upload failure"):
                candidate.execute(stamp=source.stamp)


@GPU
def test_local_candidate_counts_leases_and_same_mask_replacement(
    artifact, native_factory, local_case
):
    basis, grid, _ = local_case
    source = factors(basis, (5, 3))
    with (
        PreparedSpatialGrid(
            basis,
            grid,
            backend="cuda",
            artifact=artifact,
            tile_points=7,
            orbital_capacity=(5, 3),
            ingredients=("rho", "gradient", "sigma", "tau"),
            policy=SpatialPolicy(
                region_points=3, screening="absolute_ao_jet", cutoff=1e-8
            ),
            resource_budget=ResourceBudget(host_bytes=64 << 20, device_bytes=256 << 20),
        ) as spatial,
        PreparedXCContractions(
            native_factory("LDA_XC_PW", "potential"), basis, grid, spatial=spatial
        ) as endpoint,
    ):
        d, c = dft_density_candidates(endpoint, source, stamp=source.stamp)
        assert any(0 < row[0] < basis.nao for row in d.workload.active_ao_distribution)
        assert sum(row[2] for row in d.workload.active_ao_distribution) == len(
            grid.points
        )
        expected, _ = d.execute(stamp=source.stamp)
        value, _ = c.execute(stamp=source.stamp)
        compare(value, expected)
        with spatial.device_tasks(source, stamp=source.stamp) as tasks:
            next(tasks)
            with pytest.raises(RuntimeError, match="lease"):
                dft_density_candidates(endpoint, source, stamp=source.stamp)
        previous_mask = spatial.tasks.identity
        spatial.reconfigure(basis, grid, orbital_capacity=(7, 4))
        assert spatial.tasks.identity == previous_mask
        with pytest.raises(ValueError, match="stale"):
            c.execute(stamp=source.stamp)
