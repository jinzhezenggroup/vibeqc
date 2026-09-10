"""Opt-in local-dense CUDA parity and borrowed-buffer lifetime gates."""

import os
from pathlib import Path

import numpy as np
import pytest
from test_spatial_execution import local_case  # noqa: F401
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from vibeqc_compiler.dft.features import density_features
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in finite Slurm CUDA gate",
)


@pytest.fixture(scope="module")
def artifact():
    return compile_cuda(
        CudaCompilerAdapter(
            Path("/group/software/cuda-12.9.1/bin/nvcc"), cuda_target_info("sm_120")
        ),
        Path(".artifacts/spatial-cuda-cache"),
    )


def test_selected_cuda_jets_and_complete_density(artifact, local_case):  # noqa: F811
    basis, grid, density = local_case
    full = basis.evaluate(grid.points[:7], order=3)
    with CudaGrid(
        basis, artifact, order=3, tile_points=7, active_ao_capacity=basis.nao
    ) as cuda:
        cuda.set_density(density)
        for ids in (
            np.arange(0, basis.nao, 3),
            np.arange(basis.nao),
            np.array([], dtype=int),
        ):
            got = cuda.evaluate(grid.points[:7], ao_ids=ids, download_jets=True)
            np.testing.assert_allclose(
                got["ao_jets"], full[:, :, ids], atol=1e-11, rtol=1e-10
            )
            expected = density_features(
                full[:, :, ids], density[:, ids[:, None], ids[None, :]]
            )
            for key in expected:
                np.testing.assert_allclose(
                    got[key], expected[key], atol=1e-11, rtol=1e-10
                )


def test_device_lease_scatter_multiple_maps_and_expiry(artifact, local_case):  # noqa: F811
    basis, grid, density = local_case
    expected = np.zeros_like(density)
    rng = np.random.default_rng(2342)
    with CudaGrid(
        basis, artifact, order=1, tile_points=7, active_ao_capacity=basis.nao
    ) as cuda:
        cuda.set_density(density)
        for index, ids in enumerate(
            (np.arange(0, basis.nao, 2), np.arange(1, basis.nao, 3))
        ):
            local = rng.normal(size=(2, len(ids), len(ids)))
            local = local + local.swapaxes(1, 2)
            expected[:, ids[:, None], ids[None, :]] += local
            with cuda.task(grid.points[:7], ids) as lease:
                assert lease.view.version == 1 and lease.view.nactive == len(ids)
                assert bool(lease.view.features) and bool(lease.view.ao)
                with pytest.raises(RuntimeError, match="leased"):
                    cuda.set_density(density)
                with pytest.raises(RuntimeError, match="leased"):
                    cuda.close()
                actual = lease.scatter(local, reset=index == 0, download=True)
                np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
                np.testing.assert_array_equal(actual, actual.swapaxes(1, 2))
            with pytest.raises(RuntimeError, match="expired"):
                lease.scatter(local)


def test_prepared_cuda_fixed_masks_budgets_and_partial_device_iteration(
    artifact,
    local_case,  # noqa: F811
):
    basis, grid, density = local_case
    policy = SpatialPolicy(region_points=3, screening="absolute_ao_jet", cutoff=1e-8)
    for budget in (160 << 20, 256 << 20):
        with PreparedSpatialGrid(
            basis,
            grid,
            policy=policy,
            backend="cuda",
            artifact=artifact,
            tile_points=2,
            resource_budget=ResourceBudget(host_bytes=4 << 20, device_bytes=budget),
        ) as cuda:
            with PreparedSpatialGrid(basis, grid, policy=policy, tile_points=2) as cpu:
                for got, want in zip(
                    cuda.iter_features(density), cpu.iter_features(density), strict=True
                ):
                    for key in want.features:
                        np.testing.assert_allclose(
                            got.features[key],
                            want.features[key],
                            atol=1e-11,
                            rtol=1e-10,
                        )
            with cuda.device_tasks(density) as iterator:
                _, _, lease = next(iterator)
                with pytest.raises(RuntimeError, match="lease"):
                    cuda.reconfigure(basis, grid)
            with pytest.raises(RuntimeError, match="expired"):
                _ = lease.view
            assert len(list(cuda.iter_features(density))) > 0
