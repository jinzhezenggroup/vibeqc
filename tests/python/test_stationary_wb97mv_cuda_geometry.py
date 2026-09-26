"""Opt-in real CUDA semilocal geometry gate against independent Libxc energies.

Run with VIBEQC_TEST_RANGE_CUDA=1 inside Slurm, with nvcc on PATH and a native
VibeQC library available for basis preparation. This qualifies the semilocal
slice only; SR/LR exchange and VV10 are intentionally absent from this energy.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_cuda_b97m_geometry_matches_independent_energy_differences(
    tmp_path: Path, spin: str
) -> None:
    """Exercise actual tau/gradient pullbacks for fixed positive density matrices.

    Grid points move with their owner atoms while their explicit weights remain
    fixed. Separate center and point differences catch cancellation between the
    two sources, as well as spin and kinetic-density factors of two.
    """
    if os.environ.get("VIBEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("native CUDA compiler unavailable")
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("independent oracle is pinned to Libxc 7.0.0")
    from pyscf import gto
    from pyscf.dft import numint
    from vibeqc._stationary_cuda import _CudaSources
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.dft import NativeAO
    from vibeqc_compiler.dft.cuda import CudaGrid
    from vibeqc_compiler.dft.cuda import compile_cuda as compile_grid
    from vibeqc_compiler.method import resolve_method
    from vibeqc_compiler.method.stationary_cuda import compile_stationary_cuda
    from vibeqc_compiler.method.stationary_gradient import (
        StationaryGradientPlan,
        StationaryMeanField,
    )

    coordinates = np.array(((0.13, -0.17, -0.71), (-0.09, 0.11, 0.79)))
    # LiH supplies both s and p AOs in the admitted public basis domain.
    symbols = ("Li", "H")
    atoms = list(zip(symbols, coordinates, strict=True))
    points = np.array(
        (
            (0.31, 0.22, -0.41),
            (-0.27, 0.41, 0.65),
            (0.82, -0.31, 0.13),
            (-0.63, -0.32, -0.19),
            (0.24, 0.72, 0.31),
        )
    )
    owners = np.array((0, 1, 0, 1, 0), dtype=np.int64)
    weights = np.array((0.13, 0.27, 0.41, 0.19, 0.31))
    compiler = CudaCompilerAdapter(Path(nvcc), cuda_target_info("sm_120"))
    method = resolve_method("WB97M-V", spin=spin)
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
    )
    # No integral derivative is part of this isolated slice. Fail closed if
    # geometry execution ever calls into that unrelated provider.
    primitive = """#include <cuda_runtime.h>
__device__ bool first_derivative(unsigned, const double*, const double*, double*) {
  return false;
}
"""
    artifact = compile_stationary_cuda(
        primitive,
        functional=4,
        plan=plan,
        iterations=3,
        compiler=compiler,
        cache=tmp_path,
    )
    with NativeAO(atoms) as basis:
        rng = np.random.default_rng(1389)
        matrices = rng.normal(size=(plan.spin_blocks, basis.nao, basis.nao))
        density = np.ascontiguousarray(
            0.15 * matrices @ matrices.transpose(0, 2, 1) + 0.1 * np.eye(basis.nao)
        )
        with (
            _CudaSources(
                basis,
                artifact,
                compiler,
                0,
                len(points),
                1,
                16 << 20,
                spin_blocks=plan.spin_blocks,
            ) as sources,
            CudaGrid(
                basis,
                compile_grid(compiler, tmp_path),
                order=2,
                tile_points=len(points),
                active_ao_capacity=basis.nao,
                ingredients=("rho", "gradient", "tau"),
            ) as grid,
        ):
            grid.set_density(density[0] if plan.spin_blocks == 1 else density)
            sources.reset(1e-12, density, np.zeros_like(density))
            with grid.xc_task(points, np.arange(basis.nao), "WB97M-V") as task:
                sources.geometry(
                    task, owners, weights, np.zeros_like(weights), functional=4
                )
            actual = sources.finish()

    def energy(centers: np.ndarray, grid_points: np.ndarray) -> float:
        mol = gto.M(
            atom=list(zip(symbols, centers, strict=True)),
            basis="sto-3g",
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        ao = numint.eval_ao(mol, grid_points, deriv=1)
        rho = np.array(
            [
                numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=False)
                for dm in density
            ]
        )
        exc = libxc.eval_xc(
            "HYB_MGGA_XC_WB97M_V",
            rho[0] if plan.spin_blocks == 1 else rho,
            spin=plan.spin_blocks - 1,
            deriv=0,
        )[0]
        return float(weights @ (exc * rho[:, 0].sum(axis=0)))

    expected_basis = np.zeros((len(atoms), 3))
    expected_points = np.zeros_like(expected_basis)
    step = 2e-5
    for atom in range(len(atoms)):
        for axis in range(3):
            displacement = np.zeros_like(coordinates)
            displacement[atom, axis] = step
            motion = displacement[owners]
            expected_basis[atom, axis] = (
                energy(coordinates + displacement, points)
                - energy(coordinates - displacement, points)
            ) / (2 * step)
            expected_points[atom, axis] = (
                energy(coordinates, points + motion)
                - energy(coordinates, points - motion)
            ) / (2 * step)
    np.testing.assert_allclose(actual["xc_ao"], expected_basis, atol=3e-8, rtol=3e-7)
    np.testing.assert_allclose(actual["xc_grid"], expected_points, atol=3e-8, rtol=3e-7)
    np.testing.assert_array_equal(actual["xc_weight"], 0)
    np.testing.assert_allclose(
        (actual["xc_ao"] + actual["xc_grid"]).sum(axis=0), 0, atol=2e-12, rtol=0
    )
