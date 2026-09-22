"""Independent regression for skipped/zero-AO reference quadrature blocks."""

from __future__ import annotations

import os

import numpy as np
import pytest

from benchmarks._gpu4pyscf_grid import preserve_reference_grid_order

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GPU4PYSCF_GRID_TEST") != "1",
    reason="requires a finite Slurm GPU allocation and GPU4PySCF",
)


@pytest.mark.parametrize("functional", ["LDA_X,LDA_C_PW", "PBE", "R2SCAN"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("spin", [0, 1])
def test_empty_blocks_match_independent_density_energy_and_potential(
    functional: str, fitted: bool, spin: int
) -> None:
    """Interior/trailing empty blocks and a partial tail keep physical offsets.

    CPU PySCF is an independent numerical oracle only; no CPU timings are
    collected. Both engines receive the same density, explicit points/weights
    and, for DF, the same auxiliary basis. AO sparsity is not assumed: the test
    first checks that the dependency actually emits a zero-AO block.
    """
    assert os.environ.get("SLURM_JOB_ID"), "GPU tests must run through Slurm"
    cp = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    from gpu4pyscf.dft.numint import MIN_BLK_SIZE
    from pyscf import dft, gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="Bohr",
        basis="def2-svp",
        cart=False,
        charge=spin,
        spin=spin,
        verbose=0,
    )
    cpu = dft.UKS(mol) if spin else dft.RKS(mol)
    if fitted:
        cpu = cpu.density_fit(auxbasis="def2-universal-jkfit")
    cpu.xc = functional
    cpu.small_rho_cutoff = 0
    rng = np.random.default_rng(1079)
    near = rng.normal(scale=0.7, size=(2 * MIN_BLK_SIZE, 3))
    far = np.full((MIN_BLK_SIZE, 3), 1e6)
    points = np.concatenate(
        (near[:MIN_BLK_SIZE], far, near[MIN_BLK_SIZE:], far, far[:13])
    )
    weights = rng.uniform(1e-4, 1e-2, size=len(points))
    cpu.grids.coords, cpu.grids.weights = points, weights
    gpu = cpu.to_gpu()
    # Explicitly retain our order; conversion must not sort or rebuild the grid.
    gpu.grids.coords, gpu.grids.weights = cp.asarray(points), cp.asarray(weights)
    gpu.small_rho_cutoff = 0
    original = gpu._numint.block_loop
    assert any(
        len(indices) == 0
        for _, indices, _, _ in original(
            mol, gpu.grids, deriv=1, strict_grid_order=True
        )
    )
    preserve_reference_grid_order(gpu)
    wrapped = gpu._numint.block_loop
    preserve_reference_grid_order(gpu)
    assert gpu._numint.block_loop is wrapped
    yielded_points = sum(len(w) for _, _, w, _ in wrapped(mol, gpu.grids, deriv=1))
    assert yielded_points == len(points)
    density = cpu.get_init_guess()
    reference = cpu.get_veff(mol, density)
    actual = gpu.get_veff(mol, cp.asarray(density))
    np.testing.assert_allclose(
        actual.get(), np.asarray(reference), atol=2e-9, rtol=2e-10
    )
    assert float(actual.exc) == pytest.approx(float(reference.exc), abs=2e-10)
    assert float(actual.ecoul) == pytest.approx(float(reference.ecoul), abs=2e-10)
    # nr_rks reports the grid electron count independently of Coulomb/DF work.
    reference_integrate = cpu._numint.nr_uks if spin else cpu._numint.nr_rks
    actual_integrate = gpu._numint.nr_uks if spin else gpu._numint.nr_rks
    ref_n, ref_e, ref_v = reference_integrate(mol, cpu.grids, functional, density)
    got_n, got_e, got_v = actual_integrate(
        mol, gpu.grids, functional, cp.asarray(density)
    )
    np.testing.assert_allclose(got_n, ref_n, atol=2e-10, rtol=2e-10)
    assert float(got_e) == pytest.approx(float(ref_e), abs=2e-10)
    np.testing.assert_allclose(got_v.get(), ref_v, atol=2e-10, rtol=2e-10)
