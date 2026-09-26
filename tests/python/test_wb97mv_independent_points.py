"""Independent interior-point coverage beyond the two retained WB97M goldens."""

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc.program import build_program


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_wb97mv_random_points_match_pinned_libxc(spin: str) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("independent oracle is pinned to Libxc 7.0.0")
    rng = np.random.default_rng(7202026)
    spins = 1 if spin == "unpolarized" else 2
    count = 64
    density = np.exp(rng.uniform(np.log(0.003), np.log(4.0), (spins, count)))
    gradients = rng.normal(size=(spins, 3, count)) * density[:, None, :] ** (4 / 3)
    gradients *= 0.12
    tau = (0.5 + np.exp(rng.uniform(-1, 1, (spins, count)))) * density ** (5 / 3)
    rho = np.zeros((spins, 6, count))
    rho[:, 0], rho[:, 1:4], rho[:, 5] = density, gradients, tau
    sigma = np.sum(gradients**2, axis=1)
    if spins == 1:
        features = np.stack((density[0], sigma[0], tau[0]))
    else:
        features = np.stack(
            (
                density[0],
                density[1],
                sigma[0],
                np.sum(gradients[0] * gradients[1], axis=0),
                sigma[1],
                tau[0],
                tau[1],
            )
        )
    program = build_program(
        resolve_method("WB97M-V", spin=spin).primitives[0].functional, order=2
    )
    actual = program.unpack(program.evaluate(features))
    exc, vxc, fxc, _ = libxc.eval_xc(
        "HYB_MGGA_XC_WB97M_V",
        rho[0] if spins == 1 else rho,
        spin=spins - 1,
        deriv=2,
    )
    expected_gradient = (
        np.stack((vxc[0], vxc[1], vxc[3]))
        if spins == 1
        else np.concatenate((vxc[0], vxc[1], vxc[3]), axis=1).T
    )
    np.testing.assert_allclose(
        actual["energy_density"], exc * density.sum(axis=0), atol=3e-12, rtol=3e-11
    )
    np.testing.assert_allclose(
        actual["gradient"], expected_gradient, atol=3e-11, rtol=3e-9
    )
    if spins == 1:
        hessian = np.zeros((3, 3, count))
        hessian[0, 0], hessian[1, 1], hessian[2, 2] = fxc[0], fxc[2], fxc[4]
        hessian[0, 1] = hessian[1, 0] = fxc[1]
        hessian[0, 2] = hessian[2, 0] = fxc[6]
        hessian[1, 2] = hessian[2, 1] = fxc[9]
        np.testing.assert_allclose(actual["hessian"], hessian, atol=1e-9, rtol=1e-7)
