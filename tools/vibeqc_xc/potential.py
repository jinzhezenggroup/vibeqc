"""Feature-chain-rule coefficients for a downstream AO matrix contraction."""

import numpy as np


def potential_coefficients(spec, density_gradient, xc_gradient):
    """Return scalar, spatial-gradient and kinetic AO bilinear coefficients.

    For spin a: G_a = 2 e_sigma_aa grad(rho_a) + e_sigma_ab grad(rho_b).
    Matrix integrands are e_rho phi_mu phi_nu + G dot grad(phi_mu phi_nu)
    + (e_tau/2) grad(phi_mu) dot grad(phi_nu). The final factor is inherited
    from tau=one-half sum D_mu_nu grad(phi_mu) dot grad(phi_nu), not a spin
    degeneracy factor. The caller applies quadrature weights exactly once.
    """
    gradient, v = np.asarray(density_gradient), np.asarray(xc_gradient)
    if v.ndim != 2 or v.shape[0] != len(spec.features):
        raise ValueError("invalid XC feature gradient")
    npoint = v.shape[1]
    if spec.spin == "polarized":
        if gradient.shape != (2, npoint, 3):
            raise ValueError("density gradients require [spin,point,xyz]")
        spatial = np.stack(
            (
                2 * v[2, :, None] * gradient[0] + v[3, :, None] * gradient[1],
                v[3, :, None] * gradient[0] + 2 * v[4, :, None] * gradient[1],
            )
        )
        return {"rho": v[:2], "gradient": spatial, "tau": v[5:7] / 2}
    if gradient.shape != (npoint, 3):
        raise ValueError("unpolarized density gradient requires [point,xyz]")
    return {
        "rho": v[:1],
        "gradient": (2 * v[1, :, None] * gradient)[None],
        "tau": v[2:3] / 2,
    }
