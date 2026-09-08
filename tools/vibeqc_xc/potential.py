"""Feature-chain-rule coefficients for a downstream AO matrix contraction."""

import numpy as np

from tools.vibeqc_posthf.reference import immutable


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


def assemble_potential(spec, jets, density_gradient, xc_gradient, weights):
    """Assemble one weighted tile, returning [functional spin, AO, AO].

    All bilinears use ordinary spatial AO jets. Inputs are unweighted; this
    routine applies the complete quadrature measure exactly once. It includes
    both differentiated AO legs without doubling the scalar rho term.
    """
    jets, weights = immutable(jets), immutable(weights)
    gradient, v = immutable(density_gradient), immutable(xc_gradient)
    if jets.ndim != 3 or jets.shape[0] not in (4, 10, 20):
        raise ValueError("potential assembly requires first-order AO jets")
    if weights.shape != (jets.shape[1],) or v.shape != (
        len(spec.features),
        jets.shape[1],
    ):
        raise ValueError("potential point/weight/feature shape mismatch")
    coefficients = potential_coefficients(spec, gradient, v)
    phi, derivatives = jets[0], jets[1:4]
    matrices = []
    for rho, spatial, tau in zip(
        coefficients["rho"], coefficients["gradient"], coefficients["tau"], strict=True
    ):
        matrix = phi.T @ ((weights * rho)[:, None] * phi)
        panel = sum(spatial[:, k, None] * derivatives[k] for k in range(3))
        cross = phi.T @ (weights[:, None] * panel)
        matrix += cross + cross.T
        # The current LDA/GGA inventory has zero tau partials. Keep the
        # existing coefficient contract testable without claiming meta-GGA.
        if np.any(tau):
            for derivative in derivatives:
                matrix += derivative.T @ ((weights * tau)[:, None] * derivative)
        matrices.append(0.5 * (matrix + matrix.T))
    return immutable(matrices)
