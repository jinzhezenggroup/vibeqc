"""Independent closed-form exchange oracle (no DAG or differentiation engine)."""

import numpy as np


def exchange_reference(features, *, spin, gga):
    """Analytic energy, gradient and Hessian in the declared feature layout.

    Exchange is spin separable. In particular d²e/(d rho_b d sigma_aa) is
    identically zero, even where the reduced-variable Libxc implementation
    suffers catastrophic cancellation. This oracle provides a separate boundary
    check instead of inflating an absolute tolerance around that artifact.
    """
    x = np.asarray(features)
    nfeature, npoint = x.shape
    energy = np.zeros(npoint)
    gradient = np.zeros_like(x)
    hessian = np.zeros((nfeature, nfeature, npoint))
    q = 4 / 3
    cx = 0.75 * (6 / np.pi) ** (1 / 3)
    c = 1 / (4 * (6 * np.pi**2) ** (2 / 3))
    kappa = 0.804
    mu = 0.06672455060314922 * np.pi**2 / 3
    for r_index, s_index in ((0, 2), (1, 4)) if spin else ((0, 1),):
        r = x[r_index] if spin else x[0] / 2
        sigma = x[s_index] if spin else x[1] / 4
        u = c * sigma * r ** (-2 * q)
        den = kappa + mu * u
        f = 1 + kappa - kappa**2 / den if gga else np.ones(npoint)
        df = kappa**2 * mu / den**2 if gga else np.zeros(npoint)
        ddf = -2 * kappa**2 * mu**2 / den**3 if gga else np.zeros(npoint)
        e = -cx * r**q * f
        vr = -cx * r ** (q - 1) * (q * f - 2 * q * u * df)
        vs = -cx * c * r ** (-q) * df
        hrr = (
            -cx
            * r ** (q - 2)
            * (q * (q - 1) * f + 2 * q * u * df + 4 * q**2 * u**2 * ddf)
        )
        hrs = cx * c * q * r ** (-q - 1) * (df + 2 * u * ddf)
        hss = -cx * c**2 * r ** (-3 * q) * ddf
        energy += e if spin else 2 * e
        gradient[r_index] = vr
        gradient[s_index] = vs if spin else vs / 2
        hessian[r_index, r_index] = hrr if spin else hrr / 2
        hessian[r_index, s_index] = hessian[s_index, r_index] = hrs if spin else hrs / 4
        hessian[s_index, s_index] = hss if spin else hss / 8
    return energy, gradient, hessian
