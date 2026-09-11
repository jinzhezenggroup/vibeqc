"""Feature-chain-rule coefficients for a downstream AO matrix contraction."""

import numpy as np

from vibeqc_compiler.common.arrays import immutable

from .coefficients import coefficient_program


def potential_coefficients(spec, density_gradient, xc_gradient):
    """Return scalar, spatial-gradient and kinetic AO bilinear coefficients.

    For spin a: G_a = 2 e_sigma_aa grad(rho_a) + e_sigma_ab grad(rho_b).
    Matrix integrands are e_rho phi_mu phi_nu + G dot grad(phi_mu phi_nu)
    + (e_tau/2) grad(phi_mu) dot grad(phi_nu). The final factor is inherited
    from tau=one-half sum D_mu_nu grad(phi_mu) dot grad(phi_nu), not a spin
    degeneracy factor. The caller applies quadrature weights exactly once.
    """
    return coefficient_program(spec.spin, kinetic=True).evaluate(
        density_gradient, xc_gradient
    )


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
    return assemble_coefficients(jets, coefficients, weights)


def assemble_coefficients(jets, coefficients, weights):
    """Contract compact point coefficients with at most two GEMMs per spin.

    The scalar and gradient bilinears share this sole production assembly
    owner. LDA needs one value panel and one GEMM; GGA combines three spatial
    coefficients into one panel before its second GEMM. Optional kinetic
    coefficients preserve the legacy synthetic tau-half contract.
    """
    jets, weights = immutable(jets), immutable(weights)
    if jets.ndim != 3 or jets.shape[0] not in (1, 4, 10, 20):
        raise ValueError("invalid AO jet domain for compact assembly")
    rho = immutable(coefficients["rho"])
    if (
        rho.ndim != 2
        or rho.shape[0] not in (1, 2)
        or rho.shape[1] != jets.shape[1]
        or weights.shape != (jets.shape[1],)
    ):
        raise ValueError("coefficient point/spin/weight shape mismatch")
    if set(coefficients) - {"rho", "gradient", "tau"}:
        raise ValueError("unsupported compact coefficient")
    spatial = coefficients.get("gradient")
    kinetic = coefficients.get("tau")
    if spatial is not None:
        spatial = immutable(spatial, shape=(*rho.shape, 3))
    if kinetic is not None:
        kinetic = immutable(kinetic, shape=rho.shape)
    if (spatial is not None or kinetic is not None) and jets.shape[0] < 4:
        raise ValueError("gradient/kinetic assembly requires first AO derivatives")
    phi, derivatives = jets[0], jets[1:4]
    matrices = []
    for spin in range(len(rho)):
        matrix = phi.T @ ((weights * rho[spin])[:, None] * phi)
        if spatial is not None:
            panel = sum(spatial[spin, :, k, None] * derivatives[k] for k in range(3))
            cross = phi.T @ (weights[:, None] * panel)
            matrix += cross + cross.T
        # The current LDA/GGA inventory has zero tau partials. Keep the
        # existing coefficient contract testable without claiming meta-GGA.
        if kinetic is not None and np.any(kinetic[spin]):
            for derivative in derivatives:
                matrix += derivative.T @ (
                    (weights * kinetic[spin])[:, None] * derivative
                )
        matrices.append(0.5 * (matrix + matrix.T))
    return immutable(matrices)
