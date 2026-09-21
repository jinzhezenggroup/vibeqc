"""Feature-chain-rule coefficients for a downstream AO matrix contraction."""

import typing

import numpy as np

from vibeqc_compiler.common.arrays import immutable

from .coefficients import coefficient_program


def potential_coefficients(
    spec: typing.Any, density_gradient: typing.Any, xc_gradient: typing.Any
) -> typing.Any:
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


def assemble_potential(
    spec: typing.Any,
    jets: typing.Any,
    density_gradient: typing.Any,
    xc_gradient: typing.Any,
    weights: typing.Any,
) -> typing.Any:
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


def assemble_coefficients_directional(
    jets: typing.Any,
    directional_jets: typing.Any,
    coefficients: typing.Any,
    directional_coefficients: typing.Any,
    weights: typing.Any,
    directional_weights: typing.Any,
) -> typing.Any:
    """Differentiate the compact LDA/GGA AO matrix contraction once.

    Point coefficients, AO jets and quadrature measure may all move. This is
    the matrix-valued geometry JVP required by a stationary KS nuclear RHS; it
    deliberately excludes tau until the meta-GGA response chain is qualified.
    """
    jets = immutable(jets)
    directional_jets = immutable(directional_jets)
    weights = immutable(weights)
    directional_weights = immutable(directional_weights)
    if (
        jets.ndim != 3
        or jets.shape[0] not in (1, 4, 10, 20)
        or directional_jets.shape != jets.shape
        or weights.shape != (jets.shape[1],)
        or directional_weights.shape != weights.shape
    ):
        raise ValueError("invalid directional AO/weight domain")
    if set(coefficients) - {"rho", "gradient"} or set(directional_coefficients) - {
        "rho",
        "gradient",
    }:
        raise ValueError("directional compact assembly supports LDA/GGA only")
    rho = immutable(coefficients["rho"])
    drho = immutable(directional_coefficients["rho"], shape=rho.shape)
    if rho.ndim != 2 or rho.shape[0] not in (1, 2) or rho.shape[1] != jets.shape[1]:
        raise ValueError("invalid directional coefficient point/spin layout")
    spatial = coefficients.get("gradient")
    directional_spatial = directional_coefficients.get("gradient")
    if (spatial is None) != (directional_spatial is None):
        raise ValueError("directional gradient coefficients must match the base domain")
    if spatial is not None:
        spatial = immutable(spatial, shape=(*rho.shape, 3))
        directional_spatial = immutable(directional_spatial, shape=(*rho.shape, 3))
        if jets.shape[0] < 4:
            raise ValueError("GGA directional assembly requires first AO derivatives")

    phi, dphi = jets[0], directional_jets[0]
    derivatives, directional_derivatives = jets[1:4], directional_jets[1:4]
    matrices = []
    for spin in range(len(rho)):
        base_measure = weights * rho[spin]
        moving_measure = directional_weights * rho[spin] + weights * drho[spin]
        matrix = (
            dphi.T @ (base_measure[:, None] * phi)
            + phi.T @ (base_measure[:, None] * dphi)
            + phi.T @ (moving_measure[:, None] * phi)
        )
        if spatial is not None:
            panel = sum(spatial[spin, :, k, None] * derivatives[k] for k in range(3))
            directional_panel = sum(
                directional_spatial[spin, :, k, None] * derivatives[k]
                + spatial[spin, :, k, None] * directional_derivatives[k]
                for k in range(3)
            )
            directional_cross = (
                dphi.T @ (weights[:, None] * panel)
                + phi.T @ (directional_weights[:, None] * panel)
                + phi.T @ (weights[:, None] * directional_panel)
            )
            matrix += directional_cross + directional_cross.T
        matrices.append(0.5 * (matrix + matrix.T))
    return immutable(matrices)


def assemble_coefficients(
    jets: typing.Any, coefficients: typing.Any, weights: typing.Any
) -> typing.Any:
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
