"""One complete RHF nuclear perturbation through the shared response solver.

Full-coordinate Hessian assembly and directional consumers share these exact
metric, occupied-response and energy-weighted-density conventions.
"""

from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import GMRESOptions, RHFResponseOperator, SolveResult, solve

from .response import build_rhf_nuclear_rhs, metric_density_response_mo


@dataclass(frozen=True, eq=False)
class RHFNuclearResponse:
    """Detached response of occupied orbitals and densities to one perturbation."""

    rhs: np.ndarray
    coefficient_derivative: np.ndarray
    occupied_energy_derivative: np.ndarray
    density_derivative: np.ndarray
    energy_weighted_density_derivative: np.ndarray
    solve_result: SolveResult


def _matrix(value, n, name):
    array = np.asarray(value)
    if (
        array.shape != (n, n)
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real AO matrix")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be representable in FP64")
    if not np.allclose(array, array.T, atol=2e-10, rtol=2e-12):
        raise ValueError(f"{name} must be symmetric")
    return array


def solve_rhf_nuclear_perturbation(operator, frozen_fock, overlap, *, options=None):
    """Solve A x = -b once, retaining occupied metric and energy responses.

    ``frozen_fock`` is h'(v) + G'(v)[P] at fixed AO density, not a total Fock
    derivative that already contains electronic relaxation. Every J/K action,
    including the known metric connection, uses the operator's same backend.
    This helper owns no SCF solve, derivative provider, device or Krylov history.
    """
    if not isinstance(operator, RHFResponseOperator):
        raise TypeError("expected the shared RHFResponseOperator")
    if options is not None and not isinstance(options, GMRESOptions):
        raise TypeError("options must be GMRESOptions")
    ref = operator.problem.reference
    nmo, nocc = ref.nmo, ref.nocc
    layout = operator.problem.layout
    if layout.occupied != tuple(range(nocc)) or layout.virtual != tuple(
        range(nocc, nmo)
    ):
        raise ValueError("nuclear response requires the complete canonical RHF space")
    # Revalidate the live owner before any response operation.
    validate = getattr(operator.backend, "validate_reference", None)
    if validate is None:
        raise ValueError("nuclear response requires a reference-bound backend")
    validate(ref)
    frozen = _matrix(frozen_fock, nmo, "frozen Fock derivative")
    s1 = _matrix(overlap, nmo, "overlap derivative")
    C, eps = ref.coefficients, ref.orbital_energies
    mocc, e_i = C[:, :nocc], eps[:nocc]

    def induced_fock(density):
        coulomb, exchange = operator.backend.coulomb_exchange(density)
        return coulomb - 0.5 * exchange

    frozen_mo = C.T @ frozen @ C
    overlap_mo = C.T @ s1 @ C
    metric_dm_mo = metric_density_response_mo(overlap_mo, nocc=nocc)
    metric_fock_mo = C.T @ induced_fock(C @ metric_dm_mo @ C.T) @ C
    rhs = build_rhf_nuclear_rhs(frozen_mo, overlap_mo, metric_fock_mo, eps, nocc=nocc)
    result = solve(
        operator,
        layout.pack(-rhs),
        options=options or GMRESOptions(),
        raise_on_failure=True,
    )
    x_ia = layout.as_ia(result.solution)
    mo1 = -0.5 * overlap_mo[:, :nocc]
    mo1[nocc:, :] += x_ia.T
    c1 = C @ mo1
    density = 2.0 * (c1 @ mocc.T + mocc @ c1.T)
    hs = frozen_mo[:, :nocc] - overlap_mo[:, :nocc] * e_i[None, :]
    hs += C.T @ induced_fock(density) @ mocc
    e1 = hs[:nocc, :] + mo1[:nocc, :] * (e_i[:, None] - e_i[None, :])
    # The occupied-energy response is a full block, not merely diag(eps').
    left = (c1 * e_i[None, :]) @ mocc.T
    energy_density = 2.0 * (left + left.T + mocc @ e1 @ mocc.T)
    values = (rhs, c1, e1, density, energy_density)
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite nuclear response; no result published")
    return RHFNuclearResponse(*(immutable(value) for value in values), result)
