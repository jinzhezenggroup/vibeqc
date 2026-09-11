"""Radial Coulomb semantics and an independent interval-quadrature oracle.

The oracle is for validation, not a production integral backend. Generated
Hermite recurrences can consume these modified moments without changing the
nuclear chain rule dF_n/dT = -F_(n+1): omega and Gaussian exponents are fixed
during nuclear differentiation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CoulombKernelFamily(str, Enum):
    """Public radial kernels in atomic units; omega is inverse bohr."""

    FULL_RANGE = "full_range"
    LONG_RANGE = "long_range"
    SHORT_RANGE = "short_range"


@dataclass(frozen=True, slots=True)
class CoulombKernel:
    """Explicit 1/r, erf(omega*r)/r or erfc(omega*r)/r operator identity.

    A full-range request requires omega=0. A zero range parameter gives zero
    for the long-range kernel and full Coulomb for the short-range kernel.
    No omega derivative, density fitting or functional semantics are implied.
    """

    family: CoulombKernelFamily | str = CoulombKernelFamily.FULL_RANGE
    omega: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "family", CoulombKernelFamily(self.family))
        if isinstance(self.omega, bool):
            raise TypeError("omega must be a real inverse-bohr parameter")
        omega = float(self.omega)
        if not math.isfinite(omega) or omega < 0:
            raise ValueError("omega must be finite and nonnegative")
        if self.family == CoulombKernelFamily.FULL_RANGE and omega != 0:
            raise ValueError("full Coulomb does not accept a range parameter")
        object.__setattr__(self, "omega", 0.0 if omega == 0 else omega)

    def to_payload(self):
        """Return normalized scientific inputs for IR/cache serialization."""
        return {"version": 1, "family": self.family.value, "omega": self.omega}


def reference_moments(
    maximum_order: int, argument: float, rho: float, kernel: CoulombKernel
) -> tuple[float, ...]:
    """Integrate modified Boys moments directly for independent validation.

    With theta=omega^2/(omega^2+rho), LR integrates over [0,sqrt(theta)]
    and SR over [sqrt(theta),1]. Direct interval integration avoids subtracting
    two nearly equal full/LR moments. SciPy is imported only by this oracle;
    generating or executing an integral must not depend on it.
    """
    if type(maximum_order) is not int or not 0 <= maximum_order <= 13:
        raise ValueError("the first-derivative through-f oracle supports orders 0..13")
    if (
        not math.isfinite(argument)
        or argument < 0
        or not math.isfinite(rho)
        or rho <= 0
    ):
        raise ValueError(
            "Boys argument must be nonnegative and rho positive, both finite"
        )
    if not isinstance(kernel, CoulombKernel):
        raise TypeError("expected explicit CoulombKernel semantics")
    from scipy.integrate import quad

    radius = math.hypot(kernel.omega, math.sqrt(rho))
    boundary = kernel.omega / radius
    if kernel.family == CoulombKernelFamily.FULL_RANGE:
        lower, width = 0.0, 1.0
    elif kernel.family == CoulombKernelFamily.LONG_RANGE:
        lower, width = 0.0, boundary
    else:
        lower = boundary
        # 1 - omega/hypot(omega,sqrt(rho)), without cancellation or omega^2.
        ratio = math.sqrt(rho) / radius
        width = ratio * (ratio / (1 + boundary))
    if width == 0:
        return (0.0,) * (maximum_order + 1)

    # Bound a negligible Gaussian tail before integrating a scaled unit
    # interval. This also resolves the narrow peak at enormous Boys arguments.
    cutoff = 90.0 + 2 * maximum_order
    if argument * width * (2 * lower + width) > cutoff:
        scaled = cutoff / argument
        width = scaled / (math.hypot(lower, math.sqrt(scaled)) + lower)
    decay = math.exp(-argument * lower * lower)
    if decay == 0:
        return (0.0,) * (maximum_order + 1)
    upper = lower + width
    values = []
    for order in range(maximum_order + 1):
        # Extract small powers/decay so QUADPACK's relative error criterion
        # controls the shape integral even when the physical moment is tiny.
        def integrand(point, order=order):
            delta = width * point
            return ((lower + delta) / upper) ** (2 * order) * math.exp(
                -argument * delta * (2 * lower + delta)
            )

        value, _ = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=2e-13, limit=200)
        values.append(width * decay * upper ** (2 * order) * value)
    return tuple(values)
