"""Deterministic physical inputs for Libxc production-domain qualification.

The versioned admission profile owns which cases are required.  This module only
instantiates finite physical coordinates for the numerical rho/sigma/tau cases;
it deliberately does not evaluate XC mathematics, import Libxc/PySCF, or grant
production capability.  Control cases remain separate policy checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .libxc_production_domain import ProductionDomainProfile

Vector3 = tuple[float, float, float]
CONTROL_PREFIX = "control/"


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _scale(vector: Vector3, factor: float) -> Vector3:
    return (
        factor * vector[0],
        factor * vector[1],
        factor * vector[2],
    )


def _tau_uniform(rho: float, *, polarized: bool) -> float:
    coefficient = 0.3 * ((6.0 if polarized else 3.0) * math.pi**2) ** (2.0 / 3.0)
    return coefficient * rho ** (5.0 / 3.0)


def _tau_von_weizsaecker(rho: float, gradient: Vector3) -> float:
    if rho == 0.0:
        return 0.0
    return _dot(gradient, gradient) / (8.0 * rho)


@dataclass(frozen=True)
class ProductionDomainCase:
    """One finite physical coordinate owned independently of the XC formula."""

    spin: str
    case_id: str
    rho: tuple[float, ...]
    gradient: tuple[Vector3, ...]
    tau: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = 2 if self.spin == "polarized" else 1
        if self.spin not in ("polarized", "unpolarized"):
            raise ValueError(f"unsupported production-domain spin {self.spin!r}")
        if len(self.rho) != expected or len(self.gradient) != expected:
            raise ValueError("production-domain density/gradient spin arity mismatch")
        if len(self.tau) != expected:
            raise ValueError("production-domain tau spin arity mismatch")
        values = [
            *self.rho,
            *self.tau,
            *(component for vector in self.gradient for component in vector),
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("production-domain physical cases must remain finite")
        if any(value < 0.0 for value in self.rho) or any(
            value < 0.0 for value in self.tau
        ):
            raise ValueError(
                "production-domain rho/tau coordinates must be nonnegative"
            )

    def runtime_features(
        self, profile: ProductionDomainProfile
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Project the physical coordinate into the compact rho/sigma/tau ABI."""
        if self.case_id not in profile.case_ids_for_spin(self.spin):
            raise ValueError(
                f"case {self.spin!r}:{self.case_id!r} is outside the profile"
            )
        ingredients = frozenset(profile.required_ingredients)
        if self.spin == "polarized":
            names = ["rho_a", "rho_b"]
            values = [*self.rho]
            if "sigma" in ingredients:
                aa = _dot(self.gradient[0], self.gradient[0])
                ab = _dot(self.gradient[0], self.gradient[1])
                bb = _dot(self.gradient[1], self.gradient[1])
                names.extend(("sigma_aa", "sigma_ab", "sigma_bb"))
                values.extend((aa, ab, bb))
            if "tau" in ingredients:
                names.extend(("tau_a", "tau_b"))
                values.extend(self.tau)
        else:
            names = ["rho"]
            values = [self.rho[0]]
            if "sigma" in ingredients:
                names.append("sigma")
                values.append(_dot(self.gradient[0], self.gradient[0]))
            if "tau" in ingredients:
                names.append("tau")
                values.append(self.tau[0])
        return tuple(names), tuple(values)

    def pyscf_rho(self) -> tuple[tuple[float, ...], ...]:
        """Return density/gradient/laplacian/tau rows for an independent oracle."""
        return tuple(
            (rho, *gradient, 0.0, tau)
            for rho, gradient, tau in zip(
                self.rho, self.gradient, self.tau, strict=True
            )
        )


def _polarized_case(case_id: str) -> ProductionDomainCase:
    rho: tuple[float, ...] = (0.47, 0.31)
    gradient: tuple[Vector3, ...] = ((0.20, -0.08, 0.05), (-0.07, 0.14, 0.03))
    tau: tuple[float, ...] = (0.32, 0.21)

    if case_id == "density/vacuum":
        rho, gradient, tau = (0.0, 0.0), ((0.0, 0.0, 0.0),) * 2, (0.0, 0.0)
    elif case_id == "density/near-zero":
        rho = (7.3e-20, 2.1e-20)
        gradient = ((1.46e-19, 0.0, 0.0), (-2.1e-20, 0.0, 0.0))
        tau = tuple(0.7 * value for value in rho)
    elif case_id == "spin/balanced":
        rho = (0.4, 0.4)
        gradient = ((0.16, -0.04, 0.06), (0.16, -0.04, 0.06))
        tau = (0.26, 0.26)
    elif case_id == "spin/zero-a":
        rho = (0.0, 0.62)
        gradient = ((0.0, 0.0, 0.0), gradient[1])
        tau = (0.0, 0.36)
    elif case_id == "spin/zero-b":
        rho = (0.62, 0.0)
        gradient = (gradient[0], (0.0, 0.0, 0.0))
        tau = (0.36, 0.0)
    elif case_id == "spin/near-zero-a":
        tiny = 0.62e-14
        rho = (tiny, 0.62)
        gradient = (_scale((0.2, -0.08, 0.05), 1.0e-14), gradient[1])
        tau = (0.7 * tiny, 0.36)
    elif case_id == "spin/near-zero-b":
        tiny = 0.62e-14
        rho = (0.62, tiny)
        gradient = (gradient[0], _scale((-0.07, 0.14, 0.03), 1.0e-14))
        tau = (0.36, 0.7 * tiny)
    elif case_id == "spin/full-a":
        rho = (0.8, 0.0)
        gradient = ((0.24, -0.06, 0.08), (0.0, 0.0, 0.0))
        tau = (0.48, 0.0)
    elif case_id == "spin/full-b":
        rho = (0.0, 0.8)
        gradient = ((0.0, 0.0, 0.0), (-0.24, 0.06, -0.08))
        tau = (0.0, 0.48)
    elif case_id == "sigma/zero":
        gradient = ((0.0, 0.0, 0.0),) * 2
    elif case_id == "sigma/near-zero":
        gradient = tuple(_scale(vector, 1.0e-12) for vector in gradient)
    elif case_id == "sigma/large-finite":
        gradient = tuple(_scale(vector, 1.0e6) for vector in gradient)
    elif case_id == "sigma/cancellation":
        gradient = ((0.2, 0.1, 0.0), (-0.2, -0.1 + 1.0e-13, 0.0))
    elif case_id == "tau/uniform-gas":
        gradient = ((0.0, 0.0, 0.0),) * 2
        tau = tuple(_tau_uniform(value, polarized=True) for value in rho)
    elif case_id == "tau/isoorbital":
        tau = tuple(
            _tau_von_weizsaecker(value, vector)
            for value, vector in zip(rho, gradient, strict=True)
        )
    elif case_id == "tau/near-isoorbital":
        tau = tuple(
            _tau_von_weizsaecker(value, vector) * (1.0 + 1.0e-12)
            for value, vector in zip(rho, gradient, strict=True)
        )
    elif case_id == "tau/large-finite":
        tau = (1.0e8, 7.0e7)
    else:
        raise ValueError(f"unsupported polarized numerical case {case_id!r}")
    return ProductionDomainCase("polarized", case_id, rho, gradient, tau)


def _unpolarized_case(case_id: str) -> ProductionDomainCase:
    rho: tuple[float, ...] = (0.78,)
    gradient: tuple[Vector3, ...] = ((0.13, -0.11, 0.07),)
    tau: tuple[float, ...] = (0.48,)

    if case_id == "density/vacuum":
        rho, gradient, tau = (0.0,), ((0.0, 0.0, 0.0),), (0.0,)
    elif case_id == "density/near-zero":
        rho = (9.4e-20,)
        gradient = ((1.88e-19, 0.0, 0.0),)
        tau = (0.7 * rho[0],)
    elif case_id == "sigma/zero":
        gradient = ((0.0, 0.0, 0.0),)
    elif case_id == "sigma/near-zero":
        gradient = (_scale(gradient[0], 1.0e-12),)
    elif case_id == "sigma/large-finite":
        gradient = (_scale(gradient[0], 1.0e6),)
    elif case_id == "sigma/cancellation":
        gradient = ((1.0e-14, -1.0e-14, 1.0e-16),)
    elif case_id == "tau/uniform-gas":
        gradient = ((0.0, 0.0, 0.0),)
        tau = (_tau_uniform(rho[0], polarized=False),)
    elif case_id == "tau/isoorbital":
        tau = (_tau_von_weizsaecker(rho[0], gradient[0]),)
    elif case_id == "tau/near-isoorbital":
        tau = (
            _tau_von_weizsaecker(rho[0], gradient[0]) * (1.0 + 1.0e-12),
        )
    elif case_id == "tau/large-finite":
        tau = (1.0e8,)
    else:
        raise ValueError(f"unsupported unpolarized numerical case {case_id!r}")
    return ProductionDomainCase("unpolarized", case_id, rho, gradient, tau)


def numerical_cases(
    profile: ProductionDomainProfile, *, spin: str
) -> tuple[ProductionDomainCase, ...]:
    """Instantiate every non-control row in the exact profile matrix."""
    result = []
    for case_id in profile.case_ids_for_spin(spin):
        if case_id.startswith(CONTROL_PREFIX):
            continue
        result.append(
            _polarized_case(case_id)
            if spin == "polarized"
            else _unpolarized_case(case_id)
        )
    return tuple(result)


def control_case_ids(
    profile: ProductionDomainProfile, *, spin: str
) -> tuple[str, ...]:
    """Return policy/control rows that must not be fabricated as numeric probes."""
    return tuple(
        case_id
        for case_id in profile.case_ids_for_spin(spin)
        if case_id.startswith(CONTROL_PREFIX)
    )


__all__ = [
    "ProductionDomainCase",
    "control_case_ids",
    "numerical_cases",
]
