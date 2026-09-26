"""Versioned VV10/rVV10 scientific definitions shared across compiler layers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from .exact import require_fraction
from .provenance import canonical_hash

NONLOCAL_CORRELATION_VERSION = "vv10-rvv10-2010-2013-v1"
VV10 = "vv10"
RVV10 = "rvv10"
_VARIANTS = (VV10, RVV10)


class UnsupportedNonlocalCorrelation(ValueError):
    """The requested nonlocal-correlation definition is not audited."""


@dataclass(frozen=True)
class NonlocalCorrelationSpec:
    """Immutable scientific identity for one VV10-family kernel definition."""

    variant: str
    b: Fraction
    c: Fraction
    version: str = NONLOCAL_CORRELATION_VERSION

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise UnsupportedNonlocalCorrelation(
                f"unsupported nonlocal-correlation variant {self.variant!r}"
            )
        if self.version != NONLOCAL_CORRELATION_VERSION:
            raise UnsupportedNonlocalCorrelation(
                "unsupported nonlocal-correlation definition version"
            )
        require_fraction(
            self.b, "VV10 b", UnsupportedNonlocalCorrelation, role="parameter"
        )
        require_fraction(
            self.c, "VV10 C", UnsupportedNonlocalCorrelation, role="parameter"
        )
        if self.b <= 0 or self.c <= 0:
            raise UnsupportedNonlocalCorrelation(
                "VV10 b and C parameters must be positive"
            )

    @property
    def source(self) -> typing.Any:
        if self.variant == VV10:
            return "Vydrov-Van-Voorhis/JCP-133-244103-2010"
        return "Sabatini-Gorni-de-Gironcoli/PRB-87-041108R-2013"

    @property
    def kernel_convention(self) -> typing.Any:
        if self.variant == RVV10:
            return "finite-system-rvv10-q-kappa-total-density-v2"
        return "finite-system-real-space-total-density-v1"

    def to_payload(self) -> typing.Any:
        return {
            "variant": self.variant,
            "version": self.version,
            "b": str(self.b),
            "C": str(self.c),
            "source": self.source,
            "kernel_convention": self.kernel_convention,
            "density": "total-spin-density",
            "gradient": "cartesian-gradient-of-total-density",
            "quadrature": "real-space-weighted-point-pairs-v1",
            "regularization": "none-positive-density-domain-v1",
            "pair_integration": "full-double-integral-with-one-half-v1",
            "units": "atomic",
        }

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())


ORIGINAL_NONLOCAL_CORRELATION = MappingProxyType(
    {
        VV10: NonlocalCorrelationSpec(VV10, Fraction("5.9"), Fraction("0.0093")),
        RVV10: NonlocalCorrelationSpec(RVV10, Fraction("6.3"), Fraction("0.0093")),
    }
)


def original_nonlocal_correlation(variant: typing.Any) -> typing.Any:
    """Return an audited original VV10 or rVV10 parameterization."""
    try:
        return ORIGINAL_NONLOCAL_CORRELATION[variant]
    except KeyError as error:
        raise UnsupportedNonlocalCorrelation(
            f"unknown nonlocal-correlation variant {variant!r}"
        ) from error
