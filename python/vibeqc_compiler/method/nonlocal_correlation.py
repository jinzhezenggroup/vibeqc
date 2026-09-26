"""VV10/rVV10 MethodIR primitive for #491."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

from vibeqc_compiler.common.exact import require_fraction
from vibeqc_compiler.common.nonlocal_correlation import (
    NonlocalCorrelationSpec,
    UnsupportedNonlocalCorrelation,
)


@dataclass(frozen=True)
class NonlocalCorrelationPrimitive:
    """Distinct MethodIR node for genuinely nonlocal correlation."""

    spec: NonlocalCorrelationSpec
    coefficient: Fraction = Fraction(1)
    kind: ClassVar[str] = "nonlocal_correlation"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, NonlocalCorrelationSpec):
            raise TypeError("nonlocal primitive requires NonlocalCorrelationSpec")
        require_fraction(
            self.coefficient,
            "nonlocal-correlation coefficient",
            UnsupportedNonlocalCorrelation,
            role="parameter",
        )
        if self.coefficient <= 0:
            raise UnsupportedNonlocalCorrelation(
                "nonlocal-correlation coefficient must be positive"
            )

    @property
    def derivative_capabilities(self) -> typing.Any:
        # Energy and the self-consistent KS/Fock potential share one definition.
        return ("energy", "ks-potential", "nuclear-gradient")

    @property
    def required_ingredients(self) -> typing.Any:
        return ("rho", "sigma")

    def semantic_payload(self) -> typing.Any:
        return {
            "kind": self.kind,
            "coefficient": str(self.coefficient),
            "spec": self.spec.to_payload(),
            "derivative_capabilities": self.derivative_capabilities,
            "required_ingredients": self.required_ingredients,
        }

    def to_payload(self) -> typing.Any:
        return self.semantic_payload()
