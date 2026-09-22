"""Lightweight XC family dispatch shared by runtime and code generation."""

from __future__ import annotations

import typing

from .expressions import energy_expression as semilocal_energy_expression
from .rsh_expressions import energy_expression as rsh_energy_expression
from .spec import (
    AUTO_BULK_COMPONENTS,
    SPECIAL_EXPRESSION_COMPONENTS,
    WB97MV_COMPONENTS,
    UnsupportedXC,
)
from .wb97mv_maple import energy_expression as wb97mv_energy_expression


def build_energy_expression(
    spec: typing.Any, *, production: bool = False
) -> typing.Any:
    """Build the family-selected scalar energy DAG without runtime dependencies."""
    if type(production) is not bool:
        raise TypeError("production must be bool")
    active = {name for name, coefficient in spec.components if coefficient}
    # Representation alone cannot bypass domain qualification through either
    # runtime programs or the shared native source lowerer.
    if active & set(AUTO_BULK_COMPONENTS):
        raise UnsupportedXC(
            "bulk Libxc component is represented and pointwise-validated "
            "but not production-domain admitted"
        )
    if active & set(WB97MV_COMPONENTS):
        if not active <= set(WB97MV_COMPONENTS):
            raise UnsupportedXC(
                "omegaB97M-V semilocal components cannot be mixed with another XC family"
            )
        return wb97mv_energy_expression(spec)
    if active & set(SPECIAL_EXPRESSION_COMPONENTS):
        return rsh_energy_expression(spec, production=production)
    return semilocal_energy_expression(spec, production=production)
