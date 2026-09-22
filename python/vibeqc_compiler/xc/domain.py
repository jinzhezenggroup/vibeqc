"""Static feature-domain assumptions for XC lowering."""

from __future__ import annotations

import typing

from vibeqc_compiler.integral.expr import ScalarDomain


def feature_domains(
    spec: typing.Any, *, sanitized_mgga: bool = False
) -> dict[str, ScalarDomain]:
    """Infer scalar domains from the XC feature ABI.

    Physical rho, same-spin sigma and tau are nonnegative and may reach zero.
    ``work_mgga`` sanitization promotes those values to strictly positive work
    inputs while cross-spin sigma remains signed.
    """

    result: dict[str, ScalarDomain] = {}
    for name in spec.features:
        if name in ("sigma_ab",):
            result[name] = ScalarDomain.REAL
        elif name.startswith(("rho", "sigma", "tau")):
            result[name] = (
                ScalarDomain.POSITIVE if sanitized_mgga else ScalarDomain.NONNEGATIVE
            )
        else:
            result[name] = ScalarDomain.REAL
    return result
