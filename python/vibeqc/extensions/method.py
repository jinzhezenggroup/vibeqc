"""Public method-composition boundary over canonical MethodSpec/MethodIR."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from vibeqc_compiler.method import (
    BackendCapability,
    D3Spec,
    MethodIR,
    MethodSpec,
    MethodTypeError,
    TypedMethodIR,
    UnsupportedMethod,
    verify_method_ir,
)
from vibeqc_compiler.method import (
    resolve_method as _resolve_method,
)
from vibeqc_compiler.xc.spec import FunctionalSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

API_VERSION = 1
Coefficient = int | str | Fraction


def compose(
    identifier: str,
    *,
    xc: FunctionalSpec | None = None,
    semilocal_components: (
        Mapping[str, Coefficient] | Iterable[tuple[str, Coefficient]] | None
    ) = None,
    exact_exchange: Coefficient | None = None,
    dispersion: D3Spec | None = None,
    spin: str | None = None,
) -> MethodIR:
    """Compose supported primitives directly into canonical scientific IR."""
    # Selecting the method surface must not activate the XC extension. Reuse
    # its canonical coefficient helpers only when composition is requested.
    from .xc import _coefficient, _normalize_components

    if xc is not None and semilocal_components is not None:
        raise ValueError("provide xc or semilocal_components, not both")

    inherited_exchange = Fraction(0)
    if xc is not None:
        if not isinstance(xc, FunctionalSpec):
            raise TypeError("xc must be a FunctionalSpec from vibeqc.extensions.xc")
        if spin is not None and spin != xc.spin:
            raise UnsupportedMethod(
                f"requested spin {spin!r} conflicts with XC spin {xc.spin!r}"
            )
        spin = xc.spin
        if xc.range_omega or xc.long_range_exchange:
            raise UnsupportedMethod(
                "range-separated XC metadata is not representable by MethodIR yet"
            )
        components = _normalize_components(xc.components)
        inherited_exchange = xc.exact_exchange
    elif semilocal_components is None:
        components = ()
    else:
        # Exact exchange can remain after all semilocal fragments cancel.
        components = _normalize_components(semilocal_components, allow_empty=True)

    if exact_exchange is None:
        exchange = inherited_exchange
    else:
        exchange = _coefficient(exact_exchange, "exact exchange")
        if inherited_exchange and exchange != inherited_exchange:
            raise UnsupportedMethod(
                "explicit exact exchange conflicts with XC exchange metadata"
            )

    spec = MethodSpec(
        identifier=identifier,
        semilocal_components=tuple(components),
        exact_exchange=exchange,
        dispersion=dispersion,
    )
    return _resolve_method(spec, spin="unpolarized" if spin is None else spin)


def resolve(value: str | MethodSpec | MethodIR, *, spin: str | None = None) -> MethodIR:
    """Resolve a built-in or custom declaration to the canonical scientific IR."""
    if isinstance(value, MethodIR):
        if spin is not None and value.spin != spin:
            raise UnsupportedMethod(
                f"existing MethodIR spin {value.spin!r} does not match {spin!r}"
            )
        return value
    return _resolve_method(value, spin="unpolarized" if spin is None else spin)


def named(identifier: str, *, spin: str = "unpolarized") -> MethodIR:
    """Resolve one built-in catalog method through the canonical IR."""
    return _resolve_method(identifier, spin=spin)


def inspect(value: str | MethodSpec | MethodIR, *, spin: str | None = None) -> dict:
    """Return semantic and manifest identities without lowering or execution."""
    ir = resolve(value, spin=spin)
    return {
        "extension_api_version": API_VERSION,
        "kind": "method",
        "identity": ir.identity,
        "manifest_identity": ir.manifest_identity,
        "requirements": ir.requirements,
        "ir": ir.to_payload(),
    }


def verify(
    value: str | MethodSpec | MethodIR,
    *,
    capability: BackendCapability,
    spin: str | None = None,
    dtype: str = "float64",
    derivative_order: int = 0,
) -> TypedMethodIR:
    """Type-check represented science against an explicit backend capability."""
    return verify_method_ir(
        resolve(value, spin=spin),
        capability=capability,
        dtype=dtype,
        derivative_order=derivative_order,
    )


__all__ = [
    "API_VERSION",
    "BackendCapability",
    "D3Spec",
    "MethodIR",
    "MethodSpec",
    "MethodTypeError",
    "TypedMethodIR",
    "UnsupportedMethod",
    "compose",
    "inspect",
    "named",
    "resolve",
    "verify",
]
