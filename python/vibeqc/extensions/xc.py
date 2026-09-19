"""Public builders for exact, auditable XC compositions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction

from vibeqc_compiler.xc.spec import COMPONENTS, FunctionalSpec, UnsupportedXC
from vibeqc_compiler.xc.spec import functional as _functional

API_VERSION = 1
Coefficient = int | str | Fraction


def _coefficient(value: Coefficient, label: str) -> Fraction:
    if type(value) not in (int, str, Fraction):
        raise TypeError(
            f"{label} requires an exact integer, rational string, or Fraction"
        )
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid exact coefficient for {label}") from exc


def _normalize_components(
    components: Mapping[str, Coefficient] | Iterable[tuple[str, Coefficient]],
    *,
    allow_empty: bool = False,
) -> tuple[tuple[str, Fraction], ...]:
    items = components.items() if isinstance(components, Mapping) else components
    totals: dict[str, Fraction] = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("XC components require (component, coefficient) pairs")
        name, value = item
        if name not in COMPONENTS:
            raise UnsupportedXC(f"unsupported XC component {name!r}")
        totals[name] = totals.get(name, Fraction(0)) + _coefficient(
            value, f"component {name}"
        )
    result = tuple((name, value) for name, value in sorted(totals.items()) if value)
    if not result and not allow_empty:
        raise UnsupportedXC("functional composition cannot be empty")
    return result


def compose(
    identifier: str,
    components: Mapping[str, Coefficient] | Iterable[tuple[str, Coefficient]],
    *,
    spin: str = "unpolarized",
    exact_exchange: Coefficient = 0,
    range_omega: Coefficient = 0,
    long_range_exchange: Coefficient = 0,
) -> FunctionalSpec:
    """Build a deterministic FunctionalSpec without approximate float metadata."""
    return FunctionalSpec(
        identifier=identifier,
        components=_normalize_components(components),
        spin=spin,
        exact_exchange=_coefficient(exact_exchange, "exact exchange"),
        range_omega=_coefficient(range_omega, "range omega"),
        long_range_exchange=_coefficient(long_range_exchange, "long-range exchange"),
    )


def named(identifier: str, *, spin: str = "unpolarized") -> FunctionalSpec:
    """Resolve one audited built-in XC definition through the same spec type."""
    return _functional(identifier, spin=spin)


def inspect(spec: FunctionalSpec) -> dict:
    """Return a detached, versioned description of a public XC extension."""
    if not isinstance(spec, FunctionalSpec):
        raise TypeError("XC inspection requires FunctionalSpec")
    return {
        "extension_api_version": API_VERSION,
        "kind": "xc",
        "identity": spec.identity,
        "spec": spec.to_payload(),
    }


__all__ = [
    "API_VERSION",
    "Coefficient",
    "FunctionalSpec",
    "UnsupportedXC",
    "compose",
    "inspect",
    "named",
]
