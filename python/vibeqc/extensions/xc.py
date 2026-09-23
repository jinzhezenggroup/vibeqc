"""Public builders for exact, auditable XC compositions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

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


def capability(
    identifier: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report evidence-backed support levels for one imported XC registration.

    This is an observation-only public view over the compiler-owned Libxc
    capability record. Representation, pointwise validation, target compilation,
    production-domain qualification, molecular validation, derivatives, and
    public promotion remain distinct; no later tier is inferred from an earlier
    one. Custom compositions therefore do not acquire production support merely
    by choosing an identifier that resembles a built-in name.
    """
    # Capability inspection is itself an explicit extension action. Keep the
    # evidence registry and Maple importer dormant for ordinary XC composition.
    from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
    from vibeqc_compiler.xc.libxc_maple import MapleImportError

    if not isinstance(identifier, str) or not identifier.strip():
        raise TypeError("XC capability queries require a non-empty identifier")
    try:
        record = functional_capability(identifier, evidence=evidence)
    except MapleImportError as exc:
        raise UnsupportedXC(str(exc)) from exc

    qualified = set(record.qualified_stages)
    return {
        "extension_api_version": API_VERSION,
        "kind": "xc-capability",
        "name": record.name,
        "identity": record.identity,
        "representable": "graph-imported" in qualified,
        "pointwise_validated": record.claim_level in qualified,
        "compiled_targets": {
            "cpu": "compiled-cpu" in qualified,
            "cuda": "compiled-cuda" in qualified,
        },
        "production_domain_qualified": "production-domain" in qualified,
        "cuda_runtime_validated": "gpu-runtime" in qualified,
        "molecular_validated": "molecular-scf" in qualified,
        "derivative_validation": {
            "forces": "forces" in qualified,
            "response": "response" in qualified,
        },
        "production_promoted": record.public_dft,
        "qualified_stages": list(record.qualified_stages),
        "ready_stages": list(record.ready_stages),
        "unqualified_stages": list(record.unqualified_stages),
        "stage_evidence": [item.to_payload() for item in record.stage_evidence],
    }


__all__ = [
    "API_VERSION",
    "Coefficient",
    "FunctionalSpec",
    "UnsupportedXC",
    "capability",
    "compose",
    "inspect",
    "named",
]
