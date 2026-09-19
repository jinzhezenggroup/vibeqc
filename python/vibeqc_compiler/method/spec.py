"""Canonical DFT method composition IR and audited named manifests (#396).

This module is intentionally representation-only.  It resolves method names or
explicit :class:`MethodSpec` declarations into a backend-neutral graph of typed
primitives.  Runtime/provider capability remains a separate concern: representing
PBE0 here does not make hybrid KS execution available automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc.spec import COMPONENTS, FunctionalSpec
from vibeqc_compiler.xc.spec import VERSION as XC_VERSION

from .dispersion import D3Spec, DispersionCorrectionPrimitive
from .nonlocal_correlation import (
    NonlocalCorrelationPrimitive,
    NonlocalCorrelationSpec,
)

METHOD_IR_VERSION = "dft-method-ir-v1"
METHOD_CATALOG_VERSION = "dft-method-catalog-v1"
FULL_RANGE = "full-range"
_SPINS = ("polarized", "unpolarized")
_INGREDIENT_ORDER = ("rho", "sigma", "tau")


class UnsupportedMethod(ValueError):
    """The requested method composition cannot be represented by this IR."""


def _require_fraction(value, label):
    if not isinstance(value, Fraction):
        raise UnsupportedMethod(f"{label} requires an exact Fraction coefficient")
    return value


def _canonical_components(components):
    totals = {}
    for name, coefficient in components:
        totals[name] = totals.get(name, Fraction(0)) + coefficient
    return tuple(
        (name, coefficient)
        for name, coefficient in sorted(totals.items())
        if coefficient
    )


@dataclass(frozen=True)
class MethodSpec:
    """Declarative method composition before canonical primitive resolution.

    ``semilocal_components`` uses the audited scalar-XC component IDs owned by
    :mod:`vibeqc_compiler.xc`.  Duplicate component entries are legal here so a
    caller can compose fragments; resolution combines them exactly and removes
    cancellations before constructing ``MethodIR``.
    """

    identifier: str
    semilocal_components: tuple[tuple[str, Fraction], ...]
    exact_exchange: Fraction = Fraction(0)
    version: str = METHOD_CATALOG_VERSION
    dispersion: D3Spec | None = None
    nonlocal_correlation: NonlocalCorrelationSpec | None = None

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise UnsupportedMethod("method requires a non-empty identifier")
        if self.version != METHOD_CATALOG_VERSION:
            raise UnsupportedMethod("unsupported method manifest version")
        if not isinstance(self.semilocal_components, tuple):
            raise UnsupportedMethod("semilocal components must be an immutable tuple")
        for item in self.semilocal_components:
            if not isinstance(item, tuple) or len(item) != 2:
                raise UnsupportedMethod(
                    "semilocal components require immutable (ID, Fraction) pairs"
                )
            name, coefficient = item
            if name not in COMPONENTS:
                raise UnsupportedMethod(f"unsupported semilocal component {name!r}")
            _require_fraction(coefficient, f"component {name}")
            if not coefficient:
                raise UnsupportedMethod("zero-valued manifest components are ambiguous")
        if self.nonlocal_correlation is not None and not isinstance(
            self.nonlocal_correlation, NonlocalCorrelationSpec
        ):
            raise TypeError("nonlocal correlation requires NonlocalCorrelationSpec")
        if self.dispersion is not None and not isinstance(self.dispersion, D3Spec):
            raise TypeError("dispersion requires a D3Spec")
        _require_fraction(self.exact_exchange, "exact exchange")
        if self.exact_exchange < 0:
            raise UnsupportedMethod("exact-exchange coefficient must be nonnegative")
        if (
            not self.semilocal_components
            and not self.exact_exchange
            and self.nonlocal_correlation is None
        ):
            raise UnsupportedMethod("method composition cannot be empty")

    def to_payload(self):
        return {
            "identifier": self.identifier,
            "version": self.version,
            "semilocal_components": [
                [name, str(coefficient)]
                for name, coefficient in self.semilocal_components
            ],
            "exact_exchange": str(self.exact_exchange),
            **(
                {"nonlocal_correlation": self.nonlocal_correlation.to_payload()}
                if self.nonlocal_correlation
                else {}
            ),
            **({"dispersion": self.dispersion.to_payload()} if self.dispersion else {}),
        }


@dataclass(frozen=True)
class SemilocalXCPrimitive:
    """One canonical semilocal-XC node backed by the #161 expression compiler.

    Direct construction normalizes component order and removes inactive terms,
    just like ``resolve_method``, while retaining the functional's provenance.
    """

    functional: FunctionalSpec
    kind: ClassVar[str] = "semilocal_xc"

    def __post_init__(self):
        if not isinstance(self.functional, FunctionalSpec):
            raise TypeError("semilocal primitive requires FunctionalSpec")
        if any(
            (
                self.functional.exact_exchange,
                self.functional.range_omega,
                self.functional.long_range_exchange,
            )
        ):
            raise UnsupportedMethod(
                "semilocal primitive cannot hide exchange-operator metadata"
            )
        # FunctionalSpec intentionally preserves its declaration order. Normalize
        # at this boundary so catalog specs and explicit MethodIR construction
        # share a semantic identity without mutating the caller's XC declaration.
        components = _canonical_components(self.functional.components)
        if components != self.functional.components:
            object.__setattr__(
                self, "functional", replace(self.functional, components=components)
            )

    @property
    def derivative_capabilities(self):
        return ("energy-density", "feature-gradient", "feature-hessian")

    def semantic_payload(self):
        functional = self.functional.to_payload()
        # The functional identifier is descriptive.  MethodIR semantic identity is
        # determined by audited expressions, coefficients, spin and provenance.
        functional.pop("identifier", None)
        return {
            "kind": self.kind,
            "functional": functional,
            "derivative_capabilities": self.derivative_capabilities,
        }

    def to_payload(self):
        return {
            **self.semantic_payload(),
            "functional_identifier": self.functional.identifier,
        }


@dataclass(frozen=True)
class ExactExchangePrimitive:
    """Structural exact-exchange node; provider selection stays runtime-owned."""

    coefficient: Fraction
    operator: str = FULL_RANGE
    kind: ClassVar[str] = "exact_exchange"

    def __post_init__(self):
        _require_fraction(self.coefficient, "exact exchange")
        if self.coefficient <= 0:
            raise UnsupportedMethod(
                "exact-exchange primitive requires a positive weight"
            )
        if self.operator != FULL_RANGE:
            raise UnsupportedMethod(
                "only full-range exact exchange is representable in the first MethodIR slice"
            )

    @property
    def derivative_capabilities(self):
        # These are representation/provider requests, not public method guarantees.
        return ("energy", "fock")

    def semantic_payload(self):
        return {
            "kind": self.kind,
            "operator": self.operator,
            "coefficient": str(self.coefficient),
            "derivative_capabilities": self.derivative_capabilities,
        }

    def to_payload(self):
        return self.semantic_payload()


MethodPrimitive = (
    SemilocalXCPrimitive
    | ExactExchangePrimitive
    | NonlocalCorrelationPrimitive
    | DispersionCorrectionPrimitive
)


@dataclass(frozen=True)
class MethodIR:
    """Canonical, backend-neutral DFT method graph.

    ``identity`` hashes semantic composition and provenance but intentionally not
    the descriptive manifest name.  ``manifest_identity`` includes that name for
    result provenance, so aliases can remain auditable without fragmenting caches
    for mathematically identical graphs.
    """

    identifier: str
    spin: str
    primitives: tuple[MethodPrimitive, ...]
    version: str = METHOD_IR_VERSION

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise UnsupportedMethod("MethodIR requires a non-empty identifier")
        if self.spin not in _SPINS:
            raise UnsupportedMethod(f"unsupported spin mode {self.spin!r}")
        if self.version != METHOD_IR_VERSION:
            raise UnsupportedMethod("unsupported MethodIR version")
        if not isinstance(self.primitives, tuple) or not self.primitives:
            raise UnsupportedMethod("MethodIR requires at least one primitive")
        allowed = (
            SemilocalXCPrimitive,
            ExactExchangePrimitive,
            NonlocalCorrelationPrimitive,
            DispersionCorrectionPrimitive,
        )
        if not all(isinstance(primitive, allowed) for primitive in self.primitives):
            raise UnsupportedMethod("MethodIR contains an unsupported primitive")

        def primitive_order(primitive: MethodPrimitive) -> int:
            if isinstance(primitive, SemilocalXCPrimitive):
                return 0
            if isinstance(primitive, ExactExchangePrimitive):
                return 1
            if isinstance(primitive, NonlocalCorrelationPrimitive):
                return 2
            return 3

        keys = [primitive_order(primitive) for primitive in self.primitives]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise UnsupportedMethod(
                "MethodIR primitives must be canonical and unique by primitive family"
            )
        semilocal = [
            primitive
            for primitive in self.primitives
            if isinstance(primitive, SemilocalXCPrimitive)
        ]
        if semilocal and semilocal[0].functional.spin != self.spin:
            raise UnsupportedMethod("semilocal primitive spin does not match MethodIR")

    @property
    def reference(self):
        return "unrestricted" if self.spin == "polarized" else "restricted"

    @property
    def requirements(self):
        ingredients = set()
        operators = []
        for primitive in self.primitives:
            if isinstance(primitive, SemilocalXCPrimitive):
                ingredients.update(primitive.functional.ingredients)
                operators.append("semilocal-xc")
            elif isinstance(primitive, ExactExchangePrimitive):
                operators.append(primitive.operator + "-exchange")
            elif isinstance(primitive, NonlocalCorrelationPrimitive):
                ingredients.update(primitive.required_ingredients)
                operators.append("nonlocal-correlation")
            else:
                operators.append("geometry-d3-bj")
        return {
            "spin": self.spin,
            "reference": self.reference,
            "ingredients": tuple(
                name for name in _INGREDIENT_ORDER if name in ingredients
            ),
            "operators": tuple(operators),
        }

    def semantic_payload(self):
        return {
            "version": self.version,
            "spin": self.spin,
            "reference": self.reference,
            "primitives": [
                primitive.semantic_payload() for primitive in self.primitives
            ],
        }

    def to_payload(self):
        return {
            "identifier": self.identifier,
            **self.semantic_payload(),
            "requirements": self.requirements,
            "primitives": [primitive.to_payload() for primitive in self.primitives],
        }

    @property
    def identity(self):
        return canonical_hash(self.semantic_payload())

    @property
    def manifest_identity(self):
        return canonical_hash(self.to_payload())


METHOD_CATALOG = MappingProxyType(
    {
        "LDA_XC_PW": MethodSpec(
            "LDA_XC_PW",
            (("LDA_X", Fraction(1)), ("LDA_C_PW", Fraction(1))),
        ),
        "PBE": MethodSpec(
            "PBE",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        ),
        "R2SCAN": MethodSpec(
            "R2SCAN",
            (("MGGA_X_R2SCAN", Fraction(1)), ("MGGA_C_R2SCAN", Fraction(1))),
        ),
        "PBE0": MethodSpec(
            "PBE0",
            (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 4),
        ),
    }
)


def resolve_method(method, *, spin="unpolarized"):
    """Resolve an audited name or explicit ``MethodSpec`` into canonical MethodIR."""
    if spin not in _SPINS:
        raise UnsupportedMethod(f"unsupported spin mode {spin!r}")
    if isinstance(method, str):
        try:
            spec = METHOD_CATALOG[method]
        except KeyError as error:
            raise UnsupportedMethod(f"unknown DFT method {method!r}") from error
    elif isinstance(method, MethodSpec):
        spec = method
    else:
        raise TypeError("method must be a catalog name or MethodSpec")

    components = _canonical_components(spec.semilocal_components)
    primitives: list[MethodPrimitive] = []
    if components:
        functional = FunctionalSpec(
            identifier="method-ir-semilocal",
            components=components,
            spin=spin,
            version=XC_VERSION,
        )
        primitives.append(SemilocalXCPrimitive(functional))
    if spec.exact_exchange:
        primitives.append(ExactExchangePrimitive(spec.exact_exchange))
    if spec.nonlocal_correlation is not None:
        primitives.append(NonlocalCorrelationPrimitive(spec.nonlocal_correlation))
    if not primitives:
        raise UnsupportedMethod("method components cancel to an empty graph")
    if spec.dispersion is not None:
        primitives.append(DispersionCorrectionPrimitive(spec.dispersion))
    return MethodIR(spec.identifier, spin, tuple(primitives))
