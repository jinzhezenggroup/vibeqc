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

from .basis_binding import BasisBinding, r2scan3c_def2_mtzvpp_h_ar
from .dispersion import (
    D3Spec,
    D4Spec,
    DispersionCorrectionPrimitive,
    pbe0_d3_bj_spec,
    pbe_d3_bj_spec,
    r2scan3c_d4_eeq,
)
from .gcp import GCPSpec, GeometricCounterpoisePrimitive, r2scan3c_gcp
from .nonlocal_correlation import (
    NonlocalCorrelationPrimitive,
    NonlocalCorrelationSpec,
)

METHOD_IR_VERSION = "dft-method-ir-v1"
METHOD_CATALOG_VERSION = "dft-method-catalog-v1"
FULL_RANGE = "full-range"
SHORT_RANGE = "short-range"
LONG_RANGE = "long-range"
_RANGE_OPERATORS = (SHORT_RANGE, LONG_RANGE)
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
    short_range_exchange: Fraction = Fraction(0)
    long_range_exchange: Fraction = Fraction(0)
    range_omega: Fraction = Fraction(0)
    version: str = METHOD_CATALOG_VERSION
    dispersion: D3Spec | D4Spec | None = None
    nonlocal_correlation: NonlocalCorrelationSpec | None = None
    basis: BasisBinding | None = None
    gcp: GCPSpec | None = None

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
        if self.dispersion is not None and not isinstance(
            self.dispersion, (D3Spec, D4Spec)
        ):
            raise TypeError("dispersion requires a D3Spec or D4Spec")
        if self.basis is not None and not isinstance(self.basis, BasisBinding):
            raise TypeError("basis requires BasisBinding")
        if self.gcp is not None and not isinstance(self.gcp, GCPSpec):
            raise TypeError("gCP requires GCPSpec")
        if self.gcp is not None and self.basis is None:
            raise UnsupportedMethod("gCP requires an explicit basis binding")
        if (
            self.gcp is not None
            and self.basis is not None
            and self.gcp.basis != self.basis.name
        ):
            raise UnsupportedMethod("gCP basis does not match the method basis binding")
        for label, value in (
            ("exact exchange", self.exact_exchange),
            ("short-range exchange", self.short_range_exchange),
            ("long-range exchange", self.long_range_exchange),
            ("range omega", self.range_omega),
        ):
            _require_fraction(value, label)
            if value < 0:
                raise UnsupportedMethod(f"{label} must be nonnegative")
        has_range_exchange = bool(self.short_range_exchange or self.long_range_exchange)
        has_range_semilocal = any(
            name == "GGA_X_ITYH" and coefficient
            for name, coefficient in self.semilocal_components
        )
        if bool(self.range_omega) != (has_range_exchange or has_range_semilocal):
            raise UnsupportedMethod(
                "range-separated composition requires one positive range_omega, "
                "which is otherwise forbidden"
            )
        if not (
            self.semilocal_components
            or self.exact_exchange
            or has_range_exchange
            or self.nonlocal_correlation
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
            "short_range_exchange": str(self.short_range_exchange),
            "long_range_exchange": str(self.long_range_exchange),
            "range_omega": str(self.range_omega),
            **({"dispersion": self.dispersion.to_payload()} if self.dispersion else {}),
            **({"basis": self.basis.to_payload()} if self.basis else {}),
            **({"gcp": self.gcp.to_payload()} if self.gcp else {}),
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
                self.functional.long_range_exchange,
            )
        ):
            raise UnsupportedMethod(
                "semilocal primitive cannot hide exact-exchange operators"
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
class RangeSeparatedExchangePrimitive:
    """Explicit SR/LR exact exchange with one immutable inverse-Bohr omega."""

    coefficient: Fraction
    omega: Fraction
    operator: str
    kind: ClassVar[str] = "range_separated_exchange"

    def __post_init__(self):
        _require_fraction(self.coefficient, "range-separated exchange")
        _require_fraction(self.omega, "range omega")
        if self.coefficient <= 0 or self.omega <= 0:
            raise UnsupportedMethod(
                "range-separated exchange requires positive coefficient and omega"
            )
        if self.operator not in _RANGE_OPERATORS:
            raise UnsupportedMethod(
                f"unsupported range-separated operator {self.operator!r}"
            )

    @property
    def derivative_capabilities(self):
        return ("energy", "fock")

    def semantic_payload(self):
        return {
            "kind": self.kind,
            "operator": self.operator,
            "coefficient": str(self.coefficient),
            "omega": str(self.omega),
            "omega_units": "bohr^-1",
            "derivative_capabilities": self.derivative_capabilities,
        }

    def to_payload(self):
        return self.semantic_payload()


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
    | RangeSeparatedExchangePrimitive
    | DispersionCorrectionPrimitive
    | GeometricCounterpoisePrimitive
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
    basis: BasisBinding | None = None
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
            RangeSeparatedExchangePrimitive,
            DispersionCorrectionPrimitive,
            GeometricCounterpoisePrimitive,
        )
        if not all(isinstance(primitive, allowed) for primitive in self.primitives):
            raise UnsupportedMethod("MethodIR contains an unsupported primitive")

        def primitive_order(primitive: MethodPrimitive) -> int:
            if isinstance(primitive, SemilocalXCPrimitive):
                return 0
            if isinstance(primitive, ExactExchangePrimitive):
                return 1
            if isinstance(primitive, RangeSeparatedExchangePrimitive):
                return 2 if primitive.operator == SHORT_RANGE else 3
            if isinstance(primitive, NonlocalCorrelationPrimitive):
                return 4
            if isinstance(primitive, DispersionCorrectionPrimitive):
                return 5
            return 6

        keys = [primitive_order(primitive) for primitive in self.primitives]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise UnsupportedMethod(
                "MethodIR primitives must be canonical and unique by operator family"
            )
        semilocal = [
            primitive
            for primitive in self.primitives
            if isinstance(primitive, SemilocalXCPrimitive)
        ]
        if semilocal and semilocal[0].functional.spin != self.spin:
            raise UnsupportedMethod("semilocal primitive spin does not match MethodIR")
        if self.basis is not None and not isinstance(self.basis, BasisBinding):
            raise TypeError("MethodIR basis requires BasisBinding")
        gcp = [
            p for p in self.primitives if isinstance(p, GeometricCounterpoisePrimitive)
        ]
        if gcp:
            if self.basis is None:
                raise UnsupportedMethod(
                    "gCP primitive requires a MethodIR basis binding"
                )
            if gcp[0].specification.basis != self.basis.name:
                raise UnsupportedMethod(
                    "gCP primitive basis does not match MethodIR basis"
                )
        if self.identifier == "R2SCAN-3c":
            canonical = METHOD_CATALOG["R2SCAN-3c"]
            if self.basis != canonical.basis or self.primitives != _method_primitives(
                canonical, self.spin
            ):
                raise UnsupportedMethod(
                    "R2SCAN-3c is a canonical manifest; changed defining "
                    "components require a different explicit identifier"
                )

    @property
    def reference(self):
        return "unrestricted" if self.spin == "polarized" else "restricted"

    def preflight_atomic_numbers(self, atomic_numbers):
        """Reject unsupported chemistry before lowering or correction execution."""

        values = tuple(atomic_numbers)
        if any(type(z) is not int or not 1 <= z <= 118 for z in values):
            raise UnsupportedMethod("atomic numbers must be integers in [1, 118]")
        if self.basis is not None:
            try:
                self.basis.require_atomic_numbers(values)
            except (TypeError, ValueError) as error:
                raise UnsupportedMethod(str(error)) from error
        for primitive in self.primitives:
            if isinstance(primitive, GeometricCounterpoisePrimitive):
                unsupported = tuple(
                    sorted(
                        set(values)
                        - set(primitive.specification.supported_atomic_numbers)
                    )
                )
                if unsupported:
                    raise UnsupportedMethod(
                        f"gCP does not support atomic numbers {unsupported} "
                        "in this canonical method domain"
                    )
        return values

    @property
    def requirements(self):
        ingredients = set()
        operators = []
        for primitive in self.primitives:
            if isinstance(primitive, SemilocalXCPrimitive):
                ingredients.update(primitive.functional.ingredients)
                operators.append("semilocal-xc")
            elif isinstance(
                primitive, (ExactExchangePrimitive, RangeSeparatedExchangePrimitive)
            ):
                operators.append(primitive.operator + "-exchange")
            elif isinstance(primitive, NonlocalCorrelationPrimitive):
                ingredients.update(primitive.required_ingredients)
                operators.append("nonlocal-correlation")
            elif isinstance(primitive, DispersionCorrectionPrimitive):
                if isinstance(primitive.specification, D3Spec):
                    operators.append("geometry-d3-bj")
                else:
                    operators.append("geometry-d4-bj-eeq")
            else:
                operators.append("geometry-gcp")
        result = {
            "spin": self.spin,
            "reference": self.reference,
            "ingredients": tuple(
                name for name in _INGREDIENT_ORDER if name in ingredients
            ),
            "operators": tuple(operators),
        }
        if self.basis is not None:
            result["basis"] = self.basis.to_payload()
        return result

    def semantic_payload(self):
        return {
            "version": self.version,
            "spin": self.spin,
            "reference": self.reference,
            "basis": self.basis.semantic_payload() if self.basis else None,
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
        "PW91": MethodSpec(
            "PW91",
            (("GGA_X_PW91", Fraction(1)), ("GGA_C_PW91", Fraction(1))),
        ),
        "PW91PW91": MethodSpec(
            "PW91PW91",
            (("GGA_X_PW91", Fraction(1)), ("GGA_C_PW91", Fraction(1))),
        ),
        "R2SCAN": MethodSpec(
            "R2SCAN",
            (("MGGA_X_R2SCAN", Fraction(1)), ("MGGA_C_R2SCAN", Fraction(1))),
        ),
        "R2SCANH": MethodSpec(
            "R2SCANH",
            (("MGGA_X_R2SCAN", Fraction(9, 10)), ("MGGA_C_R2SCAN", Fraction(1))),
            exact_exchange=Fraction(1, 10),
        ),
        "R2SCAN0": MethodSpec(
            "R2SCAN0",
            (("MGGA_X_R2SCAN", Fraction(3, 4)), ("MGGA_C_R2SCAN", Fraction(1))),
            exact_exchange=Fraction(1, 4),
        ),
        "R2SCAN50": MethodSpec(
            "R2SCAN50",
            (("MGGA_X_R2SCAN", Fraction(1, 2)), ("MGGA_C_R2SCAN", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        "R2SCAN-3c": MethodSpec(
            "R2SCAN-3c",
            (("MGGA_X_R2SCAN", Fraction(1)), ("MGGA_C_R2SCAN", Fraction(1))),
            dispersion=r2scan3c_d4_eeq(),
            basis=r2scan3c_def2_mtzvpp_h_ar(),
            gcp=r2scan3c_gcp(),
        ),
        "PBE0": MethodSpec(
            "PBE0",
            (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 4),
        ),
        "PBE1PBE": MethodSpec(
            "PBE1PBE",
            (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 4),
        ),
        "PBEH": MethodSpec(
            "PBEH",
            (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 4),
        ),
        "PBE50": MethodSpec(
            "PBE50",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        "BLYP": MethodSpec(
            "BLYP",
            (("GGA_X_B88", Fraction(1)), ("GGA_C_LYP", Fraction(1))),
        ),
        "BP86": MethodSpec(
            "BP86",
            (("GGA_X_B88", Fraction(1)), ("GGA_C_P86", Fraction(1))),
        ),
        "B3P86": MethodSpec(
            "B3P86",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN_RPA", Fraction(19, 100)),
                ("GGA_C_P86", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "B3P86G": MethodSpec(
            "B3P86G",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN_RPA", Fraction(19, 100)),
                ("GGA_C_P86", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "B3P86V5": MethodSpec(
            "B3P86V5",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_P86", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "B3LYP": MethodSpec(
            "B3LYP",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN_RPA", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "B3LYPG": MethodSpec(
            "B3LYPG",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN_RPA", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        # Keep the VWN5 variant explicit: it is a distinct Libxc/PySCF
        # composition from the Gaussian-compatible VWN-RPA B3LYP above.
        "B3LYP5": MethodSpec(
            "B3LYP5",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "B3PW91": MethodSpec(
            "B3PW91",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(18, 25)),
                ("LDA_C_PW", Fraction(19, 100)),
                ("GGA_C_PW91", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 5),
        ),
        "X3LYP": MethodSpec(
            "X3LYP",
            (
                ("LDA_X", Fraction(73, 1000)),
                ("GGA_X_B88", Fraction(108477, 200000)),
                ("GGA_X_PW91", Fraction(33323, 200000)),
                ("LDA_C_VWN_RPA", Fraction(129, 1000)),
                ("GGA_C_LYP", Fraction(871, 1000)),
            ),
            exact_exchange=Fraction(109, 500),
        ),
        "X3LYPG": MethodSpec(
            "X3LYPG",
            (
                ("LDA_X", Fraction(73, 1000)),
                ("GGA_X_B88", Fraction(108477, 200000)),
                ("GGA_X_PW91", Fraction(33323, 200000)),
                ("LDA_C_VWN_RPA", Fraction(129, 1000)),
                ("GGA_C_LYP", Fraction(871, 1000)),
            ),
            exact_exchange=Fraction(109, 500),
        ),
        "X3LYP5": MethodSpec(
            "X3LYP5",
            (
                ("LDA_X", Fraction(73, 1000)),
                ("GGA_X_B88", Fraction(108477, 200000)),
                ("GGA_X_PW91", Fraction(33323, 200000)),
                ("LDA_C_VWN", Fraction(129, 1000)),
                ("GGA_C_LYP", Fraction(871, 1000)),
            ),
            exact_exchange=Fraction(109, 500),
        ),
        "B5050LYP": MethodSpec(
            "B5050LYP",
            (
                ("LDA_X", Fraction(2, 25)),
                ("GGA_X_B88", Fraction(21, 50)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            exact_exchange=Fraction(1, 2),
        ),
        "BHANDH": MethodSpec(
            "BHANDH",
            (("LDA_X", Fraction(1, 2)), ("GGA_C_LYP", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        "BHANDHLYP": MethodSpec(
            "BHANDHLYP",
            (("GGA_X_B88", Fraction(1, 2)), ("GGA_C_LYP", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        "BHHLYP": MethodSpec(
            "BHHLYP",
            (("GGA_X_B88", Fraction(1, 2)), ("GGA_C_LYP", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        "PBE-D3(BJ)": MethodSpec(
            "PBE-D3(BJ)",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            dispersion=pbe_d3_bj_spec(),
        ),
        "PBE0-D3(BJ)": MethodSpec(
            "PBE0-D3(BJ)",
            (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 4),
            dispersion=pbe0_d3_bj_spec(),
        ),
        "CAM-B3LYP": MethodSpec(
            "CAM-B3LYP",
            (
                ("GGA_X_B88", Fraction(35, 100)),
                ("GGA_X_ITYH", Fraction(46, 100)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            short_range_exchange=Fraction(19, 100),
            long_range_exchange=Fraction(65, 100),
            range_omega=Fraction(33, 100),
        ),
        "CAMB3LYP": MethodSpec(
            "CAMB3LYP",
            (
                ("GGA_X_B88", Fraction(35, 100)),
                ("GGA_X_ITYH", Fraction(46, 100)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            short_range_exchange=Fraction(19, 100),
            long_range_exchange=Fraction(65, 100),
            range_omega=Fraction(33, 100),
        ),
        "CAMH-B3LYP": MethodSpec(
            "CAMH-B3LYP",
            (
                ("GGA_X_B88", Fraction(50, 100)),
                ("GGA_X_ITYH", Fraction(31, 100)),
                ("LDA_C_VWN", Fraction(19, 100)),
                ("GGA_C_LYP", Fraction(81, 100)),
            ),
            short_range_exchange=Fraction(19, 100),
            long_range_exchange=Fraction(50, 100),
            range_omega=Fraction(33, 100),
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
        if spec.identifier == "R2SCAN-3c" and spec != METHOD_CATALOG["R2SCAN-3c"]:
            raise UnsupportedMethod(
                "R2SCAN-3c is a canonical manifest; changed defining "
                "components require a different explicit identifier"
            )
    else:
        raise TypeError("method must be a catalog name or MethodSpec")

    return MethodIR(
        spec.identifier, spin, _method_primitives(spec, spin), basis=spec.basis
    )


def _method_primitives(spec, spin):
    """One construction path shared by resolution and canonical graph validation."""
    components = _canonical_components(spec.semilocal_components)
    primitives: list[MethodPrimitive] = []
    if components:
        functional = FunctionalSpec(
            identifier="method-ir-semilocal",
            components=components,
            spin=spin,
            version=XC_VERSION,
            range_omega=spec.range_omega,
        )
        primitives.append(SemilocalXCPrimitive(functional))
    if spec.exact_exchange:
        primitives.append(ExactExchangePrimitive(spec.exact_exchange))
    if spec.short_range_exchange:
        primitives.append(
            RangeSeparatedExchangePrimitive(
                spec.short_range_exchange, spec.range_omega, SHORT_RANGE
            )
        )
    if spec.long_range_exchange:
        primitives.append(
            RangeSeparatedExchangePrimitive(
                spec.long_range_exchange, spec.range_omega, LONG_RANGE
            )
        )
    if spec.nonlocal_correlation is not None:
        primitives.append(NonlocalCorrelationPrimitive(spec.nonlocal_correlation))
    if not primitives:
        raise UnsupportedMethod("method components cancel to an empty graph")
    if spec.dispersion is not None:
        primitives.append(DispersionCorrectionPrimitive(spec.dispersion))
    if spec.gcp is not None:
        primitives.append(GeometricCounterpoisePrimitive(spec.gcp))
    return tuple(primitives)
