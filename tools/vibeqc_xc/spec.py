"""Versioned semilocal functional identity and feature/derivative contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from vibeqc.profiles import canonical_hash, file_hash

VERSION = "libxc-7.0.0/interior-v1"
POLARIZED = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b")
UNPOLARIZED = ("rho", "sigma", "tau")
COMPONENTS = ("LDA_X", "LDA_C_PW", "LDA_C_PW_MOD", "GGA_X_PBE", "GGA_C_PBE")
CATALOG = {
    **{name: ((name, Fraction(1)),) for name in COMPONENTS},
    "LDA_XC_PW": (("LDA_X", Fraction(1)), ("LDA_C_PW", Fraction(1))),
    "PBE": (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
}


class UnsupportedXC(ValueError):
    """The requested expression, derivative, or physical domain is unavailable."""


@dataclass(frozen=True)
class FunctionalSpec:
    """Exact semilocal composition; exchange metadata contributes no local energy.

    Coefficients must be rational (use Fraction with a decimal string). Hybrid
    consumers must apply the separately declared exact-exchange term themselves;
    it is never inferred from the name or added to these semilocal kernels.
    """

    identifier: str
    components: tuple[tuple[str, Fraction], ...]
    spin: str = "polarized"
    version: str = VERSION
    exact_exchange: Fraction = Fraction(0)
    range_omega: Fraction = Fraction(0)
    long_range_exchange: Fraction = Fraction(0)

    def __post_init__(self):
        if (
            not isinstance(self.identifier, str)
            or not self.identifier.strip()
            or self.spin not in ("polarized", "unpolarized")
        ):
            raise UnsupportedXC(
                "functional requires an identifier and supported spin mode"
            )
        if (
            self.version != VERSION
            or not isinstance(self.components, tuple)
            or not self.components
        ):
            raise UnsupportedXC("unsupported expression version or empty composition")
        names = []
        for item in self.components:
            if not isinstance(item, tuple) or len(item) != 2:
                raise UnsupportedXC("components require immutable (ID, Fraction) pairs")
            name, coefficient = item
            if name not in COMPONENTS or not isinstance(coefficient, Fraction):
                raise UnsupportedXC(
                    "components require audited IDs and exact Fraction coefficients"
                )
            names.append(name)
        if len(set(names)) != len(names) or not any(c for _, c in self.components):
            raise UnsupportedXC("duplicate or identically zero composition")
        for value in (self.exact_exchange, self.range_omega, self.long_range_exchange):
            if not isinstance(value, Fraction) or value < 0:
                raise UnsupportedXC(
                    "exchange metadata requires nonnegative exact fractions"
                )

    @property
    def features(self):
        """Feature-major arrays; sigma_ab has no factor two, tau has one-half."""
        return POLARIZED if self.spin == "polarized" else UNPOLARIZED

    @property
    def ingredients(self):
        return (
            ("rho", "sigma")
            if any(n.startswith("GGA") and c for n, c in self.components)
            else ("rho",)
        )

    def to_payload(self):
        """Complete identity including parameter/license provenance and units."""
        root = Path(__file__).resolve().parents[2]
        payload = asdict(self)
        payload["components"] = [[n, str(c)] for n, c in self.components]
        for name in ("exact_exchange", "range_omega", "long_range_exchange"):
            payload[name] = str(getattr(self, name))
        return {
            **payload,
            "ingredients": self.ingredients,
            "features": self.features,
            "derivative_orders": [0, 1, 2],
            "energy": "hartree/bohr^3; e_xc=(rho_a+rho_b)*epsilon_xc",
            "license": "MPL-2.0",
            "source_manifest_sha256": file_hash(
                root / "external/libxc-7.0.0/manifest.json"
            ),
            "expression_source_sha256": file_hash(
                Path(__file__).with_name("expressions.py")
            ),
            "domain": "interior-v1: no clipping; see docs/xc_expressions.md",
        }

    @property
    def identity(self):
        return canonical_hash(self.to_payload())


def functional(identifier, *, spin="polarized"):
    """Resolve only explicit audited catalog names; unknown aliases fail closed."""
    if identifier not in CATALOG:
        raise UnsupportedXC(f"unknown functional {identifier!r}")
    return FunctionalSpec(identifier, CATALOG[identifier], spin)
