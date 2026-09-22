"""Versioned semilocal functional identity and feature/derivative contracts."""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash

from .b88_vwn_maple import b88_vwn_maple_provenance
from .p86_pz_maple import p86_pz_maple_provenance
from .pbe_maple import pbe_maple_provenance
from .pw91_maple import pw91_maple_provenance
from .pw_maple import pw_maple_provenance
from .rsh_maple import rsh_maple_provenance
from .scan_maple import scan_maple_provenance
from .wb97mv_maple import wb97mv_maple_provenance

VERSION = "libxc-7.0.0/interior-v1"
POLARIZED = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b")
UNPOLARIZED = ("rho", "sigma", "tau")
PUBLIC_COMPONENTS = (
    "LDA_X",
    "LDA_C_PW",
    "LDA_C_PW_MOD",
    "GGA_X_PBE",
    "GGA_C_PBE",
    "MGGA_X_SCAN",
    "MGGA_C_SCAN",
    "MGGA_X_R2SCAN",
    "MGGA_C_R2SCAN",
)
RSH_COMPONENTS = (
    "GGA_X_B88",
    "GGA_X_ITYH",
    "LDA_C_VWN",
    "LDA_C_VWN_RPA",
    "GGA_C_LYP",
)
PW91_COMPONENTS = ("GGA_X_PW91", "GGA_C_PW91")
P86_COMPONENTS = ("LDA_C_PZ", "GGA_C_P86")
WB97MV_COMPONENTS = ("MGGA_X_WB97M_V", "MGGA_C_WB97M_V")
SPECIAL_EXPRESSION_COMPONENTS = RSH_COMPONENTS + PW91_COMPONENTS + P86_COMPONENTS
COMPONENTS = PUBLIC_COMPONENTS + SPECIAL_EXPRESSION_COMPONENTS + WB97MV_COMPONENTS
CATALOG = {
    **{name: ((name, Fraction(1)),) for name in PUBLIC_COMPONENTS},
    "LDA_XC_PW": (("LDA_X", Fraction(1)), ("LDA_C_PW", Fraction(1))),
    "PBE": (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    "SCAN": (
        ("MGGA_X_SCAN", Fraction(1)),
        ("MGGA_C_SCAN", Fraction(1)),
    ),
    "R2SCAN": (
        ("MGGA_X_R2SCAN", Fraction(1)),
        ("MGGA_C_R2SCAN", Fraction(1)),
    ),
}


class UnsupportedXC(ValueError):
    """The requested expression, derivative, or physical domain is unavailable."""


@dataclass(frozen=True)
class FunctionalSpec:
    """Exact semilocal composition with explicit scalar range parameters.

    Coefficients and range parameters must be rational. range_omega is consumed
    by ITYH when that semilocal component is present, while generic public
    compositions may also carry it as exact range-exchange metadata. Exchange
    operator weights remain separate method primitives and are never inferred
    from a functional name.
    """

    identifier: str
    components: tuple[tuple[str, Fraction], ...]
    spin: str = "polarized"
    version: str = VERSION
    exact_exchange: Fraction = Fraction(0)
    range_omega: Fraction = Fraction(0)
    long_range_exchange: Fraction = Fraction(0)

    def __post_init__(self) -> None:
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
        has_range_semilocal = any(
            name in ("GGA_X_ITYH", "MGGA_X_WB97M_V") and coefficient
            for name, coefficient in self.components
        )
        if has_range_semilocal and not self.range_omega:
            raise UnsupportedXC(
                "range-dependent semilocal exchange requires one positive range_omega"
            )

    @property
    def features(self) -> typing.Any:
        """Feature-major arrays; sigma_ab has no factor two, tau has one-half."""
        return POLARIZED if self.spin == "polarized" else UNPOLARIZED

    @property
    def ingredients(self) -> typing.Any:
        if any(
            name.startswith("MGGA") and coefficient
            for name, coefficient in self.components
        ):
            return ("rho", "sigma", "tau")
        if any(
            name.startswith("GGA") and coefficient
            for name, coefficient in self.components
        ):
            return ("rho", "sigma")
        return ("rho",)

    def to_payload(self) -> typing.Any:
        """Complete identity including parameter/license provenance and units."""
        payload = asdict(self)
        payload["components"] = [[n, str(c)] for n, c in self.components]
        for name in ("exact_exchange", "range_omega", "long_range_exchange"):
            payload[name] = str(getattr(self, name))
        special = any(
            name in SPECIAL_EXPRESSION_COMPONENTS and coefficient
            for name, coefficient in self.components
        )
        wb97mv = any(
            name in WB97MV_COMPONENTS and coefficient
            for name, coefficient in self.components
        )
        if wb97mv:
            manifest = "wb97mv-manifest.json"
            expression_source = "wb97mv_maple.py"
        else:
            manifest = "rsh-manifest.json" if special else "manifest.json"
            expression_source = "rsh_expressions.py" if special else "expressions.py"
        sources = tuple(
            record
            for record in (
                pbe_maple_provenance(self.components),
                pw_maple_provenance(self.components),
                pw91_maple_provenance(self.components),
                p86_pz_maple_provenance(self.components),
                b88_vwn_maple_provenance(self.components),
                rsh_maple_provenance(self.components),
                scan_maple_provenance(self.components),
                wb97mv_maple_provenance(self.components, self.range_omega),
            )
            if record is not None
        )
        expression_provenance = sources[0] if sources else None
        if len(sources) > 1:
            for record in sources[1:]:
                for key in ("kind", "importer_semantics", "importer_sha256"):
                    if record[key] != sources[0][key]:
                        raise UnsupportedXC("incompatible Libxc Maple provenance")
            expression_provenance = {
                **sources[0],
                "adapter_sha256": canonical_hash(
                    sorted(record["adapter_sha256"] for record in sources)
                ),
                "components": {
                    key: value
                    for record in sources
                    for key, value in record["components"].items()
                },
            }
        return {
            **payload,
            "ingredients": self.ingredients,
            "features": self.features,
            "derivative_orders": [0, 1, 2],
            "energy": "hartree/bohr^3; e_xc=(rho_a+rho_b)*epsilon_xc",
            "license": "MPL-2.0",
            "source_manifest_sha256": file_hash(
                asset_path(f"external/libxc-7.0.0/{manifest}")
            ),
            "expression_source_sha256": file_hash(
                Path(__file__).with_name(expression_source)
            ),
            **(
                {"expression_provenance": expression_provenance}
                if expression_provenance is not None
                else {}
            ),
            "domain": (
                "wb97mv-interior-v1: Libxc-7.0 B97M polynomial plus direct "
                "erf attenuation branch a<1.35; no clipping"
                if wb97mv
                else (
                    "rsh-interior-v1: interior-v1 plus explicit ITYH attenuation "
                    "branch support; no clipping"
                    if self.range_omega
                    else "interior-v1: no clipping; see docs/xc_expressions.md"
                )
            ),
        }

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())


def functional(identifier: typing.Any, *, spin: typing.Any = "polarized") -> typing.Any:
    """Resolve only explicit audited catalog names; unknown aliases fail closed."""
    if identifier not in CATALOG:
        raise UnsupportedXC(f"unknown functional {identifier!r}")
    return FunctionalSpec(identifier, CATALOG[identifier], spin)
