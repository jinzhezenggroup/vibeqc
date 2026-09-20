"""Explicit two-body D3(BJ) composition and production correction semantics."""

import math
import re
import typing
from dataclasses import asdict, dataclass
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash

D3_TABLE_SHA256 = "9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e"
D3_RADII_SHA256 = "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"


@dataclass(frozen=True)
class D3Spec:
    """Versioned two-body D3(BJ) model and numerical approximation identity.

    Cutoffs are in bohr; None means no cutoff. The GFN1 compatibility values
    (25/50 bohr and a 0.05-bohr pair switch) must be requested explicitly.
    Nonzero ATM and other damping variants are rejected in this initial slice.
    Table identities hash the actual canonical data, not the functional name.
    """

    s6: float
    s8: float
    a1: float
    a2: float
    table_sha256: str
    radii_sha256: str
    s9: float = 0.0
    damping: str = "bj"
    cn_cutoff: float | None = None
    pair_cutoff: float | None = None
    pair_switch_width: float = 0.0
    version: str = "d3-bj-spec-v1"

    def __post_init__(self) -> None:
        if self.version != "d3-bj-spec-v1" or self.damping != "bj":
            raise ValueError("only the versioned two-body D3(BJ) model is supported")
        for field in ("s6", "s8", "a1", "a2", "s9", "pair_switch_width"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a finite real scalar")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
            object.__setattr__(self, field, float(value) if value else 0.0)
        if self.s9 != 0:
            raise ValueError("ATM is not implemented; nonzero s9 cannot be dropped")
        if self.a1 == self.a2 == 0:
            raise ValueError("D3(BJ) requires a positive damping radius")
        for field in ("cn_cutoff", "pair_cutoff"):
            value = getattr(self, field)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{field} must be None or a finite real scalar")
                if not math.isfinite(value) or not 0 < value <= 1e6:
                    raise ValueError(f"{field} must be in (0, 1e6] bohr")
                object.__setattr__(self, field, float(value))
        if self.pair_switch_width and (
            self.pair_cutoff is None or self.pair_switch_width >= self.pair_cutoff
        ):
            raise ValueError("pair switch requires 0 < width < finite pair cutoff")
        for field in ("table_sha256", "radii_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} requires a lowercase SHA-256 digest")

    def to_payload(self) -> typing.Any:
        return asdict(self)

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class D4Spec:
    """Versioned molecular D4(BJ)-EEQ correction identity."""

    s6: float
    s8: float
    s9: float
    a1: float
    a2: float
    ga: float
    gc: float
    table_sha256: str
    charge_parameter_sha256: str
    profile: str = "standard"
    reference_model: str = "eeq"
    charge_model: str = "eeq2019"
    cn_cutoff: float = 30.0
    pair_cutoff: float = 60.0
    atm_cutoff: float = 40.0
    charge_cn_cutoff: float = 25.0
    version: str = "d4-bj-eeq-spec-v1"

    def __post_init__(self) -> None:
        if self.version != "d4-bj-eeq-spec-v1":
            raise ValueError("unsupported D4 specification version")
        if self.reference_model != "eeq" or self.charge_model != "eeq2019":
            raise ValueError("only the pinned D4 EEQ2019 charge model is supported")
        if self.profile not in {"standard", "r2scan3c"}:
            raise ValueError("unsupported D4 EEQ reference profile")
        for field in ("s6", "s8", "s9", "a1", "a2", "ga", "gc"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a finite real scalar")
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, float(value) if value else 0.0)
        if self.s6 < 0 or self.s9 < 0 or self.a1 < 0 or self.a2 <= 0:
            raise ValueError("invalid D4 damping coefficient")
        if self.ga <= 0 or self.gc <= 0:
            raise ValueError("D4 zeta parameters must be positive")
        expected = {"standard": (3.0, 2.0), "r2scan3c": (2.0, 1.0)}[self.profile]
        if (self.ga, self.gc) != expected:
            raise ValueError("D4 profile and zeta parameters disagree")
        for field in ("cn_cutoff", "pair_cutoff", "atm_cutoff", "charge_cn_cutoff"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a finite real scalar")
            if not math.isfinite(value) or not 0 < value <= 1e6:
                raise ValueError(f"{field} must be in (0, 1e6] bohr")
            object.__setattr__(self, field, float(value))
        for field in ("table_sha256", "charge_parameter_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} requires a lowercase SHA-256 digest")

    def to_payload(self) -> typing.Any:
        return asdict(self)

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())


def r2scan3c_d4_eeq() -> D4Spec:
    """Exact pinned D4 part of r2SCAN-3c; the electronic method is separate."""
    return D4Spec(
        s6=1.0,
        s8=0.0,
        s9=2.0,
        a1=0.42,
        a2=5.65,
        ga=2.0,
        gc=1.0,
        profile="r2scan3c",
        table_sha256="d1691a6cf08748e7c35a340f78824a4a8da1813b0ca32d1074c346bfb3874871",
        charge_parameter_sha256="02b8bee49c10b4c31914caf149d9f58164e58d6dc7ae2ab21e9d12f5bc22797a",
    )


@dataclass(frozen=True)
class DispersionCorrectionPrimitive:
    """Geometry-only correction request, separate from semilocal XC and Fock.

    Listed derivatives are representable requests, not promoted native KS
    capabilities. The migrated qualification provider is deliberately separate.
    """

    specification: D3Spec | D4Spec
    kind: ClassVar[str] = "dispersion_correction"

    def __post_init__(self) -> None:
        if not isinstance(self.specification, (D3Spec, D4Spec)):
            raise TypeError("dispersion primitive requires a D3Spec or D4Spec")

    @property
    def derivative_capabilities(self) -> typing.Any:
        return ("energy", "nuclear-gradient")

    def semantic_payload(self) -> typing.Any:
        return {
            "kind": self.kind,
            "specification": self.specification.to_payload(),
            "derivative_capabilities": self.derivative_capabilities,
        }

    def to_payload(self) -> typing.Any:
        return self.semantic_payload()


def pbe_d3_bj_spec() -> typing.Any:
    """Audited PBE-D3(BJ) two-body parameters from simple-dftd3 1.4.0."""
    return D3Spec(
        s6=1.0,
        s8=0.7875,
        a1=0.4289,
        a2=4.4407,
        table_sha256=D3_TABLE_SHA256,
        radii_sha256=D3_RADII_SHA256,
    )


def pbe0_d3_bj_spec() -> typing.Any:
    """Audited PBE0-D3(BJ) two-body parameters from simple-dftd3 1.4.0."""
    return D3Spec(
        s6=1.0,
        s8=1.2177,
        a1=0.4145,
        a2=4.8593,
        table_sha256=D3_TABLE_SHA256,
        radii_sha256=D3_RADII_SHA256,
    )
