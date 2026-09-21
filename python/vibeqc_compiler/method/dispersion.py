"""Explicit D3 variant composition and production correction semantics."""

import math
import re
import typing
from dataclasses import asdict, dataclass, replace
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash

from . import _generated_parameters as _parameters

D3_TABLE_SHA256 = _parameters.D3_TABLE_SHA256
D3_RADII_SHA256 = _parameters.D3_RADII_SHA256
D3_BJ_PARAMETER_SETS = _parameters.D3_BJ_PARAMETER_SETS
D3_ZERO_PARAMETER_SETS = _parameters.D3_ZERO_PARAMETER_SETS
D4_PARAMETER_SETS = _parameters.D4_PARAMETER_SETS


@dataclass(frozen=True)
class D3Spec:
    """Versioned D3 model with separately admitted damping/ATM capabilities.

    Cutoffs are in bohr; None means no cutoff. Merely having a parameter record
    does not imply executable capability: BJ two-body, BJ+ATM, and zero-damping
    two-body have distinct versions and semantic identities. Zero+ATM remains
    fail-closed until separately qualified.
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
    rs6: float = 0.0
    rs8: float = 0.0
    alp: float = 0.0
    atm_cutoff: float | None = None
    atm_switch_width: float = 0.0

    def __post_init__(self) -> None:
        if self.damping not in {"bj", "zero"}:
            raise ValueError("unsupported D3 damping variant")
        for field in (
            "s6",
            "s8",
            "a1",
            "a2",
            "s9",
            "pair_switch_width",
            "rs6",
            "rs8",
            "alp",
            "atm_switch_width",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a finite real scalar")
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, float(value) if value else 0.0)
        if (
            self.s6 < 0
            or self.s9 < 0
            or self.pair_switch_width < 0
            or self.atm_switch_width < 0
        ):
            raise ValueError("D3 scaling/switch parameters must be nonnegative")

        for field in ("cn_cutoff", "pair_cutoff", "atm_cutoff"):
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
        if self.atm_switch_width and (
            self.atm_cutoff is None or self.atm_switch_width >= self.atm_cutoff
        ):
            raise ValueError("ATM switch requires 0 < width < finite ATM cutoff")

        if self.damping == "bj":
            if self.a1 < 0 or self.a2 < 0 or self.s8 < 0 or self.a1 == self.a2 == 0:
                raise ValueError("invalid D3(BJ) damping coefficient")
            if any(value != 0.0 for value in (self.rs6, self.rs8, self.alp)):
                raise ValueError("D3(BJ) must not carry zero-damping parameters")
            if self.s9 == 0:
                if self.version != "d3-bj-spec-v1":
                    raise ValueError("two-body D3(BJ) requires d3-bj-spec-v1")
                if self.atm_cutoff is not None or self.atm_switch_width != 0:
                    raise ValueError("two-body D3(BJ) must not carry ATM options")
            elif self.version != "d3-bj-atm-spec-v1":
                raise ValueError("D3(BJ)-ATM requires d3-bj-atm-spec-v1")
        else:
            if self.version != "d3-zero-spec-v1":
                raise ValueError("zero-damping D3 requires d3-zero-spec-v1")
            if self.s9 != 0:
                raise ValueError("zero-damping D3 plus ATM is not qualified")
            if self.a1 != 0 or self.a2 != 0:
                raise ValueError("zero-damping D3 must not carry BJ radii")
            if self.rs6 <= 0 or self.rs8 <= 0 or self.alp <= 0:
                raise ValueError("zero-damping D3 requires positive rs6/rs8/alp")
            if self.atm_cutoff is not None or self.atm_switch_width != 0:
                raise ValueError("zero-damping D3 must not carry ATM options")

        for field in ("table_sha256", "radii_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} requires a lowercase SHA-256 digest")

    def to_payload(self) -> typing.Any:
        # Preserve the established two-body BJ identity byte-for-byte in the
        # semantic payload. New capability fields exist only for new variants.
        if self.damping == "bj" and self.s9 == 0:
            return {
                "s6": self.s6,
                "s8": self.s8,
                "a1": self.a1,
                "a2": self.a2,
                "table_sha256": self.table_sha256,
                "radii_sha256": self.radii_sha256,
                "s9": self.s9,
                "damping": self.damping,
                "cn_cutoff": self.cn_cutoff,
                "pair_cutoff": self.pair_cutoff,
                "pair_switch_width": self.pair_switch_width,
                "version": self.version,
            }
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
    return D4Spec(**_parameters.d4_parameters("r2SCAN-3c"))


def pbe_d4_eeq_spec() -> D4Spec:
    """Audited PBE-D4(BJ-EEQ-ATM) parameters from the pinned D4 catalog."""
    return D4Spec(**_parameters.d4_parameters("PBE-D4(BJ-EEQ-ATM)"))


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


def pbe_d3_bj_spec() -> D3Spec:
    """Audited PBE-D3(BJ) two-body parameters from simple-dftd3 1.4.0."""
    return D3Spec(**_parameters.d3_parameters("PBE-D3(BJ)"))


def pbe0_d3_bj_spec() -> D3Spec:
    """Audited PBE0-D3(BJ) two-body parameters from simple-dftd3 1.4.0."""
    return D3Spec(**_parameters.d3_parameters("PBE0-D3(BJ)"))


def pbe_d3_bj_atm_spec() -> D3Spec:
    """Explicit PBE-D3(BJ)-ATM capability using the pinned BJ pair model."""
    return replace(pbe_d3_bj_spec(), s9=1.0, version="d3-bj-atm-spec-v1")


def _zero_spec(name: str) -> D3Spec:
    parameters = _parameters.d3_zero_parameters(name)
    return D3Spec(
        s6=parameters["s6"],
        s8=parameters["s8"],
        a1=0.0,
        a2=0.0,
        table_sha256=parameters["table_sha256"],
        radii_sha256=parameters["radii_sha256"],
        s9=0.0,
        damping="zero",
        rs6=parameters["rs6"],
        rs8=parameters["rs8"],
        alp=parameters["alp"],
        version="d3-zero-spec-v1",
    )


def pbe_d3_zero_spec() -> D3Spec:
    """Audited PBE-D3 zero-damping two-body parameters."""
    return _zero_spec("PBE-D3(0)")


def pbe0_d3_zero_spec() -> D3Spec:
    """Audited PBE0-D3 zero-damping two-body parameters."""
    return _zero_spec("PBE0-D3(0)")
