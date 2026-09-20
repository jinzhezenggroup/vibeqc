"""Versioned geometric counterpoise (gCP) correction manifests."""

from __future__ import annotations

import math
import re
import typing
from dataclasses import dataclass
from typing import ClassVar

from ._generated_parameters import gcp_parameters

GCP_SPEC_VERSION = "gcp-spec-v1"


@dataclass(frozen=True)
class GCPSpec:
    """Audited gCP parameter set; the correction remains outside XC/SCF."""

    basis: str
    sigma: float
    eta: float
    eta_spec: float
    alpha: float
    beta: float
    damping_scale: float
    damping_exponent: float
    parameter_sha256: str
    implementation_sha256: str
    vdw_radii_sha256: str
    data_sha256: str
    source_revision: str
    supported_atomic_numbers: tuple[int, ...]
    damping: bool = True
    source: str = "dftd3/simple-dftd3"
    license: str = "LGPL-3.0-or-later"
    profile: str = "r2scan3c"
    version: str = GCP_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != GCP_SPEC_VERSION:
            raise ValueError("unsupported gCP specification version")
        if not all(
            isinstance(v, str) and v
            for v in (
                self.basis,
                self.source,
                self.source_revision,
                self.license,
                self.profile,
            )
        ):
            raise ValueError(
                "gCP specification requires provenance and profile metadata"
            )
        for name in (
            "sigma",
            "eta",
            "eta_spec",
            "alpha",
            "beta",
            "damping_scale",
            "damping_exponent",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"gCP {name} must be finite and positive")
        for name in (
            "parameter_sha256",
            "implementation_sha256",
            "vdw_radii_sha256",
            "data_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"gCP {name} requires a SHA-256 digest")
        if (
            not isinstance(self.supported_atomic_numbers, tuple)
            or not self.supported_atomic_numbers
        ):
            raise ValueError("gCP requires a finite supported-element domain")
        if (
            tuple(sorted(set(self.supported_atomic_numbers)))
            != self.supported_atomic_numbers
        ):
            raise ValueError("gCP supported elements must be sorted and unique")

    def to_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "basis": self.basis,
            "profile": self.profile,
            "sigma": self.sigma,
            "eta": self.eta,
            "eta_spec": self.eta_spec,
            "alpha": self.alpha,
            "beta": self.beta,
            "damping": self.damping,
            "damping_scale": self.damping_scale,
            "damping_exponent": self.damping_exponent,
            "parameter_sha256": self.parameter_sha256,
            "implementation_sha256": self.implementation_sha256,
            "vdw_radii_sha256": self.vdw_radii_sha256,
            "data_sha256": self.data_sha256,
            "source": self.source,
            "source_revision": self.source_revision,
            "license": self.license,
            "supported_atomic_numbers": list(self.supported_atomic_numbers),
        }


def r2scan3c_gcp() -> typing.Any:
    """Exact def2-mTZVPP gCP profile used by r2SCAN-3c, scoped to H-Ar."""
    return GCPSpec(**gcp_parameters("r2SCAN-3c"))


@dataclass(frozen=True)
class GeometricCounterpoisePrimitive:
    """Typed external correction node evaluated once outside the electronic SCF."""

    specification: GCPSpec
    kind: ClassVar[str] = "geometric_counterpoise"

    def __post_init__(self) -> None:
        if not isinstance(self.specification, GCPSpec):
            raise TypeError("gCP primitive requires GCPSpec")

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
