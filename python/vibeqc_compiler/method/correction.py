"""Uniform result contract for geometry-only method corrections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_CORRECTION_API_VERSION = "correction-result-v1"


@dataclass(frozen=True)
class CorrectionProvenance:
    source: str
    source_revision: str
    data_sha256: str
    license: str
    implementation: str
    version: str = _CORRECTION_API_VERSION

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v
            for v in (
                self.source,
                self.source_revision,
                self.license,
                self.implementation,
            )
        ):
            raise ValueError("correction provenance requires nonempty source metadata")
        if not re.fullmatch(r"[0-9a-f]{64}", self.data_sha256):
            raise ValueError("correction provenance requires a SHA-256 data digest")
        if self.version != _CORRECTION_API_VERSION:
            raise ValueError("unsupported correction result version")

    def to_payload(self):
        return {
            "version": self.version,
            "source": self.source,
            "source_revision": self.source_revision,
            "data_sha256": self.data_sha256,
            "license": self.license,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class CorrectionResult:
    """Per-item correction result with one centralized force-sign convention."""

    component: str
    energy: float
    gradient: tuple[tuple[float, float, float], ...]
    status: str
    parameters: tuple[tuple[str, object], ...]
    provenance: CorrectionProvenance
    energy_unit: str = "hartree"
    gradient_unit: str = "hartree/bohr"
    version: str = _CORRECTION_API_VERSION

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("correction result requires a component name")
        if self.status != "ok":
            raise ValueError("successful correction results must use status='ok'")
        if not isinstance(self.energy, (int, float)) or not math.isfinite(self.energy):
            raise ValueError("correction energy must be finite")
        if not isinstance(self.gradient, tuple) or any(
            not isinstance(row, tuple)
            or len(row) != 3
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
            for row in self.gradient
        ):
            raise ValueError("correction gradient requires finite immutable xyz rows")
        if not isinstance(self.parameters, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            for item in self.parameters
        ):
            raise ValueError("correction parameters require immutable key/value pairs")
        if not isinstance(self.provenance, CorrectionProvenance):
            raise TypeError("correction result requires CorrectionProvenance")
        if (self.energy_unit, self.gradient_unit, self.version) != (
            "hartree",
            "hartree/bohr",
            _CORRECTION_API_VERSION,
        ):
            raise ValueError("unsupported correction result units/version")

    @property
    def forces(self):
        """Cartesian forces in Hartree/bohr; force = -dE/dR exactly once here."""

        return tuple(tuple(-v for v in row) for row in self.gradient)

    def to_payload(self):
        return {
            "version": self.version,
            "component": self.component,
            "energy": self.energy,
            "gradient": [list(row) for row in self.gradient],
            "forces": [list(row) for row in self.forces],
            "status": self.status,
            "parameters": dict(self.parameters),
            "provenance": self.provenance.to_payload(),
            "energy_unit": self.energy_unit,
            "gradient_unit": self.gradient_unit,
        }
