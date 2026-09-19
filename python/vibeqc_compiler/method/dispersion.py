"""Explicit D3(BJ) composition semantics; no public runtime capability (#492)."""

import math
import re
from dataclasses import asdict, dataclass
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash


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

    def __post_init__(self):
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

    def to_payload(self):
        return asdict(self)

    @property
    def identity(self):
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class DispersionCorrectionPrimitive:
    """Geometry-only correction request, separate from semilocal XC and Fock.

    Listed derivatives are representable requests, not promoted native KS
    capabilities. The migrated qualification provider is deliberately separate.
    """

    specification: D3Spec
    kind: ClassVar[str] = "dispersion_correction"

    def __post_init__(self):
        if not isinstance(self.specification, D3Spec):
            raise TypeError("dispersion primitive requires a D3Spec")

    @property
    def derivative_capabilities(self):
        return ("energy", "nuclear-gradient")

    def semantic_payload(self):
        return {
            "kind": self.kind,
            "specification": self.specification.to_payload(),
            "derivative_capabilities": self.derivative_capabilities,
        }

    def to_payload(self):
        return self.semantic_payload()
