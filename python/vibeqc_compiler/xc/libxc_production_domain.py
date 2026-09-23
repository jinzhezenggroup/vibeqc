"""Versioned production-domain qualification profiles for bulk Libxc XC.

This module defines the evidence contract for promoting an imported semilocal
Libxc registration beyond the interior pointwise domain.  It deliberately does
not evaluate a functional and never grants production admission by itself.

The profile is structural: every successful production-domain claim must prove
the same named boundary matrix for the registration's ingredient set and both
spin layouts.  Numerical fixtures and runners live downstream, but their
evidence is rejected unless it matches this exact versioned profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vibeqc_compiler.common.evidence import canonical_hash

SCHEMA = "vibeqc.libxc-production-domain-profile.v1"
PROFILE = "semilocal-boundary-matrix/v1"
SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))

_DENSITY_CASES = (
    "density/vacuum",
    "density/near-zero",
)
_SPIN_CASES = (
    "spin/balanced",
    "spin/zero-a",
    "spin/zero-b",
    "spin/near-zero-a",
    "spin/near-zero-b",
    "spin/full-a",
    "spin/full-b",
)
_SIGMA_CASES = (
    "sigma/zero",
    "sigma/near-zero",
    "sigma/large-finite",
    "sigma/cancellation",
)
_TAU_CASES = (
    "tau/uniform-gas",
    "tau/isoorbital",
    "tau/near-isoorbital",
    "tau/large-finite",
)
_CONTROL_CASES = (
    "control/lazy-inactive-branch",
    "control/invalid-nonfinite",
)


@dataclass(frozen=True)
class ProductionDomainProfile:
    """Exact boundary matrix required before production-domain promotion."""

    family: str
    required_ingredients: tuple[str, ...]
    case_ids: tuple[str, ...]
    blocker: str | None = None
    schema: str = SCHEMA
    profile: str = PROFILE
    spin_layouts: tuple[str, ...] = ("polarized", "unpolarized")
    outputs: tuple[str, ...] = ("energy", "vxc", "fxc")

    @property
    def eligible(self) -> bool:
        """Whether this ingredient set can enter the current admission runner."""
        return self.blocker is None

    @property
    def identity(self) -> str:
        """Stable identity for evidence binding and deterministic regeneration."""
        return canonical_hash(
            {
                "schema": self.schema,
                "profile": self.profile,
                "family": self.family,
                "required_ingredients": self.required_ingredients,
                "case_ids": self.case_ids,
                "spin_layouts": self.spin_layouts,
                "outputs": self.outputs,
                "blocker": self.blocker,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical detached profile payload."""
        return {
            "schema": self.schema,
            "profile": self.profile,
            "identity": self.identity,
            "family": self.family,
            "required_ingredients": list(self.required_ingredients),
            "case_ids": list(self.case_ids),
            "spin_layouts": list(self.spin_layouts),
            "outputs": list(self.outputs),
            "eligible": self.eligible,
            "blocker": self.blocker,
        }


def qualification_profile(
    family: str, required_ingredients: tuple[str, ...]
) -> ProductionDomainProfile:
    """Return the deterministic production-domain matrix for one registration."""
    if family not in ("lda", "gga", "mgga"):
        raise ValueError(f"unsupported semilocal family {family!r}")
    if (
        not isinstance(required_ingredients, tuple)
        or not required_ingredients
        or required_ingredients[0] != "rho"
        or len(set(required_ingredients)) != len(required_ingredients)
    ):
        raise ValueError(
            "production-domain ingredients require a unique rho-first tuple"
        )

    ingredients = frozenset(required_ingredients)
    unsupported = tuple(sorted(ingredients - SUPPORTED_INGREDIENTS))
    blocker = (
        "unsupported-ingredients:" + ",".join(unsupported) if unsupported else None
    )

    cases = [*_DENSITY_CASES, *_SPIN_CASES]
    if "sigma" in ingredients:
        cases.extend(_SIGMA_CASES)
    if "tau" in ingredients:
        cases.extend(_TAU_CASES)
    cases.extend(_CONTROL_CASES)
    return ProductionDomainProfile(
        family=family,
        required_ingredients=required_ingredients,
        case_ids=tuple(cases),
        blocker=blocker,
    )


def validate_qualification(
    value: Mapping[str, Any] | None, expected: ProductionDomainProfile
) -> None:
    """Require evidence to cover exactly the current versioned boundary profile."""
    if not isinstance(value, Mapping):
        raise ValueError("production-domain pass requires qualification profile")
    canonical = expected.to_payload()
    if value.get("schema") != SCHEMA or value.get("profile") != PROFILE:
        raise ValueError(
            "production-domain qualification has unsupported schema/profile"
        )
    if value.get("identity") != expected.identity:
        raise ValueError("production-domain qualification profile identity mismatch")
    if dict(value) != canonical:
        raise ValueError("production-domain qualification does not cover exact profile")
    if not expected.eligible:
        raise ValueError(
            "production-domain qualification is blocked: " + str(expected.blocker)
        )
