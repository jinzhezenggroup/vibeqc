"""Evidence-gated analytic-force admission for automatic bulk Libxc XC.

This module owns only the qualification/binding decision for Issue #1122 D1.
It does not assemble a molecular gradient and never promotes a force from the
existence of a pointwise Graph, a method name, or energy-only endpoint evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vibeqc_compiler.xc.endpoint_capability import (
    EndpointCapabilityResolution,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    functional_capability,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from .spec import UnsupportedMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

BULK_FORCE_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-force-resolution.v1"
_SUPPORTED_FORCE_INGREDIENTS = {
    ("rho",): "lda",
    ("rho", "sigma"): "gga",
    ("rho", "sigma", "tau"): "mgga",
}


@dataclass(frozen=True)
class BulkForceResolution:
    """Exact force qualification for one automatic semilocal registration."""

    endpoint: EndpointCapabilityResolution
    family: str
    required_ingredients: tuple[str, ...]
    production_domain_identity: str

    @property
    def name(self) -> str:
        return self.endpoint.name

    @property
    def identity(self) -> str:
        return self.endpoint.identity

    @property
    def backend(self) -> str:
        return self.endpoint.backend

    @property
    def spin(self) -> str:
        return self.endpoint.spin

    @property
    def public_dft(self) -> bool:
        return self.endpoint.public_dft

    @property
    def tau_generalized_ks(self) -> bool:
        return "tau" in self.required_ingredients

    def to_payload(self) -> dict[str, Any]:
        """Return detached provenance for the exact admitted force endpoint."""
        return {
            "schema": BULK_FORCE_RESOLUTION_SCHEMA,
            "name": self.name,
            "identity": self.identity,
            "backend": self.backend,
            "spin": self.spin,
            "family": self.family,
            "required_ingredients": list(self.required_ingredients),
            "production_domain_identity": self.production_domain_identity,
            "tau_generalized_ks": self.tau_generalized_ks,
            "public_dft": self.public_dft,
            "endpoint": self.endpoint.to_payload(),
        }


def _validate_force_ingredient_contract(
    capability: BulkFunctionalCapability,
) -> tuple[str, ...]:
    ingredients = capability.required_ingredients
    unsupported = tuple(
        ingredient
        for ingredient in ingredients
        if ingredient not in ("rho", "sigma", "tau")
    )
    if unsupported:
        raise UnsupportedMethod(
            "automatic bulk analytic forces do not support derivative ingredients "
            f"{unsupported!r}; Laplacian/current/nonlocal operators require "
            "separately qualified derivative owners"
        )

    expected_family = _SUPPORTED_FORCE_INGREDIENTS.get(ingredients)
    if expected_family is None or capability.family != expected_family:
        raise UnsupportedMethod(
            "automatic bulk analytic forces require exactly rho, rho/sigma, or "
            "rho/sigma/tau with the matching LDA/GGA/meta-GGA family"
        )

    profile = capability.production_domain_profile
    if not profile.eligible:
        raise UnsupportedMethod(
            "automatic bulk analytic forces are blocked by the production-domain "
            f"profile: {profile.blocker}"
        )
    return ingredients


def resolve_bulk_force_capability(
    name: str,
    *,
    backend: str,
    spin: str,
    evidence: Mapping[str, Any] | None = None,
    require_public: bool = False,
) -> BulkForceResolution:
    """Resolve a generic semilocal force only from exact retained evidence.

    The endpoint resolver supplies backend/spin/product evidence gates. This
    adapter adds the structural ingredient contract required by the existing
    stationary LDA/GGA/tau-gradient machinery. No functional-name whitelist can
    grant a force: even a representable registration must carry an exact force
    endpoint pass covering the requested backend and spin layout.
    """
    capability = functional_capability(name, evidence=evidence)
    ingredients = _validate_force_ingredient_contract(capability)

    if capability.name not in AUTO_BULK_COMPONENTS:
        raise UnsupportedMethod(
            "automatic bulk analytic forces require one non-curated pure semilocal "
            "AUTO_BULK_COMPONENTS registration; curated, exact-exchange, "
            "range-separated, and nonlocal compositions require their separately "
            "qualified MethodIR force owners"
        )

    endpoint = resolve_endpoint_capability(
        capability.name,
        backend=backend,
        product="forces",
        spin=spin,
        require_public=require_public,
        evidence=evidence,
    )
    if endpoint.identity != capability.identity:
        raise RuntimeError(
            "bulk Libxc capability identity changed during force resolution"
        )

    return BulkForceResolution(
        endpoint=endpoint,
        family=capability.family,
        required_ingredients=ingredients,
        production_domain_identity=capability.production_domain_profile.identity,
    )
