"""Evidence-gated analytic-force admission for automatic bulk Libxc XC.

This module owns only the qualification/binding decision for Issue #1122 D1.
It does not assemble a molecular gradient and never promotes a force from the
existence of a pointwise Graph, a method name, or energy-only endpoint evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc.bulk_runtime import (
    BulkRuntimeProgram,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.endpoint_capability import (
    EndpointCapabilityResolution,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    functional_capability,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, FunctionalSpec, functional

from .spec import UnsupportedMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

BULK_FORCE_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-force-resolution.v1"
BULK_FORCE_GEOMETRY_SCHEMA = "vibeqc.bulk-libxc-force-geometry-diagnostic.v1"
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
    key = name.upper() if isinstance(name, str) else name
    if key not in AUTO_BULK_COMPONENTS:
        raise UnsupportedMethod(
            "automatic bulk analytic forces require one non-curated pure semilocal "
            "AUTO_BULK_COMPONENTS registration; curated, exact-exchange, "
            "range-separated, and nonlocal compositions require their separately "
            "qualified MethodIR force owners"
        )

    capability = functional_capability(key, evidence=evidence)
    ingredients = _validate_force_ingredient_contract(capability)

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


@dataclass(frozen=True)
class BulkForceGeometryDiagnostic:
    """Interior point-provider proof feeding the common geometry pullback.

    Force endpoint evidence is resolved before construction, but the point
    executor retained here is still the existing bulk interior Graph runtime.
    This object therefore validates the D2 contraction seam only; it does not
    claim boundary-domain or public molecular-force execution.
    """

    resolution: BulkForceResolution
    functional: FunctionalSpec
    point_program: BulkRuntimeProgram
    contraction: ExternalPointContraction

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": BULK_FORCE_GEOMETRY_SCHEMA,
            "force_resolution": self.resolution.to_payload(),
            "functional_identity": self.functional.identity,
            "point_expression_identity": self.point_program.expression_hash,
            "point_execution_domain": self.point_program.spec.domain,
            "geometry_contract_identity": self.contraction.contract.identity,
            "claim": "interior-fixed-density-geometry-diagnostic",
        }

    def _point_rows(self, features: dict[str, Any]) -> dict[tuple[int, ...], Any]:
        packed = self.contraction.pack_features(features)
        indices = {name: i for i, name in enumerate(self.functional.features)}
        try:
            selected = np.stack(
                [packed[indices[name]] for name in self.point_program.spec.features]
            )
        except KeyError as error:
            raise RuntimeError(
                "bulk point program feature layout disagrees with functional ABI"
            ) from error
        values = self.point_program.evaluate(selected)
        return dict(zip(self.point_program.outputs, values, strict=True))

    def point_energy(self, features: dict[str, Any]) -> Any:
        """Return provider-owned per-point energy for an independent FD oracle."""
        return self._point_rows(features)[()]

    def geometry(
        self,
        jets: Any,
        density: Any,
        weights: Any,
        *,
        ao_atoms: Any,
        natom: Any,
    ) -> Any:
        """Pull one bulk point differential back through the common AO owner."""
        features = self.contraction.features(jets, density)
        rows = self._point_rows(features)
        return self.contraction.geometry_from_feature_rows(
            jets,
            density,
            weights,
            features,
            rows,
            ao_atoms=ao_atoms,
            natom=natom,
        )


def resolve_bulk_force_geometry_diagnostic(
    name: str,
    *,
    spin: str,
    evidence: Mapping[str, Any] | None = None,
    require_public: bool = False,
) -> BulkForceGeometryDiagnostic:
    """Build the CPU interior D2 seam after exact force capability resolution."""
    resolution = resolve_bulk_force_capability(
        name,
        backend="cpu",
        spin=spin,
        evidence=evidence,
        require_public=require_public,
    )
    spec = functional(resolution.name, spin=spin)
    contraction = ExternalPointContraction(spec, "geometry")
    point_program = build_bulk_runtime_program(
        resolution.name,
        spin=spin,
        order=1,
        outputs=contraction.contract.scalar_outputs,
    )
    if point_program.spec.capability_identity != resolution.identity:
        raise RuntimeError("bulk point runtime capability identity mismatch")
    if point_program.spec.ingredients != resolution.required_ingredients:
        raise RuntimeError("bulk point runtime ingredient contract mismatch")
    if point_program.spec.spin != resolution.spin:
        raise RuntimeError("bulk point runtime spin contract mismatch")
    return BulkForceGeometryDiagnostic(
        resolution=resolution,
        functional=spec,
        point_program=point_program,
        contraction=contraction,
    )
