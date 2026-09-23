"""Evidence-gated MethodIR/KS resolution for automatic bulk Libxc registrations.

This module is deliberately a composition boundary, not an evidence producer.
A bulk registration reaches a KS plan only after the existing capability owner
has qualified the exact CPU molecular-SCF product requested here.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.xc.capability_resolution import (
    CapabilityResolution,
    resolve_capability,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, functional

from .ks_execution import KsExecutionPlan, compile_ks_execution_plan
from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod

BULK_KS_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-ks-resolution.v1"
_CPU_REQUIRED_STAGES = ("compiled-cpu", "production-domain", "molecular-scf")
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


@dataclass(frozen=True)
class BulkKsResolution:
    """One evidence-qualified pure-semilocal bulk Libxc KS composition."""

    capability: CapabilityResolution
    method: MethodIR
    plan: KsExecutionPlan
    required_ingredients: tuple[str, ...]
    backend: str = "cpu"

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a detached provenance record for the resolved composition."""
        return {
            "schema": BULK_KS_RESOLUTION_SCHEMA,
            "backend": self.backend,
            "capability": self.capability.to_payload(),
            "required_ingredients": list(self.required_ingredients),
            "method_identity": self.method.identity,
            "method_identifier": self.method.identifier,
            "plan_identity": self.plan.identity,
            "spin": self.method.spin,
            "reference": self.method.reference,
            "required_lowerers": list(self.plan.required_lowerers),
            "public_dft": self.capability.public_dft,
        }


def resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve one qualified automatic Libxc registration into a pure KS plan.

    This first integration slice is intentionally CPU-only.  CPU admission
    requires explicit ``compiled-cpu``, ``production-domain`` and
    ``molecular-scf`` evidence.  Representation, pointwise validation, CUDA
    compilation, or a ready-but-unqualified stage never grants this resolver.

    ``identifier`` is descriptive MethodIR provenance.  It does not alter the
    scientific identity of an otherwise identical resolved registration.
    """
    if backend != "cpu":
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution is currently qualified only for CPU"
        )

    capability = functional_capability(name, evidence=evidence)
    if capability.name not in AUTO_BULK_COMPONENTS:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution requires a non-curated "
            "AUTO_BULK_COMPONENTS registration"
        )

    unsupported = tuple(
        ingredient
        for ingredient in capability.required_ingredients
        if ingredient not in _SUPPORTED_INGREDIENTS
    )
    if unsupported:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution does not support ingredients "
            f"{unsupported!r}"
        )

    qualified = resolve_capability(
        capability.name,
        required_stages=_CPU_REQUIRED_STAGES,
        evidence=evidence,
    )
    if qualified.identity != capability.identity:
        raise RuntimeError("bulk Libxc capability identity changed during KS resolution")

    functional_spec = functional(capability.name, spin=spin)
    method = MethodIR(
        identifier=identifier or f"LIBXC:{capability.name}",
        spin=spin,
        primitives=(SemilocalXCPrimitive(functional_spec),),
    )
    plan = compile_ks_execution_plan(method)
    if plan.exchange or plan.nonlocal_correlation is not None or plan.post_scf:
        raise RuntimeError("pure semilocal bulk KS resolution produced extra primitives")

    return BulkKsResolution(
        capability=qualified,
        method=method,
        plan=plan,
        required_ingredients=capability.required_ingredients,
    )
