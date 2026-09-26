"""Evidence-gated MethodIR/KS resolution for automatic bulk Libxc registrations.

This module is deliberately a composition boundary, not an evidence producer.
Qualification producers may resolve an execution candidate after compiled-CPU
and production-domain evidence exists; ordinary consumers require the additional
molecular-SCF evidence produced by executing that candidate.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.xc.capability_resolution import (
    CapabilityResolution,
    resolve_capability,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import validate_qualification
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    functional_capability,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, functional

from .ks_execution import KsExecutionPlan, compile_ks_execution_plan
from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod

BULK_KS_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-ks-resolution.v2"
_CPU_EXECUTION_STAGES = ("compiled-cpu", "production-domain")
_CPU_PROMOTION_STAGES = (*_CPU_EXECUTION_STAGES, "molecular-scf")
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


@dataclass(frozen=True)
class BulkKsResolution:
    """One evidence-qualified pure-semilocal bulk Libxc KS composition."""

    capability: CapabilityResolution
    method: MethodIR
    plan: KsExecutionPlan
    required_ingredients: tuple[str, ...]
    compiled_cpu_binding_identity: str
    compiled_cpu_result_identity: str
    backend: str = "cpu"

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a detached provenance record for the resolved composition."""
        return {
            "schema": BULK_KS_RESOLUTION_SCHEMA,
            "backend": self.backend,
            "capability": self.capability.to_payload(),
            "required_ingredients": list(self.required_ingredients),
            "compiled_cpu_binding_identity": self.compiled_cpu_binding_identity,
            "compiled_cpu_result_identity": self.compiled_cpu_result_identity,
            "method_identity": self.method.identity,
            "method_identifier": self.method.identifier,
            "plan_identity": self.plan.identity,
            "spin": self.method.spin,
            "reference": self.method.reference,
            "required_lowerers": list(self.plan.required_lowerers),
            "public_dft": self.capability.public_dft,
        }


def _require_exact_compiled_cpu(
    capability: BulkFunctionalCapability,
) -> dict[str, typing.Any]:
    stage = next(
        (
            item
            for item in capability.stage_evidence
            if item.stage == "compiled-cpu" and item.status == "pass"
        ),
        None,
    )
    if stage is None:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires passing compiled-CPU evidence"
        )
    try:
        return validate_qualification(capability.name, stage.qualification)
    except (TypeError, ValueError) as exc:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires exact compiled-CPU qualification"
        ) from exc


def _resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
    required_stages: tuple[str, ...],
) -> BulkKsResolution:
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
        required_stages=required_stages,
        evidence=evidence,
    )
    if qualified.identity != capability.identity:
        raise RuntimeError(
            "bulk Libxc capability identity changed during KS resolution"
        )
    compiled_cpu = _require_exact_compiled_cpu(capability)

    functional_spec = functional(capability.name, spin=spin)
    method = MethodIR(
        identifier=(
            identifier if identifier is not None else f"LIBXC:{capability.name}"
        ),
        spin=spin,
        primitives=(SemilocalXCPrimitive(functional_spec),),
    )
    plan = compile_ks_execution_plan(method)
    if plan.exchange or plan.nonlocal_correlation is not None or plan.post_scf:
        raise RuntimeError(
            "pure semilocal bulk KS resolution produced extra primitives"
        )

    return BulkKsResolution(
        capability=qualified,
        method=method,
        plan=plan,
        required_ingredients=capability.required_ingredients,
        compiled_cpu_binding_identity=compiled_cpu["binding_identity"],
        compiled_cpu_result_identity=compiled_cpu["result_identity"],
    )


def resolve_bulk_ks_candidate(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve the CPU candidate used to produce molecular-SCF evidence.

    Candidate execution remains fail-closed on compiled-CPU and complete
    production-domain evidence. Requiring molecular-SCF here would be circular:
    this is the exact plan that the qualification runner must execute to create
    that evidence.
    """
    return _resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
        required_stages=_CPU_EXECUTION_STAGES,
    )


def resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve one promoted automatic Libxc registration into a pure KS plan.

    Ordinary consumers require compiled-CPU, production-domain and molecular-SCF
    evidence. Qualification producers must use resolve_bulk_ks_candidate instead
    of manufacturing the final endpoint stage.
    """
    return _resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
        required_stages=_CPU_PROMOTION_STAGES,
    )
