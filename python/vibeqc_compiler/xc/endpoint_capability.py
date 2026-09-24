"""Exact backend/product/spin admission for bulk Libxc molecular endpoints.

Stage promotion alone cannot prove that a molecular result belongs to a
particular backend, spin layout, or requested product. This module layers a
small fail-closed coverage contract over the existing capability DAG without
creating a second scientific support registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .capability_resolution import (
    CapabilityNotQualified,
    resolve_capability,
)
from .libxc_bulk_capabilities import (
    CAPABILITY_STAGES,
    SPIN_LAYOUTS,
    functional_capability,
)

if TYPE_CHECKING:
    from .capability_resolution import CapabilityResolution
    from .libxc_bulk_capabilities import BulkFunctionalCapability, StageEvidence

ENDPOINT_COVERAGE_SCHEMA = "vibeqc.libxc-endpoint-coverage.v1"
ENDPOINT_RESOLUTION_SCHEMA = "vibeqc.libxc-endpoint-resolution.v1"
ENDPOINT_BACKENDS = ("cpu", "cuda")
ENDPOINT_PRODUCTS = ("energy", "forces", "response")
_ENDPOINT_PRODUCT_STAGE = {
    "energy": "molecular-scf",
    "forces": "forces",
    "response": "response",
}


@dataclass(frozen=True)
class EndpointCapabilityResolution:
    """One exact backend/product/spin endpoint admitted by retained evidence."""

    capability: CapabilityResolution
    backend: str
    product: str
    spin: str
    public_dft: bool

    @property
    def name(self) -> str:
        return self.capability.name

    @property
    def identity(self) -> str:
        return self.capability.identity

    @property
    def required_stages(self) -> tuple[str, ...]:
        return self.capability.required_stages

    def to_payload(self) -> dict[str, Any]:
        """Return a detached machine-readable endpoint resolution record."""
        return {
            "schema": ENDPOINT_RESOLUTION_SCHEMA,
            "name": self.name,
            "identity": self.identity,
            "backend": self.backend,
            "product": self.product,
            "spin": self.spin,
            "public_dft": self.public_dft,
            "capability": self.capability.to_payload(),
        }


def _normalize_endpoint_value(
    value: str,
    *,
    field: str,
    allowed: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    if value not in allowed:
        raise ValueError(f"unsupported {field} {value!r}; expected one of {allowed!r}")
    return value


def _endpoint_required_stages(
    *,
    backend: str,
    product: str,
    require_public: bool,
) -> tuple[str, ...]:
    requested = {"production-domain", "molecular-scf"}
    if backend == "cpu":
        requested.add("compiled-cpu")
    else:
        requested.update(("compiled-cuda", "gpu-runtime"))
    if product != "energy":
        requested.add(_ENDPOINT_PRODUCT_STAGE[product])
    if require_public:
        requested.add("public-method")
    return tuple(stage for stage in CAPABILITY_STAGES if stage in requested)


def _endpoint_coverage_reason(
    stage_evidence: StageEvidence | None,
    *,
    backend: str,
    spin: str,
    product: str,
) -> str | None:
    if stage_evidence is None or stage_evidence.status != "pass":
        return "missing pass evidence"

    qualification = stage_evidence.qualification
    if not isinstance(qualification, Mapping):
        return "missing endpoint coverage qualification"
    if qualification.get("schema") != ENDPOINT_COVERAGE_SCHEMA:
        return "endpoint coverage qualification has unsupported schema"

    raw_coverage = qualification.get("coverage")
    if isinstance(raw_coverage, (str, bytes)) or not isinstance(
        raw_coverage,
        Sequence,
    ):
        return "endpoint coverage qualification requires a coverage sequence"
    if not raw_coverage:
        return "endpoint coverage qualification requires non-empty coverage"

    normalized: list[tuple[str, str, tuple[str, ...]]] = []
    for index, item in enumerate(raw_coverage):
        if not isinstance(item, Mapping):
            return f"endpoint coverage[{index}] must be a mapping"
        item_backend = item.get("backend")
        item_spin = item.get("spin")
        products = item.get("products")
        if item_backend not in ENDPOINT_BACKENDS:
            return f"endpoint coverage[{index}] has invalid backend"
        if item_spin not in SPIN_LAYOUTS:
            return f"endpoint coverage[{index}] has invalid spin"
        if isinstance(products, (str, bytes)) or not isinstance(products, Sequence):
            return f"endpoint coverage[{index}] products must be a sequence"
        product_tuple = tuple(products)
        if (
            not product_tuple
            or any(
                not isinstance(entry, str) or entry not in ENDPOINT_PRODUCTS
                for entry in product_tuple
            )
            or len(set(product_tuple)) != len(product_tuple)
        ):
            return f"endpoint coverage[{index}] has invalid products"
        normalized.append((item_backend, item_spin, product_tuple))

    if any(
        item_backend == backend and item_spin == spin and product in products
        for item_backend, item_spin, products in normalized
    ):
        return None
    return (
        f"pass evidence does not cover backend={backend} spin={spin} product={product}"
    )


def _exact_endpoint_public(
    capability: BulkFunctionalCapability,
    *,
    backend: str,
    spin: str,
    product: str,
) -> bool:
    evidence_by_stage = {item.stage: item for item in capability.stage_evidence}
    return (
        "public-method" in capability.qualified_stages
        and _endpoint_coverage_reason(
            evidence_by_stage.get("public-method"),
            backend=backend,
            spin=spin,
            product=product,
        )
        is None
    )


def resolve_endpoint_capability(
    name: str,
    *,
    backend: str,
    product: str,
    spin: str,
    require_public: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> EndpointCapabilityResolution:
    """Resolve one exact molecular endpoint from stage plus coverage evidence.

    Endpoint-scoped pass evidence must attach an
    ``ENDPOINT_COVERAGE_SCHEMA`` qualification. This prevents CPU molecular
    evidence from being combined with unrelated CUDA compilation/runtime proof,
    and prevents energy-only or one-spin evidence from inflating force,
    response, or public capability.
    """
    backend = _normalize_endpoint_value(
        backend,
        field="backend",
        allowed=ENDPOINT_BACKENDS,
    )
    product = _normalize_endpoint_value(
        product,
        field="product",
        allowed=ENDPOINT_PRODUCTS,
    )
    spin = _normalize_endpoint_value(
        spin,
        field="spin",
        allowed=SPIN_LAYOUTS,
    )
    if not isinstance(require_public, bool):
        raise TypeError("require_public must be a bool")

    required = _endpoint_required_stages(
        backend=backend,
        product=product,
        require_public=require_public,
    )
    resolved = resolve_capability(
        name,
        required_stages=required,
        evidence=evidence,
    )
    capability = functional_capability(name, evidence=evidence)
    if capability.identity != resolved.identity:
        raise RuntimeError("capability identity changed during endpoint resolution")

    evidence_by_stage = {item.stage: item for item in capability.stage_evidence}
    coverage_checks = [("molecular-scf", "energy")]
    if product != "energy":
        coverage_checks.append((_ENDPOINT_PRODUCT_STAGE[product], product))
    if require_public:
        coverage_checks.append(("public-method", product))

    blockers = []
    for stage, covered_product in coverage_checks:
        reason = _endpoint_coverage_reason(
            evidence_by_stage.get(stage),
            backend=backend,
            spin=spin,
            product=covered_product,
        )
        if reason is not None:
            blockers.append((stage, reason))
    if blockers:
        raise CapabilityNotQualified(
            name=capability.name,
            identity=capability.identity,
            blockers=tuple(blockers),
        )

    return EndpointCapabilityResolution(
        capability=resolved,
        backend=backend,
        product=product,
        spin=spin,
        public_dft=_exact_endpoint_public(
            capability,
            backend=backend,
            spin=spin,
            product=product,
        ),
    )
