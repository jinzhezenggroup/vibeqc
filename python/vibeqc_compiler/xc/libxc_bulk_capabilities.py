"""Machine-readable qualification claims for the bulk Libxc compiler inventory.

The #912 bulk importer proves a compiler-facing semilocal representation. The
source-controlled independent fixtures and tests additionally qualify energy,
vxc and packed fxc for both spin layouts on the declared interior domain.

Higher qualifications are evidence-driven and fail closed. Source emission is
not compilation, compilation is not runtime validation, and no molecular or
public-method claim is inferred from an earlier stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vibeqc_compiler.common import paths
from vibeqc_compiler.common.evidence import canonical_hash, validate_outcome

from . import libxc_bulk
from .libxc_maple import MapleImportError

CAPABILITY_SCHEMA = "vibeqc.libxc-bulk-capability.v2"
STAGE_EVIDENCE_SCHEMA = "vibeqc.libxc-bulk-stage-evidence.v1"
CLAIM_LEVEL = "pointwise-validated"
POINTWISE_EVIDENCE = "pyscf-2.14.0/libxc-7.0.0-both-spin-e-vxc-fxc"
SPIN_LAYOUTS = ("polarized", "unpolarized")
VALIDATED_OUTPUTS = ("energy", "vxc", "fxc")
SOURCE_EMITTERS = ("c", "cuda")
CAPABILITY_STAGES = (
    "graph-imported",
    CLAIM_LEVEL,
    "compiled-cpu",
    "compiled-cuda",
    "production-domain",
    "gpu-runtime",
    "molecular-scf",
    "forces",
    "response",
    "public-method",
)
BASE_QUALIFIED_STAGES = ("graph-imported", CLAIM_LEVEL)
# Each outer tuple is an AND requirement. Entries within one inner tuple are OR.
STAGE_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "graph-imported": (),
    CLAIM_LEVEL: (("graph-imported",),),
    "compiled-cpu": ((CLAIM_LEVEL,),),
    "compiled-cuda": ((CLAIM_LEVEL,),),
    "production-domain": ((CLAIM_LEVEL,),),
    "gpu-runtime": (("compiled-cuda",),),
    "molecular-scf": (
        ("production-domain",),
        ("compiled-cpu", "gpu-runtime"),
    ),
    "forces": (("molecular-scf",),),
    "response": (("molecular-scf",),),
    "public-method": (("molecular-scf",),),
}
UNQUALIFIED_STAGES = tuple(
    stage for stage in CAPABILITY_STAGES if stage not in BASE_QUALIFIED_STAGES
)


@dataclass(frozen=True)
class StageEvidence:
    """One validated evidence envelope attached to a qualification stage."""

    stage: str
    status: str
    evidence: str | None
    reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BulkFunctionalCapability:
    """Exact qualification boundary for one imported Libxc registration."""

    name: str
    family: str
    libxc_id: int
    entry: str
    owner: str
    identity: str
    qualified_stages: tuple[str, ...]
    stage_evidence: tuple[StageEvidence, ...] = ()
    domain: str = libxc_bulk.BULK_SEMANTICS
    claim_level: str = CLAIM_LEVEL
    spin_layouts: tuple[str, ...] = SPIN_LAYOUTS
    validated_outputs: tuple[str, ...] = VALIDATED_OUTPUTS
    source_emitters: tuple[str, ...] = SOURCE_EMITTERS
    evidence: str = POINTWISE_EVIDENCE

    @property
    def unqualified_stages(self) -> tuple[str, ...]:
        return tuple(
            stage for stage in CAPABILITY_STAGES if stage not in self.qualified_stages
        )

    @property
    def ready_stages(self) -> tuple[str, ...]:
        """Return unqualified stages whose prerequisite groups are satisfied."""
        qualified = set(self.qualified_stages)
        return tuple(
            stage
            for stage in CAPABILITY_STAGES
            if stage not in qualified and _requirements_met(stage, qualified)
        )

    @property
    def public_dft(self) -> bool:
        """Public admission is explicit evidence, never inferred from SCF alone."""
        return "public-method" in self.qualified_stages

    def to_payload(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable capability record."""
        return {
            "schema": CAPABILITY_SCHEMA,
            "identity": self.identity,
            "name": self.name,
            "family": self.family,
            "libxc_id": self.libxc_id,
            "entry": self.entry,
            "owner": self.owner,
            "domain": self.domain,
            "claim_level": self.claim_level,
            "qualified_stages": list(self.qualified_stages),
            "ready_stages": list(self.ready_stages),
            "spin_layouts": list(self.spin_layouts),
            "validated_outputs": list(self.validated_outputs),
            "source_emitters": list(self.source_emitters),
            "evidence": self.evidence,
            "stage_evidence": [
                stage_evidence.to_payload() for stage_evidence in self.stage_evidence
            ],
            "unqualified_stages": list(self.unqualified_stages),
            "public_dft": self.public_dft,
        }


def _imported_record(
    name: str, catalog: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise MapleImportError("bulk capability requires a non-empty registration name")
    if catalog is None:
        catalog = libxc_bulk.read_catalog()
    records = {record["name"]: record for record in catalog["registrations"]}
    key = name.upper()
    if key not in records:
        raise MapleImportError(f"unknown bulk Libxc registration: {name!r}")
    record = records[key]
    if record["graph_status"] != "imported":
        raise MapleImportError(
            f"bulk Libxc registration is not claimable: {record['reason']}"
        )
    return record


def _capability_source_identity(catalog: Mapping[str, Any]) -> str:
    """Bind proof to pinned scientific inputs and the executing compiler bytes.

    This conservative source inventory uses stable logical paths in both wheels
    and checkouts. Compute it once per bulk query, without lowering any Graph or
    importing a native runtime. Source edits invalidate previously attached proof.
    """
    return canonical_hash(
        {
            "schema": "vibeqc.libxc-bulk-capability-sources.v1",
            "provider": libxc_bulk.SOURCE_ASSET,
            "upstream": catalog["upstream"],
            "source_files": catalog["source_files"],
            "importer_semantics": catalog["importer_semantics"],
            "parameter_binding_semantics": catalog["parameter_binding_semantics"],
            "compiler_sources": paths.source_hashes("common", "integral", "xc"),
        }
    )


def _capability_identity(record: Mapping[str, Any], source_identity: str) -> str:
    return canonical_hash(
        {
            "schema": CAPABILITY_SCHEMA,
            "registration": {
                key: value
                for key, value in record.items()
                if key not in ("graph_status", "graph_nodes")
            },
            "source_identity": source_identity,
            "domain": libxc_bulk.BULK_SEMANTICS,
        }
    )


def _requirements_met(stage: str, qualified: set[str]) -> bool:
    return all(
        any(prerequisite in qualified for prerequisite in alternatives)
        for alternatives in STAGE_REQUIREMENTS[stage]
    )


def _normalize_stage_evidence(
    identity: str, evidence: Mapping[str, Any] | None
) -> tuple[StageEvidence, ...]:
    if evidence is None:
        return ()
    if not isinstance(evidence, Mapping):
        raise TypeError("stage evidence must be a stage-to-envelope mapping")

    unknown = set(evidence) - set(CAPABILITY_STAGES)
    if unknown:
        raise ValueError(f"unknown capability stage evidence: {sorted(unknown)!r}")
    intrinsic = set(evidence) & set(BASE_QUALIFIED_STAGES)
    if intrinsic:
        raise ValueError(
            "graph-imported and pointwise-validated are intrinsic bulk evidence "
            "and cannot be overridden"
        )

    normalized = []
    for stage in CAPABILITY_STAGES:
        if stage not in evidence:
            continue
        payload = evidence[stage]
        if not isinstance(payload, dict):
            raise TypeError(f"{stage} evidence must be a mapping")
        if payload.get("schema") != STAGE_EVIDENCE_SCHEMA:
            raise ValueError(f"{stage} evidence has unsupported schema")
        if payload.get("subject_identity") != identity:
            raise ValueError(f"{stage} evidence subject identity mismatch")
        if payload.get("stage") != stage:
            raise ValueError(f"{stage} evidence stage mismatch")
        validate_outcome(payload)
        evidence_ref = payload.get("evidence")
        if payload["status"] == "pass" and (
            not isinstance(evidence_ref, str) or not evidence_ref.strip()
        ):
            raise ValueError(f"{stage} pass requires a non-empty evidence reference")
        normalized.append(
            StageEvidence(
                stage=stage,
                status=payload["status"],
                evidence=evidence_ref,
                reason=payload.get("reason"),
            )
        )
    return tuple(normalized)


def _qualified_stages(evidence: tuple[StageEvidence, ...]) -> tuple[str, ...]:
    qualified = set(BASE_QUALIFIED_STAGES)
    by_stage = {item.stage: item for item in evidence}
    for stage in CAPABILITY_STAGES:
        if stage in qualified:
            continue
        item = by_stage.get(stage)
        if (
            item is not None
            and item.status == "pass"
            and _requirements_met(stage, qualified)
        ):
            qualified.add(stage)
    return tuple(stage for stage in CAPABILITY_STAGES if stage in qualified)


def _functional_capability(
    record: Mapping[str, Any], source_identity: str, evidence: Mapping[str, Any] | None
) -> BulkFunctionalCapability:
    identity = _capability_identity(record, source_identity)
    stage_evidence = _normalize_stage_evidence(identity, evidence)
    return BulkFunctionalCapability(
        name=record["name"],
        family=record["family"],
        libxc_id=record["id"],
        entry=record["entry"],
        owner=record["owner"],
        identity=identity,
        qualified_stages=_qualified_stages(stage_evidence),
        stage_evidence=stage_evidence,
    )


def functional_capability(
    name: str, *, evidence: Mapping[str, Any] | None = None
) -> BulkFunctionalCapability:
    """Return qualification state for one imported registration."""
    catalog = libxc_bulk.read_catalog()
    record = _imported_record(name, catalog)
    return _functional_capability(
        record, _capability_source_identity(catalog), evidence
    )


def _normalize_evidence_inventory(
    evidence_by_functional: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if evidence_by_functional is None:
        return {}
    if not isinstance(evidence_by_functional, Mapping):
        raise TypeError("bulk evidence inventory must be a functional-to-stage mapping")
    result: dict[str, Mapping[str, Any]] = {}
    for name, evidence in evidence_by_functional.items():
        record = _imported_record(name)
        if not isinstance(evidence, Mapping):
            raise TypeError(f"evidence for {name!r} must be a stage mapping")
        result[record["name"]] = evidence
    return result


def available_capabilities(
    evidence_by_functional: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[BulkFunctionalCapability, ...]:
    """Return every imported functional with evidence-driven qualification state."""
    catalog = libxc_bulk.read_catalog()
    evidence_inventory = _normalize_evidence_inventory(evidence_by_functional)
    source_identity = _capability_source_identity(catalog)
    return tuple(
        _functional_capability(
            record, source_identity, evidence_inventory.get(record["name"])
        )
        for record in catalog["registrations"]
        if record["graph_status"] == "imported"
    )


def claimable_functionals(
    level: str = CLAIM_LEVEL,
    evidence_by_functional: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Return names satisfying one exact capability stage.

    Known higher stages with no evidence return an empty inventory. Unknown
    stage names are rejected so typos cannot silently create claims.
    """
    if level not in CAPABILITY_STAGES:
        raise ValueError(f"unknown bulk capability stage {level!r}")
    return tuple(
        capability.name
        for capability in available_capabilities(evidence_by_functional)
        if level in capability.qualified_stages
    )
