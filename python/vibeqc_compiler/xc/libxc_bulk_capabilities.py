"""Machine-readable qualification claims for the bulk Libxc compiler inventory.

The #912 bulk importer proves a compiler-facing semilocal representation.  The
source-controlled independent fixtures and tests additionally qualify energy,
vxc and packed fxc for both spin layouts on the declared interior domain.

These claims are intentionally below public MethodIR/Calculator admission:
source emission is not binary/runtime qualification, and no molecular SCF,
force, response or complete-domain support is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import libxc_bulk
from .libxc_maple import MapleImportError

CAPABILITY_SCHEMA = "vibeqc.libxc-bulk-capability.v1"
CLAIM_LEVEL = "pointwise-validated"
POINTWISE_EVIDENCE = "pyscf-2.14.0/libxc-7.0.0-both-spin-e-vxc-fxc"
SPIN_LAYOUTS = ("polarized", "unpolarized")
VALIDATED_OUTPUTS = ("energy", "vxc", "fxc")
SOURCE_EMITTERS = ("c", "cuda")
UNQUALIFIED_STAGES = (
    "production-domain",
    "compiled-cpu",
    "compiled-cuda",
    "gpu-runtime",
    "molecular-scf",
    "forces",
    "response",
    "public-method",
)


@dataclass(frozen=True)
class BulkFunctionalCapability:
    """Exact qualification boundary for one imported Libxc registration."""

    name: str
    family: str
    libxc_id: int
    entry: str
    owner: str
    domain: str = libxc_bulk.BULK_SEMANTICS
    claim_level: str = CLAIM_LEVEL
    spin_layouts: tuple[str, ...] = SPIN_LAYOUTS
    validated_outputs: tuple[str, ...] = VALIDATED_OUTPUTS
    source_emitters: tuple[str, ...] = SOURCE_EMITTERS
    evidence: str = POINTWISE_EVIDENCE
    unqualified_stages: tuple[str, ...] = UNQUALIFIED_STAGES
    public_dft: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable capability record."""
        return {
            "schema": CAPABILITY_SCHEMA,
            "name": self.name,
            "family": self.family,
            "libxc_id": self.libxc_id,
            "entry": self.entry,
            "owner": self.owner,
            "domain": self.domain,
            "claim_level": self.claim_level,
            "spin_layouts": list(self.spin_layouts),
            "validated_outputs": list(self.validated_outputs),
            "source_emitters": list(self.source_emitters),
            "evidence": self.evidence,
            "unqualified_stages": list(self.unqualified_stages),
            "public_dft": self.public_dft,
        }


def _imported_record(name: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise MapleImportError("bulk capability requires a non-empty registration name")
    records = {
        record["name"]: record for record in libxc_bulk.read_catalog()["registrations"]
    }
    key = name.upper()
    if key not in records:
        raise MapleImportError(f"unknown bulk Libxc registration: {name!r}")
    record = records[key]
    if record["graph_status"] != "imported":
        raise MapleImportError(
            f"bulk Libxc registration is not claimable: {record['reason']}"
        )
    return record


def functional_capability(name: str) -> BulkFunctionalCapability:
    """Return the pointwise qualification record for one imported registration."""
    record = _imported_record(name)
    return BulkFunctionalCapability(
        name=record["name"],
        family=record["family"],
        libxc_id=record["id"],
        entry=record["entry"],
        owner=record["owner"],
    )


def available_capabilities() -> tuple[BulkFunctionalCapability, ...]:
    """Return every automatically claimable bulk functional, deterministically."""
    return tuple(
        functional_capability(name) for name in libxc_bulk.available_functionals()
    )


def claimable_functionals(level: str = CLAIM_LEVEL) -> tuple[str, ...]:
    """Return names satisfying a machine-readable qualification level.

    Today graph import and independent pointwise qualification cover the same
    221-registration inventory.  Higher stages stay fail-closed until separate
    evidence is wired into this registry.
    """
    if level not in ("graph-imported", CLAIM_LEVEL):
        raise ValueError(
            "bulk automatic claims are limited to graph-imported or "
            f"{CLAIM_LEVEL!r}; runtime/public admission is separate"
        )
    return tuple(capability.name for capability in available_capabilities())
