"""Separate XC representation/codegen/runtime/evidence stages without promotion."""

from vibeqc_compiler.common.evidence import validate_evidence
from vibeqc_compiler.common.provenance import file_hash

from .cuda_emit import emit_cuda


def query_capability(program, *, schedule=None, artifact=None, evidence=None):
    """Report exact consumer stages; successful compilation proves no accuracy.

    Evidence is a diagnostic report tied to both generated identity and binary.
    This interface never promotes a schedule or registers a public DFT method.
    """
    _, contract, _ = emit_cuda(program, schedule)
    compiled = bool(
        artifact is not None
        and artifact.contract["identity"] == contract["identity"]
        and artifact.runtime.metadata["binary_sha256"]
        == file_hash(artifact.runtime.library)
    )
    if evidence is not None:
        validate_evidence(evidence)
    validated = bool(
        compiled
        and evidence
        and evidence.get("identity") == contract["identity"]
        and evidence.get("binary_sha256") == artifact.runtime.metadata["binary_sha256"]
        and evidence.get("inputs_hash") == program.expression_hash
        and evidence.get("backend_selected") == "cuda"
        and evidence.get("stages", {}).get("compilation", {}).get("status") == "pass"
        and evidence.get("stages", {}).get("numerical", {}).get("status") == "pass"
        and evidence.get("block_errors")
        and all(
            error.get("passed") is True and 0 <= error.get("max_scaled_error", 2) <= 1
            for error in evidence["block_errors"].values()
        )
    )
    return {
        "functional": program.spec.to_payload(),
        "expression_hash": program.expression_hash,
        "identity": contract["identity"],
        "represented": True,
        "emitted": True,
        "compiled": compiled,
        "validated": validated,
        "promoted": False,
        "public_dft": False,
        "exact_exchange_evaluated": False,
        "hartree_evaluated": False,
        "nuclear_energy_evaluated": False,
        "reason": "explicit XC kernel candidate; promotion requires downstream workflow evidence",
    }
