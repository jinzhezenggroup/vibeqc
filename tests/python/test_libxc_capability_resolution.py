"""Regression coverage for exact bulk Libxc capability resolution."""

from __future__ import annotations

import pytest
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.capability_resolution import (
    RESOLUTION_SCHEMA,
    CapabilityNotQualified,
    resolve_capability,
)


def _stage_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    status: str = "pass",
    reason: str | None = None,
) -> dict:
    return {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": status,
        "reason": reason,
        "evidence": f"test://{capability.name}/{stage}/{status}",
    }


def test_resolve_capability_records_exact_identity_and_requested_stages() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}

    resolved = resolve_capability(
        "gga_x_pbe_sol",
        required_stages=("compiled-cpu", "pointwise-validated"),
        evidence=evidence,
    )

    assert resolved.name == base.name
    assert resolved.identity == base.identity
    assert resolved.required_stages == ("pointwise-validated", "compiled-cpu")
    assert resolved.public_dft is False
    assert resolved.to_payload() == {
        "schema": RESOLUTION_SCHEMA,
        "name": base.name,
        "identity": base.identity,
        "required_stages": ["pointwise-validated", "compiled-cpu"],
        "qualified_stages": list(resolved.qualified_stages),
        "public_dft": False,
    }


def test_resolve_capability_does_not_treat_ready_as_qualified() -> None:
    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_capability("GGA_X_PBE_SOL", required_stages=("compiled-cpu",))

    assert caught.value.missing_stages == ("compiled-cpu",)
    assert caught.value.blockers == (("compiled-cpu", "missing pass evidence"),)


def test_resolve_capability_keeps_cpu_and_cuda_requests_distinct() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}

    resolve_capability(base.name, required_stages=("compiled-cpu",), evidence=evidence)
    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_capability(
            base.name, required_stages=("compiled-cuda",), evidence=evidence
        )

    assert caught.value.missing_stages == ("compiled-cuda",)
    assert caught.value.blockers == (("compiled-cuda", "missing pass evidence"),)


def test_resolve_capability_surfaces_explicit_blocker_reason() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        "compiled-cpu": _stage_evidence(
            base,
            "compiled-cpu",
            status="fail",
            reason="compiler qualification failed",
        )
    }

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_capability(base.name, required_stages=("compiled-cpu",), evidence=evidence)

    assert caught.value.identity == base.identity
    assert caught.value.blockers == (("compiled-cpu", "compiler qualification failed"),)


def test_resolve_capability_reports_unmet_prerequisites_without_inference() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        "molecular-scf": _stage_evidence(base, "molecular-scf"),
    }

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_capability(
            base.name, required_stages=("molecular-scf",), evidence=evidence
        )

    reason = dict(caught.value.blockers)["molecular-scf"]
    assert "production-domain" in reason
    assert "compiled-cpu or gpu-runtime" in reason


@pytest.mark.parametrize(
    ("required_stages", "error", "match"),
    [
        ((), ValueError, "must not be empty"),
        (("compiled-cpu", "compiled-cpu"), ValueError, "duplicates"),
        (("invented-stage",), ValueError, "unknown capability stages"),
        ("compiled-cpu", TypeError, "sequence of capability stage names"),
    ],
)
def test_resolve_capability_rejects_ambiguous_stage_requests(
    required_stages: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        resolve_capability(
            "GGA_X_PBE_SOL",
            required_stages=required_stages,  # type: ignore[arg-type]
        )
