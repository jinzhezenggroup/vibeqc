"""Public XC extension capability queries remain evidence-backed."""

from __future__ import annotations

import pytest
from vibeqc.extensions import xc
from vibeqc_compiler.xc import libxc_bulk_capabilities


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


def test_capability_keeps_public_support_levels_distinct() -> None:
    report = xc.capability("GGA_X_PBE_SOL")

    assert report["kind"] == "xc-capability"
    assert report["representable"] is True
    assert report["pointwise_validated"] is True
    assert report["compiled_targets"] == {"cpu": False, "cuda": False}
    assert report["production_domain_qualified"] is False
    assert report["cuda_runtime_validated"] is False
    assert report["molecular_validated"] is False
    assert report["derivative_validation"] == {"forces": False, "response": False}
    assert report["production_promoted"] is False
    assert report["qualified_stages"] == ["graph-imported", "pointwise-validated"]
    assert "compiled-cpu" in report["ready_stages"]


def test_capability_does_not_promote_cpu_compile_evidence_to_cuda() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    report = xc.capability(
        "gga_x_pbe_sol",
        evidence={"compiled-cpu": _stage_evidence(base, "compiled-cpu")},
    )

    assert report["identity"] == base.identity
    assert report["compiled_targets"] == {"cpu": True, "cuda": False}
    assert report["molecular_validated"] is False
    assert report["production_promoted"] is False


def test_capability_retains_explicit_failed_evidence_without_promotion() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    report = xc.capability(
        base.name,
        evidence={
            "compiled-cuda": _stage_evidence(
                base,
                "compiled-cuda",
                status="fail",
                reason="device compiler qualification failed",
            )
        },
    )

    assert report["compiled_targets"]["cuda"] is False
    assert report["production_promoted"] is False
    assert report["stage_evidence"] == [
        {
            "stage": "compiled-cuda",
            "status": "fail",
            "evidence": f"test://{base.name}/compiled-cuda/fail",
            "reason": "device compiler qualification failed",
        }
    ]


def test_capability_rejects_unknown_or_empty_registration() -> None:
    with pytest.raises(xc.UnsupportedXC, match="unknown bulk Libxc registration"):
        xc.capability("not-a-registered-functional")
    with pytest.raises(TypeError, match="non-empty identifier"):
        xc.capability("")
