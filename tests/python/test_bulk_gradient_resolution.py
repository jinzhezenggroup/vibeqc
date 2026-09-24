"""Regression coverage for automatic bulk Libxc force qualification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vibeqc_compiler.method import bulk_gradient
from vibeqc_compiler.method.bulk_gradient import (
    BULK_FORCE_RESOLUTION_SCHEMA,
    resolve_bulk_force_capability,
)
from vibeqc_compiler.method.spec import UnsupportedMethod
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.capability_resolution import CapabilityNotQualified
from vibeqc_compiler.xc.endpoint_capability import ENDPOINT_COVERAGE_SCHEMA


def _coverage(
    *rows: tuple[str, str, tuple[str, ...]],
) -> dict:
    return {
        "schema": ENDPOINT_COVERAGE_SCHEMA,
        "coverage": [
            {
                "backend": backend,
                "spin": spin,
                "products": list(products),
            }
            for backend, spin, products in rows
        ],
    }


def _stage_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    qualification: dict | None = None,
) -> dict:
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": "pass",
        "reason": None,
        "evidence": f"test://{capability.name}/{stage}",
    }
    if stage == "production-domain" and qualification is None:
        qualification = capability.production_domain_profile.to_payload()
    if qualification is not None:
        payload["qualification"] = qualification
    return payload


def _force_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    *,
    backend: str,
    spin: str,
    force_products: tuple[str, ...] = ("forces",),
) -> dict:
    evidence = {
        "production-domain": _stage_evidence(capability, "production-domain"),
        "molecular-scf": _stage_evidence(
            capability,
            "molecular-scf",
            qualification=_coverage((backend, spin, ("energy",))),
        ),
        "forces": _stage_evidence(
            capability,
            "forces",
            qualification=_coverage((backend, spin, force_products)),
        ),
    }
    if backend == "cpu":
        evidence["compiled-cpu"] = _stage_evidence(capability, "compiled-cpu")
    else:
        evidence["compiled-cuda"] = _stage_evidence(capability, "compiled-cuda")
        evidence["gpu-runtime"] = _stage_evidence(capability, "gpu-runtime")
    return evidence


def test_bulk_force_resolution_binds_endpoint_and_ingredient_identity() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _force_evidence(base, backend="cpu", spin="unpolarized")

    resolved = resolve_bulk_force_capability(
        base.name,
        backend="cpu",
        spin="unpolarized",
        evidence=evidence,
    )

    assert resolved.identity == base.identity
    assert resolved.backend == "cpu"
    assert resolved.spin == "unpolarized"
    assert resolved.family == "gga"
    assert resolved.required_ingredients == ("rho", "sigma")
    assert (
        resolved.production_domain_identity == base.production_domain_profile.identity
    )
    assert resolved.tau_generalized_ks is False
    assert resolved.public_dft is False

    payload = resolved.to_payload()
    assert payload["schema"] == BULK_FORCE_RESOLUTION_SCHEMA
    assert payload["identity"] == base.identity
    assert payload["required_ingredients"] == ["rho", "sigma"]
    assert payload["endpoint"]["product"] == "forces"


def test_energy_only_capability_cannot_promote_bulk_force() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _force_evidence(base, backend="cpu", spin="unpolarized")
    evidence.pop("forces")

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_bulk_force_capability(
            base.name,
            backend="cpu",
            spin="unpolarized",
            evidence=evidence,
        )

    assert "forces" in caught.value.missing_stages


def test_force_stage_requires_exact_force_product_coverage() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _force_evidence(
        base,
        backend="cpu",
        spin="unpolarized",
        force_products=("energy",),
    )

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_bulk_force_capability(
            base.name,
            backend="cpu",
            spin="unpolarized",
            evidence=evidence,
        )

    assert caught.value.blockers == (
        (
            "forces",
            "pass evidence does not cover backend=cpu spin=unpolarized product=forces",
        ),
    )


def test_tau_mgga_uses_same_generic_force_resolver() -> None:
    base = libxc_bulk_capabilities.functional_capability("MGGA_X_R2SCAN01")
    evidence = _force_evidence(base, backend="cpu", spin="polarized")

    resolved = resolve_bulk_force_capability(
        base.name,
        backend="cpu",
        spin="polarized",
        evidence=evidence,
    )

    assert resolved.family == "mgga"
    assert resolved.required_ingredients == ("rho", "sigma", "tau")
    assert resolved.tau_generalized_ks is True


def test_cpu_and_cuda_share_scientific_capability_identity() -> None:
    base = libxc_bulk_capabilities.functional_capability("LDA_C_VWN_4")
    cpu = resolve_bulk_force_capability(
        base.name,
        backend="cpu",
        spin="unpolarized",
        evidence=_force_evidence(base, backend="cpu", spin="unpolarized"),
    )
    cuda = resolve_bulk_force_capability(
        base.name,
        backend="cuda",
        spin="unpolarized",
        evidence=_force_evidence(base, backend="cuda", spin="unpolarized"),
    )

    assert cpu.identity == cuda.identity == base.identity
    assert cpu.production_domain_identity == cuda.production_domain_identity
    assert cpu.required_ingredients == cuda.required_ingredients == ("rho",)
    assert cpu.backend == "cpu"
    assert cuda.backend == "cuda"


def test_unsupported_gradient_ingredient_fails_before_endpoint_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    fake = SimpleNamespace(
        name=base.name,
        family="mgga",
        required_ingredients=("rho", "sigma", "laplacian"),
    )
    monkeypatch.setattr(
        bulk_gradient,
        "functional_capability",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        bulk_gradient,
        "resolve_endpoint_capability",
        lambda *args, **kwargs: pytest.fail("endpoint promotion must stay inactive"),
    )

    with pytest.raises(UnsupportedMethod, match="laplacian"):
        resolve_bulk_force_capability(
            base.name,
            backend="cpu",
            spin="unpolarized",
        )


def test_curated_registration_cannot_enter_automatic_force_lane() -> None:
    with pytest.raises(UnsupportedMethod, match="non-curated pure semilocal"):
        resolve_bulk_force_capability(
            "LDA_X",
            backend="cpu",
            spin="unpolarized",
        )


@pytest.mark.parametrize(
    "name", ["LDA_X", "lda_x", "GGA_X_PBE", "MGGA_X_SCAN", "HYB_GGA_XC_PBE0"]
)
def test_nonautomatic_registration_rejected_before_bulk_import(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    assert name.upper() not in bulk_gradient.AUTO_BULK_COMPONENTS
    monkeypatch.setattr(
        bulk_gradient,
        "functional_capability",
        lambda *args, **kwargs: pytest.fail("nonautomatic graph must not be imported"),
    )
    with pytest.raises(UnsupportedMethod, match="non-curated pure semilocal"):
        resolve_bulk_force_capability(name, backend="cpu", spin="unpolarized")
