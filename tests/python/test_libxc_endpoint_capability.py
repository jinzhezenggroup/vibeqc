"""Regression coverage for exact bulk Libxc endpoint capability resolution."""

from __future__ import annotations

import pytest
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.capability_resolution import CapabilityNotQualified
from vibeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    ENDPOINT_RESOLUTION_SCHEMA,
    resolve_endpoint_capability,
)


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


def _cpu_energy_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    *,
    spin: str = "unpolarized",
) -> dict:
    return {
        "compiled-cpu": _stage_evidence(capability, "compiled-cpu"),
        "production-domain": _stage_evidence(capability, "production-domain"),
        "molecular-scf": _stage_evidence(
            capability,
            "molecular-scf",
            qualification=_coverage(("cpu", spin, ("energy",))),
        ),
    }


def test_exact_cpu_energy_resolution_records_endpoint_identity() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)

    resolved = resolve_endpoint_capability(
        base.name,
        backend="cpu",
        product="energy",
        spin="unpolarized",
        evidence=evidence,
    )

    assert resolved.identity == base.identity
    assert resolved.required_stages == (
        "compiled-cpu",
        "production-domain",
        "molecular-scf",
    )
    assert resolved.backend == "cpu"
    assert resolved.product == "energy"
    assert resolved.spin == "unpolarized"
    assert resolved.public_dft is False
    payload = resolved.to_payload()
    assert payload["schema"] == ENDPOINT_RESOLUTION_SCHEMA
    assert payload["identity"] == base.identity
    assert payload["capability"]["identity"] == base.identity


def test_cpu_molecular_evidence_cannot_inflate_cuda_endpoint() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)
    evidence.update(
        {
            "compiled-cuda": _stage_evidence(base, "compiled-cuda"),
            "gpu-runtime": _stage_evidence(base, "gpu-runtime"),
        }
    )

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_endpoint_capability(
            base.name,
            backend="cuda",
            product="energy",
            spin="unpolarized",
            evidence=evidence,
        )

    assert caught.value.blockers == (
        (
            "molecular-scf",
            "pass evidence does not cover backend=cuda spin=unpolarized product=energy",
        ),
    )


def test_energy_coverage_cannot_inflate_force_capability() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)
    evidence["forces"] = _stage_evidence(
        base,
        "forces",
        qualification=_coverage(("cpu", "unpolarized", ("energy",))),
    )

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_endpoint_capability(
            base.name,
            backend="cpu",
            product="forces",
            spin="unpolarized",
            evidence=evidence,
        )

    assert caught.value.blockers == (
        (
            "forces",
            "pass evidence does not cover backend=cpu spin=unpolarized product=forces",
        ),
    )


def test_spin_coverage_is_exact() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base, spin="polarized")

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_endpoint_capability(
            base.name,
            backend="cpu",
            product="energy",
            spin="unpolarized",
            evidence=evidence,
        )

    assert caught.value.blockers == (
        (
            "molecular-scf",
            "pass evidence does not cover backend=cpu spin=unpolarized product=energy",
        ),
    )


def test_public_force_requires_exact_public_product_coverage() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)
    evidence["forces"] = _stage_evidence(
        base,
        "forces",
        qualification=_coverage(("cpu", "unpolarized", ("forces",))),
    )
    evidence["public-method"] = _stage_evidence(
        base,
        "public-method",
        qualification=_coverage(("cpu", "unpolarized", ("energy",))),
    )

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_endpoint_capability(
            base.name,
            backend="cpu",
            product="forces",
            spin="unpolarized",
            require_public=True,
            evidence=evidence,
        )

    assert caught.value.blockers == (
        (
            "public-method",
            "pass evidence does not cover backend=cpu spin=unpolarized product=forces",
        ),
    )


def test_matching_public_energy_coverage_is_admitted_exactly() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)
    evidence["public-method"] = _stage_evidence(
        base,
        "public-method",
        qualification=_coverage(("cpu", "unpolarized", ("energy",))),
    )

    resolved = resolve_endpoint_capability(
        base.name,
        backend="cpu",
        product="energy",
        spin="unpolarized",
        require_public=True,
        evidence=evidence,
    )

    assert resolved.public_dft is True
    assert "public-method" in resolved.required_stages


def test_missing_endpoint_qualification_fails_closed() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = _cpu_energy_evidence(base)
    evidence["molecular-scf"] = _stage_evidence(base, "molecular-scf")

    with pytest.raises(CapabilityNotQualified) as caught:
        resolve_endpoint_capability(
            base.name,
            backend="cpu",
            product="energy",
            spin="unpolarized",
            evidence=evidence,
        )

    assert caught.value.blockers == (
        ("molecular-scf", "missing endpoint coverage qualification"),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"backend": "metal", "product": "energy", "spin": "unpolarized"},
            ValueError,
            "unsupported backend",
        ),
        (
            {"backend": "cpu", "product": "hessian", "spin": "unpolarized"},
            ValueError,
            "unsupported product",
        ),
        (
            {"backend": "cpu", "product": "energy", "spin": "restricted"},
            ValueError,
            "unsupported spin",
        ),
        (
            {
                "backend": "cpu",
                "product": "energy",
                "spin": "unpolarized",
                "require_public": 1,
            },
            TypeError,
            "require_public must be a bool",
        ),
    ],
)
def test_endpoint_request_rejects_ambiguous_inputs(
    kwargs: dict,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        resolve_endpoint_capability("GGA_X_PBE_SOL", **kwargs)
