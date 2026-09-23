"""Admission-boundary tests for machine-readable bulk Libxc claims."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from vibeqc_compiler.xc import (
    UnsupportedXC,
    build_program,
    functional,
    libxc_bulk,
    libxc_bulk_capabilities,
)
from vibeqc_compiler.xc.libxc_maple import MapleImportError
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, COMPONENTS

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests/data/xc/libxc-bulk"


def _fixtures() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_ROOT.glob("*.json"))
    ]


def _stage_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    status: str = "pass",
    reason: str | None = None,
) -> dict:
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": status,
        "reason": reason,
        "evidence": f"test://{capability.name}/{stage}" if status == "pass" else None,
    }
    if stage == "production-domain" and status == "pass":
        payload["qualification"] = capability.production_domain_profile.to_payload()
    return payload


def test_bulk_capability_inventory_is_exact_imported_inventory() -> None:
    capabilities = libxc_bulk_capabilities.available_capabilities()
    imported = libxc_bulk.available_functionals()
    catalog = libxc_bulk.read_catalog()

    assert tuple(item.name for item in capabilities) == imported
    assert libxc_bulk_capabilities.claimable_functionals("graph-imported") == imported
    assert (
        libxc_bulk_capabilities.claimable_functionals("pointwise-validated") == imported
    )
    assert len(capabilities) == catalog["counts"]["graph_imported_registrations"]

    for capability in capabilities:
        payload = capability.to_payload()
        assert payload["schema"] == "vibeqc.libxc-bulk-capability.v2"
        assert capability.domain == libxc_bulk.BULK_SEMANTICS
        assert capability.spin_layouts == ("polarized", "unpolarized")
        assert capability.validated_outputs == ("energy", "vxc", "fxc")
        assert capability.source_emitters == ("c", "cuda")
        assert capability.flags
        assert capability.required_ingredients
        assert capability.required_ingredients[0] == "rho"
        assert capability.claim_level == "pointwise-validated"
        assert capability.qualified_stages == (
            "graph-imported",
            "pointwise-validated",
        )
        expected_ready = ["compiled-cpu", "compiled-cuda"]
        if capability.production_domain_profile.eligible:
            expected_ready.append("production-domain")
        assert capability.ready_stages == tuple(expected_ready)
        assert capability.public_dft is False
        assert "molecular-scf" in capability.unqualified_stages
        assert "forces" in capability.unqualified_stages
        assert "response" in capability.unqualified_stages
        assert "public-method" in capability.unqualified_stages


def test_every_pointwise_claim_has_both_spin_independent_reference_evidence() -> None:
    imported = set(libxc_bulk_capabilities.claimable_functionals())
    fixture_payloads = _fixtures()

    assert fixture_payloads
    assert {
        (case["name"], case["spin"])
        for payload in fixture_payloads
        for case in payload["cases"]
    } == {(name, spin) for name in imported for spin in ("polarized", "unpolarized")}

    for payload in fixture_payloads:
        assert payload["oracle"]["pyscf"] == "2.14.0"
        assert payload["oracle"]["libxc"] == "7.0.0"
        for case in payload["cases"]:
            assert len(case["features"]) == 2
            assert len(case["expected"]) == 2


def test_bulk_capability_lookup_is_case_insensitive_and_fail_closed() -> None:
    capability = libxc_bulk_capabilities.functional_capability("gga_x_pbe_sol")
    assert capability.name == "GGA_X_PBE_SOL"

    catalog = libxc_bulk.read_catalog()
    blocked = next(
        item["name"]
        for item in catalog["registrations"]
        if item["graph_status"] == "blocked"
    )
    for name in ("NOT_A_FUNCTIONAL", blocked):
        with pytest.raises(MapleImportError):
            libxc_bulk_capabilities.functional_capability(name)

    for level in ("compiled-cuda", "gpu-runtime", "molecular-scf", "public-method"):
        assert libxc_bulk_capabilities.claimable_functionals(level) == ()

    with pytest.raises(ValueError, match="unknown bulk capability stage"):
        libxc_bulk_capabilities.claimable_functionals("not-a-stage")


def test_stage_promotion_requires_prerequisites_and_never_jumps() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        "molecular-scf": _stage_evidence(base, "molecular-scf"),
        "public-method": _stage_evidence(base, "public-method"),
    }

    capability = libxc_bulk_capabilities.functional_capability(
        base.name, evidence=evidence
    )
    assert capability.qualified_stages == (
        "graph-imported",
        "pointwise-validated",
    )
    assert capability.public_dft is False
    assert "molecular-scf" not in capability.ready_stages
    assert "public-method" not in capability.ready_stages


def test_cpu_evidence_promotes_scf_and_public_without_force_or_response() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        stage: _stage_evidence(base, stage)
        for stage in (
            "compiled-cpu",
            "production-domain",
            "molecular-scf",
            "public-method",
        )
    }

    capability = libxc_bulk_capabilities.functional_capability(
        base.name, evidence=evidence
    )
    assert capability.qualified_stages == (
        "graph-imported",
        "pointwise-validated",
        "compiled-cpu",
        "production-domain",
        "molecular-scf",
        "public-method",
    )
    assert capability.public_dft is True
    assert "forces" in capability.ready_stages
    assert "response" in capability.ready_stages
    assert "compiled-cuda" in capability.ready_stages


def test_gpu_runtime_is_alternative_execution_prerequisite_for_scf() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        stage: _stage_evidence(base, stage)
        for stage in (
            "compiled-cuda",
            "production-domain",
            "gpu-runtime",
            "molecular-scf",
        )
    }

    capability = libxc_bulk_capabilities.functional_capability(
        base.name, evidence=evidence
    )
    assert "compiled-cpu" not in capability.qualified_stages
    assert "compiled-cuda" in capability.qualified_stages
    assert "gpu-runtime" in capability.qualified_stages
    assert "molecular-scf" in capability.qualified_stages


def test_claimable_inventory_is_computed_from_attached_evidence() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {
        base.name: {
            "compiled-cpu": _stage_evidence(base, "compiled-cpu"),
        }
    }

    assert libxc_bulk_capabilities.claimable_functionals("compiled-cpu", evidence) == (
        base.name,
    )
    assert (
        libxc_bulk_capabilities.claimable_functionals("compiled-cuda", evidence) == ()
    )


def test_stage_evidence_is_identity_bound_and_schema_checked() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    good = _stage_evidence(base, "compiled-cpu")

    bad_identity = {**good, "subject_identity": "0" * 64}
    with pytest.raises(ValueError, match="subject identity mismatch"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"compiled-cpu": bad_identity}
        )

    bad_stage = {**good, "stage": "compiled-cuda"}
    with pytest.raises(ValueError, match="stage mismatch"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"compiled-cpu": bad_stage}
        )

    no_reference = {**good, "evidence": ""}
    with pytest.raises(ValueError, match="non-empty evidence reference"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"compiled-cpu": no_reference}
        )

    with pytest.raises(ValueError, match="cannot be overridden"):
        libxc_bulk_capabilities.functional_capability(
            base.name,
            evidence={
                "graph-imported": {
                    **good,
                    "stage": "graph-imported",
                }
            },
        )


@pytest.mark.parametrize(
    "change", ("binding", "source", "upstream", "binding-semantics")
)
def test_stage_evidence_rejects_changed_scientific_inputs(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Same name and owner must not reuse proof for different scientific input."""
    import copy

    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}
    catalog = copy.deepcopy(libxc_bulk.read_catalog())
    record = next(
        item for item in catalog["registrations"] if item["name"] == base.name
    )
    if change == "binding":
        record["bindings"]["params_a_mu"] = "0.5"
    elif change == "source":
        catalog["source_files"][record["entry"]] = "0" * 64
    elif change == "upstream":
        catalog["upstream"] = {"revision": "changed-source-revision"}
    else:
        record["binding_semantics"] = "changed-parameter-interpretation"
    monkeypatch.setattr(libxc_bulk, "read_catalog", lambda: catalog)
    with pytest.raises(ValueError, match="subject identity mismatch"):
        libxc_bulk_capabilities.functional_capability(base.name, evidence=evidence)


def test_stage_evidence_rejects_changed_compiler_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implementation drift invalidates proof without importing native runtime."""
    from vibeqc_compiler.common import paths

    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    evidence = {"compiled-cpu": _stage_evidence(base, "compiled-cpu")}
    original = paths.source_hashes

    def changed(*families: str, assets: tuple[str, ...] = ()) -> dict[str, str]:
        result = original(*families, assets=assets)
        result["python/vibeqc_compiler/integral/scalar_c.py"] = "0" * 64
        return result

    monkeypatch.setattr(paths, "source_hashes", changed)
    with pytest.raises(ValueError, match="subject identity mismatch"):
        libxc_bulk_capabilities.functional_capability(base.name, evidence=evidence)


def test_pointwise_lda_gga_components_are_automatically_representable() -> None:
    represented = libxc_bulk_capabilities.claimable_components(families=("lda", "gga"))
    assert represented
    assert "GGA_X_PBE_SOL" in represented
    assert set(represented) <= set(COMPONENTS)
    assert "GGA_X_PBE_SOL" in AUTO_BULK_COMPONENTS

    spec = functional("GGA_X_PBE_SOL", spin="unpolarized")
    assert spec.components == (("GGA_X_PBE_SOL", Fraction(1)),)
    payload = spec.to_payload()
    assert payload["qualification"] == "pointwise-validated"
    assert payload["production_admitted"] is False
    assert payload["expression_provenance"]["kind"] == "libxc-bulk-pointwise"

    with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
        build_program(spec)


def test_tau_only_mgga_is_automatically_representable_but_not_production() -> None:
    representable = libxc_bulk_capabilities.claimable_components(
        families=("mgga",),
        supported_ingredients=("rho", "sigma", "tau"),
    )
    assert "MGGA_X_R2SCAN01" in representable
    assert "MGGA_X_R2SCAN01" in AUTO_BULK_COMPONENTS

    spec = functional("MGGA_X_R2SCAN01", spin="unpolarized")
    assert spec.ingredients == ("rho", "sigma", "tau")
    assert spec.to_payload()["production_admitted"] is False
    with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
        build_program(spec)


def test_laplacian_mgga_remains_fail_closed_until_feature_ir_exists() -> None:
    all_mgga = libxc_bulk_capabilities.claimable_components(families=("mgga",))
    representable = libxc_bulk_capabilities.claimable_components(
        families=("mgga",),
        supported_ingredients=("rho", "sigma", "tau"),
    )
    assert "MGGA_X_JK" in all_mgga
    assert "MGGA_X_JK" not in representable
    assert "MGGA_X_JK" not in AUTO_BULK_COMPONENTS
    capability = libxc_bulk_capabilities.functional_capability("MGGA_X_JK")
    assert "laplacian" in capability.required_ingredients
    with pytest.raises(UnsupportedXC, match="unknown functional"):
        functional("MGGA_X_JK")


def test_automatic_component_filters_are_fail_closed() -> None:
    for families in ((), ("hybrid",), ("lda", "hybrid")):
        with pytest.raises(ValueError, match="lda/gga/mgga"):
            libxc_bulk_capabilities.claimable_components(families=families)
    for ingredients in ((), ("current",), ("rho", "current")):
        with pytest.raises(ValueError, match="rho/sigma/laplacian/tau"):
            libxc_bulk_capabilities.claimable_components(
                supported_ingredients=ingredients
            )


def test_production_domain_profiles_are_ingredient_driven_and_versioned() -> None:
    lda = libxc_bulk_capabilities.functional_capability("LDA_X")
    gga = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    mgga = libxc_bulk_capabilities.functional_capability("MGGA_X_R2SCAN01")

    assert lda.production_domain_profile.eligible is True
    assert gga.production_domain_profile.eligible is True
    assert mgga.production_domain_profile.eligible is True

    assert "sigma/zero" not in lda.production_domain_profile.case_ids
    assert "sigma/zero" in gga.production_domain_profile.case_ids
    assert "tau/isoorbital" not in gga.production_domain_profile.case_ids
    assert "tau/isoorbital" in mgga.production_domain_profile.case_ids

    for capability in (lda, gga, mgga):
        profile = capability.production_domain_profile
        payload = profile.to_payload()
        assert payload["schema"] == "vibeqc.libxc-production-domain-profile.v1"
        assert payload["profile"] == "semilocal-boundary-matrix/v1"
        assert payload["identity"] == profile.identity
        assert payload["spin_layouts"] == ["polarized", "unpolarized"]
        assert payload["outputs"] == ["energy", "vxc", "fxc"]
        assert "density/vacuum" in payload["case_ids"]
        assert "spin/zero-a" in payload["case_ids"]
        assert "control/lazy-inactive-branch" in payload["case_ids"]
        assert "control/invalid-nonfinite" in payload["case_ids"]


def test_laplacian_profile_is_explicitly_blocked_from_production_readiness() -> None:
    capability = libxc_bulk_capabilities.functional_capability("MGGA_X_JK")
    profile = capability.production_domain_profile

    assert profile.eligible is False
    assert profile.blocker == "unsupported-ingredients:laplacian"
    assert "production-domain" not in capability.ready_stages
    assert capability.to_payload()["production_domain_profile"]["blocker"] == (
        "unsupported-ingredients:laplacian"
    )


def test_production_domain_pass_requires_exact_versioned_qualification() -> None:
    base = libxc_bulk_capabilities.functional_capability("GGA_X_PBE_SOL")
    generic = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": base.identity,
        "stage": "production-domain",
        "status": "pass",
        "reason": None,
        "evidence": "test://generic-production-pass",
    }
    with pytest.raises(ValueError, match="requires qualification profile"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"production-domain": generic}
        )

    good = _stage_evidence(base, "production-domain")
    capability = libxc_bulk_capabilities.functional_capability(
        base.name, evidence={"production-domain": good}
    )
    assert "production-domain" in capability.qualified_stages
    stage = next(
        item for item in capability.stage_evidence if item.stage == "production-domain"
    )
    assert stage.qualification == base.production_domain_profile.to_payload()


def test_production_domain_pass_rejects_partial_or_stale_boundary_matrix() -> None:
    base = libxc_bulk_capabilities.functional_capability("MGGA_X_R2SCAN01")
    good = _stage_evidence(base, "production-domain")

    partial = {
        **good,
        "qualification": {
            **good["qualification"],
            "case_ids": good["qualification"]["case_ids"][:-1],
        },
    }
    with pytest.raises(ValueError, match="does not cover exact profile"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"production-domain": partial}
        )

    stale = {
        **good,
        "qualification": {
            **good["qualification"],
            "profile": "semilocal-boundary-matrix/v0",
        },
    }
    with pytest.raises(ValueError, match="unsupported schema/profile"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"production-domain": stale}
        )


def test_blocked_ingredient_cannot_be_promoted_with_forged_profile() -> None:
    base = libxc_bulk_capabilities.functional_capability("MGGA_X_JK")
    evidence = _stage_evidence(base, "production-domain")

    with pytest.raises(ValueError, match="qualification is blocked"):
        libxc_bulk_capabilities.functional_capability(
            base.name, evidence={"production-domain": evidence}
        )
