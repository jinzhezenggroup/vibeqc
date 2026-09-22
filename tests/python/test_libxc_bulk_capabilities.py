"""Admission-boundary tests for machine-readable bulk Libxc claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeqc_compiler.xc import libxc_bulk, libxc_bulk_capabilities
from vibeqc_compiler.xc.libxc_maple import MapleImportError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests/data/xc/libxc-bulk"


def _fixtures() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_ROOT.glob("*.json"))
    ]


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
        assert payload["schema"] == "vibeqc.libxc-bulk-capability.v1"
        assert capability.domain == libxc_bulk.BULK_SEMANTICS
        assert capability.spin_layouts == ("polarized", "unpolarized")
        assert capability.validated_outputs == ("energy", "vxc", "fxc")
        assert capability.source_emitters == ("c", "cuda")
        assert capability.claim_level == "pointwise-validated"
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
        with pytest.raises(ValueError):
            libxc_bulk_capabilities.claimable_functionals(level)
