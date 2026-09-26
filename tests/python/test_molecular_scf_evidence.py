"""Exact lifecycle receipts for automatic bulk Libxc molecular SCF."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from vibeqc_compiler.common.evidence import canonical_hash
from vibeqc_compiler.xc.endpoint_capability import ENDPOINT_COVERAGE_SCHEMA
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from vibeqc_compiler.xc.molecular_scf_evidence import (
    PHASES,
    RESULT_SCHEMA,
    build_result,
    required_matrix,
    stage_evidence,
    validate_result,
)

NAME = "GGA_X_PBE_SOL"


def _resolution(spin: str) -> SimpleNamespace:
    capability = functional_capability(NAME)
    binding_identity = ("a" if spin == "unpolarized" else "b") * 64
    result_identity = ("c" if spin == "unpolarized" else "d") * 64
    payload = {
        "schema": "vibeqc.bulk-libxc-ks-resolution.v2",
        "backend": "cpu",
        "capability": {
            "name": capability.name,
            "identity": capability.identity,
        },
        "required_ingredients": ["rho", "sigma"],
        "compiled_cpu_binding_identity": binding_identity,
        "compiled_cpu_result_identity": result_identity,
        "method_identity": "1" * 64,
        "method_identifier": f"LIBXC:{NAME}",
        "plan_identity": "2" * 64,
        "spin": spin,
        "reference": "restricted" if spin == "unpolarized" else "unrestricted",
        "required_lowerers": ["semilocal-xc"],
        "public_dft": False,
    }
    return SimpleNamespace(
        backend="cpu",
        capability=SimpleNamespace(name=capability.name, identity=capability.identity),
        method=SimpleNamespace(spin=spin),
        compiled_cpu_binding_identity=binding_identity,
        compiled_cpu_result_identity=result_identity,
        to_payload=lambda: deepcopy(payload),
    )


def _resolutions() -> dict[str, SimpleNamespace]:
    return {spin: _resolution(spin) for spin in ("polarized", "unpolarized")}


def _rows() -> list[dict]:
    rows = []
    for spin, phase in required_matrix():
        cold_geometry = ("3" if spin == "unpolarized" else "4") * 64
        changed_geometry = ("5" if spin == "unpolarized" else "6") * 64
        energy = -1.0 if spin == "unpolarized" else -0.9
        reference = energy + 2.0e-10
        rows.append(
            {
                "spin": spin,
                "phase": phase,
                "status": "pass",
                "reason": None,
                "fixture_identity": ("7" if spin == "unpolarized" else "8") * 64,
                "geometry_identity": (
                    changed_geometry if phase == "changed-geometry" else cold_geometry
                ),
                "reference_identity": ("9" if spin == "unpolarized" else "e") * 64,
                "converged": True,
                "iterations": 7 if phase == "cold" else 2,
                "energy_hartree": energy,
                "reference_energy_hartree": reference,
                "absolute_energy_error_hartree": abs(energy - reference),
                "energy_tolerance_hartree": 1.0e-8,
                "physical_residual_rms": 2.0e-11,
                "residual_tolerance": 1.0e-9,
            }
        )
    return rows


def test_complete_dual_spin_lifecycle_promotes_molecular_scf() -> None:
    result = build_result(
        NAME,
        _resolutions(),
        _rows(),
        evidence="artifact://libxc-scf/GGA_X_PBE_SOL.json",
    )
    envelope = stage_evidence(NAME, result)

    assert result["schema"] == RESULT_SCHEMA
    assert validate_result(NAME, result) == result
    assert envelope["status"] == "pass"
    assert envelope["reason"] is None
    assert envelope["qualification"]["schema"] == ENDPOINT_COVERAGE_SCHEMA
    assert envelope["qualification"]["coverage"] == [
        {"backend": "cpu", "spin": "polarized", "products": ["energy"]},
        {"backend": "cpu", "spin": "unpolarized", "products": ["energy"]},
    ]
    assert result["identity"] in envelope["evidence"]


def test_molecular_scf_requires_exact_spin_phase_matrix() -> None:
    with pytest.raises(ValueError, match="exact lifecycle matrix"):
        build_result(
            NAME,
            _resolutions(),
            _rows()[:-1],
            evidence="test://partial",
        )
    assert required_matrix() == tuple(
        (spin, phase) for spin in ("polarized", "unpolarized") for phase in PHASES
    )


def test_warm_and_changed_geometry_lifecycle_is_fail_closed() -> None:
    warm_drift = _rows()
    warm = next(
        row
        for row in warm_drift
        if row["spin"] == "polarized" and row["phase"] == "warm-replay"
    )
    warm["geometry_identity"] = "f" * 64
    with pytest.raises(ValueError, match="warm molecular-SCF row changed geometry"):
        build_result(NAME, _resolutions(), warm_drift, evidence="test://warm-drift")

    changed_drift = _rows()
    cold = next(
        row
        for row in changed_drift
        if row["spin"] == "unpolarized" and row["phase"] == "cold"
    )
    changed = next(
        row
        for row in changed_drift
        if row["spin"] == "unpolarized" and row["phase"] == "changed-geometry"
    )
    changed["geometry_identity"] = cold["geometry_identity"]
    with pytest.raises(ValueError, match="reused cold geometry"):
        build_result(
            NAME, _resolutions(), changed_drift, evidence="test://changed-drift"
        )


def test_passing_row_must_match_independent_energy_and_residual_gates() -> None:
    rows = _rows()
    rows[0]["absolute_energy_error_hartree"] = 0.0
    with pytest.raises(ValueError, match="energy error is inconsistent"):
        build_result(NAME, _resolutions(), rows, evidence="test://bad-error")

    rows = _rows()
    rows[0]["physical_residual_rms"] = 2.0e-8
    with pytest.raises(ValueError, match="exceeds residual tolerance"):
        build_result(NAME, _resolutions(), rows, evidence="test://bad-residual")


def test_failed_row_remains_nonpromoting_and_actionable() -> None:
    rows = _rows()
    row = rows[0]
    row.update(
        {
            "status": "fail",
            "reason": "independent energy mismatch",
            "converged": False,
        }
    )
    result = build_result(NAME, _resolutions(), rows, evidence="test://failed")
    envelope = stage_evidence(NAME, result)

    assert envelope["status"] == "fail"
    assert "independent energy mismatch" in envelope["reason"]
    assert "qualification" not in envelope


def test_stored_resolution_tampering_invalidates_receipt() -> None:
    result = build_result(NAME, _resolutions(), _rows(), evidence="test://valid")
    tampered = deepcopy(result)
    tampered["resolutions"]["polarized"]["payload"]["compiled_cpu_result_identity"] = (
        "0" * 64
    )

    with pytest.raises(ValueError, match="stored resolution identity mismatch"):
        validate_result(NAME, tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "vibeqc.bulk-libxc-ks-resolution.v1"),
        ("method_identity", None),
        ("plan_identity", "not-a-plan"),
        ("reference", "restricted"),
        ("required_ingredients", ["rho"]),
        ("required_lowerers", ["exact-exchange"]),
    ],
)
def test_rehashed_resolution_must_still_satisfy_v2_contract(
    field: str, value: object
) -> None:
    result = build_result(NAME, _resolutions(), _rows(), evidence="test://wrong-v2")
    item = result["resolutions"]["polarized"]
    item["payload"][field] = value
    item["identity"] = canonical_hash(item["payload"])
    result["identity"] = canonical_hash(
        {key: value for key, value in result.items() if key != "identity"}
    )
    with pytest.raises(ValueError):
        stage_evidence(NAME, result)


def test_builder_rejects_legacy_resolution_schema() -> None:
    resolutions = _resolutions()
    resolution = resolutions["polarized"]
    payload = resolution.to_payload()
    payload["schema"] = "vibeqc.bulk-libxc-ks-resolution.v1"
    resolution.to_payload = lambda: payload
    with pytest.raises(ValueError, match="resolution schema"):
        build_result(NAME, resolutions, _rows(), evidence="test://legacy")


def test_energy_gate_checks_recomputed_error_even_below_reporting_precision() -> None:
    rows = _rows()
    rows[0].update(
        energy_hartree=0.0,
        reference_energy_hartree=5.0e-16,
        absolute_energy_error_hartree=0.0,
        energy_tolerance_hartree=0.0,
    )
    with pytest.raises(ValueError, match="exceeds energy tolerance"):
        build_result(NAME, _resolutions(), rows, evidence="test://underreported")


def test_boolean_numeric_gate_is_not_a_physical_tolerance() -> None:
    rows = _rows()
    rows[0]["residual_tolerance"] = True
    with pytest.raises(ValueError, match="must be finite"):
        build_result(NAME, _resolutions(), rows, evidence="test://boolean")
