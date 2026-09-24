"""Conventional MP2 actual-path coverage for the cross-method production ledger."""

from __future__ import annotations

import typing

from tools.check_electronic_production_paths import DEFAULT_LEDGER, load_and_validate


def _mp2_row() -> dict[str, object]:
    payload, errors = load_and_validate(DEFAULT_LEDGER)
    assert errors == []
    rows = typing.cast("list[dict[str, object]]", payload["rows"])
    matches = [row for row in rows if row["id"] == "mp2-energy-cpu-conventional"]
    assert len(matches) == 1
    return matches[0]


def test_cpu_conventional_mp2_actual_path_anchors_are_explicit() -> None:
    row = _mp2_row()
    assert row["method_family"] == "mp2"
    assert row["product"] == "energy"
    assert row["backend"] == "cpu"
    assert row["domain"] == "canonical-conventional-mp2"
    assert row["status"] == "production"
    assert row["scientific_owner"] == "tools/vibeqc_mp2/equations.py"
    assert row["provider_owner"] == "src/posthf/native_provider.cpp"
    assert row["execution_owner"] == "src/methods/mp2_method.cpp"
    assert row["selector"] == "src/methods/registry.cpp"
    assert row["state_owner"] == "src/core/electronic_reference.hpp"
    assert row["resource_owner"] == "src/posthf/mp2_energy.cpp"


def test_cpu_conventional_mp2_evidence_does_not_promote_cuda_or_force() -> None:
    row = _mp2_row()
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])

    assert levels["represented"]["state"] == "present"
    assert levels["compiled-cpu"]["state"] == "present"
    assert levels["domain-qualified"]["state"] == "present"
    assert levels["molecular"]["state"] == "present"
    assert levels["public"]["state"] == "present"

    assert levels["compiled-cuda"]["state"] == "not-applicable"
    assert levels["device-executed"]["state"] == "not-applicable"
    assert levels["derivative"]["state"] == "not-applicable"
    assert levels["compiled-cuda"]["evidence"] == []
    assert levels["device-executed"]["evidence"] == []
    assert levels["derivative"]["evidence"] == []
