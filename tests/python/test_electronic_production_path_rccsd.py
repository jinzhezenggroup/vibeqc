"""RCCSD actual-path coverage for the cross-method production ledger."""

from __future__ import annotations

import typing

from tools.check_electronic_production_paths import DEFAULT_LEDGER, load_and_validate


def _rccsd_row() -> dict[str, object]:
    payload, errors = load_and_validate(DEFAULT_LEDGER)
    assert errors == []
    rows = typing.cast("list[dict[str, object]]", payload["rows"])
    matches = [row for row in rows if row["id"] == "cc-energy-cpu-rccsd"]
    assert len(matches) == 1
    return matches[0]


def test_cpu_rccsd_actual_path_anchors_are_explicit() -> None:
    row = _rccsd_row()
    assert row["method_family"] == "cc"
    assert row["product"] == "energy"
    assert row["backend"] == "cpu"
    assert row["domain"] == "canonical-rccsd"
    assert row["status"] == "production"
    assert row["scientific_owner"] == "tools/vibeqc_cc/doubles.py"
    assert row["provider_owner"] == "src/posthf/native_provider.cpp"
    assert row["execution_owner"] == "src/methods/rccsd_method.cpp"
    assert row["selector"] == "src/methods/registry.cpp"
    assert row["state_owner"] == "src/cc/solver.hpp"
    assert row["resource_owner"] == "src/cc/solver.cpp"


def test_cpu_rccsd_evidence_does_not_inflate_cuda_or_derivatives() -> None:
    row = _rccsd_row()
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
