"""GFN2-xTB CPU path coverage for the production ledger."""

from __future__ import annotations

import typing

from tools.check_electronic_production_paths import DEFAULT_LEDGER, load_and_validate


def _gfn2_row() -> dict[str, object]:
    payload, errors = load_and_validate(DEFAULT_LEDGER)
    assert errors == []
    rows = typing.cast("list[dict[str, object]]", payload["rows"])
    matches = [row for row in rows if row["id"] == "gfn2-energy-cpu-scc"]
    assert len(matches) == 1
    return matches[0]


def test_gfn2_cpu_actual_path_is_explicit() -> None:
    row = _gfn2_row()
    assert row["method_family"] == "gfn"
    assert row["product"] == "energy"
    assert row["backend"] == "cpu"
    assert row["domain"] == "gfn2-xtb-restricted-scc"
    assert row["status"] == "production"
    assert row["public_entry"] == "python/vibeqc/calculator.py"
    assert row["selector"] == "src/methods/xtb_method.cpp"
    assert row["scientific_owner"] == "src/xtb/native/src/model/gfn2/scc_driver.cpp"
    assert row["execution_owner"] == "src/xtb/native/src/runtime/gfn2_cpu_execution.cpp"
    assert row["state_owner"] == "src/xtb/native/src/model/gfn2/scc_driver.hpp"


def test_gfn2_cpu_evidence_stays_scoped() -> None:
    row = _gfn2_row()
    levels = typing.cast("dict[str, dict[str, object]]", row["evidence_levels"])

    for level in (
        "represented",
        "compiled-cpu",
        "domain-qualified",
        "molecular",
        "public",
    ):
        assert levels[level]["state"] == "present"
    for level in ("compiled-cuda", "device-executed", "derivative"):
        assert levels[level]["state"] == "not-applicable"
        assert levels[level]["evidence"] == []

    evidence = typing.cast("list[str]", row["evidence"])
    assert "tests/python/test_gfn2_runtime_bridge_boundary.py" in evidence
    assert "tests/python/test_gfn2_xtb.py" in evidence
