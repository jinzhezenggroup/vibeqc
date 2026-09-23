"""Small real-import gates; the full catalog census is an opt-in offline tool."""

from __future__ import annotations

from dataclasses import replace

import pytest
from vibeqc_compiler.xc import bulk_aot, libxc_bulk


@pytest.mark.parametrize("name", ["LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"])
def test_census_reuses_existing_real_emitters(name: str) -> None:
    program = libxc_bulk.build_bulk_program(name, spin="unpolarized")
    for backend in bulk_aot.BACKENDS:
        item = bulk_aot.inspect_program(
            program, 0, backend, domain=libxc_bulk.BULK_SEMANTICS,
        )
        assert item.source == program.emit_source(0, cuda=backend == "cuda")
        assert item.energy_nodes > 0
        assert item.ssa["root_count"] == 1
        # A separate provenance record is not a separate compilation task.
        alias = replace(item, name="CENSUS_ALIAS", import_identity="another-owner")
        plan = bulk_aot.plan_package([item, alias])
        assert plan["summary"]["unique_artifacts"] == 1
        assert len(plan["artifacts"][0]["registrations"]) == 2
        assert plan["capability_claims"] == []


def test_real_catalog_selection_records_emission_without_compilation() -> None:
    plan = bulk_aot.census_catalog(
        names=["LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"],
        spins=["unpolarized"], derivative_orders=[0], backends=["cpu"],
        budget=bulk_aot.PackageBudget(max_artifacts=2),
    )
    assert plan["summary"]["registration_variants"] == 3
    assert plan["summary"]["selected_artifacts"] <= 2
    assert all(row["status"] == "emitted" for row in plan["observations"])
    assert "measurements" not in plan
    assert plan["capability_claims"] == []
