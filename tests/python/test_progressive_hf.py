"""Projected seeds must converge the actual target Hamiltonian and forces."""

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator
from vibeqc.checkpoint import CheckpointError
from vibeqc.progressive import projected_singlepoint
from vibeqc.projection import ProjectionPolicy, ProjectionRejected

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("fitted", [False, True])
def test_small_to_large_matches_independent_target_energy_force(
    method, charge, multiplicity, fitted
):
    options = {
        "method": method,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    source = Calculator(basis="sto-3g", **options)
    target = Calculator(
        basis="def2-svp", density_fitting="cpu" if fitted else "none", **options
    )
    result = projected_singlepoint(
        target, source, ATOMS, charge=charge, multiplicity=multiplicity
    )
    cold = target.singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
    assert result.target.restart_origin == "basis_projection"
    assert result.target.fock_builds == result.target.iterations + 2
    assert (
        result.target.converged
        and result.diagnostics["target_verification"] == "executed"
    )
    assert (
        result.diagnostics["items"][0]["source_model_identity"]
        != result.diagnostics["items"][0]["target_model_identity"]
    )
    np.testing.assert_allclose(result.target.energy, cold.energy, atol=3e-10, rtol=0)
    np.testing.assert_allclose(result.target.forces, cold.forces, atol=3e-9, rtol=0)
    references = json.loads(
        (Path(__file__).parents[1] / "data/basis_projection_reference.json").read_text()
    )["hf"]
    reference = next(
        r for r in references if r["method"] == method and r["fitted"] == fitted
    )
    np.testing.assert_allclose(
        result.target.energy, reference["energy"], atol=3e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.target.forces, reference["forces"], atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(
        result.target_density, reference["density"], atol=1e-7, rtol=0
    )
    assert (
        result.diagnostics["total_seconds"]
        >= result.diagnostics["source_seconds"] + result.diagnostics["target_seconds"]
    )


def test_projected_density_cannot_be_exported_as_a_converged_target_checkpoint(
    tmp_path,
):
    with (
        Calculator().prepare_batch([ATOMS]) as source,
        Calculator(basis="def2-svp").prepare_batch([ATOMS]) as target,
    ):
        source.execute(strict=True)
        report = target.initialize_from(source)
        assert report["target_verification"] == "pending_execute"
        with pytest.raises(CheckpointError, match="target solve"):
            target.save_checkpoint(tmp_path / "proposal.vqcp")
        with pytest.raises(ProjectionRejected, match="already has a seed"):
            target.initialize_from(source)
        target.execute(strict=True)
        target.save_checkpoint(tmp_path / "converged.vqcp")


def test_geometry_order_charge_and_failed_source_cannot_silently_seed_target():
    with Calculator().prepare_batch([ATOMS]) as source:
        source.execute(strict=True)
        moved = [(z, (r[0] + 0.01, r[1], r[2])) for z, r in ATOMS]
        with Calculator().prepare_batch([moved]) as target:
            with pytest.raises(ProjectionRejected, match="geometry"):
                target.initialize_from(source)
            reports = target.initialize_from(source, strict=False)
            assert not reports["items"][0]["accepted"]
            assert target.execute(strict=True).items[0].restart_origin == "cold"
        with (
            Calculator().prepare_batch([ATOMS], charges=[-2]) as target,
            pytest.raises(ProjectionRejected, match="electrons"),
        ):
            target.initialize_from(source)


def test_failed_source_and_strict_projection_loss_fall_back_completely():
    source = Calculator(basis="def2-svp", max_iterations=1)
    target = Calculator()
    result = projected_singlepoint(target, source, ATOMS)
    assert not result.source.succeeded
    assert result.target.converged and result.target.restart_origin == "cold"
    assert "successful" in result.diagnostics["items"][0]["reason"]
    result = projected_singlepoint(
        Calculator(basis="def2-svp"),
        Calculator(),
        ATOMS,
        policy=ProjectionPolicy(maximum_residual=1e-8),
    )
    assert result.target.restart_origin == "cold"
    assert "projection residual" in result.diagnostics["items"][0]["reason"]


def test_fleet_projection_budget_rejection_precedes_seed_install():
    with (
        Calculator().prepare_batch([ATOMS]) as source,
        Calculator(basis="def2-svp").prepare_batch([ATOMS]) as target,
    ):
        source.execute(strict=True)
        with pytest.raises(MemoryError, match="maximum_host_bytes"):
            target.initialize_from(source, maximum_host_bytes=1)
        assert target.execute(strict=True).items[0].restart_origin == "cold"


def test_failed_fleet_neighbor_does_not_discard_a_valid_projection():
    other = [("He", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    with (
        Calculator(max_iterations=3).prepare_batch(
            [ATOMS, other], charges=[0, 1]
        ) as source,
        Calculator(basis="def2-svp").prepare_batch(
            [ATOMS, other], charges=[0, 1]
        ) as target,
    ):
        initial = source.execute()
        assert initial.items[0].succeeded and not initial.items[1].succeeded
        report = target.initialize_from(source, strict=False)
        assert [item["accepted"] for item in report["items"]] == [True, False]
        result = target.execute(strict=True)
        assert [item.restart_origin for item in result.items] == [
            "basis_projection",
            "cold",
        ]


@pytest.mark.parametrize(
    "source_basis,target_basis", [("sto-3g", "sto-3g"), ("def2-svp", "sto-3g")]
)
def test_same_and_reverse_basis_transfers_still_rebuild_target_equations(
    source_basis, target_basis
):
    source, target = Calculator(basis=source_basis), Calculator(basis=target_basis)
    result = projected_singlepoint(target, source, ATOMS)
    cold = target.singlepoint(ATOMS)
    assert result.target.restart_origin == "basis_projection"
    assert result.target.energy == pytest.approx(cold.energy, abs=3e-10)
    np.testing.assert_allclose(result.target.forces, cold.forces, atol=3e-9, rtol=0)
