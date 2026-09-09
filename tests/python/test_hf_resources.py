"""HF resource planning resolves actual inputs before allocating solve tensors."""

import json

import numpy as np
import pytest
from vibeqc import Calculator, ResourceBudget, estimate_hf_resources

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


def test_large_infeasible_dry_run_never_initializes_native_or_allocates_tensors(
    monkeypatch,
):
    from vibeqc import _native

    def forbidden(*args, **kwargs):
        pytest.fail(
            "resource dry-run attempted native execution or a numerical allocation"
        )

    monkeypatch.setattr(_native, "load_library", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    atoms = [(1, (0.0, 0.0, 2.0 * i)) for i in range(500)]
    plan = estimate_hf_resources([atoms], budget=ResourceBudget(host_bytes=1024))
    assert plan.status == "infeasible"
    assert plan.peak_bytes["host"] > 10**12
    assert "largest serialized item" in plan.diagnostic


def test_cpu_df_inventory_includes_existing_four_center_preparation():
    direct = estimate_hf_resources([H2])
    fitted = estimate_hf_resources([H2], density_fitting="cpu")
    assert fitted.peak_bytes["host"] >= direct.peak_bytes["host"]
    inventory = json.loads(
        dict(fitted.requests[0].candidates[0].decisions)["item_phase_inventory"]
    )
    assert inventory[0]["integral_state"] > 8 * (2**2 * 2 + 2**2)


def test_ragged_fleet_sums_retained_state_but_shares_serial_workspace():
    one = estimate_hf_resources([H2])
    many = estimate_hf_resources([H2] * 4)
    assert many.resident_bytes["host"] == 4 * one.resident_bytes["host"]
    assert (
        many.peak_bytes["host"] - many.resident_bytes["host"]
        == one.peak_bytes["host"] - one.resident_bytes["host"]
    )


def test_geometry_reuses_topology_while_basis_spin_and_controls_invalidate():
    base = estimate_hf_resources([H2])
    moved = [(1, (0.0, 0.0, -0.8)), (1, (0.0, 0.0, 0.8))]
    assert estimate_hf_resources([moved]).identity == base.identity
    assert estimate_hf_resources([H2], basis="def2-svp").identity != base.identity
    assert (
        estimate_hf_resources([H2], method="uhf", multiplicities=[3]).identity
        != base.identity
    )
    assert estimate_hf_resources([H2], diis_history=16).identity != base.identity


@pytest.mark.parametrize("fitted", [False, True])
def test_prepared_cpu_budget_gates_before_native_context_and_preserves_results(fitted):
    controls = {"density_fitting": "cpu" if fitted else "none"}
    baseline = Calculator(**controls).singlepoint(H2)
    probe = estimate_hf_resources([H2], **controls)
    calculator = Calculator(
        **controls, resource_budget=ResourceBudget(host_bytes=probe.peak_bytes["host"])
    )
    with calculator.prepare_batch([H2]) as batch:
        assert batch.resource_plan.status == "feasible"
        first = batch.execute(strict=True).items[0]
        observed = batch.resource_diagnostics["observation"]
        assert observed["status"] == "observed"
        assert observed["cpu_worker_limit"] == 1
        assert (
            0
            < observed["sampled_item_peak_host_bytes"]
            <= batch.resource_plan.peak_bytes["host"]
        )
        assert not observed["complete_plan_peak"]
        same_plan = batch.resource_plan.identity
        second = batch.execute(strict=True).items[0]
        assert batch.resource_plan.identity == same_plan
        assert first.energy == baseline.energy
        np.testing.assert_array_equal(first.forces, baseline.forces)
        assert second.warm_start_used
        assert abs(second.energy - baseline.energy) < 1e-10
    constrained = Calculator(
        **controls,
        resource_budget=ResourceBudget(host_bytes=probe.peak_bytes["host"] - 1),
    )
    with pytest.raises(MemoryError, match="no supported plan fits"):
        constrained.prepare_batch([H2])
    with pytest.raises(MemoryError, match="no supported plan fits"):
        constrained.singlepoint(H2)


def test_bounded_fleet_samples_every_serial_item_and_restores_thread_scope():
    from vibeqc.resources import CpuResourceObservation

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=10**6))
    with calculator.prepare_batch([H2] * 4) as batch:
        result = batch.execute(strict=True)
        observed = batch.resource_diagnostics["observation"]
        assert observed["samples"] == sum(item.iterations for item in result.items)
    with CpuResourceObservation(calculator._library) as observation:
        assert observation.cpu_workers == 0
    assert observation.samples is None


def test_failed_solve_retains_available_resource_samples():
    calculator = Calculator(
        max_iterations=1, resource_budget=ResourceBudget(host_bytes=10**6)
    )
    with calculator.prepare_batch([H2]) as batch:
        result = batch.execute()
        assert not result.items[0].succeeded
        observed = batch.resource_diagnostics["observation"]
        assert observed["samples"] == 1
        assert observed["sampled_item_peak_host_bytes"] > 0
    with pytest.raises(RuntimeError) as failure:
        calculator.singlepoint(H2)
    assert failure.value.resource_diagnostics["observation"]["samples"] == 1


def test_unverified_native_resource_contract_fails_before_preparation(monkeypatch):
    from types import SimpleNamespace

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=10**6))
    monkeypatch.setattr(calculator, "_library", SimpleNamespace())
    assert calculator.estimate_resources([H2]).status == "unsupported"
    with pytest.raises(NotImplementedError, match="allocation inventory"):
        calculator.prepare_batch([H2])


def test_explicit_global_plan_must_match_prepared_scientific_inputs():
    calculator = Calculator()
    foreign = estimate_hf_resources([H2], basis="def2-svp")
    with pytest.raises(ValueError, match="HF inputs differ"):
        calculator.prepare_batch([H2], resource_plan=foreign)


def test_unavailable_cuda_hf_query_is_diagnosed_without_cpu_fallback():
    from types import SimpleNamespace

    plan = estimate_hf_resources(
        [H2],
        backend="cuda",
        library=SimpleNamespace(),
        budget=ResourceBudget(device_bytes=10**9),
    )
    assert plan.status == "unsupported"
    assert plan.requests[0].identity.backend == "cuda"


def test_cli_resource_dry_run_reads_real_xyz_and_reports_infeasibility(
    monkeypatch, tmp_path, capsys
):
    from vibeqc import _native
    from vibeqc.__main__ import main

    path = tmp_path / "h2.xyz"
    path.write_text("2\nbohr fixture\nH 0 0 -0.7\nH 0 0 0.7\n")

    def forbidden(*args, **kwargs):
        pytest.fail("CLI resource estimation touched a native runtime")

    monkeypatch.setattr(_native, "load_library", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        ["vibeqc", "resources", str(path), "--units", "bohr", "--host-bytes", "1"],
    )
    assert main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "infeasible"
    assert output["peak_bytes"]["host"] > 1
