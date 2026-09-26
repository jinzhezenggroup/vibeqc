"""HF resource planning resolves actual inputs before allocating solve tensors."""

import json
import typing

import numpy as np
import pytest
from vibeqc import Calculator, ResourceBudget, estimate_hf_resources

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


@pytest.mark.parametrize("allocation", [False, True])
def test_singlepoint_native_detail_preserves_structured_resource_error(
    monkeypatch: typing.Any, allocation: typing.Any
) -> None:
    """Adding context detail must retain the exception's retry/evidence payload."""
    from vibeqc import resources_native
    from vibeqc_compiler.common.resources import ResourceAllocationError

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=1 << 30))
    failure = (
        ResourceAllocationError("host", "allocation failed")
        if allocation
        else RuntimeError("numerical failure")
    )
    evidence = {"sentinel": "resource evidence"}
    failure.resource_diagnostics = evidence

    def reject(*args: typing.Any) -> typing.Any:
        raise failure

    monkeypatch.setattr(resources_native, "check_resource_status", reject)
    monkeypatch.setattr(
        calculator._library,
        "vibeqc_context_get_last_detail",
        lambda context: b"native diagnostic",
    )
    with pytest.raises(type(failure), match="native diagnostic") as caught:
        calculator.singlepoint(H2)
    assert caught.value is failure
    assert caught.value.resource_diagnostics is evidence
    if allocation:
        assert caught.value.space == "host"


def test_large_infeasible_dry_run_never_initializes_native_or_allocates_tensors(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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


def test_cpu_df_inventory_includes_existing_four_center_preparation() -> None:
    direct = estimate_hf_resources([H2])
    fitted = estimate_hf_resources([H2], density_fitting="cpu")
    assert fitted.peak_bytes["host"] >= direct.peak_bytes["host"]
    inventory = json.loads(
        dict(fitted.requests[0].candidates[0].decisions)["item_phase_inventory"]
    )
    assert inventory[0]["integral_state"] > 8 * (2**2 * 2 + 2**2)


def test_ragged_fleet_sums_retained_state_but_shares_serial_workspace() -> None:
    one = estimate_hf_resources([H2])
    many = estimate_hf_resources([H2] * 4)
    assert many.resident_bytes["host"] == 4 * one.resident_bytes["host"]
    assert (
        many.peak_bytes["host"] - many.resident_bytes["host"]
        == one.peak_bytes["host"] - one.resident_bytes["host"]
    )


def test_geometry_reuses_topology_while_basis_spin_and_controls_invalidate() -> None:
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
def test_prepared_cpu_budget_gates_before_native_context_and_preserves_results(
    fitted: typing.Any,
) -> None:
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


def test_bounded_fleet_samples_every_serial_item_and_restores_thread_scope() -> None:
    from vibeqc_compiler.common.resources import CpuResourceObservation

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=10**6))
    with calculator.prepare_batch([H2] * 4) as batch:
        result = batch.execute(strict=True)
        observed = batch.resource_diagnostics["observation"]
        assert observed["samples"] == sum(item.iterations for item in result.items)
    with CpuResourceObservation(calculator._library) as observation:
        assert observation.cpu_workers == 0
    assert observation.samples is None


def test_failed_solve_retains_available_resource_samples() -> None:
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


def test_unverified_native_resource_contract_fails_before_preparation(
    monkeypatch: typing.Any,
) -> None:
    from types import SimpleNamespace

    calculator = Calculator(resource_budget=ResourceBudget(host_bytes=10**6))
    monkeypatch.setattr(calculator, "_library", SimpleNamespace())
    assert calculator.estimate_resources([H2]).status == "unsupported"
    with pytest.raises(NotImplementedError, match="allocation inventory"):
        calculator.prepare_batch([H2])


def test_explicit_global_plan_must_match_prepared_scientific_inputs() -> None:
    calculator = Calculator()
    foreign = estimate_hf_resources([H2], basis="def2-svp")
    with pytest.raises(ValueError, match="HF inputs differ"):
        calculator.prepare_batch([H2], resource_plan=foreign)


def test_unavailable_cuda_hf_query_is_diagnosed_without_cpu_fallback() -> None:
    from types import SimpleNamespace

    plan = estimate_hf_resources(
        [H2],
        backend="cuda",
        library=SimpleNamespace(),
        budget=ResourceBudget(device_bytes=10**9),
    )
    assert plan.status == "unsupported"
    assert plan.requests[0].identity.backend == "cuda"


def test_cuda_hf_inventory_marshals_exact_shell_topology_to_v2() -> None:
    from types import SimpleNamespace

    seen: dict[str, typing.Any] = {}

    def query(
        nbf: int,
        direct_nbf: int,
        atoms: int,
        shells: int,
        angular: typing.Any,
        primitive_counts: typing.Any,
        diis_history: int,
        spins: int,
        precision: int,
        energy_tolerance: float,
        screening_tolerance: float,
        output: typing.Any,
        count: int,
    ) -> int:
        seen.update(
            nbf=nbf,
            direct_nbf=direct_nbf,
            atoms=atoms,
            shells=shells,
            angular=tuple(angular[:shells]),
            primitive_counts=tuple(primitive_counts[:shells]),
            diis_history=diis_history,
            spins=spins,
            precision=precision,
            energy_tolerance=energy_tolerance,
            screening_tolerance=screening_tolerance,
            count=count,
        )
        output[0] = 4096
        output[1] = 256
        return 0

    library = SimpleNamespace(vibeqc_resource_small_hf_cuda_v2=query)
    plan = estimate_hf_resources([H2], backend="cuda", library=library)
    assert plan.status == "feasible"
    assert seen == {
        "nbf": 2,
        "direct_nbf": 2,
        "atoms": 2,
        "shells": 2,
        "angular": (0, 0),
        "primitive_counts": (3, 3),
        "diis_history": 8,
        "spins": 1,
        "precision": 0,
        "energy_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "count": 2,
    }


def test_cuda_hf_inventory_refuses_aggregate_v1_fallback() -> None:
    from types import SimpleNamespace

    legacy = SimpleNamespace(vibeqc_resource_small_hf_cuda_v1=lambda *args: 0)
    plan = estimate_hf_resources([H2], backend="cuda", library=legacy)
    assert plan.status == "unsupported"
    assert "topology-aware CUDA HF allocation inventory v2" in plan.diagnostic


def test_retired_df_math_control_does_not_change_cuda_resource_identity(
    monkeypatch: typing.Any,
) -> None:
    """A removed selector cannot invalidate a prepared CUDA schedule."""
    from types import SimpleNamespace

    variable = "VIBEQC_DF_SHELL_MATH_000"
    # Identity construction precedes the optional native inventory query; no
    # CUDA context or real device is needed to check environment sensitivity.
    library = SimpleNamespace()
    monkeypatch.delenv(variable, raising=False)
    baseline = estimate_hf_resources([H2], backend="cuda", library=library)
    monkeypatch.setenv(variable, "rys")
    changed = estimate_hf_resources([H2], backend="cuda", library=library)
    assert changed.requests[0].identity == baseline.requests[0].identity
    assert changed.identity == baseline.identity


def test_cli_resource_dry_run_reads_real_xyz_and_reports_infeasibility(
    monkeypatch: typing.Any, tmp_path: typing.Any, capsys: typing.Any
) -> None:
    from vibeqc import _native
    from vibeqc.__main__ import main

    path = tmp_path / "h2.xyz"
    path.write_text("2\nbohr fixture\nH 0 0 -0.7\nH 0 0 0.7\n")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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
