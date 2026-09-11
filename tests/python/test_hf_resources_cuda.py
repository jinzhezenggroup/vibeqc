"""Direct-HF budget evidence; run only in a scheduler-assigned GPU job."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, ResourceBudget, estimate_hf_resources

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_small_direct_cuda_global_budget_covers_all_ragged_caches(method):
    reference = Calculator(device="cuda", method=method)
    systems = [H2, WATER, H2]
    probe = reference.estimate_resources(systems).require_feasible()
    budget = ResourceBudget(
        device_bytes=probe.peak_bytes["device"], host_bytes=probe.peak_bytes["host"]
    )
    calculator = Calculator(device="cuda", method=method, resource_budget=budget)
    # Match cold/warm history to the ordinary prepared endpoint. Comparing a
    # warm replay with a cold singlepoint also compares different densities
    # within the solver's stopping tolerance, obscuring a resource regression.
    with (
        calculator.prepare_batch(systems) as batch,
        reference.prepare_batch(systems) as ordinary,
    ):
        for replay in range(2):
            expected = ordinary.execute(strict=True).items
            result = batch.execute(strict=True)
            observed = batch.resource_diagnostics["observation"]
            assert (
                0
                < observed["sampled_cuda_arena_peak_bytes"]
                <= probe.resident_bytes["device"]
            )
            assert observed["cuda_arena_samples"] >= 2
            ledger = observed["device_ledger"]
            assert (
                0
                < ledger["live_bytes"]
                <= ledger["peak_bytes"]
                <= ledger["limit_bytes"]
            )
            assert ledger["rejected_allocations"] == 0
            assert (ledger["allocations"] == 0) == bool(replay)
            for actual, target in zip(result.items, expected, strict=True):
                assert actual.energy == pytest.approx(target.energy, abs=1e-10)
                np.testing.assert_allclose(
                    actual.forces, target.forces, atol=1e-9, rtol=1e-8
                )
        identity = batch.resource_plan.identity
        moved = [
            [(z, (x, y, zz + 0.01)) for z, (x, y, zz) in atoms] for atoms in systems
        ]
        coordinates = [[position for _, position in atoms] for atoms in moved]
        batch.execute(coordinates=coordinates, strict=True)
        assert batch.resource_plan.identity == identity
    constrained = Calculator(
        device="cuda",
        method=method,
        resource_budget=ResourceBudget(device_bytes=budget.device_bytes - 1),
    )
    with pytest.raises(MemoryError, match="no supported plan fits"):
        constrained.prepare_batch(systems)


def test_cuda_dry_run_query_never_calls_profile_or_context(monkeypatch):
    from vibeqc import _native, profiles

    library = _native.load_library(device="cpu")

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run resource query initialized a CUDA execution context")

    monkeypatch.setattr(profiles, "select_library", forbidden)
    monkeypatch.setattr(library, "vibeqc_context_create", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    plan = estimate_hf_resources([WATER], backend="cuda", library=library)
    assert plan.status == "feasible"
    assert plan.peak_bytes["device"] > 0
    oversized = estimate_hf_resources(
        [WATER], backend="cuda", basis="def2-svp", library=library
    )
    assert oversized.status == "unsupported"
    assert "<=16" in oversized.diagnostic


@pytest.mark.parametrize(
    "variable,value",
    [
        ("VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE", "graph_native"),
        ("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE", "profile"),
        ("VIBEQC_FINAL_FOCK_REBUILD", "1"),
    ],
)
def test_cuda_execution_rejects_changed_resource_schedule(monkeypatch, variable, value):
    calculator = Calculator(
        device="cuda", resource_budget=ResourceBudget(device_bytes=1 << 20)
    )
    with calculator.prepare_batch([H2]) as batch:
        monkeypatch.setenv(variable, value)
        with pytest.raises(ValueError, match="schedule changed"):
            batch.execute()


@pytest.mark.parametrize("fitted", [False, True])
def test_native_ledger_rejects_unplanned_arena_and_releases_failed_state(fitted):
    """Fault the assigned capacity to exercise the native allocation boundary."""
    calculator = Calculator(
        device="cuda",
        density_fitting="cuda" if fitted else "none",
        resource_budget=ResourceBudget(),
    )
    with calculator.prepare_batch([H2]) as batch:
        ledger = batch._resource_ledger
        ledger.close()
        ledger.handle = calculator._library.vibeqc_resource_ledger_create_v1(1, 0)
        failed = batch.execute()
        assert not failed.items[0].succeeded
        observation = batch.resource_diagnostics["observation"]["device_ledger"]
        assert observation["rejected_allocations"] >= 1
        assert observation["live_bytes"] == 0
        ledger.close()
        ledger.handle = calculator._library.vibeqc_resource_ledger_create_v1(
            ledger.limit, 0
        )
        result = batch.execute(strict=True)
        assert result.items[0].executed_backend == "cuda"
        # Keep the metadata handle alive while closing the native cache, so
        # the assertion observes actual charge release after destruction.
        batch._resource_ledger = None
    try:
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        ledger.close()


@pytest.mark.parametrize("mode", ["resident", "recomputed"])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_cuda_df_global_candidates_bind_execution_and_respect_host_device_caps(
    mode, method
):
    from vibeqc.resources import ResourcePlan, plan_resources

    systems = [H2, WATER, H2]
    options = {
        "method": method,
        "density_fitting": "cuda",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    reference = Calculator(device="cuda", **options)
    request = reference._resource_request(systems)
    request_plan = plan_resources([request], ResourceBudget()).require_feasible()
    candidate = next(c for c in request.candidates if c.mode == mode)
    selected = ResourcePlan(
        ResourceBudget(), (request,), (("hf", candidate.name),), "feasible"
    )
    budget = ResourceBudget(
        host_bytes=selected.peak_bytes["host"],
        device_bytes=selected.peak_bytes["device"],
    )
    plan = plan_resources([request], budget).require_feasible()
    assert dict(plan.selections)["hf"] == candidate.name
    if mode == "recomputed":
        assert selected.peak_bytes["host"] < request_plan.peak_bytes["host"]
        assert "CPU DIIS/eigensolvers" in dict(candidate.decisions)["scf_driver"]
    native_budget = int(
        dict(candidate.decisions)["density_fitting_memory_budget_bytes"]
    )
    ordinary = Calculator(
        device="cuda", density_fitting_memory_budget_bytes=native_budget, **options
    )
    calculator = Calculator(device="cuda", resource_budget=budget, **options)
    cpu_options = {**options, "density_fitting": "cpu"}
    cpu = Calculator(**cpu_options)
    expected_cpu = [cpu.singlepoint(atoms) for atoms in systems]
    with (
        calculator.prepare_batch(systems) as batch,
        ordinary.prepare_batch(systems) as baseline,
    ):
        for _ in range(2):
            result, expected = batch.execute(strict=True), baseline.execute(strict=True)
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert (
                0
                < ledger["live_bytes"]
                <= ledger["peak_bytes"]
                <= ledger["limit_bytes"]
            )
            assert ledger["rejected_allocations"] == 0
            for actual, native, oracle in zip(
                result.items, expected.items, expected_cpu, strict=True
            ):
                assert actual.energy == pytest.approx(native.energy, abs=1e-10)
                assert actual.energy == pytest.approx(oracle.energy, abs=1e-9)
                np.testing.assert_allclose(
                    actual.forces, native.forces, atol=1e-9, rtol=1e-8
                )
                np.testing.assert_allclose(
                    actual.forces, oracle.forces, atol=2e-8, rtol=1e-7
                )
            assert all(
                d.streamed == (mode == "recomputed")
                for d in batch.last_density_fitting_metric_diagnostics()
            )
    infeasible = plan_resources(
        [request],
        ResourceBudget(
            host_bytes=min(
                ResourcePlan(
                    ResourceBudget(), (request,), (("hf", c.name),), "feasible"
                ).peak_bytes["host"]
                for c in request.candidates
            )
            - 1
        ),
    )
    assert infeasible.status == "infeasible"


@pytest.mark.parametrize("mode", ["resident", "recomputed"])
def test_cuda_df_distinct_auxiliary_basis_and_open_shell_inventory(mode):
    """Orbital dimensions cannot substitute for auxiliary or spin dimensions."""
    from vibeqc.resources import ResourcePlan

    options = {
        "method": "uhf",
        "basis": "sto-3g",
        "auxiliary_basis": "def2-svp",
        "density_fitting": "cuda",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    reference = Calculator(device="cuda", **options)
    request = reference._resource_request([H2], charges=[1], multiplicities=[2])
    candidate = next(c for c in request.candidates if c.mode == mode)
    selected = ResourcePlan(
        ResourceBudget(), (request,), (("hf", candidate.name),), "feasible"
    )
    calculator = Calculator(
        device="cuda",
        resource_budget=ResourceBudget(
            host_bytes=selected.peak_bytes["host"],
            device_bytes=selected.peak_bytes["device"],
        ),
        **options,
    )
    cpu = Calculator(**{**options, "density_fitting": "cpu"})
    expected = cpu.singlepoint(H2, charge=1, multiplicity=2)
    with calculator.prepare_batch([H2], charges=[1], multiplicities=[2]) as batch:
        assert dict(batch.resource_plan.selections)["hf"] == candidate.name
        for _ in range(2):
            actual = batch.execute(strict=True).items[0]
            assert actual.energy == pytest.approx(expected.energy, abs=1e-9)
            np.testing.assert_allclose(
                actual.forces, expected.forces, atol=2e-8, rtol=1e-7
            )
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]
            assert ledger["rejected_allocations"] == 0
