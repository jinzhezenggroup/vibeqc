"""Method-level KS capacity gates include preparation and retained replay state."""

import ctypes
import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator, ResourceBudget, estimate_ks_resources

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
HE = [(2, (0.0, 0.0, 0.0))]
H = [(1, (0.0, 0.0, 0.0))]
WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]
CUDA = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU job",
)


def test_dry_run_uses_only_metadata_and_tracks_all_retained_items(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("KS dry run attempted native execution or a numerical allocation")

    monkeypatch.setattr(_native, "load_library", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    one = estimate_ks_resources([H2])
    many = estimate_ks_resources([H2] * 4)
    assert many.resident_bytes["host"] == 4 * one.resident_bytes["host"]
    assert (
        many.peak_bytes["host"] - many.resident_bytes["host"]
        == one.peak_bytes["host"] - one.resident_bytes["host"]
    )
    large = [(1, (0.0, 0.0, 2.0 * i)) for i in range(500)]
    plan = estimate_ks_resources([large], budget=ResourceBudget(host_bytes=1024))
    assert plan.status == "infeasible"
    assert plan.peak_bytes["host"] > 10**12


def test_topology_controls_and_spin_bind_the_resource_identity() -> None:
    base = estimate_ks_resources([H2])
    moved = [(1, (0.0, 0.0, -0.8)), (1, (0.0, 0.0, 0.8))]
    assert estimate_ks_resources([moved]).identity == base.identity
    for options in (
        {"basis": "def2-svp"},
        {"method": "lda-rks"},
        {"method": "pbe-uks", "charges": [1], "multiplicities": [2]},
        {"max_iterations": 200},
        {"diis_history": 16},
        {"energy_tolerance": 1e-12},
    ):
        assert estimate_ks_resources([H2], **options).identity != base.identity


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_cpu_budget_covers_prepare_replay_rebuild_and_singlepoint(
    method: typing.Any,
) -> None:
    uks = method.endswith("uks")
    systems = [H2, H if uks else HE]
    charges, multiplicities = ([1, 0], [2, 2]) if uks else ([0, 0], [1, 1])
    reference = Calculator(method=method)
    probe = reference.estimate_resources(
        systems, charges=charges, multiplicities=multiplicities
    ).require_feasible()
    calculator = Calculator(
        method=method,
        resource_budget=ResourceBudget(host_bytes=probe.peak_bytes["host"]),
    )
    with calculator.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities
    ) as batch:
        preparation = batch.resource_diagnostics["preparation"]
        assert preparation["status"] == "observed"
        assert (
            0
            < preparation["sampled_item_peak_host_bytes"]
            <= probe.resident_bytes["host"]
        )
        cold = batch.execute(strict=True)
        replay = batch.execute(strict=True)
        assert all(item.warm_start_used for item in replay.items)
        assert replay.energies == pytest.approx(cold.energies, abs=1e-9)
        assert batch.resource_diagnostics["preparation"] == preparation
        moved = batch.execute([[(0.0, 0.0, -0.8), (0.0, 0.0, 0.8)], None], strict=True)
        assert moved.items[0].succeeded
        observed = batch.resource_diagnostics["observation"]
        assert observed["samples"] > 0
        assert 0 < observed["sampled_item_peak_host_bytes"] <= probe.peak_bytes["host"]
        assert not observed["complete_plan_peak"]
    actual = calculator.singlepoint(
        H2, charge=charges[0], multiplicity=multiplicities[0]
    )
    expected = reference.singlepoint(
        H2, charge=charges[0], multiplicity=multiplicities[0]
    )
    assert actual.energy == expected.energy
    assert actual.resource_diagnostics["preparation"]["status"] == "observed"


@pytest.mark.parametrize("batch", (False, True))
def test_one_byte_short_rejects_before_context_or_preparation(
    monkeypatch: typing.Any, batch: typing.Any
) -> None:
    probe = estimate_ks_resources([H2])
    calculator = Calculator(
        method="pbe-rks",
        resource_budget=ResourceBudget(host_bytes=probe.peak_bytes["host"] - 1),
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("an infeasible KS plan attempted native preparation")

    for name in (
        "vibeqc_context_create",
        "vibeqc_calculation_prepare",
        "vibeqc_batch_prepare",
    ):
        monkeypatch.setattr(calculator._library, name, forbidden)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        calculator.prepare_batch([H2]) if batch else calculator.singlepoint(H2)


def test_failed_preparation_keeps_evidence_and_failed_scf_keeps_samples(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native
    from vibeqc_compiler.common.resources import ResourceAllocationError

    calculator = Calculator(method="pbe-uks", resource_budget=ResourceBudget())
    # CPU allocations have no limiter. Inject the native status at the prepare
    # boundary to verify that an OOM retains its phase and host attribution.
    with monkeypatch.context() as patch:
        patch.setattr(
            calculator._library,
            "vibeqc_batch_prepare",
            lambda *args: _native.STATUS_OUT_OF_MEMORY,
        )
        with pytest.raises(ResourceAllocationError) as failed:
            calculator.prepare_batch([H2])
    assert failed.value.space == "host"
    assert failed.value.resource_diagnostics["phase"] == "preparation"
    assert failed.value.resource_diagnostics["owner"] == "ks"
    limited = Calculator(
        method="pbe-uks", max_iterations=1, resource_budget=ResourceBudget()
    )
    with pytest.raises(RuntimeError) as exhausted:
        limited.singlepoint(H2, charge=1, multiplicity=2)
    diagnostics = exhausted.value.resource_diagnostics
    assert diagnostics["phase"] == "observation"
    assert diagnostics["preparation"]["status"] == "observed"
    assert diagnostics["observation"]["sampled_item_peak_host_bytes"] > 0


def test_cli_ks_dry_run_does_not_load_a_native_runtime(
    monkeypatch: typing.Any, tmp_path: typing.Any, capsys: typing.Any
) -> None:
    import json

    from vibeqc import _native
    from vibeqc.__main__ import main

    path = tmp_path / "h2.xyz"
    path.write_text("2\nbohr\nH 0 0 -0.7\nH 0 0 0.7\n")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("KS CLI dry run loaded a native runtime")

    monkeypatch.setattr(_native, "load_library", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vibeqc",
            "resources",
            str(path),
            "--method",
            "pbe-uks",
            "--charge",
            "1",
            "--multiplicity",
            "2",
            "--units",
            "bohr",
            "--host-bytes",
            "1",
        ],
    )
    assert main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "infeasible"
    assert output["requests"][0]["name"] == "ks"


@pytest.mark.parametrize(
    ("has_semantic_abi", "diagnostic"),
    ((False, "semantic KS execution-plan ABI"), (True, "allocation inventory")),
)
def test_missing_inventory_and_foreign_plan_reject_before_preparation(
    monkeypatch: typing.Any, has_semantic_abi: bool, diagnostic: str
) -> None:
    from types import SimpleNamespace

    calculator = Calculator(method="pbe-rks", resource_budget=ResourceBudget())
    foreign = estimate_ks_resources([H2], method="lda-rks")
    with pytest.raises(ValueError, match="KS inputs differ"):
        calculator.prepare_batch([H2], resource_plan=foreign)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsupported KS library attempted native preparation")

    library = SimpleNamespace(
        vibeqc_context_create=forbidden,
        vibeqc_calculation_prepare=forbidden,
        vibeqc_batch_prepare=forbidden,
    )
    if has_semantic_abi:
        library.vibeqc_ks_options_version = lambda: 1
    monkeypatch.setattr(calculator, "_library", library)
    assert calculator.estimate_resources([H2]).status == "unsupported"
    with pytest.raises(NotImplementedError, match=diagnostic):
        calculator.prepare_batch([H2])


@CUDA
@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_cuda_ledger_owns_prepare_replay_rebuild_and_release(
    method: typing.Any,
) -> None:
    uks = method.endswith("uks")
    systems = [H2, H if uks else HE, H2]
    charges, multiplicities = ([1, 0, 1], [2, 2, 2]) if uks else ([0] * 3, [1] * 3)
    reference = Calculator(method=method, device="cuda")
    probe = reference.estimate_resources(
        systems, charges=charges, multiplicities=multiplicities
    ).require_feasible()
    calculator = Calculator(
        method=method,
        device="cuda",
        resource_budget=ResourceBudget(
            host_bytes=probe.peak_bytes["host"], device_bytes=probe.peak_bytes["device"]
        ),
    )
    ledger = None
    try:
        with calculator.prepare_batch(
            systems, charges=charges, multiplicities=multiplicities
        ) as batch:
            ledger = batch._resource_ledger
            prep = batch.resource_diagnostics["preparation"]["device_ledger"]
            assert prep["allocations"] > 0 and prep["rejected_allocations"] == 0
            assert prep["live_bytes"] == probe.resident_bytes["device"]
            for replay in range(2):
                result = batch.execute(strict=True)
                observed = batch.resource_diagnostics["observation"]["device_ledger"]
                assert observed["allocations"] == 0
                assert observed["live_bytes"] == prep["live_bytes"]
                assert all(item.executed_backend == "cuda" for item in result.items)
                assert all(
                    item.warm_start_used == bool(replay) for item in result.items
                )
            batch.execute([[(0, 0, -0.8), (0, 0, 0.8)], None, None], strict=True)
            rebuilt = batch.resource_diagnostics["observation"]["device_ledger"]
            assert rebuilt["allocations"] > 0
            assert rebuilt["rejected_allocations"] == 0
            assert rebuilt["live_bytes"] == prep["live_bytes"]
            assert rebuilt["peak_bytes"] <= rebuilt["limit_bytes"]
            # Keep the metadata handle until the actual native owner is closed.
            batch._resource_ledger = None
        assert ledger.to_dict()["live_bytes"] == 0
    finally:
        if ledger is not None:
            ledger.close()


@CUDA
def test_cuda_shape_queries_need_no_execution_context_and_cover_large_solver(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _native, profiles

    library = _native.load_library(device="cpu")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("KS shape query attempted scientific execution")

    with monkeypatch.context() as patch:
        patch.setattr(profiles, "select_library", forbidden)
        patch.setattr(library, "vibeqc_context_create", forbidden)
        patch.setattr(np, "empty", forbidden)
        probe = estimate_ks_resources(
            [WATER], basis="def2-svp", backend="cuda", library=library
        ).require_feasible()
    calculator = Calculator(
        method="pbe-rks",
        basis="def2-svp",
        device="cuda",
        resource_budget=ResourceBudget(
            host_bytes=probe.peak_bytes["host"], device_bytes=probe.peak_bytes["device"]
        ),
    )
    result = calculator.singlepoint(WATER)
    assert result.converged and result.executed_backend == "cuda"
    diagnostics = result.resource_diagnostics
    assert (
        diagnostics["preparation"]["device_ledger"]["live_bytes"]
        == probe.resident_bytes["device"]
    )
    assert diagnostics["observation"]["device_ledger"]["allocations"] == 0


@CUDA
@pytest.mark.parametrize("partial", (False, True))
def test_cuda_failed_preparation_retains_ledger_rejection_and_releases_buffers(
    monkeypatch: typing.Any,
    partial: typing.Any,
) -> None:
    from vibeqc_compiler.common.resources import ResourceAllocationError

    calculator = Calculator(
        method="pbe-rks", device="cuda", resource_budget=ResourceBudget()
    )
    create = calculator._library.vibeqc_resource_ledger_create_v1
    create.argtypes = [ctypes.c_size_t, ctypes.c_int]
    create.restype = ctypes.c_void_p
    # The second case allows one complete owner before rejecting its neighbor.
    # Constructor unwinding must release both its state and any partial upload.
    capacity = (
        calculator.estimate_resources([H2]).resident_bytes["device"] if partial else 1
    )
    # Fault the assigned capacity only, retaining the real native allocator and
    # exception path so this exercises preparation cleanup rather than a mock.
    monkeypatch.setattr(
        calculator._library,
        "vibeqc_resource_ledger_create_v1",
        lambda requested, device: create(capacity, device),
    )
    with pytest.raises(ResourceAllocationError) as failed:
        calculator.prepare_batch([H2, H2])
    assert failed.value.space == "device:0"
    ledger = failed.value.resource_diagnostics["preparation"]["device_ledger"]
    assert ledger["rejected_allocations"] > 0
    assert ledger["live_bytes"] == 0
    assert bool(ledger["allocations"]) == partial
