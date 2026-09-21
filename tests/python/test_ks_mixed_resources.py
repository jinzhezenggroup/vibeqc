"""Mixed-J planning must budget both nonlinear stages without using a GPU."""

import json
import typing
from types import SimpleNamespace

import pytest
from vibeqc import Calculator, KsOptions, _native, resources_ks

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


def _inventory_library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    library = SimpleNamespace(
        vibeqc_ks_resource_inventory_version_v1=lambda: 1,
        vibeqc_ks_options_version=lambda: 3,
    )
    monkeypatch.setattr(resources_ks, "_cuda_library_identity", lambda _: {})
    monkeypatch.setattr(
        resources_ks,
        "_cuda_item_inventory",
        lambda *args, **kwargs: {"state": 8, "xc": 8, "coulomb": 8, "setup": 24},
    )
    return library


@pytest.mark.parametrize("iterations", [0, 1, 37, 100])
def test_cuda_auto_history_and_resource_identity_include_refinement(
    monkeypatch: pytest.MonkeyPatch, iterations: int
) -> None:
    library = _inventory_library(monkeypatch)
    arguments = {"backend": "cuda", "library": library, "max_iterations": iterations}
    strict = resources_ks.ks_resource_request([H2], precision="fp64", **arguments)
    automatic = resources_ks.ks_resource_request([H2], precision="auto", **arguments)
    assert automatic.identity != strict.identity
    assert (
        strict.identity == resources_ks.ks_resource_request([H2], **arguments).identity
    )
    count = iterations or 100
    for request, expected in ((strict, count), (automatic, 2 * count + 4)):
        decisions = dict(request.candidates[0].decisions)
        inventory = json.loads(decisions["item_host_inventory"])
        assert inventory[0]["history"] == 256 * expected


def test_calculator_forwards_mixed_policy_to_ks_capacity_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator = Calculator(method="pbe-rks")
    calculator._device_name = "cuda"
    calculator._precision_mode = _native.PRECISION_AUTO
    captured = {}
    sentinel = object()

    def capture(systems: typing.Any, **kwargs: typing.Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(resources_ks, "ks_resource_request", capture)
    assert calculator._resource_request([H2]) is sentinel
    assert captured["precision"] == "auto"


def test_cpu_auto_capacity_is_not_advertised() -> None:
    with pytest.raises(NotImplementedError, match="CUDA"):
        resources_ks.ks_resource_request([H2], backend="cpu", precision="auto")


def test_cuda_host_unfused_resource_plan_moves_xc_out_of_device_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _inventory_library(monkeypatch)
    fused = resources_ks.ks_resource_request([H2], backend="cuda", library=library)
    unfused = resources_ks.ks_resource_request(
        [H2],
        backend="cuda",
        library=library,
        ks_options=KsOptions(xc_schedule="host_unfused"),
    )
    fused_decisions = dict(fused.candidates[0].decisions)
    unfused_decisions = dict(unfused.candidates[0].decisions)
    fused_device = json.loads(fused_decisions["item_device_inventory"])[0]
    unfused_device = json.loads(unfused_decisions["item_device_inventory"])[0]
    fused_host = json.loads(fused_decisions["item_host_inventory"])[0]
    unfused_host = json.loads(unfused_decisions["item_host_inventory"])[0]

    assert fused.identity != unfused.identity
    assert fused_device["xc"] == 8
    assert unfused_device["xc"] == 0
    assert fused_host["xc_schedule_staging"] == 0
    assert unfused_host["xc_schedule_staging"] > 0
    assert unfused_host["scf_workspace"] > fused_host["scf_workspace"]


@pytest.mark.parametrize("precision", ["fp32", "mixed", False, None])
def test_unknown_ks_precision_rejected(precision: typing.Any) -> None:
    with pytest.raises(ValueError, match="precision"):
        resources_ks.ks_resource_request([H2], precision=precision)
