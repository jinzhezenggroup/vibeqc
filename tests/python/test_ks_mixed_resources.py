"""Mixed-J planning must budget both nonlinear stages without using a GPU."""

import json
import typing
from types import SimpleNamespace

import pytest
from vibeqc import Calculator, _native, resources_ks

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


def _inventory_library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    library = SimpleNamespace(vibeqc_ks_resource_inventory_version_v1=lambda: 1)
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


@pytest.mark.parametrize("precision", ["fp32", "mixed", False, None])
def test_unknown_ks_precision_rejected(precision: typing.Any) -> None:
    with pytest.raises(ValueError, match="precision"):
        resources_ks.ks_resource_request([H2], precision=precision)
