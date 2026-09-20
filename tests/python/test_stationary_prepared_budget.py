"""Resident reuse must obey each call's admission caps before any device work."""

from types import SimpleNamespace

import pytest
from vibeqc import _stationary_cuda as runtime


@pytest.mark.parametrize(
    "failed,changed", [(False, False), (False, True), (True, False)]
)
def test_retained_host_budget_is_rechecked_before_rebind(
    monkeypatch: pytest.MonkeyPatch, failed: bool, changed: bool
) -> None:
    monkeypatch.setattr(runtime, "_basis_topology_identity", lambda basis: "topology")
    source = SimpleNamespace(backend="cuda")
    identity = SimpleNamespace(
        method="pbe-rks", functional_identity="xc", regularization_identity="reg"
    )
    state = SimpleNamespace(identity=identity, _source=source)
    spec = SimpleNamespace(partition_iterations=3)
    grid_plan = SimpleNamespace(allocation_bytes=10)
    target = SimpleNamespace(to_payload=dict)
    owner = runtime.PreparedStationaryCudaExecution()
    owner._key = (
        "plan",
        "pbe-rks",
        "gga",
        "unpolarized",
        False,
        "topology",
        "cuda",
        "xc",
        "reg",
        repr(("all-electron",)),
        0,
        repr({}),
        repr(spec),
        3,
        4,
        5,
        6,
        10,
        (),
    )
    owner._bound_basis_identity = "old"
    owner._failed = failed
    owner.host_bound = 100
    owner.device_peak_bound = 30
    owner.artifacts = ()
    owner.sources = SimpleNamespace(
        rebind_geometry=lambda basis: pytest.fail("rebind before admission")
    )
    kwargs = {
        "state": state,
        "basis": SimpleNamespace(identity="new" if changed else "old"),
        "contract": SimpleNamespace(family="gga", spin="unpolarized"),
        "plan": SimpleNamespace(identity="plan"),
        "tensor_plans": {},
        "compiler": SimpleNamespace(target=target),
        "cache": "unused",
        "requests": (),
        "functional": 1,
        "ecp": False,
        "device": 0,
        "spec": spec,
        "grid_plan": grid_plan,
        "source_bytes": 20,
        "tile_points": 4,
        "primitive_tile": 5,
        "integral_terms": 6,
        "max_device_bytes": 30,
        "max_host_bytes": 99,
        "host_bound": 80,
    }
    with pytest.raises(ValueError, match="host budget"):
        owner.ensure(**kwargs)
    # An admission failure must leave an unchanged compatible owner reusable.
    owner._failed = False
    kwargs["basis"] = SimpleNamespace(identity="old")
    kwargs["max_host_bytes"] = 100
    owner.ensure(**kwargs)
    assert owner._bound_basis_identity == "old"
