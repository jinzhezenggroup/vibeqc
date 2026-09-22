"""Resident reuse must obey each call's admission caps before any device work."""

from types import SimpleNamespace

import pytest
from vibeqc import _stationary_cuda as runtime
from vibeqc_compiler.common.prepared_execution import PreparedArtifactBinding
from vibeqc_compiler.common.provenance import canonical_hash


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
    target = SimpleNamespace(to_payload=dict)
    owner = runtime.PreparedStationaryCudaExecution()
    owner._bound_basis_identity = "old"
    owner.artifacts = ()
    owner.sources = SimpleNamespace(
        rebind_geometry=lambda basis: pytest.fail("rebind before admission")
    )
    owner.grid = SimpleNamespace(
        _rebind_centers=lambda centers: pytest.fail("grid rebind before admission")
    )
    basis = SimpleNamespace(
        identity="new" if changed else "old",
        packed=[],
        natom=0,
        nao=0,
    )
    kwargs = {
        "state": state,
        "basis": basis,
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
        "grid_plan": SimpleNamespace(allocation_bytes=10, peak_bytes=10),
        "source_bytes": 20,
        "tile_points": 4,
        "primitive_tile": 5,
        "integral_terms": 6,
        "work_budget": 7,
        "max_device_bytes": 30,
        "max_host_bytes": 99,
        "host_bound": 80,
    }
    request = owner._request(
        state=state,
        basis=basis,
        contract=kwargs["contract"],
        plan=kwargs["plan"],
        tensor_plans=kwargs["tensor_plans"],
        target=target,
        aot_directory=None,
        native_grid_library=None,
        ecp=False,
        device=0,
        spec=spec,
        grid_plan=kwargs["grid_plan"],
        source_bytes=20,
        tile_points=4,
        primitive_tile=5,
        integral_terms=6,
        work_budget=7,
    )
    owner._lease.install(
        request,
        (
            PreparedArtifactBinding(
                canonical_hash("artifact-key"), canonical_hash("artifact-binary")
            ),
        ),
        host_bytes=100,
        device_bytes=30,
    )
    if failed:
        owner._lease.mark_failure()
    with pytest.raises(ValueError, match="host budget"):
        owner.ensure(**kwargs)
    # An admission failure must leave an unchanged compatible owner reusable.
    if owner._lease.failed:
        owner._lease.mark_refresh()
    kwargs["basis"] = SimpleNamespace(identity="old", packed=[], natom=0, nao=0)
    kwargs["max_host_bytes"] = 100
    owner.ensure(**kwargs)
    assert owner._bound_basis_identity == "old"
