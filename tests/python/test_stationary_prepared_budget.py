"""Replay admission must honor current caps before touching retained owners."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
from vibeqc import _stationary_cuda as runtime


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("cap", ["max_host_bytes", "max_device_bytes"])
def test_lowered_replay_cap_rejects_before_rebind(
    monkeypatch: pytest.MonkeyPatch, rebind: bool, cap: str
) -> None:
    touched = []
    owner = NS(
        rebind_geometry=lambda basis: touched.append("source"),
        _rebind_centers=lambda centers: touched.append("grid"),
    )
    artifact = NS(library="unused", metadata={"key": "k", "binary_sha256": "h"})
    monkeypatch.setattr(runtime, "_basis_topology_identity", lambda basis: "fixed")
    monkeypatch.setattr(runtime, "emit_first_derivative_cuda", lambda req: "source")
    for name in ("compile_stationary_cuda", "compile_grid", "compile_cuda"):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: artifact)
    for name in ("_CudaSources", "CudaGrid", "PreparedCuda"):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: nullcontext(owner))
    monkeypatch.setattr(runtime, "file_hash", lambda path: "h")
    basis = NS(identity="initial", natom=1, nao=1, packed=np.zeros(3))
    kwargs = {
        "state": NS(
            identity=NS(
                method="pbe-rks",
                functional_identity="pbe",
                regularization_identity="strict",
            ),
            _source=NS(backend="cuda"),
        ),
        "basis": basis,
        "contract": NS(family="gga", spin="unpolarized"),
        "plan": NS(identity="p"),
        "tensor_plans": {"x": NS(identity="t", peak_bytes=40, host_bytes=30)},
        "compiler": NS(target=NS(to_payload=lambda: {"arch": "sm_120"})),
        "cache": "unused",
        "requests": (),
        "pbe": True,
        "ecp": False,
        "device": 0,
        "spec": NS(partition_iterations=3),
        "grid_plan": NS(allocation_bytes=10, peak_bytes=10),
        "source_bytes": 20,
        "tile_points": 1,
        "primitive_tile": 1,
        "integral_terms": 1,
        "max_device_bytes": 70,
        "max_host_bytes": 130,
        "host_bound": 100,
    }
    with runtime.PreparedStationaryCudaExecution() as prepared:
        prepared.ensure(**kwargs)
        assert prepared.host_bound == 130 and prepared.device_peak_bound == 70
        if rebind:
            basis.identity = "moved"
        lower = {**kwargs, cap: kwargs[cap] - 1}
        with pytest.raises(ValueError, match="budget"):
            prepared.ensure(**lower)
        assert not touched and prepared._bound_basis_identity == "initial"
        prepared.ensure(**kwargs)
        assert touched == (["source", "grid"] if rebind else [])
