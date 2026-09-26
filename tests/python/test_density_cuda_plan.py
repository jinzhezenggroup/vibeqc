"""Device-free capacity and preflight gates for the D/C feature owner."""

import typing
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.resources import ResourceBudget, plan_resources
from vibeqc_compiler.dft.cuda import CudaGrid
from vibeqc_compiler.dft.plan import plan_tiles


def basis() -> typing.Any:
    return SimpleNamespace(nao=32, natom=3, numeric_bytes=8192, packed=np.zeros(256))


def test_occupied_capacity_does_not_materialize_full_point_by_orbital_data() -> None:
    small, large = [
        plan_tiles(
            basis(),
            backend="cuda",
            tile_points=37,
            orbital_capacity=(n, n),
            orbital_tile=7,
        )
        for n in (11, 97)
    ]
    assert small.orbital_buffers["orbital_psi"] == large.orbital_buffers["orbital_psi"]
    assert (
        small.orbital_buffers["orbital_pack"] == large.orbital_buffers["orbital_pack"]
    )
    assert (
        large.orbital_buffers["orbital_factors"]
        > small.orbital_buffers["orbital_factors"]
    )
    for plan in (small, large):
        composed = plan_resources(
            (plan.resource_request(("rho", "sigma")),), ResourceBudget()
        )
        assert composed.peak_bytes["device"] == plan.device_bytes
        assert composed.peak_bytes["host"] == plan.host_bytes
        assert {e.name for e in composed.estimates} >= {
            "orbital_factors",
            "orbital_pack",
            "orbital_psi",
        }


@pytest.mark.parametrize(
    "budget", [ResourceBudget(host_bytes=1), ResourceBudget(device_bytes=1)]
)
def test_shared_budget_rejects_before_loading_or_allocating_cuda(
    budget: typing.Any, monkeypatch: typing.Any
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("CUDA must not load before resource preflight")

    monkeypatch.setattr("vibeqc_compiler.dft.cuda.ct.CDLL", forbidden)
    with pytest.raises(MemoryError):
        CudaGrid(basis(), None, orbital_capacity=(7, 0), resource_budget=budget)


@pytest.mark.parametrize(
    "counts,tile", [((-1, 0), 3), ((2, 3, 4), 3), ((True, 1), 3), ((3, 2), 0)]
)
def test_invalid_orbital_topology_fails_before_allocation(
    counts: typing.Any, tile: typing.Any
) -> None:
    with pytest.raises(ValueError):
        plan_tiles(basis(), backend="cuda", orbital_capacity=counts, orbital_tile=tile)


def test_legacy_and_shared_budget_are_not_silently_combined() -> None:
    with pytest.raises(ValueError, match="shared resource budget or"):
        CudaGrid(
            basis(), None, budget_bytes=128 << 20, resource_budget=ResourceBudget()
        )
