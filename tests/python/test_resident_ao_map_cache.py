"""Host protocol gates for scientific identity, numeric budgets and fallbacks."""

import typing
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._resident_ao_maps import ResidentAoMapCache, ResidentAoMapDomain


def immutable(ids: typing.Iterable[int]) -> np.ndarray:
    return np.frombuffer(np.asarray(ids, dtype=np.uintp).tobytes(), dtype=np.uintp)


class Grid:
    def __init__(self) -> None:
        self.plan = SimpleNamespace(
            nao=10, active_ao_capacity=10, order=2, tile_points=4
        )
        self.basis_identity = "basis"
        self.basis_generation = 0
        self.geometry_generation = 0
        self.device_id = 0
        self._lock = RLock()
        self.calls = []
        self.closed = False
        self.answer = immutable([1, 4, 8])

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("closed")

    def select_ao_device_points(
        self, pointer: int, count: int, *, cutoff: float
    ) -> np.ndarray:
        self.calls.append((pointer, count, cutoff))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def domain() -> ResidentAoMapDomain:
    return ResidentAoMapDomain("basis", "geometry", "grid", 0, 8192, 10, 4, 2)


def test_retained_maps_work_and_density_independent_reuse() -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    first = cache.select(grid, domain(), 4, 4)
    assert grid.calls == [(8192 + 4 * 24, 4, 1e-16)]
    assert not first.flags.writeable
    cache.reset_work()
    assert cache.select(grid, domain(), 4, 4) is first
    assert len(grid.calls) == 1
    assert cache.work["cache_hits"] == 1
    assert cache.work["discoveries"] == 0
    assert cache.work["point_ao_square_sum"] == 4 * 3**2
    assert cache.work["dense_point_ao_square_sum"] == 4 * 10**2
    cache.select(grid, domain(), 8, 2)
    assert cache.work["discovery_ao_jet_values"] == 2 * 10 * 10
    assert cache.work["discovery_density_contractions"] == 0
    assert cache.work["retained_map_bytes"] == 48
    assert cache.work["active_aos_min"] == cache.work["active_aos_max"] == 3
    assert cache.work["active_aos_sum"] == 6
    assert cache.work["tile_count"] == 2
    assert cache.work["empty_tile_count"] == 0
    assert cache.work["numeric_peak_bound_bytes"] <= 512
    for name, value in (
        ("cutoff", 1e-3),
        ("budget_bytes", 1 << 20),
        ("domain", domain()),
    ):
        with pytest.raises(AttributeError):
            setattr(cache, name, value)


@pytest.mark.parametrize("variant", ["empty", "full"])
def test_empty_and_identity_maps_have_no_retained_numeric_cost(variant: str) -> None:
    grid = Grid()
    grid.answer = immutable([] if variant == "empty" else range(10))
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    for _ in range(2):
        ids = cache.select(grid, domain(), 0, 4)
        assert ids is None if variant == "full" else len(ids) == 0
    assert len(grid.calls) == 1
    assert cache.work["retained_map_bytes"] == 0


@pytest.mark.parametrize("budget", [0, 199, 200, 279])
def test_budget_miss_is_dense_before_discovery(budget: int) -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=budget)
    assert cache.select(grid, domain(), 0, 4) is None
    assert not grid.calls
    assert cache.work["dense_budget_tiles"] == 1
    assert cache.work["numeric_peak_bound_bytes"] <= budget


def test_retention_cannot_exceed_budget_and_keeps_existing_maps() -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=280)
    first = cache.select(grid, domain(), 0, 4)
    assert cache.select(grid, domain(), 4, 4) is None
    assert cache.select(grid, domain(), 0, 4) is first
    assert len(grid.calls) == 1
    assert cache.work["numeric_peak_bound_bytes"] <= 280


def test_capability_miss_is_dense_but_device_failure_is_not_hidden() -> None:
    grid = Grid()
    grid.answer = NotImplementedError("old artifact")
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    assert cache.select(grid, domain(), 0, 4) is None
    assert cache.select(grid, domain(), 4, 4) is None
    assert len(grid.calls) == 1
    assert cache.work["dense_capability_tiles"] == 2
    grid.answer = RuntimeError("device error")
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    with pytest.raises(RuntimeError, match="device error"):
        cache.select(grid, domain(), 0, 4)
    assert cache.work["discoveries"] == 0
    grid.answer = immutable([2])
    assert list(cache.select(grid, domain(), 0, 4)) == [2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("basis_identity", "changed"),
        ("geometry_identity", "moved"),
        ("grid_identity", "reordered"),
        ("point_pointer", 16384),
        ("derivative_order", 1),
        ("tile_points", 2),
        ("device", 1),
        ("point_count", 9),
    ],
)
def test_every_domain_component_is_checked_before_cache_hit(
    field: str, value: typing.Any
) -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    cache.select(grid, domain(), 0, 4)
    with pytest.raises(ValueError, match="binding mismatch"):
        cache.select(grid, replace(domain(), **{field: value}), 0, 4)
    assert len(grid.calls) == 1


@pytest.mark.parametrize("field", ["geometry_generation", "basis_generation"])
def test_grid_rebinding_revokes_cache_even_when_pointers_are_reused(field: str) -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    cache.select(grid, domain(), 0, 4)
    setattr(grid, field, 1)
    with pytest.raises(ValueError, match="binding mismatch"):
        cache.select(grid, domain(), 0, 4)


def test_closed_owner_and_malformed_tiles_fail_even_on_cache_hit() -> None:
    grid = Grid()
    cache = ResidentAoMapCache(grid, domain(), cutoff=1e-16, budget_bytes=512)
    cache.select(grid, domain(), 0, 4)
    for begin, count in ((1, 4), (0, 3), (8, 4), (10, 0), (-4, 4)):
        with pytest.raises(ValueError, match="tile domain"):
            cache.select(grid, domain(), begin, count)
    grid.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        cache.select(grid, domain(), 0, 4)
