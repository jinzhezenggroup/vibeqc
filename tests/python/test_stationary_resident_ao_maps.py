"""Ordinary force mask admission, identity, and replay remain caller-owned."""

import ast
import inspect
import sys
import typing
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
from generativeqc import _stationary_cuda as runtime


@pytest.mark.parametrize("cutoff", [None, 1e-16])
@pytest.mark.parametrize("available", [0, 15, 128])
def test_optional_map_storage_never_displaces_dense_fallback(
    cutoff: float | None, available: int
) -> None:
    reserve = runtime._stationary_ao_map_reserve(cutoff, 64, 100, 100 + available)
    assert reserve == (0 if cutoff is None else min(64, available))


@pytest.mark.parametrize("cutoff", [True, 0, -1, float("nan"), float("inf"), "1e-16"])
def test_invalid_cutoff_fails_closed(cutoff: typing.Any) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        runtime._stationary_ao_map_reserve(cutoff, 64, 100, 200)


@pytest.mark.parametrize("budget", [True, -1, 1.5, (1 << 40) + 1])
def test_invalid_map_budget_fails_closed_even_if_disabled(budget: typing.Any) -> None:
    with pytest.raises(ValueError, match="cache budget"):
        runtime._stationary_ao_map_reserve(None, budget, 100, 200)


def test_map_reserve_does_not_hide_existing_host_overcommit() -> None:
    with pytest.raises(ValueError, match="additional-host"):
        runtime._stationary_ao_map_reserve(1e-16, 0, 101, 100)


@dataclass(frozen=True)
class _Domain:
    basis_identity: str
    geometry_identity: str
    grid_identity: str
    device: int
    point_pointer: int
    point_count: int
    tile_points: int
    derivative_order: int


@pytest.fixture
def bindings(monkeypatch: pytest.MonkeyPatch) -> typing.Iterator[typing.Any]:
    """Test the caller without inheriting implementation details of the cache."""

    class Cache:
        def __init__(
            self, grid: typing.Any, domain: _Domain, *, cutoff: float, budget_bytes: int
        ) -> None:
            self.grid, self.domain = grid, domain
            self.cutoff, self.budget_bytes = cutoff, budget_bytes
            self.resets = 0

        def reset_work(self) -> None:
            self.resets += 1

    module = ModuleType("generativeqc._resident_ao_maps")
    module.ResidentAoMapDomain = _Domain
    module.ResidentAoMapCache = Cache
    monkeypatch.setitem(sys.modules, module.__name__, module)
    checks = []
    state = SimpleNamespace(
        _source=SimpleNamespace(check_current=lambda: checks.append(True)),
        identity=SimpleNamespace(geometry_identity="geometry", grid_identity="grid"),
    )
    grid = SimpleNamespace(
        basis_identity="initial-basis",
        basis_generation=1,
        geometry_generation=1,
        plan=SimpleNamespace(tile_points=256, order=2),
    )
    resident = SimpleNamespace(device=0, points=1000, point_count=512)
    owner = runtime.PreparedStationaryCudaExecution()
    yield owner, grid, state, resident, checks
    owner.close()


def _cache(bindings: typing.Any, **kwargs: typing.Any) -> typing.Any:
    owner, grid, state, resident, _ = bindings
    return runtime._stationary_resident_ao_cache(
        owner,
        grid,
        state,
        resident,
        **{"cutoff": 1e-16, "budget_bytes": 64, **kwargs},
    )


def test_replay_rechecks_snapshot_but_not_density_independent_maps(
    bindings: typing.Any,
) -> None:
    first = _cache(bindings)
    state = bindings[2]
    state.identity.density_generation = 7
    assert _cache(bindings) is first
    assert first.resets == 2
    assert bindings[4] == [True, True]
    assert first.domain.derivative_order == 2
    assert first.domain.basis_identity == "initial-basis"


@pytest.mark.parametrize(
    "change",
    [
        "geometry",
        "points",
        "grid",
        "order",
        "tile",
        "basis-generation",
        "geometry-generation",
    ],
)
def test_map_replay_rebuilds_for_every_changed_binding(
    bindings: typing.Any, change: str
) -> None:
    first = _cache(bindings)
    _, grid, state, resident, _ = bindings
    if change == "geometry":
        state.identity.geometry_identity = "changed"
    elif change == "points":
        resident.points += 64
    elif change == "grid":
        state.identity.grid_identity = "changed"
    elif change == "order":
        grid.plan.order = 1
    elif change == "tile":
        grid.plan.tile_points = 128
    elif change == "basis-generation":
        grid.basis_generation += 1
    else:
        grid.geometry_generation += 1
    assert _cache(bindings) is not first


def test_changed_grid_owner_cannot_reuse_same_pointer_domain(
    bindings: typing.Any,
) -> None:
    first = _cache(bindings)
    owner, grid, state, resident, checks = bindings
    replacement = SimpleNamespace(**vars(grid))
    assert _cache((owner, replacement, state, resident, checks)) is not first


@pytest.mark.parametrize("changed", [{"cutoff": 1e-18}, {"budget_bytes": 0}])
def test_changed_explicit_policy_does_not_reuse_maps(
    bindings: typing.Any, changed: dict[str, typing.Any]
) -> None:
    first = _cache(bindings)
    assert _cache(bindings, **changed) is not first


def test_revoked_snapshot_is_checked_before_cache_hit(bindings: typing.Any) -> None:
    _cache(bindings)

    def revoked() -> typing.NoReturn:
        raise RuntimeError("stale snapshot")

    bindings[2]._source.check_current = revoked
    with pytest.raises(RuntimeError, match="stale snapshot"):
        _cache(bindings)


@pytest.mark.parametrize("disabled", [True, False])
def test_dense_fallback_releases_retained_maps(
    bindings: typing.Any, disabled: bool
) -> None:
    _cache(bindings)
    owner, grid, state, resident, _ = bindings
    assert (
        runtime._stationary_resident_ao_cache(
            owner,
            grid,
            state,
            resident if disabled else None,
            cutoff=None if disabled else 1e-16,
            budget_bytes=64,
        )
        is None
    )
    assert owner._resident_ao_maps is None
    assert owner._resident_ao_map_key is None


def test_unprepared_endpoint_does_not_retain_a_cache(bindings: typing.Any) -> None:
    _, grid, state, resident, _ = bindings
    first = runtime._stationary_resident_ao_cache(
        None, grid, state, resident, cutoff=1e-16, budget_bytes=64
    )
    second = runtime._stationary_resident_ao_cache(
        None, grid, state, resident, cutoff=1e-16, budget_bytes=64
    )
    assert first is not second


def test_closing_prepared_owner_releases_maps(bindings: typing.Any) -> None:
    _cache(bindings)
    bindings[0].close()
    assert bindings[0]._resident_ao_maps is None


def test_default_and_public_forwarding_do_not_enable_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        runtime,
        "_complete_rks_cuda_gradient_diagnostic",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    runtime.complete_rks_cuda_gradient_diagnostic(None, None, compiler=None, cache=None)
    assert calls[-1]["resident_ao_cutoff"] is None
    runtime.complete_rks_cuda_gradient_diagnostic(
        None,
        None,
        compiler=None,
        cache=None,
        resident_ao_cutoff=1e-16,
        resident_ao_cache_bytes=0,
    )
    assert calls[-1]["resident_ao_cutoff"] == 1e-16
    assert calls[-1]["resident_ao_cache_bytes"] == 0


def test_actual_resident_loop_passes_map_to_native_geometry_lease() -> None:
    source = ast.parse(
        inspect.getsource(runtime._complete_rks_cuda_gradient_diagnostic)
    )
    leases = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "feature_task_device_points"
    ]
    assert len(leases) == 1
    assert ast.unparse(leases[0].args[2]) == "selected_ao_ids"
