"""Public full-grid admission keeps finite work and bounded submission fences."""

from __future__ import annotations

import ast
import typing
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import numpy as np
import pytest
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_grid_work,
    stationary_cuda_requires_native_integrals,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("atoms", "visits"),
    [(24, 325583124), (48, 2661287016), (96, 21516784080)],
)
def test_public_complete_grid_is_finite_not_a_diagnostic_total_work_cap(
    atoms: int, visits: int
) -> None:
    points = atoms * 48 * 16 * 32
    with pytest.raises(ValueError, match="work budget"):
        plan_stationary_cuda_grid_work(atoms=atoms, grid_points=points, tile_points=256)
    plan = plan_stationary_cuda_grid_work(
        atoms=atoms,
        grid_points=points,
        tile_points=256,
        max_grid_points=None,
        max_grid_pair_visits=None,
    )
    assert plan.grid_pair_visits == visits
    assert plan.chunk_pair_visits <= 100_000_000
    assert plan.chunk_points <= 64 * plan.tile_points
    assert plan.tile_count == (points + 255) // 256
    assert sum(end - begin for begin, end in plan.chunks()) == points
    assert len(list(plan.chunks())) == plan.chunk_count
    assert list(plan.chunks())[-1][1] == points
    assert stationary_cuda_requires_native_integrals(
        atoms=atoms, aos=8 * atoms, primitives=atoms * 22 // 3
    )


@pytest.mark.parametrize("atoms", [1, 2, 32, 96, 128])
@pytest.mark.parametrize("points", [1, 255, 256, 257, 16384, 16385, 32771])
def test_chunk_tail_preserves_point_order_and_original_tile_boundaries(
    atoms: int, points: int
) -> None:
    plan = plan_stationary_cuda_grid_work(
        atoms=atoms,
        grid_points=points,
        tile_points=256,
        max_grid_points=None,
        max_grid_pair_visits=None,
    )
    actual = [
        (begin, min(begin + 256, stop))
        for start, stop in plan.chunks()
        for begin in range(start, stop, 256)
    ]
    assert actual == [
        (begin, min(begin + 256, points)) for begin in range(0, points, 256)
    ]
    assert (
        sum(
            2 * (stop - start) * atoms * (atoms - 1) // 2
            for start, stop in plan.chunks()
        )
        + atoms * (atoms - 1) // 2
        == plan.grid_pair_visits
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atoms": 129},
        {"grid_points": 0},
        {"grid_points": (1 << 40) + 1},
        {"tile_points": 4097},
        {"max_grid_points": True},
        {"max_grid_pair_visits": 0},
        {"max_pending_tiles": 0},
        {"max_pending_pair_visits": 0},
    ],
)
def test_grid_work_rejects_invalid_shape_and_limits(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        plan_stationary_cuda_grid_work(
            **({"atoms": 96, "grid_points": 1024, "tile_points": 256} | kwargs)
        )


def test_per_window_work_cap_cannot_be_disabled_with_whole_grid_guards() -> None:
    with pytest.raises(ValueError, match="tile exceeds pending"):
        plan_stationary_cuda_grid_work(
            atoms=96,
            grid_points=1024,
            tile_points=256,
            max_grid_points=None,
            max_grid_pair_visits=None,
            max_pending_pair_visits=256 * 96 * 95 - 1,
        )
    plan = plan_stationary_cuda_grid_work(
        atoms=96,
        grid_points=1024,
        tile_points=256,
        max_grid_points=None,
        max_grid_pair_visits=None,
        max_pending_pair_visits=256 * 96 * 95,
    )
    assert plan.chunk_count == 4


@pytest.mark.parametrize(
    ("shape", "required"),
    [
        ((32, 128, 4096), False),
        ((33, 128, 4096), True),
        ((32, 129, 4096), True),
        ((32, 128, 4097), True),
        ((128, 1024, 16384), True),
    ],
)
def test_native_owner_required_only_beyond_legacy_diagnostic_shape(
    shape: tuple[int, int, int], required: bool
) -> None:
    assert (
        stationary_cuda_requires_native_integrals(
            atoms=shape[0], aos=shape[1], primitives=shape[2]
        )
        is required
    )


@pytest.mark.parametrize(
    "shape", [(129, 1024, 16384), (128, 1025, 16384), (128, 1024, 16385)]
)
def test_native_owner_caps_remain_fail_closed(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="resource caps"):
        stationary_cuda_requires_native_integrals(
            atoms=shape[0], aos=shape[1], primitives=shape[2]
        )


def _geometry_loop(scope: dict[str, typing.Any]) -> None:
    """Compile the actual production submission loop with explicit fake owners."""
    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_complete_rks_cuda_gradient_diagnostic"
    )
    loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [elt.id for elt in node.target.elts] == ["chunk_begin", "chunk_end"]
    )
    wrapper = ast.parse("def run():\n    pass").body[0]
    wrapper.body = [loop]
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(value for value in code.co_consts if isinstance(value, CodeType))
    FunctionType(compiled, scope)()


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("fail_at_drain", [None, 2])
@pytest.mark.parametrize("selected", [None, (), (1, 4)])
def test_actual_geometry_loop_preserves_offsets_and_stops_at_failed_window(
    resident: bool,
    profile: bool,
    fail_at_drain: int | None,
    selected: tuple[int, ...] | None,
) -> None:
    points, tile = 1003, 64
    plan = plan_stationary_cuda_grid_work(
        atoms=96,
        grid_points=points,
        tile_points=tile,
        max_grid_points=None,
        max_grid_pair_visits=None,
        max_pending_tiles=3,
    )
    grid = SimpleNamespace(
        points=np.arange(points * 3).reshape(points, 3),
        weights=np.arange(points) + 100,
        owners=np.arange(points),
    )
    weights = np.arange(points) + 200
    lease = SimpleNamespace(points=10000, weights=20000, atomic_weights=30000)
    seen, events = [], []
    active = False
    mask = None if selected is None else np.asarray(selected, dtype=np.uintp)
    selections = []

    def select(owner: typing.Any, domain: str, begin: int, count: int) -> np.ndarray:
        assert not active and domain == "order-two-grid"
        assert owner is scope["ao"]
        selections.append((begin, count))
        return mask

    @contextmanager
    def task(*args: typing.Any, **kwargs: typing.Any) -> typing.Iterator[typing.Any]:
        nonlocal active
        active = True
        yield args
        active = False

    def geometry(
        view: typing.Any,
        owners: np.ndarray,
        weighted: np.ndarray,
        atomic: np.ndarray,
        *,
        functional: int,
    ) -> None:
        assert active and functional == 1
        begin, count = int(owners[0]), len(owners)
        np.testing.assert_array_equal(view[0], grid.points[begin : begin + count])
        np.testing.assert_array_equal(weighted, grid.weights[begin : begin + count])
        np.testing.assert_array_equal(atomic, weights[begin : begin + count])
        seen.append((begin, count))
        events.append("tile")

    def resident_geometry(
        view: typing.Any,
        begin: int,
        points_per_atom: int,
        weighted: int,
        host_weighted: np.ndarray | None,
        atomic: int,
        host_atomic: np.ndarray | None,
        *,
        functional: int,
    ) -> None:
        assert active and functional == 1 and points_per_atom == 17
        assert view[0] == lease.points + 24 * begin
        assert view[2] is mask
        assert weighted == lease.weights + 8 * begin
        assert atomic == lease.atomic_weights + 8 * begin
        if profile:
            np.testing.assert_array_equal(
                host_weighted, grid.weights[begin : begin + view[1]]
            )
            np.testing.assert_array_equal(host_atomic, weights[begin : begin + view[1]])
        else:
            assert host_weighted is host_atomic is None
        seen.append((begin, view[1]))
        events.append("tile")

    def drain() -> None:
        assert not active
        events.append("drain")
        if events.count("drain") == fail_at_drain:
            raise RuntimeError("injected deferred geometry error")

    scope = {
        "grid_work": plan,
        "tile_points": tile,
        "grid": grid,
        "state": SimpleNamespace(_source=SimpleNamespace(atomic_weights=weights)),
        "resident_grid": lease if resident else None,
        "ao": SimpleNamespace(feature_task=task, feature_task_device_points=task),
        "ao_maps": (
            None
            if selected is None
            else SimpleNamespace(domain="order-two-grid", select=select)
        ),
        "sources": SimpleNamespace(
            geometry=geometry,
            geometry_molecular_resident_weights=resident_geometry,
            drain_geometry=drain,
        ),
        "timeline": SimpleNamespace(phase=lambda _: nullcontext()),
        "ingredients": ("rho", "sigma"),
        "points_per_atom": 17,
        "functional": 1,
        "profile_device": profile,
        "np": np,
    }
    if fail_at_drain:
        with pytest.raises(RuntimeError, match="deferred geometry"):
            _geometry_loop(scope)
        stop = fail_at_drain * plan.chunk_points
    else:
        _geometry_loop(scope)
        stop = points
    assert seen == [
        (begin, min(tile, points - begin)) for begin in range(0, stop, tile)
    ]
    assert events[-1] == "drain"
    assert events.count("drain") == (fail_at_drain or plan.chunk_count)
    assert selections == (seen if resident and selected is not None else [])


def _resource_preflight(basis: typing.Any, host_budget: int = 256 << 20) -> dict:
    """Execute production resource admission without compiling or allocating CUDA."""
    from generativeqc import _stationary_cuda as runtime
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_complete_rks_cuda_gradient_diagnostic"
    )

    def assigns(node: ast.stmt, name: str) -> bool:
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )

    start = next(
        index for index, node in enumerate(function.body) if assigns(node, "grid_plan")
    )
    stop = next(
        index for index, node in enumerate(function.body) if assigns(node, "cache")
    )
    wrapper = ast.parse("def preflight():\n    pass").body[0]
    wrapper.body = (
        function.body[start:stop]
        + ast.parse(
            "return dict(grid=grid_plan, host=host_bound, source=source_resources, reserve=native_geometry_reserve)"
        ).body
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(value for value in code.co_consts if isinstance(value, CodeType))
    scope = vars(runtime) | {
        "basis": basis,
        "na": basis.natom,
        "n": basis.nao,
        "tile_points": 256,
        "primitive_tile": 4096,
        "integral_terms": 32,
        "needs_first": True,
        "max_device_bytes": 512 << 20,
        "max_host_bytes": host_budget,
        "prepared": None,
        "resident_ao_cutoff": None,
        "resident_ao_cache_bytes": 16 << 20,
        "ecp": False,
        "plan": SimpleNamespace(spin_blocks=1),
        "source_names": tuple(range(8)),
        "state": SimpleNamespace(
            _source=SimpleNamespace(cuda_integral_derivatives=lambda *_: None)
        ),
        "target": cuda_target_info("sm_120"),
    }
    return FunctionType(compiled, scope)()


@pytest.mark.parametrize(
    ("atoms", "aos", "primitives", "basis_bytes", "host", "source", "reserve"),
    [
        (24, 192, 176, 62592, 24714848, 2229248, 328064),
        (48, 384, 352, 125184, 65054240, 4938496, 1245888),
        (96, 768, 704, 250368, 197047712, 13978880, 4851008),
    ],
)
def test_actual_resource_admission_keeps_24_48_96_inside_unchanged_byte_caps(
    atoms: int,
    aos: int,
    primitives: int,
    basis_bytes: int,
    host: int,
    source: int,
    reserve: int,
) -> None:
    basis = SimpleNamespace(
        natom=atoms,
        nao=aos,
        nprimitive=primitives,
        numeric_bytes=basis_bytes,
        packed=SimpleNamespace(size=3 * atoms + 2 * primitives + 16 * aos),
    )
    plan = _resource_preflight(basis)
    assert plan["host"] == host < 256 << 20
    assert plan["source"].allocation_bytes == source
    assert plan["reserve"] == reserve
    assert plan["grid"].peak_bytes + source + reserve < 512 << 20
    assert plan["source"].becke_threads_per_point == 32
    assert _resource_preflight(basis, host)["host"] == host
    with pytest.raises(ValueError, match="additional-host byte budget"):
        _resource_preflight(basis, host - 1)


@pytest.mark.parametrize("executed_points", [0, 1, 2])
def test_publication_checks_points_even_for_zero_atom_pair_work(
    executed_points: int,
) -> None:
    path = ROOT / "python/generativeqc/_stationary_cuda.py"
    tree = ast.parse(path.read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Constant)
            and child.value
            == "CUDA executed work disagrees with admitted source coverage"
            for child in ast.walk(node)
        )
    )
    wrapper = ast.parse("def validate():\n    pass").body[0]
    wrapper.body = [guard]
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
        str(path),
        "exec",
    )
    compiled = next(value for value in code.co_consts if isinstance(value, CodeType))
    scope = {
        "work": {
            "xc_points": executed_points,
            "primitive_records": 0,
            "grid_pair_visits": 0,
        },
        "records": 0,
        "pair_visits": 0,
        "grid_work": SimpleNamespace(grid_points=1),
    }
    validate = FunctionType(compiled, scope)
    if executed_points == 1:
        validate()
    else:
        with pytest.raises(RuntimeError, match="source coverage"):
            validate()
