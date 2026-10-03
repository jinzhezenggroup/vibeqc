"""Device-free protocol tests for the production resident nonlocal join."""

from __future__ import annotations

import importlib.util
import typing
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resident_nonlocal_join", ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(
    stream: int = 31, seed_stream: int = 31
) -> tuple[dict[str, typing.Any], list[tuple[typing.Any, ...]]]:
    events: list[tuple[typing.Any, ...]] = []
    resident_grid = SimpleNamespace(
        device=0, points=8192, weights=12288, atomic_weights=16384, point_count=6
    )
    source = SimpleNamespace(
        grid_spec=SimpleNamespace(coincident_tolerance=1e-12),
        cuda_resident_grid=lambda: resident_grid,
    )
    state = SimpleNamespace(
        grid=SimpleNamespace(
            points=np.arange(18, dtype=float).reshape(6, 3),
            weights=np.arange(6, dtype=float) + 0.5,
            owners=np.array([0, 0, 0, 1, 1, 1]),
        ),
        density=np.eye(2)[None, ...],
        weighted_density=np.eye(2)[None, ...],
        _source=source,
    )

    class Grid:
        device_id = 0

        @contextmanager
        def feature_task_device_points(
            self,
            pointer: int,
            point_count: int,
            ids: typing.Any,
            ingredients: tuple[str, ...],
        ) -> typing.Iterator[SimpleNamespace]:
            assert ids is None
            assert ingredients == ("rho", "gradient", "tau")
            begin = (pointer - resident_grid.points) // (3 * 8)
            assert pointer == resident_grid.points + 3 * begin * 8
            assert 0 <= begin < resident_grid.point_count
            lease = SimpleNamespace(
                alive=True,
                view=SimpleNamespace(stream=stream),
                _owner=SimpleNamespace(device_id=0),
            )
            events.append(("borrow", point_count))
            events.append(("point_pointer", begin))
            try:
                yield lease
            finally:
                lease.alive = False
                events.append(("release", point_count))

        def feature_task(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.NoReturn:
            raise AssertionError("host point upload reintroduced")

        def feature_task_with_features(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("host feature export reintroduced")

        def xc_task(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("functional-name dispatch reintroduced")

    class Nonlocal:
        point_count = 6
        executed = False
        collected_points = 0
        seeds = SimpleNamespace(
            pointer=4096, stride=6, stream=seed_stream, generation=7
        )

        def diagnostic(self) -> SimpleNamespace:
            return SimpleNamespace(
                executed=self.executed, collected_points=self.collected_points
            )

        def reset(self) -> None:
            assert self.executed, "native reset rejects a never-executed owner"
            self.executed = False
            self.collected_points = 0
            events.append(("nlc_reset",))

        def collect(self, *_args: typing.Any) -> typing.NoReturn:
            raise AssertionError("tile feature collection reintroduced")

        def seed_from_snapshot(
            self, snapshot: SimpleNamespace, task: SimpleNamespace
        ) -> None:
            assert snapshot is source and task.alive
            self.collected_points = self.point_count
            events.append(("resident_seed",))

        def execute(self) -> SimpleNamespace:
            assert self.collected_points == self.point_count
            events.append(("pairs",))
            self.executed = True
            return self.seeds

    class Sources:
        natom = 2

        def __init__(self, label: str, base: float) -> None:
            self.label = label
            self.base = base
            self.finishes = 0

        def geometry_molecular_resident_weights(
            self,
            task: SimpleNamespace,
            begin: int,
            points_per_atom: int,
            device_weights: int,
            weights: np.ndarray,
            device_raw: int,
            host_raw: np.ndarray | None,
            *,
            functional: int,
        ) -> None:
            assert self.label == "local"
            assert task.alive and functional == 4
            assert points_per_atom == 3
            assert device_weights == resident_grid.weights + begin * 8
            assert device_raw == resident_grid.atomic_weights + begin * 8
            assert host_raw is None
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(weights)]
            )
            events.append(("weight_pointer", begin))
            events.append(("raw_pointer", begin))
            events.append(("local", len(weights)))

        def geometry_external_device_molecular_resident_weights(
            self,
            task: SimpleNamespace,
            begin: int,
            points_per_atom: int,
            device_weights: int,
            weights: np.ndarray,
            device_raw: int,
            host_raw: np.ndarray | None,
            pointer: int,
            stride: int,
            seed_begin: int,
        ) -> None:
            assert self.label == "nonlocal"
            assert task.alive and points_per_atom == 3 and seed_begin == begin
            assert (pointer, stride) == (4096, 6)
            assert device_weights == resident_grid.weights + begin * 8
            assert device_raw == resident_grid.atomic_weights + begin * 8
            assert host_raw is None
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(weights)]
            )
            events.append(("weight_pointer", begin))
            events.append(("raw_pointer", begin))
            events.append(("external", begin))

        def finish_span(self, names: tuple[str, ...]) -> dict[str, np.ndarray]:
            events.append((self.label + "_finish",))
            events.append((self.label + "_finish_span", names))
            self.finishes += 1
            inventory = (
                "xc_ao",
                "xc_grid",
                "xc_weight",
                "overlap_pulay",
                "nuclear",
            )
            return {
                key: np.full((2, 3), self.base + inventory.index(key), dtype=float)
                for key in names
            }

    args = {
        "grid": Grid(),
        "sources": Sources("local", 1.0),
        "nonlocal_sources": Sources("nonlocal", 10.0),
        "nonlocal_owner": Nonlocal(),
        "state": state,
        "raw_weights": np.ones(6),
        "tile_points": 2,
        "ao_count": 2,
        "functional": 4,
        "ingredients": ("rho", "gradient", "tau"),
    }
    return args, events


def test_complete_join_collocates_each_grid_tile_once() -> None:
    args, events = fixture()
    components, seconds, work = MODULE.resident_nonlocal_geometry(**args)
    assert set(components) == {
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "nuclear",
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
    }
    assert [event for event in events if event[0] == "borrow"] == [
        ("borrow", 2),
        ("borrow", 2),
        ("borrow", 2),
    ]
    assert [event for event in events if event[0] == "local"] == [
        ("local", 2),
        ("local", 2),
        ("local", 2),
    ]
    assert [event for event in events if event[0] == "external"] == [
        ("external", 0),
        ("external", 2),
        ("external", 4),
    ]
    assert events.index(("local", 2)) < events.index(("resident_seed",))
    assert events.index(("resident_seed",)) < events.index(("pairs",))
    assert events.index(("pairs",)) < events.index(("external", 0))
    assert work["nonlocal_feature_source"] == "exact-final-scf-device-binding"
    assert work["nonlocal_feature_d2d_bytes"] == 192
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["grid_owner_source"] == "implicit-atom-major-index"
    assert work["grid_owner_h2d_bytes"] == 0
    assert work["grid_point_source"] == "exact-native-resident-grid"
    assert work["grid_point_h2d_bytes"] == 0
    assert work["grid_weight_source"] == "exact-native-resident-grid"
    assert work["grid_weight_h2d_bytes"] == 0
    assert work["grid_atomic_measure_source"] == "exact-native-resident-grid"
    assert work["grid_atomic_measure_h2d_bytes"] == 0
    assert [event[1] for event in events if event[0] == "point_pointer"] == [0, 2, 4]
    assert [event[1] for event in events if event[0] == "weight_pointer"] == [
        0,
        0,
        2,
        2,
        4,
        4,
    ]
    assert [event[1] for event in events if event[0] == "raw_pointer"] == [
        0,
        0,
        2,
        2,
        4,
        4,
    ]
    assert work["ao_collocation_point_visits"] == 6
    assert work["geometry_point_visits"] == 12
    assert (
        "local_finish_span",
        ("xc_ao", "xc_grid", "xc_weight", "overlap_pulay", "nuclear"),
    ) in events
    assert ("nonlocal_finish_span", ("xc_ao", "xc_grid", "xc_weight")) in events
    assert work["stationary_source_d2h_policy"] == "contiguous-live-spans"
    assert work["semilocal_stationary_source_d2h_bytes"] == 240
    assert work["nonlocal_stationary_source_d2h_bytes"] == 144
    assert work["stationary_source_full_arena_d2h_bytes_avoided"] == 288
    assert work["nonlocal_dense_pair_capacity"] == 36
    assert "nonlocal_pair_evaluations" not in work
    assert "resident_feature_seed_enqueue" in seconds
    assert "single_pass_geometry_and_pair_drain" in seconds
    np.testing.assert_array_equal(components["xc_ao"], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(components["nonlocal_ao"], np.full((2, 3), 10.0))


@pytest.mark.parametrize(
    "field,value", [("pointer", 0), ("stride", 4), ("stream", 0), ("generation", 0)]
)
def test_invalid_seed_view_never_reaches_geometry(field: str, value: int) -> None:
    args, events = fixture()
    setattr(args["nonlocal_owner"].seeds, field, value)
    with pytest.raises(ValueError, match="invalid seed lease"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["sources"].finishes == 0
    assert args["nonlocal_sources"].finishes == 0


def test_cross_stream_seed_is_rejected_before_consumption() -> None:
    args, events = fixture(seed_stream=32)
    with pytest.raises(ValueError, match="streams differ"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)


def test_join_refuses_missing_consumer_dependency() -> None:
    args, events = fixture()
    args["nonlocal_sources"].geometry_external_device_molecular_resident_weights = None
    with pytest.raises(TypeError, match="lacks the resident"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def test_join_refuses_missing_snapshot_seed_dependency() -> None:
    args, events = fixture()
    args["nonlocal_owner"].seed_from_snapshot = None
    with pytest.raises(TypeError, match="final-state feature handoff"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def test_join_replays_complete_grid_after_reset() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    MODULE.resident_nonlocal_geometry(**args)
    assert events.count(("nlc_reset",)) == 1
    assert events.count(("resident_seed",)) == 2
    assert events.count(("borrow", 2)) == 6


def test_production_driver_uses_shared_pass_accumulators() -> None:
    driver = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()
    join = (ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py").read_text()
    assert "resident_nonlocal_geometry(" in driver
    assert "nonlocal_sources=self.nonlocal_sources" in driver
    assert "_ResidentNonlocalForceOwner(" in driver
    assert "seed_from_snapshot(state._source, task)" in join
    assert join.count("with grid.feature_task_device_points(") == 1
    assert "with grid.feature_task(" not in join
    assert "except NotImplementedError:" in join
    assert join.count("nonlocal_owner.collect(") == 1
    assert "state.grid.owners" not in join
    assert "geometry_molecular_resident_weights(" in join
    assert "geometry_external_device_molecular_resident_weights(" in join
    for retired in (
        "NonlocalFixedGridPlan",
        "feature_task_with_features(",
        "rho[active]",
        "seeds[:, begin:end]",
    ):
        assert retired not in driver


def test_fresh_owner_is_not_reset_before_first_seed() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    assert ("nlc_reset",) not in events


def test_partial_previous_collection_is_not_reused() -> None:
    args, events = fixture()
    args["nonlocal_owner"].collected_points = 2
    with pytest.raises(ValueError, match="incomplete previous"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def fallback_fixture() -> tuple[dict[str, typing.Any], list[tuple[typing.Any, ...]]]:
    args, events = fixture()
    owner = args["nonlocal_owner"]

    def unavailable(snapshot: typing.Any, task: typing.Any) -> typing.NoReturn:
        assert snapshot is args["state"]._source and task.alive
        events.append(("resident_unavailable",))
        raise NotImplementedError("final state has no device-resident features")

    def collect(task: typing.Any, begin: int) -> None:
        assert task.alive and begin == owner.collected_points
        length = min(args["tile_points"], owner.point_count - begin)
        owner.collected_points += length
        events.append(("collect", begin))

    owner.seed_from_snapshot = unavailable
    owner.collect = collect
    return args, events


@pytest.mark.parametrize("tile_points", [1, 2, 8])
def test_missing_resident_features_preserves_bounded_device_fallback(
    tile_points: int,
) -> None:
    args, events = fallback_fixture()
    args["tile_points"] = tile_points
    components, seconds, work = MODULE.resident_nonlocal_geometry(**args)
    offsets = list(range(0, 6, tile_points))
    assert [event[1] for event in events if event[0] == "collect"] == offsets
    assert [event[1] for event in events if event[0] == "external"] == offsets
    assert sum(event[1] for event in events if event[0] == "borrow") == 12
    assert sum(event[1] for event in events if event[0] == "local") == 6
    assert events.count(("resident_unavailable",)) == 1
    assert events.index(("collect", offsets[-1])) < events.index(("pairs",))
    assert events.index(("local_finish",)) < events.index(("pairs",))
    assert events.index(("pairs",)) < events.index(("external", 0))
    assert args["sources"].finishes == args["nonlocal_sources"].finishes == 1
    assert work["ao_collocation_point_visits"] == 12
    assert work["geometry_point_visits"] == 12
    assert work["nonlocal_feature_source"] == "bounded-grid-feature-collection"
    assert work["nonlocal_feature_d2d_bytes"] == 0
    assert work["nonlocal_feature_collection_point_visits"] == 6
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["grid_owner_source"] == "implicit-atom-major-index"
    assert work["grid_owner_h2d_bytes"] == 0
    assert work["grid_weight_source"] == "exact-native-resident-grid"
    assert work["grid_weight_h2d_bytes"] == 0
    assert work["grid_atomic_measure_source"] == "exact-native-resident-grid"
    assert work["grid_atomic_measure_h2d_bytes"] == 0
    assert "two_pass_geometry_and_pair_drain" in seconds
    assert "single_pass_geometry_and_pair_drain" not in seconds
    np.testing.assert_array_equal(components["xc_ao"], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(components["nonlocal_ao"], np.full((2, 3), 10.0))


def test_fallback_replays_from_an_empty_collection() -> None:
    args, events = fallback_fixture()
    MODULE.resident_nonlocal_geometry(**args)
    MODULE.resident_nonlocal_geometry(**args)
    assert events.count(("nlc_reset",)) == 1
    assert events.count(("collect", 0)) == 2
    assert events.count(("pairs",)) == 2
    assert args["nonlocal_owner"].collected_points == 6


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, MemoryError])
def test_snapshot_errors_are_not_hidden_by_capability_fallback(
    error_type: type[Exception],
) -> None:
    args, events = fallback_fixture()

    def invalid(*args: typing.Any) -> typing.NoReturn:
        raise error_type("stale token, device mismatch or failed CUDA allocation")

    args["nonlocal_owner"].seed_from_snapshot = invalid
    with pytest.raises(error_type, match="stale token"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] in ("collect", "pairs", "external") for event in events)
    assert args["sources"].finishes == args["nonlocal_sources"].finishes == 0
    assert events[-1][0] == "release"


@pytest.mark.parametrize("fallback", [False, True])
def test_pair_failure_is_not_retried_as_a_feature_capability_miss(
    fallback: bool,
) -> None:
    args, events = fallback_fixture() if fallback else fixture()

    def invalid() -> typing.NoReturn:
        raise NotImplementedError("pair execution rejected")

    args["nonlocal_owner"].execute = invalid
    with pytest.raises(NotImplementedError, match="pair execution rejected"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["nonlocal_sources"].finishes == 0


@pytest.mark.parametrize("field,value", [("pointer", 0), ("stream", 32)])
def test_fallback_rejects_bad_seeds_before_nonlocal_consumption(
    field: str, value: int
) -> None:
    args, events = fallback_fixture()
    setattr(args["nonlocal_owner"].seeds, field, value)
    with pytest.raises(ValueError, match="invalid seed lease|streams differ"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["nonlocal_sources"].finishes == 0


@pytest.mark.parametrize("fallback", [False, True])
def test_both_schedules_keep_host_intervals_exclusive(
    monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    args, _ = fallback_fixture() if fallback else fixture()
    ticks = iter((0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 7.0, 15.0))
    monkeypatch.setattr(MODULE, "perf_counter", lambda: next(ticks))
    _, seconds, _ = MODULE.resident_nonlocal_geometry(**args)
    assert sum(seconds.values()) == pytest.approx(15.0)
    assert seconds["resident_feature_seed_enqueue"] == pytest.approx(2.0)
    assert seconds["vv10_pair_enqueue"] == pytest.approx(3.0)
    assert all(value >= 0 for value in seconds.values())


def test_capability_miss_cannot_reuse_a_partially_seeded_owner() -> None:
    args, events = fallback_fixture()
    owner = args["nonlocal_owner"]

    def partial(*args: typing.Any) -> typing.NoReturn:
        owner.collected_points = 1
        raise NotImplementedError("partially changed owner")

    owner.seed_from_snapshot = partial
    with pytest.raises(RuntimeError, match="modified the nonlocal owner"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "collect" for event in events)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("local_profile", [False, True])
@pytest.mark.parametrize("nonlocal_profile", [False, True])
def test_profile_fallback_reports_actual_grid_uploads(
    fallback: bool, local_profile: bool, nonlocal_profile: bool
) -> None:
    args, _events = fallback_fixture() if fallback else fixture()
    args["sources"].profile_device = local_profile
    args["nonlocal_sources"].profile_device = nonlocal_profile
    grid_lease = args["state"]._source.cuda_resident_grid()
    atomic_resident = hasattr(grid_lease, "atomic_weights")
    if atomic_resident:
        # The existing fixture validates the resident-pointer fast path. Also
        # validate the host raw measure supplied to either profiling consumer,
        # then let that fixture check every pointer/offset and source mapping.
        def wrap(original: typing.Any) -> typing.Any:
            def invoke(*values: typing.Any, **keywords: typing.Any) -> None:
                host_raw = values[6]
                if local_profile or nonlocal_profile:
                    np.testing.assert_array_equal(host_raw, np.ones(len(values[4])))
                else:
                    assert host_raw is None
                original(*values[:6], None, *values[7:], **keywords)

            return invoke

        for owner, name in (
            (args["sources"], "geometry_molecular_resident_weights"),
            (
                args["nonlocal_sources"],
                "geometry_external_device_molecular_resident_weights",
            ),
        ):
            setattr(owner, name, wrap(getattr(owner, name)))
    _parts, _seconds, work = MODULE.resident_nonlocal_geometry(**args)
    profiled = int(local_profile) + int(nonlocal_profile)
    expected = profiled * grid_lease.point_count * 8
    assert work["grid_owner_h2d_bytes"] == expected
    assert work["grid_weight_h2d_bytes"] == expected
    assert work["grid_point_h2d_bytes"] == 0
    assert work["nonlocal_seed_h2d_bytes"] == 0
    assert work["ao_collocation_point_visits"] == (2 if fallback else 1) * 6
    assert ("profile-host" in work["grid_owner_source"]) == bool(profiled)
    assert ("profile-host" in work["grid_weight_source"]) == bool(profiled)
    if atomic_resident:
        assert work["grid_atomic_measure_h2d_bytes"] == expected
        assert ("profile-host" in work["grid_atomic_measure_source"]) == bool(profiled)


@pytest.mark.parametrize("empty", [False, True])
def test_local_ao_maps_reach_both_consumers_without_skipping_points(
    empty: bool,
) -> None:
    """Even an empty AO tile still owns nonlocal seeds and grid response work."""
    args, events = fixture()
    state = args["state"]
    state.grid.identity = "grid"
    state.identity = SimpleNamespace(geometry_identity="geometry")
    selected = np.frombuffer(
        np.array([] if empty else [1], dtype=np.uintp).tobytes(), dtype=np.uintp
    )
    domain = SimpleNamespace(
        point_pointer=8192,
        point_count=6,
        device=0,
        grid_identity="grid",
        geometry_identity="geometry",
    )
    original = args["grid"].feature_task_device_points

    @contextmanager
    def feature(
        pointer: int, count: int, ids: typing.Any, ingredients: tuple[str, ...]
    ) -> typing.Iterator[SimpleNamespace]:
        assert ids is selected
        events.append(("selected", len(ids)))
        with original(pointer, count, None, ingredients) as task:
            yield task

    class Maps:
        work: typing.ClassVar[dict[str, int]] = {"retained_map_bytes": selected.nbytes}

        def reset_work(self) -> None:
            events.append(("mask_reset",))

        def select(
            self, grid: typing.Any, actual_domain: typing.Any, begin: int, count: int
        ) -> np.ndarray:
            assert actual_domain is domain
            assert grid is args["grid"] and count == 2
            events.append(("mask", begin))
            return selected

    args["grid"].feature_task_device_points = feature
    args.update(ao_maps=Maps(), ao_domain=domain)
    _, _, work = MODULE.resident_nonlocal_geometry(**args)
    assert [event for event in events if event[0] == "mask"] == [
        ("mask", 0),
        ("mask", 2),
        ("mask", 4),
    ]
    assert len([event for event in events if event[0] == "local"]) == 3
    assert len([event for event in events if event[0] == "external"]) == 3
    assert work["geometry_point_visits"] == 12
    assert work["active_ao_maps"] == {"retained_map_bytes": selected.nbytes}


def test_local_ao_domain_is_checked_against_actual_snapshot_grid() -> None:
    args, events = fixture()
    args["state"].grid.identity = "grid"
    args["state"].identity = SimpleNamespace(geometry_identity="geometry")
    args.update(
        ao_maps=object(),
        ao_domain=SimpleNamespace(
            point_pointer=16384,
            point_count=6,
            device=0,
            grid_identity="grid",
            geometry_identity="geometry",
        ),
    )
    with pytest.raises(ValueError, match="current grid lease"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events
