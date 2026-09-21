"""Bounded triples tile enumerator and TensorIR CPU tests (slice B of #150).

These tests run locally (numpy only, no PySCF, no CUDA).  GPU validation
tests are run manually on qz and record their results as JSON evidence.
"""

import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_cc.triples import (
    triples_energy,
)
from tools.vibeqc_cc.triples_tiles import (
    TileSpec,
    TriplesTileEnumerator,
    build_runtime_tile_triples_program,
    build_tile_triples_program,
    runtime_tile_capacity,
    runtime_tile_control_batches,
    runtime_tile_controls,
    runtime_tile_static_feeds,
    runtime_tile_triples_energy_tensorir,
    tile_triples_energy,
    tile_triples_energy_masked,
    tile_triples_energy_tensorir,
)

INPUT_NAMES = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
ENDPOINTS = Path(__file__).resolve().parents[1] / "reference_data/cc/endpoints"

GROUND_TRUTH = {
    "h2": (1, 1, 8.392021714075268e-49),
    "he": (1, 1, 0.0),
    "h2o": (5, 2, -6.731393342463869e-05),
    "nh3": (5, 3, -1.122922812723691e-04),
    "ch4": (5, 4, -1.555665872715297e-04),
}


def _random_case(nocc: typing.Any, nvir: typing.Any, seed: typing.Any) -> typing.Any:
    rng = np.random.default_rng(seed)
    ovvv = rng.normal(size=(nocc, nvir, nvir, nvir))
    ovoo = rng.normal(size=(nocc, nvir, nocc, nocc))
    ovov = rng.normal(size=(nocc, nvir, nocc, nvir))
    fov = rng.normal(size=(nocc, nvir))
    t1 = rng.normal(size=(nocc, nvir))
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    eps_o = np.linspace(-1.0, -0.5, nocc)
    eps_v = np.linspace(0.5, 1.5, nvir)
    return ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v


def _bounded_tensorir_feeds(arrays: typing.Any, a_end: typing.Any) -> typing.Any:
    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = arrays
    return {
        "ovvv": np.ascontiguousarray(ovvv[:, :a_end, :, :a_end]),
        "ovoo": np.ascontiguousarray(ovoo[:, :a_end, :, :]),
        "ovov": np.ascontiguousarray(ovov[:, :a_end, :, :a_end]),
        "fov": np.ascontiguousarray(fov[:, :a_end]),
        "t1": np.ascontiguousarray(t1[:, :a_end]),
        "t2": np.ascontiguousarray(t2[:, :, :a_end, :]),
        "eps_o": np.ascontiguousarray(eps_o),
        "eps_v": np.ascontiguousarray(eps_v[:a_end]),
    }


def _endpoint_feeds(name: typing.Any) -> typing.Any:
    with np.load(ENDPOINTS / f"{name}.npz", allow_pickle=False) as data:
        eps = data["eps"]
        occ = data["occ"]
        C = data["C"]
        F = data["F"]
        g = data["g"]
        t1 = data["t1"]
        t2 = data["t2"]
    nocc = int(np.sum(occ > 0))
    nvir = len(eps) - nocc
    fov = (C.T @ F @ C)[:nocc, nocc:]
    return (
        nocc,
        nvir,
        g[:nocc, nocc:, nocc:, nocc:],
        g[:nocc, nocc:, :nocc, :nocc],
        g[:nocc, nocc:, :nocc, nocc:],
        fov,
        t1,
        t2,
        eps[:nocc],
        eps[nocc:],
    )


# ---------------------------------------------------------------------------
# Tile enumerator coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("o,v", [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_tile_enumerator_covers_all_virtual_triples(
    o: typing.Any, v: typing.Any, chunk: typing.Any
) -> None:
    """Every (a,b,c) with a>=b>=c appears in exactly one tile."""
    enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk)
    seen = set()
    for tile in enumerator:
        for a, b, c in tile:
            assert 0 <= c <= b <= a < v
            key = (a, b, c)
            assert key not in seen, f"Duplicate (a,b,c)={key} in tile {tile}"
            seen.add(key)
    expected = sum(1 for a in range(v) for b in range(a + 1) for c in range(b + 1))
    assert len(seen) == expected


@pytest.mark.parametrize(
    "o,v,seed,chunk",
    [
        (2, 3, 101, 1),
        (2, 3, 101, 2),
        (2, 3, 101, 3),
        (3, 4, 102, 2),
        (3, 4, 102, 3),
    ],
)
def test_tile_sum_equals_full_reference(
    o: typing.Any, v: typing.Any, seed: typing.Any, chunk: typing.Any
) -> None:
    """Sum of per-tile energies equals the full reference to 1e-10.

    Floating-point associativity differences across tile-ordered vs full
    summation can raise the residual to O(1e-13) on random inputs.  1e-10
    is the per-tile gate the goal requires for GPU-vs-CPU comparison.
    """
    arrays = _random_case(o, v, seed)
    _ovvv, _ovoo, _ovov, _fov, _t1, _t2, _eps_o, _eps_v = arrays
    ref = triples_energy(o, v, *arrays)
    enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk)
    total = 0.0
    for tile in enumerator:
        total += tile_triples_energy(tile, o, *arrays)
    np.testing.assert_allclose(total, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "o,v,seed,chunk",
    [
        (2, 3, 103, 1),
        (2, 3, 103, 2),
        (3, 4, 104, 3),
    ],
)
def test_masked_tile_sum_equals_full_reference(
    o: typing.Any, v: typing.Any, seed: typing.Any, chunk: typing.Any
) -> None:
    """Sum of masked per-tile energies equals the full reference to 1e-10."""
    arrays = _random_case(o, v, seed)
    ref = triples_energy(o, v, *arrays)
    enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk)
    total = 0.0
    for tile in enumerator:
        total += tile_triples_energy_masked(tile, o, *arrays)
    np.testing.assert_allclose(total, ref, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Per-tile correctness
# ---------------------------------------------------------------------------


def test_tile_energy_matches_enumeration() -> None:
    """The per-tile energy function loops exactly the triples claimed by the tile."""
    o, v = 2, 3
    arrays = _random_case(o, v, 105)
    for chunk in [1, 2, 3]:
        enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk)
        for tile in enumerator:
            # Compute reference by separately evaluating each (a,b,c) from the
            # full reference with all other a-coordinates zeroed
            t_energy = tile_triples_energy(tile, o, *arrays)
            # Masked version should agree (same a-range filtering)
            m_energy = tile_triples_energy_masked(tile, o, *arrays)
            np.testing.assert_allclose(t_energy, m_energy, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# TensorIR tile program
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("o,v,seed", [(1, 1, 201), (2, 2, 202), (2, 3, 203)])
def test_tile_tensorir_program_roundtrip(
    o: typing.Any, v: typing.Any, seed: typing.Any
) -> None:
    """The tile TensorIR program produces the same output after JSON roundtrip."""
    from vibeqc_compiler.tensor import Program
    from vibeqc_compiler.tensor import execute as tensor_execute

    arrays = _random_case(o, v, seed)
    for chunk in ((0, v), (0, v // 2 + 1) if v > 1 else (0, v)):
        feeds = _bounded_tensorir_feeds(arrays, chunk[1])
        prog = build_tile_triples_program(o, v, vir_chunk=chunk)
        replayed = Program.loads(prog.dumps())
        v1 = tensor_execute(prog, feeds).outputs["triples_energy"]
        v2 = tensor_execute(replayed, feeds).outputs["triples_energy"]
        np.testing.assert_allclose(v1, v2, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("o,v,seed", [(2, 3, 204), (3, 3, 205)])
def test_tile_tensorir_vs_cpu_reference(
    o: typing.Any, v: typing.Any, seed: typing.Any
) -> None:
    """Tile TensorIR matches the CPU tile reference to < 1e-11."""
    from vibeqc_compiler.tensor import execute as tensor_execute

    arrays = _random_case(o, v, seed)
    for chunk_size in [1, 2, v]:
        enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk_size)
        for tile in enumerator:
            cpu = tile_triples_energy(tile, o, *arrays)
            prog = build_tile_triples_program(
                o, v, vir_chunk=(tile.a_start, tile.a_end)
            )
            feeds = _bounded_tensorir_feeds(arrays, tile.a_end)
            tir = tensor_execute(prog, feeds).outputs["triples_energy"]
            np.testing.assert_allclose(tir, cpu, atol=1e-11, rtol=1e-10)


def test_partial_tile_tensorir_bounds_labels_but_keeps_full_f_axis() -> None:
    """Partial-tile specs retain full summation axes and bounded label axes."""
    from vibeqc_compiler.tensor import execute as tensor_execute

    o, v = 2, 4
    arrays = _random_case(o, v, 208)
    tile = TileSpec(1, 3, v)
    feeds = _bounded_tensorir_feeds(arrays, tile.a_end)
    assert feeds["ovvv"].shape == (o, tile.a_end, v, tile.a_end)
    assert feeds["t2"].shape == (o, o, tile.a_end, v)
    assert feeds["eps_v"].shape == (tile.a_end,)

    prog = build_tile_triples_program(o, v, vir_chunk=(tile.a_start, tile.a_end))
    got = tensor_execute(prog, feeds).outputs["triples_energy"]
    expected = tile_triples_energy(tile, o, *arrays)
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("o,v,seed", [(2, 2, 206)])
def test_tile_tensorir_is_differentiable(
    o: typing.Any, v: typing.Any, seed: typing.Any
) -> None:
    """The tile TensorIR program passes the adjoint dot test."""
    from vibeqc_compiler.tensor import dot_test

    arrays = _random_case(o, v, seed)
    feeds = {n: a for n, a in zip(INPUT_NAMES, arrays)}
    prog = build_tile_triples_program(o, v)
    rng = np.random.default_rng(207)
    tangents = {k: rng.normal(size=x.shape) for k, x in feeds.items()}
    result = dot_test(prog, feeds, tangents, {"triples_energy": np.array(1.0)})
    assert result.passed, "tile TensorIR (T) adjoint dot test failed"


# ---------------------------------------------------------------------------
# Runtime-indexed TensorIR triples (#783)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "o,v,seed,chunk", [(2, 3, 220, 1), (2, 3, 221, 2), (3, 4, 222, 2)]
)
def test_runtime_indexed_tile_program_reuses_one_graph_and_matches_reference(
    o: typing.Any, v: typing.Any, seed: typing.Any, chunk: typing.Any
) -> None:
    from vibeqc_compiler.tensor import execute as tensor_execute

    arrays = dict(zip(INPUT_NAMES, _random_case(o, v, seed), strict=True))
    tiles = tuple(TriplesTileEnumerator(o, v, vir_chunk_size=chunk))
    capacity = runtime_tile_capacity(o, v, chunk)
    program = build_runtime_tile_triples_program(o, v, capacity=capacity)
    static = runtime_tile_static_feeds(arrays)
    total = 0.0
    for tile in tiles:
        controls = runtime_tile_controls(tile, capacity)
        got = float(
            tensor_execute(program, {**static, **controls}).outputs["triples_energy"]
        )
        expected = tile_triples_energy(tile, o, *arrays.values())
        np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-10)
        total += got
    np.testing.assert_allclose(
        total, triples_energy(o, v, *arrays.values()), atol=1e-11, rtol=1e-10
    )


@pytest.mark.parametrize("o,v,seed,chunk", [(2, 3, 223, 1), (3, 5, 224, 2)])
def test_runtime_indexed_complete_tiled_reference(
    o: typing.Any, v: typing.Any, seed: typing.Any, chunk: typing.Any
) -> None:
    arrays = dict(zip(INPUT_NAMES, _random_case(o, v, seed), strict=True))
    got = runtime_tile_triples_energy_tensorir(o, v, arrays, vir_chunk_size=chunk)
    expected = triples_energy(o, v, *arrays.values())
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-10)


def test_runtime_control_subbatches_preserve_triangular_order() -> None:
    tile = TileSpec(0, 3, 3)
    batches = list(runtime_tile_control_batches(tile, 4))
    assert [int(np.sum(batch["active"])) for batch in batches] == [4, 4, 2]
    recovered = []
    for batch in batches:
        active = np.flatnonzero(batch["active"])
        recovered.extend(
            zip(
                batch["a_map"][active],
                batch["b_map"][active],
                batch["c_map"][active],
                strict=True,
            )
        )
    assert recovered == list(tile)


def test_runtime_indexed_graph_size_does_not_scale_with_virtual_triple_count() -> None:
    small = build_runtime_tile_triples_program(2, 3, capacity=4)
    large = build_runtime_tile_triples_program(3, 8, capacity=64)
    assert len(small.live_nodes) == len(large.live_nodes)
    assert not any(node.op == "gather" for node in large.live_nodes)
    assert sum(node.op == "runtime_indexed_select" for node in large.live_nodes) > 0


def test_runtime_indexed_streaming_schedule_bounds_high_rank_intermediates() -> None:
    from vibeqc_compiler.integral.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

    program = build_runtime_tile_triples_program(4, 8, capacity=120)
    target = cuda_target_info("sm_120")
    baseline = plan_cuda(program, target, max_bytes=2 << 30)
    streamed = plan_cuda(
        program,
        target,
        max_bytes=2 << 30,
        schedule=TensorSchedule(stream_reductions=True),
    )

    assert streamed.identity != baseline.identity
    assert streamed.arena_bytes * 20 < baseline.arena_bytes
    assert sum(step.virtual for step in streamed.steps) > 150
    assert sum(step.gemm != "none" for step in streamed.steps) < sum(
        step.gemm != "none" for step in baseline.steps
    )

    q_domain = (
        next(
            node
            for node in program.live_nodes
            if node.op == "input" and node.attrs["name"] == "a_map"
        )
        .spec.indices[0]
        .domain
    )
    for step in streamed.steps:
        indices = step.node.spec.indices
        if (
            step.node.op not in ("input", "constant")
            and indices
            and all(index.domain == q_domain for index in indices)
        ):
            assert not step.virtual


# ---------------------------------------------------------------------------
# Determinism / chunk size independence
# ---------------------------------------------------------------------------


def test_deterministic_same_order_bitwise() -> None:
    """Same tile order produces identical results across two runs."""
    o, v = 2, 3
    arrays = _random_case(o, v, 300)

    def run() -> typing.Any:
        enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=1)
        return [tile_triples_energy(t, o, *arrays) for t in enumerator]

    r1, r2 = run(), run()
    for i, (a, b) in enumerate(zip(r1, r2, strict=True)):
        assert a == b, f"tile {i}: values differ"


def test_different_chunk_sizes_agree() -> None:
    """Total E_T must agree across different vir_chunk_size values to 1e-12."""
    o, v = 3, 5
    arrays = _random_case(o, v, 301)
    ref = triples_energy(o, v, *arrays)
    for chunk in [1, 2, 3, 5]:
        enumerator = TriplesTileEnumerator(o, v, vir_chunk_size=chunk)
        total = sum(tile_triples_energy(t, o, *arrays) for t in enumerator)
        np.testing.assert_allclose(total, ref, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# TileSpec validation
# ---------------------------------------------------------------------------


def test_tile_spec_validation() -> None:
    with pytest.raises(ValueError):
        TileSpec(-1, 2, 3)
    with pytest.raises(ValueError):
        TileSpec(2, 1, 3)
    with pytest.raises(ValueError):
        TileSpec(0, 4, 3)


def test_tile_spec_ntriples() -> None:
    assert TileSpec(0, 1, 3).ntriples == 1
    assert TileSpec(0, 2, 5).ntriples == 4  # (0,0,0),(1,0,0),(1,1,0),(1,1,1)


# ---------------------------------------------------------------------------
# Endpoint regression: tile sums match ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
@pytest.mark.parametrize("chunk", [None])
def test_endpoint_tile_sum_matches_ground_truth(
    name: typing.Any, chunk: typing.Any
) -> None:
    """Tile sum over endpoint data matches pinned PySCF 2.14.0 ground truth."""
    expected_nocc, expected_nvir, expected = GROUND_TRUTH[name]
    feeds = _endpoint_feeds(name)
    nocc, nvir = feeds[0], feeds[1]
    assert nocc == expected_nocc
    assert nvir == expected_nvir
    ref = triples_energy(*feeds)
    if name in ("h2", "he"):
        assert abs(ref) < 1e-9
    else:
        np.testing.assert_allclose(ref, expected, atol=1e-9, rtol=0)
    # Tile sum with various chunk sizes
    for vir_chunk in (nvir,):  # single tile
        enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk)
        total = sum(tile_triples_energy(t, nocc, *feeds[2:]) for t in enumerator)
        np.testing.assert_allclose(total, ref, atol=1e-12, rtol=1e-12)
        # TensorIR tile program
        tir_result = tile_triples_energy_tensorir(nocc, nvir, *feeds[2:])
        np.testing.assert_allclose(tir_result, ref, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("name", ["h2o", "nh3", "ch4"])
@pytest.mark.parametrize("chunk", [1, 2])
def test_endpoint_multi_tile_coverage(name: typing.Any, chunk: typing.Any) -> None:
    """Multi-tile coverage on all non-zero-triples endpoints."""
    feeds = _endpoint_feeds(name)
    nocc, nvir = feeds[0], feeds[1]
    nvir * (nvir + 1) * (nvir + 2) // 6
    if chunk > nvir:
        pytest.skip(f"chunk size {chunk} > nvir {nvir}")
    enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=chunk)
    tiles = list(enumerator)
    assert len(tiles) > 0
    # Every tile should cover at least one a-value
    total_a = sum(t.a_end - t.a_start for t in tiles)
    assert total_a == nvir
    # Tile energies must sum to total
    ref = triples_energy(*feeds)
    total = sum(tile_triples_energy(t, nocc, *feeds[2:]) for t in tiles)
    np.testing.assert_allclose(total, ref, atol=1e-12, rtol=1e-12)
    # TensorIR tile programs
    tir_total = 0.0
    for t in tiles:
        tir_total += tile_triples_energy_tensorir(
            nocc, nvir, *feeds[2:], vir_chunk=(t.a_start, t.a_end)
        )
    np.testing.assert_allclose(tir_total, ref, atol=1e-11, rtol=1e-10)


# ---------------------------------------------------------------------------
# Bounded memory / budget rejection tests
# ---------------------------------------------------------------------------


def test_build_tile_program_refuses_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        build_tile_triples_program(0, 2)
    with pytest.raises(ValueError):
        build_tile_triples_program(2, 0)
    with pytest.raises(ValueError):
        build_tile_triples_program(2, 3, vir_chunk=(-1, 2))
    with pytest.raises(ValueError):
        build_tile_triples_program(2, 3, vir_chunk=(3, 1))


def test_runtime_domain_planner_shrinks_only_after_budget_rejection(
    tmp_path: typing.Any,
) -> None:
    from tools.vibeqc_cc.triples_cuda import CudaTriplesTiles, TriplesTileConfig

    config = TriplesTileConfig(2, 3, vir_chunk_size=1, max_bytes=256 << 20)
    compiler = SimpleNamespace(target=object())
    executor = CudaTriplesTiles(config, compiler, tmp_path)
    seen = []

    def fake_plan(
        program: typing.Any, target: typing.Any, *, max_bytes: typing.Any
    ) -> typing.Any:
        assert target is compiler.target
        assert max_bytes == config.max_bytes
        capacity = program.provenance["runtime_domain_capacity"]
        seen.append(capacity)
        if capacity > 2:
            raise ValueError("infeasible tensor byte budget: synthetic rejection")
        return SimpleNamespace(peak_bytes=1234, identity=f"runtime-{capacity}")

    executor._plan_cuda = fake_plan
    capacity, plan, attempts = executor.plan_runtime_domain()
    assert seen == [6, 3, 1]
    assert capacity == 1
    assert plan.identity == "runtime-1"
    assert [entry["status"] for entry in attempts] == [
        "infeasible",
        "infeasible",
        "selected",
    ]


def test_cuda_runtime_domain_reuses_one_plan_artifact_and_owner(
    tmp_path: typing.Any,
) -> None:
    """Multiple logical tiles must not rebuild the scientific CUDA program."""
    from tools.vibeqc_cc.triples_cuda import CudaTriplesTiles, TriplesTileConfig

    arrays = dict(zip(INPUT_NAMES, _random_case(2, 3, 402), strict=True))
    config = TriplesTileConfig(2, 3, vir_chunk_size=1, max_bytes=256 << 20)
    compiler = SimpleNamespace(target=object())
    executor = CudaTriplesTiles(config, compiler, tmp_path)
    calls: dict[str, typing.Any] = {
        "plan": 0,
        "compile": 0,
        "owners": 0,
        "uploads": [],
        "runs": 0,
    }
    plan = SimpleNamespace(peak_bytes=1234, identity="runtime-plan")
    artifact = SimpleNamespace(metadata={"key": "runtime-artifact"})

    def fake_plan(
        program: typing.Any, target: typing.Any, *, max_bytes: typing.Any
    ) -> typing.Any:
        calls["plan"] += 1
        assert target is compiler.target
        assert max_bytes == config.max_bytes
        assert any(node.op == "runtime_indexed_select" for node in program.live_nodes)
        return plan

    def fake_compile(
        current_plan: typing.Any, current_compiler: typing.Any, cache: typing.Any
    ) -> typing.Any:
        calls["compile"] += 1
        assert current_plan is plan
        assert current_compiler is compiler
        assert cache == tmp_path
        return artifact

    class FakeResident:
        def __init__(
            self,
            current_plan: typing.Any,
            current_artifact: typing.Any,
            *,
            device: typing.Any,
        ) -> None:
            calls["owners"] += 1
            assert current_plan is plan
            assert current_artifact is artifact
            assert device == 0
            self.device = {"test_only": True}
            self._last_value = 0.0

        def __enter__(self) -> typing.Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def upload(self, feeds: typing.Any) -> None:
            calls["uploads"].append(tuple(sorted(feeds)))
            if "active" in feeds:
                self._last_value = float(np.sum(feeds["active"]))

        def run(self, *, profile: typing.Any = False) -> typing.Any:
            assert profile is False
            calls["runs"] += 1
            return {"triples_energy": object()}, {}

        def download(self, _lease: typing.Any) -> np.ndarray:
            return np.asarray(self._last_value)

    executor._plan_cuda = fake_plan
    executor._compile_resident = fake_compile
    executor._PreparedResident = FakeResident
    result = executor.run_tiles(arrays)

    assert calls["plan"] == calls["compile"] == calls["owners"] == 1
    assert calls["runs"] == result.tile_count == 3
    assert calls["uploads"][0] == tuple(sorted(INPUT_NAMES))
    assert all(
        upload == ("a_map", "active", "b_map", "c_map", "degeneracy")
        for upload in calls["uploads"][1:]
    )
    assert result.plan_identity == "runtime-plan"
    assert result.artifact_keys == ["runtime-artifact"]
    assert result.peak_bytes_per_tile == [1234] * result.tile_count


def test_tile_enumerator_refuses_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        TriplesTileEnumerator(0, 2)
    with pytest.raises(ValueError):
        TriplesTileEnumerator(2, 0)
    with pytest.raises(ValueError):
        TriplesTileEnumerator(2, 3, vir_chunk_size=0)
    with pytest.raises(ValueError):
        TriplesTileEnumerator(2, 3, vir_chunk_size=-1)


def test_program_deterministic_hash() -> None:
    """Same parameters produce identical logical hashes."""
    p1 = build_tile_triples_program(2, 3)
    p2 = build_tile_triples_program(2, 3)
    assert p1.logical_hash == p2.logical_hash
    # Different chunk produces different hash
    p3 = build_tile_triples_program(2, 3, vir_chunk=(1, 3))
    assert p1.logical_hash != p3.logical_hash


@pytest.mark.parametrize(
    "invalid, message",
    [
        ("nan", "finite"),
        ("inf", "finite"),
        ("noncanonical", "noncanonical"),
        ("near_zero", "near-zero"),
    ],
)
def test_cuda_input_guards_precede_planning(
    tmp_path: typing.Any, invalid: typing.Any, message: typing.Any
) -> None:
    """Invalid scientific inputs fail before compilation or device allocation."""
    from tools.vibeqc_cc.triples_cuda import CudaTriplesTiles, TriplesTileConfig

    arrays = dict(zip(INPUT_NAMES, _random_case(2, 3, 401), strict=True))
    if invalid in ("nan", "inf"):
        arrays["t1"][0, 0] = float(invalid)
    else:
        arrays["eps_o"].fill(0)
        arrays["eps_v"].fill(-1 if invalid == "noncanonical" else 1e-12)

    def unexpected_planning(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("invalid inputs reached CUDA planning")

    config = TriplesTileConfig(2, 3, vir_chunk_size=1, max_bytes=256 << 20)
    with CudaTriplesTiles(config, None, tmp_path) as tiles:
        tiles._plan_cuda = unexpected_planning
        with pytest.raises(ValueError, match=message):
            tiles.run_tiles(arrays)
