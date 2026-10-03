"""Physical source/frame reverse gates using independent AO/metric fixtures."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.df_source import factor_embedding_vjp
from generativeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]


def embed(seeds: list[np.ndarray], o: int, n: int) -> np.ndarray:
    """Independent dense Frobenius embedding, including both cross sectors."""
    q = seeds[0].shape[0]
    full = np.zeros((q, n, n))
    full[:, :o, :o] = seeds[0]
    full[:, :o, o:] = seeds[1]
    full[:, o:, o:] = seeds[2]
    return ((full + full.transpose(0, 2, 1)) / 2).transpose(1, 2, 0)


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 1, 5), (3, 2, 4)])
def test_compiler_embedding_uses_physical_pair_metric(o: int, v: int, q: int) -> None:
    rng = np.random.default_rng(1763)
    shapes = [(q, o, o), (q, o, v), (q, v, v)]
    seeds = [rng.normal(size=s) for s in shapes]
    program = factor_embedding_vjp(o, v, q)
    result = execute(
        program, dict(zip(("bar_boo", "bar_bov", "bar_bvv"), seeds))
    ).outputs
    np.testing.assert_array_equal(result["bar_bmo"], embed(seeds, o, o + v))
    assert all(len(node.spec.indices) <= 3 for node in program.live_nodes)
    assert not any(node.op == "einsum" for node in program.live_nodes)


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_SOURCE_RESPONSE_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if not cache or not compiler:
        pytest.skip("requires ccache and C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-source-metric-response")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    cuda = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    obj, out = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
            "-c",
            str(ROOT / "tests/native/df_source_metric_response_probe.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=90,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            str(obj),
            str(library),
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(library.parent),
            "-Wl,-rpath," + str(cuda / "lib64"),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    dll = ct.CDLL(str(out))
    call = dll.df_source_metric_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_void_p,
        dp,
        ct.c_size_t,
        ct.c_double,
        ct.POINTER(dp),
        ct.POINTER(dp),
        ct.c_size_t,
        ct.c_size_t,
        ct.c_int,
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run(
    call: typing.Any,
    source: typing.Any,
    c: np.ndarray,
    o: int,
    seeds: list[np.ndarray],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
    failure: int = 0,
    threshold: float = 1e-10,
) -> tuple:
    n, q = source.nbf, source.naux
    outputs = [np.full(s, np.nan) for s in ((n, n, q), (n, n), (q, q))]
    dp = ct.POINTER(ct.c_double)
    counts = np.zeros(14, dtype=np.uintp)
    error = ct.create_string_buffer(1024)
    status = call(
        source._handle,
        c.ctypes.data_as(dp),
        o,
        threshold,
        (dp * 3)(*(a.ctypes.data_as(dp) for a in seeds)),
        (dp * 3)(*(a.ctypes.data_as(dp) for a in outputs)),
        budget,
        caller_bytes,
        failure,
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, error.value.decode(), outputs, counts


def fixture(name: str, duplicate: bool) -> tuple:
    from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments

    meta, arrays = load_fixture(name)
    args = source_arguments(meta)
    indices = list(range(len(arrays["metric"])))
    if duplicate:
        assert args["auxiliary_basis"][0].angular_momentum == 0
        args["auxiliary_basis"] += (args["auxiliary_basis"][0],)
        indices.append(0)
    c = np.ascontiguousarray(arrays["conventional_C"])
    o = meta["records"]["conventional"]["electron_count"] // 2
    raw = arrays["raw_three_center"][:, :, indices]
    metric = arrays["metric"][np.ix_(indices, indices)]
    rng = np.random.default_rng(1765)
    q, n = len(indices), len(c)
    seeds = [rng.normal(size=s) for s in ((q, o, o), (q, o, n - o), (q, n - o, n - o))]
    return args, c, o, raw, metric, seeds


def root(metric: np.ndarray) -> np.ndarray:
    eig, u = np.linalg.eigh(metric)
    kept = eig > eig.max() * 1e-10
    return (u[:, kept] / np.sqrt(eig[kept])) @ u[:, kept].T


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_physical_source_response_and_fixed_rank_differences(
    probe: typing.Any,
    name: str,
    duplicate: bool,
) -> None:
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, raw, metric, seeds = fixture(name, duplicate)
    with NativeSource(**args) as source:
        status, error, outputs, counts = run(probe, source, c, o, seeds)
    assert status == 0, error
    da, dc, dm = outputs
    b = embed(seeds, o, len(c))
    w = root(metric)
    expected_a = np.einsum("pqQ,mp,nq,PQ->mnP", b, c, c, w, optimize=True)
    expected_c = np.einsum("pqQ,nq,mnP,PQ->mp", b, c, raw, w, optimize=True)
    expected_c += np.einsum("qpQ,nq,nmP,PQ->mp", b, c, raw, w, optimize=True)
    np.testing.assert_allclose(da, expected_a, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(dc, expected_c, atol=3e-10, rtol=3e-10)
    np.testing.assert_array_equal(dm, dm.T)

    def objective(a: np.ndarray, coeff: np.ndarray, m: np.ndarray) -> float:
        return float(
            np.einsum("pqQ,mp,nq,mnP,PQ->", b, coeff, coeff, a, root(m), optimize=True)
        )

    # Exact isospectral rotations move retained/discarded subspaces without
    # lifting a discarded eigenvalue through the cutoff. Eigenvalue directions
    # test the retained response separately. This is an independent FD oracle,
    # not another implementation of the generated spectral divided differences.
    eig, u = np.linalg.eigh(metric)
    rng = np.random.default_rng(1796)
    directions = [("raw", rng.normal(size=raw.shape)), ("C", rng.normal(size=c.shape))]
    step = 2e-5
    for kind, d in directions:

        def value(t: float, d: np.ndarray = d, kind: str = kind) -> float:
            return objective(
                raw + t * d if kind == "raw" else raw,
                c + t * d if kind == "C" else c,
                metric,
            )

        fd = (
            value(-2 * step) - 8 * value(-step) + 8 * value(step) - value(2 * step)
        ) / (12 * step)
        analytic = float(np.sum((da if kind == "raw" else dc) * d))
        np.testing.assert_allclose(analytic, fd, atol=2e-8, rtol=2e-8)
    j = int(np.flatnonzero(eig > eig.max() * 1e-10)[0])
    i = 0 if duplicate else min(j + 1, len(eig) - 1)
    assert i != j
    generator = np.zeros_like(metric)
    generator[i, j], generator[j, i] = 1, -1
    derivative = u @ (generator @ np.diag(eig) - np.diag(eig) @ generator) @ u.T

    def rotated(t: float) -> np.ndarray:
        r = np.eye(len(eig))
        r[i, i] = r[j, j] = np.cos(t)
        r[i, j], r[j, i] = np.sin(t), -np.sin(t)
        return (u @ r * eig) @ (u @ r).T

    fd = (
        objective(raw, c, rotated(-2 * step))
        - 8 * objective(raw, c, rotated(-step))
        + 8 * objective(raw, c, rotated(step))
        - objective(raw, c, rotated(2 * step))
    ) / (12 * step)
    np.testing.assert_allclose(np.sum(dm * derivative), fd, atol=2e-8, rtol=2e-8)
    direction = np.outer(u[:, j], u[:, j]) * eig[j]
    fd = (
        objective(raw, c, metric - 2 * step * direction)
        - 8 * objective(raw, c, metric - step * direction)
        + 8 * objective(raw, c, metric + step * direction)
        - objective(raw, c, metric + 2 * step * direction)
    ) / (12 * step)
    np.testing.assert_allclose(np.sum(dm * direction), fd, atol=2e-8, rtol=2e-8)
    n, q = len(c), len(metric)
    assert tuple(counts[5:11]) == (
        2 * n,
        2 * n * n * q,
        n,
        n * n * q,
        3 * n + 5,
        6 * n**3 * q + 2 * n * n * q * q,
    )
    assert counts[1] == counts[12] > 0
    assert counts[13] == q - int(duplicate)


@pytest.mark.parametrize(
    "failure,message",
    [
        (1, "retained physical state"),
        (2, "identity mismatch"),
        (3, "retained physical state"),
        (4, "consume failure"),
        (5, "finish failure"),
        (6, "seed shape/value"),
    ],
)
def test_source_response_failure_is_transactional(
    probe: typing.Any, failure: int, message: str
) -> None:
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, _, _, seeds = fixture("h2", False)
    with NativeSource(**args) as source:
        status, error, outputs, _ = run(probe, source, c, o, seeds, failure=failure)
    assert status != 0 and message in error
    assert all(np.isnan(a).all() for a in outputs)


@pytest.mark.parametrize("failure", [0, 8, 9])
def test_source_response_budget_and_callback_recovery(
    probe: typing.Any, failure: int
) -> None:
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, _, _, seeds = fixture("h2", True)
    with NativeSource(**args) as source:
        status, error, expected, counts = run(
            probe, source, c, o, seeds, failure=failure
        )
        assert status == 0, error
        capacity = int(counts[0])
        status, error, outputs, _ = run(probe, source, c, o, seeds, budget=capacity - 1)
        assert status != 0 and "budget" in error
        assert all(np.isnan(a).all() for a in outputs)
        status, error, outputs, adjusted = run(
            probe, source, c, o, seeds, budget=capacity + 17, caller_bytes=17
        )
        assert status == 0, error
        assert adjusted[0] == capacity + 17
        for want, actual in zip(expected, outputs, strict=True):
            np.testing.assert_allclose(actual, want, atol=1e-12, rtol=1e-12)


def test_source_response_refuses_cutoff_crossing(probe: typing.Any) -> None:
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, _, metric, seeds = fixture("water", False)
    eig = np.linalg.eigvalsh(metric)
    with NativeSource(**args) as source:
        status, error, outputs, _ = run(
            probe, source, c, o, seeds, threshold=float(eig[1] / eig[-1])
        )
    assert status != 0 and "rank crossing" in error
    assert all(np.isnan(a).all() for a in outputs)


def test_source_response_refuses_nonfinite_seed(probe: typing.Any) -> None:
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, _, _, seeds = fixture("h2", False)
    seeds[0].flat[0] = np.nan
    with NativeSource(**args) as source:
        status, error, outputs, _ = run(probe, source, c, o, seeds)
    assert status != 0 and "seed shape/value" in error
    assert all(np.isnan(a).all() for a in outputs)


def test_native_ccsd_lambda_source_composition_against_determinant_oracle(
    probe: typing.Any,
) -> None:
    """Cold native composition, with exact two-electron energy differences."""
    from tools.generativeqc_cc.oracle import DeterminantOracle
    from tools.generativeqc_posthf.fixtures import load_fixture
    from tools.generativeqc_posthf.sources import NativeSource

    args, c, o, raw, metric, _ = fixture("h2", True)
    assert o == 1
    _, arrays = load_fixture("h2")
    f = np.ascontiguousarray(c.T @ arrays["conventional_F"] @ c)
    with NativeSource(**args) as source:
        status, error, outputs, _ = run(probe, source, c, o, [f, f, f], failure=16)
    assert status == 0, error
    da, dc, dm = outputs
    rng = np.random.default_rng(1764)
    d_raw, d_c = rng.normal(size=raw.shape), rng.normal(size=c.shape)
    d_metric = metric * 0.2  # preserves the duplicate-shell null space exactly
    analytic = np.sum(da * d_raw) + np.sum(dc * d_c) + np.sum(dm * d_metric)

    def energy(t: float) -> float:
        coeff = c + t * d_c
        b = np.einsum(
            "mp,nq,mnP,PQ->pqQ",
            coeff,
            coeff,
            raw + t * d_raw,
            root(metric + t * d_metric),
            optimize=True,
        )
        b = (b + b.transpose(1, 0, 2)) / 2
        g = np.einsum("pqQ,rsQ->pqrs", b, b)
        oracle = DeterminantOracle(f, g, o)
        return float(np.linalg.eigvalsh(oracle.hnormal)[0])

    step = 2e-5
    fd = (
        energy(-2 * step) - 8 * energy(-step) + 8 * energy(step) - energy(2 * step)
    ) / (12 * step)
    np.testing.assert_allclose(analytic, fd, atol=3e-9, rtol=3e-8)
