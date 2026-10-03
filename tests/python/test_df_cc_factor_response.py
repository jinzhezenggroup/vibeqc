"""Physical factor pullback gates independent of the compiler's Gram AD."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import sys
import typing
from math import prod
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.df_source import (
    BLOCK_FACTORS,
    PHYSICAL_FACTORS,
    retained_factor_vjp,
)
from generativeqc_compiler.tensor import execute
from test_df_cc_lambda import case as lambda_case
from test_df_cc_lambda import feeds as lambda_feeds
from test_df_cc_lambda import probe as lambda_probe  # noqa: F401 -- pytest fixture
from test_df_cc_lambda import run as run_lambda

from tools.generativeqc_cc.oracle import DeterminantOracle

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    *PHYSICAL_FACTORS,
    *("bar_" + name for name in (*BLOCK_FACTORS, "bov", "bvv")),
)


@pytest.mark.parametrize("o,v,q", [(9, 221, 488), (21, 243, 666)])
def test_large_factor_response_has_bounded_work_and_intermediates(
    o: int, v: int, q: int
) -> None:
    program = retained_factor_vjp(o, v, q)
    work = 0
    for node in program.live_nodes:
        if node.op == "input":
            continue
        assert len(node.spec.indices) <= 3
        if node.op == "einsum":
            extents = {
                label: extent
                for operand, labels in zip(
                    node.inputs, node.attrs["labels"], strict=True
                )
                for label, extent in zip(labels, operand.spec.shape, strict=True)
            }
            work += prod(extents.values())
    assert work == 2 * q * (3 * o**2 * v**2 + o**3 * v + o**4)


def test_factor_codegen_is_deterministic_without_runtime_imports(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-S",
        str(ROOT / "tools/generate_df_cc_source.py"),
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def case(o: int, v: int, q: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Nonsymmetric cotangents detect erroneous block or pair multiplicities."""
    rng = np.random.default_rng(1792 + 100 * o + 10 * v + q)
    b = rng.normal(size=(q, o + v, o + v))
    b = (b + b.transpose(0, 2, 1)) / 2
    arrays = {"boo": b[:, :o, :o], "bov": b[:, :o, o:], "bvv": b[:, o:, o:]}
    for name in BLOCK_FACTORS:
        arrays["bar_" + name] = rng.normal(
            size=tuple(o if c == "o" else v for c in name)
        )
    arrays["bar_bov"] = rng.normal(size=(q, o, v))
    arrays["bar_bvv"] = rng.normal(size=(q, v, v))
    return b, {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def dense_reference(
    b: np.ndarray, arrays: dict[str, np.ndarray], o: int
) -> dict[str, np.ndarray]:
    """Embed independent block seeds in a complete chemists' ERI cotangent."""
    n = b.shape[1]
    bar_g = np.zeros((n,) * 4)
    for name in BLOCK_FACTORS:
        slices = tuple(slice(0, o) if c == "o" else slice(o, n) for c in name)
        bar_g[slices] += arrays["bar_" + name]
    full = np.einsum("pqrs,Qrs->Qpq", bar_g, b) + np.einsum("rspq,Qrs->Qpq", bar_g, b)
    full[:, :o, o:] += arrays["bar_bov"]
    full[:, o:, o:] += arrays["bar_bvv"]
    full = (full + full.transpose(0, 2, 1)) / 2
    return {
        "bar_boo": full[:, :o, :o],
        "bar_bov": 2 * full[:, :o, o:],
        "bar_bvv": full[:, o:, o:],
    }


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 5)])
def test_generated_pullback_matches_independent_dense_gram_and_difference(
    o: int, v: int, q: int
) -> None:
    b, arrays = case(o, v, q)
    program = retained_factor_vjp(o, v, q)
    result = execute(program, arrays).outputs
    expected = dense_reference(b, arrays, o)
    for name in expected:
        np.testing.assert_allclose(result[name], expected[name], atol=3e-12, rtol=2e-13)
    direction = np.random.default_rng(1765).normal(size=b.shape)
    direction = (direction + direction.transpose(0, 2, 1)) / 2
    analytic = sum(
        float(np.sum(result["bar_" + name] * delta))
        for name, delta in zip(
            PHYSICAL_FACTORS,
            (direction[:, :o, :o], direction[:, :o, o:], direction[:, o:, o:]),
            strict=True,
        )
    )

    def objective(factors: np.ndarray) -> float:
        g = np.einsum("Qpq,Qrs->pqrs", factors, factors)
        value = 0.0
        for name in BLOCK_FACTORS:
            slices = tuple(slice(0, o) if c == "o" else slice(o, o + v) for c in name)
            value += float(np.sum(g[slices] * arrays["bar_" + name]))
        return value + float(
            np.sum(factors[:, :o, o:] * arrays["bar_bov"])
            + np.sum(factors[:, o:, o:] * arrays["bar_bvv"])
        )

    step = 1e-4
    numerical = (objective(b + step * direction) - objective(b - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(analytic, numerical, atol=3e-9, rtol=3e-10)
    # Only borrowed four-index seed blocks are live; every generated result or
    # scratch tensor has rank <= 3, independent of the virtual extent.
    assert all(
        len(node.spec.indices) <= 3 for node in program.live_nodes if node.op != "input"
    )


@pytest.fixture(scope="module")
def factor_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_LAMBDA_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if not cache or not compiler:
        pytest.skip("requires ccache and C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-factor-response")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj, out = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(ROOT / "tests/native/df_cc_factor_response_probe.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            str(obj),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(out))
    call = dll.df_cc_factor_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(dp),
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(dp),
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run_factor(
    call: typing.Any,
    arrays: dict[str, np.ndarray],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
) -> tuple:
    q, o, v = arrays["bov"].shape
    inputs = [np.ascontiguousarray(arrays[name]) for name in FIELDS]
    outputs = [np.full_like(arrays[name], np.nan) for name in PHYSICAL_FACTORS]
    counts = np.zeros(6, dtype=np.uintp)
    error = ct.create_string_buffer(1024)
    dp = ct.POINTER(ct.c_double)
    status = call(
        o,
        v,
        q,
        (dp * len(inputs))(*(x.ctypes.data_as(dp) for x in inputs)),
        budget,
        caller_bytes,
        (dp * len(outputs))(*(x.ctypes.data_as(dp) for x in outputs)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, outputs, counts, error.value.decode()


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 5)])
def test_native_complete_factor_pullback(
    factor_probe: typing.Any, o: int, v: int, q: int
) -> None:
    b, arrays = case(o, v, q)
    status, outputs, counts, error = run_factor(factor_probe, arrays, caller_bytes=1234)
    assert status == 0, error
    expected = dense_reference(b, arrays, o)
    for name, result in zip(PHYSICAL_FACTORS, outputs, strict=True):
        np.testing.assert_allclose(
            result, expected["bar_" + name], atol=3e-12, rtol=2e-13
        )
    assert counts[2] == sum(arrays[name].nbytes for name in FIELDS)
    assert counts[3] == sum(value.nbytes for value in outputs)
    assert counts[0] == 1234 + counts[1] + counts[2] + counts[3]
    assert counts[4] == 2 * q * (3 * o**2 * v**2 + o**3 * v + o**4)
    assert counts[5] > 0
    budget = int(counts[0])
    assert run_factor(factor_probe, arrays, budget=budget, caller_bytes=1234)[0] == 0
    for refused in (1, budget - 1):
        status, outputs, _, error = run_factor(
            factor_probe, arrays, budget=refused, caller_bytes=1234
        )
        assert status != 0 and "budget" in error
        assert all(np.isnan(x).all() for x in outputs)


@pytest.mark.parametrize("failure", ["nan", "asymmetry", "overflow"])
def test_factor_failures_never_publish(factor_probe: typing.Any, failure: str) -> None:
    _, arrays = case(2, 3, 4)
    if failure == "nan":
        arrays["bar_ovov"].flat[0] = np.nan
    elif failure == "asymmetry":
        arrays["boo"][0, 0, 1] += 1
    else:
        arrays["boo"][:] = 10
        arrays["bar_oooo"][:] = 1e308
    status, outputs, _, error = run_factor(factor_probe, arrays)
    assert status != 0, error
    assert all(np.isnan(x).all() for x in outputs)


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3)])
def test_native_lambda_and_factor_response_match_resolved_energy(
    factor_probe: typing.Any,
    lambda_probe: typing.Any,  # noqa: F811 -- pytest fixture
    o: int,
    v: int,
) -> None:
    """Exercise native composition through all factors, with an FCI oracle for o=1."""
    b, f, arrays = lambda_case(o, v)
    status, response, _, _, error = run_lambda(lambda_probe, arrays)
    assert status == 0, error
    inputs = {
        "boo": b[:, :o, :o],
        "bov": b[:, :o, o:],
        "bvv": b[:, o:, o:],
        **{"bar_" + name: response[7 + i] for i, name in enumerate(BLOCK_FACTORS)},
        "bar_bov": response[12],
        "bar_bvv": response[13],
    }
    status, factors, _, error = run_factor(factor_probe, inputs)
    assert status == 0, error
    delta = np.random.default_rng(1764).normal(scale=0.03, size=b.shape)
    delta = (delta + delta.transpose(0, 2, 1)) / 2
    analytic = sum(
        float(np.sum(bar * direction))
        for bar, direction in zip(
            factors, (delta[:, :o, :o], delta[:, :o, o:], delta[:, o:, o:]), strict=True
        )
    )
    step = 1e-4
    energies = []
    for sign in (-1, 1):
        displaced = b + sign * step * delta
        status, _, values, _, error = run_lambda(
            lambda_probe, lambda_feeds(displaced, f, o)
        )
        assert status == 0, error
        energy = values[0]
        if o == 1:
            oracle = DeterminantOracle(
                f, np.einsum("Qpq,Qrs->pqrs", displaced, displaced), o
            )
            energy = float(np.linalg.eigvalsh(oracle.hnormal)[0])
            np.testing.assert_allclose(values[0], energy, atol=3e-11, rtol=0)
        energies.append(energy)
    np.testing.assert_allclose(
        analytic, (energies[1] - energies[0]) / (2 * step), atol=3e-10, rtol=3e-6
    )
