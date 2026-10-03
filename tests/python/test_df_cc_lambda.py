"""DF Lambda composition, native solves and physical factor energy derivatives."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.df_equations import build_df_virtual_response_programs
from generativeqc_compiler.cc.df_lambda import (
    RETAINED_PARAMETERS,
    retained_response_programs,
)
from generativeqc_compiler.cc.lambda_equations import (
    build_lambda_programs,
    build_parameter_vjp,
)
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "foo",
    "fov",
    "fvv",
    "ovov",
    "ovvo",
    "oovv",
    "ovvv",
    "ovoo",
    "oooo",
    "vvvv",
    "d1",
    "d2",
    "t1",
    "t2",
    "bov",
    "bvv",
)
PARAMETERS = ("foo", "fov", "fvv", "ovov", "ovvo", "oovv", "ovoo", "oooo")


def case(
    o: int, v: int, q: int = 3
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(158 + o * 10 + v)
    b = rng.normal(scale=0.04, size=(q, o + v, o + v))
    b = (b + b.transpose(0, 2, 1)) / 2
    f = np.diag(np.r_[np.linspace(-1.3, -0.7, o), np.linspace(0.4, 1.1, v)])
    f[:o, o:] = rng.normal(scale=0.002, size=(o, v))
    f[o:, :o] = f[:o, o:].T
    return b, f, feeds(b, f, o)


def feeds(b: np.ndarray, f: np.ndarray, o: int) -> dict[str, np.ndarray]:
    g = np.einsum("Qpq,Qrs->pqrs", b, b)
    eps = np.diag(f)
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    result = dense_feeds(
        f, g, f[:o, o:] / d1, g[:o, o:, :o, o:].transpose(0, 2, 1, 3) / d2
    )
    result.update(d1=d1, d2=d2, bov=b[:, :o, o:], bvv=b[:, o:, o:])
    return {key: np.ascontiguousarray(value) for key, value in result.items()}


@pytest.mark.parametrize("o,v", [(1, 2), (2, 3), (3, 2)])
def test_retained_and_auxiliary_actions_reproduce_full_derivatives(
    o: int, v: int
) -> None:
    _, _, arrays = case(o, v)
    rng = np.random.default_rng(1785)
    seeds = {
        "bar_correlation_energy": np.array(-1.0),
        "bar_singles_residual": rng.normal(size=(o, v)),
        "bar_doubles_residual": rng.normal(size=(o, o, v, v)),
    }
    retained = retained_response_programs(o, v)
    dense = build_lambda_programs(o, v, form="expanded")
    virtual = build_df_virtual_response_programs(o, v)
    expected = execute(dense.residual_vjp.program, {**arrays, **seeds}).outputs
    virtual_seeds = {
        "bar_df_virtual_singles": seeds["bar_singles_residual"],
        "bar_df_virtual_doubles": seeds["bar_doubles_residual"],
    }
    for prefix in ("", "independent_"):
        actual = execute(retained[prefix + "transpose"], {**arrays, **seeds}).outputs
        for Q in range(len(arrays["bov"])):
            row = execute(
                virtual.amplitude_vjp.program,
                {
                    **arrays,
                    **virtual_seeds,
                    "bov": arrays["bov"][Q],
                    "bvv": arrays["bvv"][Q],
                },
            ).outputs
            for key in actual:
                actual[key] = actual[key] + row[key]
        for key in expected:
            np.testing.assert_allclose(
                actual[key], expected[key], atol=3e-12, rtol=3e-12
            )
        rhs = execute(retained[prefix + "rhs"], {**arrays, **seeds}).outputs
        expected_rhs = execute(dense.energy_vjp.program, {**arrays, **seeds}).outputs
        for key in rhs:
            np.testing.assert_allclose(
                rhs[key], expected_rhs[key], atol=3e-12, rtol=3e-12
            )
    for name in RETAINED_PARAMETERS:
        actual = execute(retained["parameter_" + name], {**arrays, **seeds}).outputs
        reference = execute(
            build_parameter_vjp(dense.primal, name).program, {**arrays, **seeds}
        ).outputs
        np.testing.assert_allclose(
            actual["bar_" + name], reference["bar_" + name], atol=3e-12, rtol=3e-12
        )


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_LAMBDA_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if not cache or not compiler:
        pytest.skip("requires ccache and C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("df-lambda")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj, out = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(ROOT / "tests/native/df_cc_lambda_probe.cpp"),
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
    call = dll.df_cc_lambda_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_int,
        ct.c_size_t,
        ct.POINTER(dp),
        dp,
        dp,
        ct.POINTER(dp),
        dp,
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run(
    call: typing.Any,
    arrays: dict[str, np.ndarray],
    *,
    df: bool = True,
    source: bool = False,
    budget: int = 1 << 30,
) -> tuple:
    o, v = arrays["t1"].shape
    q = len(arrays["bov"])
    inputs = [np.ascontiguousarray(arrays[name]) for name in FIELDS]
    rng = np.random.default_rng(1764)
    seeds = [
        rng.normal(scale=0.01, size=(o, v)),
        rng.normal(scale=0.01, size=(o, o, v, v)),
    ]
    dp = ct.POINTER(ct.c_double)
    ptrs = (dp * len(inputs))(*(x.ctypes.data_as(dp) for x in inputs))
    shapes = [
        (o, v),
        (o, o, v, v),
        (o, v),
        (o, o, v, v),
        *(arrays[k].shape for k in PARAMETERS),
        (q, o, v) if df else (0,),
        (q, v, v) if df else (0,),
        (0,) if df else (o, v, v, v),
        (0,) if df else (v, v, v, v),
    ]
    result = [np.full(shape, np.nan) for shape in shapes]
    outputs = (dp * len(result))(*(x.ctypes.data_as(dp) for x in result))
    values = np.full(6, np.nan)
    counts = np.zeros(10, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = call(
        o,
        v,
        q,
        int(df) + 2 * int(source),
        budget,
        ptrs,
        *(x.ctypes.data_as(dp) for x in seeds),
        outputs,
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, result, values, counts, error.value.decode()


@pytest.mark.parametrize(
    "o,v,source", [(1, 2, False), (2, 3, False), (3, 2, False), (2, 3, True)]
)
def test_native_df_lambda_and_parameters_match_complete_integral_path(
    probe: typing.Any, o: int, v: int, source: bool
) -> None:
    _, _, arrays = case(o, v)
    status, result, values, counts, error = run(probe, arrays, source=source)
    assert status == 0, error
    dense_status, dense, dense_values, _, error = run(
        probe, arrays, df=False, source=source
    )
    assert dense_status == 0, error
    np.testing.assert_allclose(values[0], dense_values[0], atol=3e-12, rtol=0)
    for actual, expected in zip(result[:12], dense[:12], strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
    assert np.max(values[1:]) < 1e-9
    # Every Q appears once in the primal replay, each GMRES action, the
    # independent transpose audit, and final factor publication.
    assert counts[7] == (counts[1] + 3) * len(arrays["bov"])
    assert counts[8] > 0 and counts[9] > 0
    # Compare virtual-factor cotangents to an independent explicit Gram chain.
    bov, bvv = arrays["bov"], arrays["bvv"]
    want_bov = np.einsum("iabc,Qbc->Qia", dense[14], bvv)
    want_bvv = (
        np.einsum("iabc,Qia->Qbc", dense[14], bov)
        + np.einsum("abcd,Qcd->Qab", dense[15], bvv)
        + np.einsum("abcd,Qab->Qcd", dense[15], bvv)
    )
    want_bvv = (want_bvv + want_bvv.transpose(0, 2, 1)) / 2
    np.testing.assert_allclose(result[12], want_bov, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(result[13], want_bvv, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("o,v", [(2, 3), (1, 3)])
def test_native_response_matches_resolved_factor_energy_derivative(
    probe: typing.Any,
    o: int,
    v: int,
) -> None:
    """Differentiate every physical DF block, including all retained Gram terms."""
    b, f, arrays = case(o, v)
    status, result, _, _, error = run(probe, arrays)
    assert status == 0, error
    direction = np.random.default_rng(1765).normal(scale=0.02, size=b.shape)
    direction = (direction + direction.transpose(0, 2, 1)) / 2
    dg = np.einsum("Qpq,Qrs->pqrs", direction, b) + np.einsum(
        "Qpq,Qrs->pqrs", b, direction
    )
    derivative_feeds = dense_feeds(
        np.zeros_like(f), dg, np.zeros((o, v)), np.zeros((o, o, v, v))
    )
    analytic = sum(
        float(np.sum(result[index + 4] * derivative_feeds[name]))
        for index, name in enumerate(PARAMETERS)
    )
    analytic += float(
        np.sum(result[12] * direction[:, :o, o:])
        + np.sum(result[13] * direction[:, o:, o:])
    )
    step = 1e-4
    energies = []
    for sign in (-1, 1):
        displaced = b + sign * step * direction
        status, _, values, _, error = run(probe, feeds(displaced, f, o))
        assert status == 0, error
        if o == 1:
            # For two electrons CCSD is exact. This independent determinant
            # Hamiltonian uses fermionic operators rather than the CC inventory
            # or its AD, so its finite difference is a separate response oracle.
            g = np.einsum("Qpq,Qrs->pqrs", displaced, displaced)
            oracle = DeterminantOracle(f, g, o)
            exact = float(np.linalg.eigvalsh(oracle.hnormal)[0])
            np.testing.assert_allclose(values[0], exact, atol=3e-11, rtol=0)
            energies.append(exact)
        else:
            energies.append(values[0])
    np.testing.assert_allclose(
        analytic, (energies[1] - energies[0]) / (2 * step), atol=3e-10, rtol=3e-6
    )


def test_failed_native_response_has_no_publication(probe: typing.Any) -> None:
    _, _, arrays = case(2, 3)
    status, reference, _, counts, error = run(probe, arrays)
    assert status == 0, error
    budget = int(counts[2])
    status, exact, _, _, error = run(probe, arrays, budget=budget)
    assert status == 0, error
    for actual, expected in zip(exact, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for refused in (1, budget - 1):
        status, result, values, _, error = run(probe, arrays, budget=refused)
        assert status != 0 and "DF Lambda" in error and "budget" in error
        assert np.isnan(values).all() and all(np.isnan(x).all() for x in result)
