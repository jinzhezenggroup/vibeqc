"""DF source staging against an independently contracted complete expression."""

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
from generativeqc_compiler.method.df_mo_source import (
    build_df_mo_source_program,
    build_df_mo_source_vjp_program,
)
from generativeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("n,q", [(2, 3), (5, 4), (3, 7)])
def test_staged_source_matches_direct_orbital_and_metric_contraction(
    n: int, q: int
) -> None:
    rng = np.random.default_rng(1770 + n + q)
    raw = rng.normal(size=(n, n, q))
    raw = (raw + raw.transpose(1, 0, 2)) / 2
    c = rng.normal(size=(n, n))
    root = rng.normal(size=(q, q))
    root = (root + root.T) / 2
    actual = execute(
        build_df_mo_source_program(n, q),
        {"raw_three_center": raw, "coefficients": c, "inverse_root": root},
    ).outputs
    expected = np.einsum("mp,nq,mnP,PQ->pqQ", c, c, raw, root, optimize=False)
    np.testing.assert_allclose(actual["whitened"], expected, atol=3e-12, rtol=2e-14)
    np.testing.assert_allclose(
        actual["whitened"], actual["whitened"].transpose(1, 0, 2), atol=3e-12, rtol=0
    )


def test_source_schedule_never_expands_both_orbital_reductions() -> None:
    for n, q in ((230, 488), (264, 666)):
        program = build_df_mo_source_program(n, q)
        work = 0
        contractions = [node for node in program.live_nodes if node.op == "einsum"]
        assert len(contractions) == 3
        for node in contractions:
            extents = {
                label: index.space.size
                for child, labels in zip(node.inputs, node.attrs["labels"], strict=True)
                for label, index in zip(labels, child.spec.indices, strict=True)
            }
            work += prod(extents.values())
        assert work == 2 * n**3 * q + n**2 * q**2
        assert all(len(node.spec.indices) <= 3 for node in program.live_nodes)


@pytest.mark.parametrize("n,q", [(0, 2), (2, 0), (True, 2), (2, 2.5)])
def test_source_rejects_invalid_dimensions(n: int, q: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_df_mo_source_program(n, q)


@pytest.fixture(scope="module")
def native_source_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the generated traversal with the real scalar CPU GEMM backend."""
    directory = tmp_path_factory.mktemp("df-mo-source")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and ccache required for native DF source probe")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    header = directory / "df_mo_source_generated.hpp"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "tools/generate_df_mo_source.py"),
        "--output",
        str(header),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    original = header.read_bytes()
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert original == header.read_bytes()
    objects = []
    for i, source in enumerate(
        (
            ROOT / "tests/native/df_mo_source_probe.cpp",
            ROOT / "src/tensor/cpu_linalg.cpp",
        )
    ):
        obj = directory / f"source-{i}.o"
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O2",
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "include"),
                "-I" + str(directory),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
            check=True,
            capture_output=True,
            timeout=90,
        )
        objects.append(str(obj))
    executable = directory / "source-probe"
    subprocess.run(
        [compiler, *objects, "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("n,q", [(1, 1), (2, 3), (5, 4), (3, 7)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_native_source_layout_and_observed_work(
    native_source_probe: Path, n: int, q: int, symmetric: bool
) -> None:
    # Non-symmetric metric/raw inputs catch transposition errors that a physical
    # symmetric source would mask. Production metric factorization is unchanged.
    rng = np.random.default_rng(1763 + n + q)
    raw, c, root = (rng.normal(size=shape) for shape in ((n, n, q), (n, n), (q, q)))
    if symmetric:
        raw = (raw + raw.transpose(1, 0, 2)) / 2
        root = (root + root.T) / 2
    data = f"{n} {q}\n" + "\n".join(
        " ".join(map(repr, values.ravel().tolist())) for values in (raw, c, root)
    )
    result = subprocess.run(
        [str(native_source_probe)],
        input=data,
        text=True,
        check=True,
        capture_output=True,
        timeout=15,
    )
    counts, transformed, whitened = result.stdout.splitlines()
    assert tuple(map(int, counts.split())) == (n, n + 2, 2 * n**3 * q + n**2 * q**2)
    expected_a = np.einsum("mp,nq,mnP->pqP", c, c, raw, optimize=False)
    expected_b = np.einsum("mp,nq,mnP,PQ->pqQ", c, c, raw, root, optimize=False)
    np.testing.assert_allclose(
        np.fromstring(transformed, sep=" ").reshape(n, n, q),
        expected_a,
        atol=3e-12,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        np.fromstring(whitened, sep=" ").reshape(n, n, q),
        expected_b,
        atol=3e-12,
        rtol=2e-14,
    )


@pytest.mark.parametrize("n,q", [(230, 488), (264, 666)])
def test_native_work_query_at_large_target_shapes(
    native_source_probe: Path, n: int, q: int
) -> None:
    result = subprocess.run(
        [str(native_source_probe), "--work"],
        input=f"{n} {q}\n",
        text=True,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert tuple(map(int, result.stdout.split())) == (
        n**2 * q,
        n,
        n + 2,
        2 * n**3 * q + n**2 * q**2,
    )


def _response_inputs(n: int, q: int, symmetric: bool) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1764 + 100 * n + q)
    raw, c, root, bar = (
        rng.normal(size=shape) for shape in ((n, n, q), (n, n), (q, q), (n, n, q))
    )
    if symmetric:
        raw = (raw + raw.transpose(1, 0, 2)) / 2
        root = (root + root.T) / 2
        bar = (bar + bar.transpose(1, 0, 2)) / 2
    return raw, c, root, bar


def _direct_source_response(
    raw: np.ndarray, c: np.ndarray, root: np.ndarray, bar: np.ndarray
) -> dict[str, np.ndarray]:
    """Independent complete expressions, without the compiler's staged AD."""
    return {
        "bar_raw_three_center": np.einsum(
            "mp,nq,PQ,pqQ->mnP", c, c, root, bar, optimize=False
        ),
        "bar_coefficients": np.einsum(
            "nq,mnP,PQ,pqQ->mp", c, raw, root, bar, optimize=False
        )
        + np.einsum("mp,mnP,PQ,pqQ->nq", c, raw, root, bar, optimize=False),
        "bar_inverse_root": np.einsum(
            "mp,nq,mnP,pqQ->PQ", c, c, raw, bar, optimize=False
        ),
    }


@pytest.mark.parametrize("n,q", [(2, 3), (5, 4), (3, 7)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_staged_source_response_matches_direct_and_directional_difference(
    n: int, q: int, symmetric: bool
) -> None:
    raw, c, root, bar = _response_inputs(n, q, symmetric)
    program = build_df_mo_source_vjp_program(n, q)
    actual = execute(
        program,
        {
            "raw_three_center": raw,
            "coefficients": c,
            "inverse_root": root,
            "bar_whitened": bar,
        },
    ).outputs
    expected = _direct_source_response(raw, c, root, bar)
    for name, result in actual.items():
        np.testing.assert_allclose(result, expected[name], atol=3e-11, rtol=3e-13)
    rng = np.random.default_rng(1765)
    deltas = [rng.normal(size=x.shape) for x in (raw, c, root)]
    analytic = sum(
        float(np.sum(actual["bar_" + name] * delta))
        for name, delta in zip(
            ("raw_three_center", "coefficients", "inverse_root"), deltas, strict=True
        )
    )

    def objective(sign: int, step: float) -> float:
        a, coefficients, w = (
            x + sign * step * d for x, d in zip((raw, c, root), deltas, strict=True)
        )
        return float(
            np.sum(
                np.einsum(
                    "mp,nq,mnP,PQ->pqQ",
                    coefficients,
                    coefficients,
                    a,
                    w,
                    optimize=False,
                )
                * bar
            )
        )

    # Perturb all three inputs together, including both appearances of C.
    step = 1e-5
    np.testing.assert_allclose(
        analytic,
        (objective(1, step) - objective(-1, step)) / (2 * step),
        atol=3e-7,
        rtol=3e-9,
    )
    assert all(len(node.spec.indices) <= 3 for node in program.live_nodes)
    work = 0
    for node in program.live_nodes:
        if node.op == "einsum":
            extents = {
                label: extent
                for child, labels in zip(node.inputs, node.attrs["labels"], strict=True)
                for label, extent in zip(labels, child.spec.shape, strict=True)
            }
            work += prod(extents.values())
    assert work == 6 * n**3 * q + 2 * n**2 * q**2


@pytest.mark.parametrize("n,q", [(1, 1), (2, 3), (5, 4), (3, 7)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_native_source_reverse_layout_reuse_and_complete_work(
    native_source_probe: Path, n: int, q: int, symmetric: bool
) -> None:
    arrays = _response_inputs(n, q, symmetric)
    data = f"{n} {q}\n" + "\n".join(
        " ".join(map(repr, x.ravel().tolist())) for x in arrays
    )
    completed = subprocess.run(
        [str(native_source_probe), "--response"],
        input=data,
        text=True,
        check=True,
        capture_output=True,
        timeout=15,
    )
    counts, *outputs = completed.stdout.splitlines()
    assert tuple(map(int, counts.split())) == (
        2 * n,
        3 * n + 5,
        6 * n**3 * q + 2 * n**2 * q**2,
        n,
    )
    expected = _direct_source_response(*arrays)
    for output, name in zip(
        outputs,
        ("bar_raw_three_center", "bar_coefficients", "bar_inverse_root"),
        strict=True,
    ):
        want = expected[name]
        np.testing.assert_allclose(
            np.fromstring(output, sep=" ").reshape(want.shape),
            want,
            atol=3e-11,
            rtol=3e-13,
        )


@pytest.mark.parametrize("n,q", [(230, 488), (264, 666)])
def test_native_source_response_work_query_at_large_target_shapes(
    native_source_probe: Path, n: int, q: int
) -> None:
    completed = subprocess.run(
        [str(native_source_probe), "--response-work"],
        input=f"{n} {q}\n",
        text=True,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert tuple(map(int, completed.stdout.split())) == (
        2 * n,
        2 * n**2 * q,
        n,
        n**2 * q,
        3 * n + 5,
        6 * n**3 * q + 2 * n**2 * q**2,
        2 * n**2 * q + 2 * n * q,
    )


@pytest.fixture(scope="module")
def cuda_source_response_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    """Use a qualified complete library, or compile the same standalone CUDA owner."""
    if os.environ.get("GENERATIVEQC_DF_MO_RESPONSE_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    nvcc = shutil.which("nvcc") or "/group/software/cuda-12.9.1/bin/nvcc"
    if not compiler or not cache or not Path(nvcc).exists():
        pytest.skip("requires ccache, C++ and CUDA compilers")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("cuda-source-response")
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_df_mo_source.py"),
            "--output",
            str(directory / "df_mo_source_generated.hpp"),
        ],
        check=True,
        capture_output=True,
    )
    cuda_root = Path(nvcc).resolve().parents[1]
    includes = [
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(directory),
        "-I" + str(cuda_root / "include"),
    ]
    complete_library = os.environ.get("GENERATIVEQC_DF_MO_RESPONSE_LIBRARY")
    objects = []
    for source, command in (
        (
            ROOT / "src/posthf/df_mo_response.cu",
            [
                nvcc,
                "-std=c++20",
                "-O2",
                "-lineinfo",
                "-arch=sm_120",
                "-Xcompiler=-fPIC",
            ],
        ),
        (
            ROOT / "tests/native/df_mo_response_probe.cpp",
            [compiler, "-std=c++20", "-O2", "-fPIC"],
        ),
    ):
        if complete_library and source.suffix == ".cu":
            continue
        obj = directory / (source.stem + ".o")
        subprocess.run(
            [cache, *command, *includes, "-c", str(source), "-o", str(obj)],
            check=True,
            capture_output=True,
            timeout=180,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(str(obj))
    if complete_library:
        full = Path(complete_library).resolve(strict=True)
        objects += [str(full), "-Wl,-rpath," + str(full.parent)]
    library = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            *objects,
            "-L" + str(cuda_root / "lib64"),
            "-lcublas",
            "-lcudart",
            "-Wl,-rpath," + str(cuda_root / "lib64"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    dll = ct.CDLL(str(library))
    call = dll.df_mo_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
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


def _run_cuda_source(
    call: typing.Any,
    arrays: tuple[np.ndarray, ...],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
    failure: int = 0,
) -> tuple:
    raw, c, root, _ = arrays
    n, _, q = raw.shape
    inputs = [np.ascontiguousarray(value) for value in arrays]
    outputs = [np.full_like(value, np.nan) for value in (raw, c, root)]
    counts = np.zeros(9, dtype=np.uintp)
    error = ct.create_string_buffer(1024)
    dp = ct.POINTER(ct.c_double)
    status = call(
        n,
        q,
        (dp * 4)(*(x.ctypes.data_as(dp) for x in inputs)),
        (dp * 3)(*(x.ctypes.data_as(dp) for x in outputs)),
        budget,
        caller_bytes,
        failure,
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, outputs, counts, error.value.decode()


@pytest.mark.parametrize("n,q", [(1, 1), (2, 3), (5, 4), (3, 7)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_cuda_streamed_source_response_matches_independent_complete_expression(
    cuda_source_response_probe: typing.Any, n: int, q: int, symmetric: bool
) -> None:
    arrays = _response_inputs(n, q, symmetric)
    status, outputs, counts, error = _run_cuda_source(
        cuda_source_response_probe, arrays, caller_bytes=1234
    )
    assert status == 0, error
    expected = _direct_source_response(*arrays)
    for actual, name in zip(
        outputs,
        ("bar_raw_three_center", "bar_coefficients", "bar_inverse_root"),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected[name], atol=3e-11, rtol=3e-13)
    assert tuple(counts[:6]) == (
        2 * n,
        2 * n * n * q,
        n,
        n * n * q,
        3 * n + 5,
        6 * n**3 * q + 2 * n * n * q * q,
    )
    assert counts[6] == (2 * n * n * q + 2 * n * q + n * n + q * q) * 8 + 4
    assert counts[8] == 4
    assert (
        _run_cuda_source(
            cuda_source_response_probe, arrays, budget=int(counts[7]), caller_bytes=1234
        )[0]
        == 0
    )
    status, failed, _, error = _run_cuda_source(
        cuda_source_response_probe, arrays, budget=int(counts[7]) - 1, caller_bytes=1234
    )
    assert status != 0 and "budget" in error
    assert all(np.isnan(x).all() for x in failed)


@pytest.mark.parametrize("failure", [1, 2, 3, 4, 5])
def test_cuda_source_response_callback_lifetime_and_ownership_failures(
    cuda_source_response_probe: typing.Any, failure: int
) -> None:
    status, outputs, _, error = _run_cuda_source(
        cuda_source_response_probe, _response_inputs(3, 5, False), failure=failure
    )
    assert status != 0, error
    assert all(np.isnan(x).all() for x in outputs)


def test_cuda_source_response_preserves_earlier_arithmetic_errors(
    cuda_source_response_probe: typing.Any,
) -> None:
    raw, c, root, bar = _response_inputs(2, 3, False)
    raw[:] = 1e308
    c[:] = 10
    bar[:] = 0
    status, outputs, _, error = _run_cuda_source(
        cuda_source_response_probe, (raw, c, root, bar)
    )
    assert status != 0 and "nonfinite" in error
    assert all(np.isnan(x).all() for x in outputs)
