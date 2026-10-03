"""DF source staging against an independently contracted complete expression."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from math import prod
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.df_mo_source import build_df_mo_source_program
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
