"""Independent triples algebra and native DF occupied-tile qualification."""

from __future__ import annotations

import ctypes as ct
import itertools
import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import (
    PERMUTATIONS,
    df_panel_program,
    energy_scalar_program,
    moment_program,
)
from generativeqc_compiler.cc.triples import _LABELS, VP, triples_energy
from generativeqc_compiler.tensor import execute

from tools.generate_df_occupied_triples import header

ROOT = Path(__file__).resolve().parents[2]
GPU = os.environ.get("GENERATIVEQC_DF_TRIPLES_CUDA_TEST") == "1"


def case(o: int, v: int, q: int = 4) -> tuple[list[np.ndarray], np.ndarray]:
    """Independent physical Gram integrals with pair-symmetric doubles."""
    rng = np.random.default_rng(1763 + 100 * o + v + q)
    b = rng.normal(scale=0.2, size=(q, o + v, o + v))
    b = (b + b.transpose(0, 2, 1)) / 2
    eri = np.einsum("Qpq,Qrs->pqrs", b, b)
    t2 = rng.normal(scale=0.03, size=(o, o, v, v))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    inputs = [
        b[:, :o, o:],
        b[:, o:, o:],
        eri[:o, o:, :o, :o],
        eri[:o, o:, :o, o:],
        rng.normal(scale=0.01, size=(o, v)),
        rng.normal(scale=0.04, size=(o, v)),
        t2,
        np.linspace(-1.1, -0.6, o),
        np.linspace(0.2, 1.3, v),
    ]
    return [np.ascontiguousarray(x) for x in inputs], eri[:o, o:, o:, o:]


def reference(inputs: list[np.ndarray], ovvv: np.ndarray) -> float:
    _, _, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    return float(
        triples_energy(len(eo), len(ev), ovvv, ovoo, ovov, fov, t1, t2, eo, ev)
    )


@pytest.fixture(scope="module")
def scalar_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    directory = tmp_path_factory.mktemp("df-triples-scalar")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    (directory / "generated.hpp").write_text(header())
    program = energy_scalar_program()
    names = tuple(
        sorted(n.attrs["name"] for n in program.live_nodes if n.op == "input")
    )
    args = ",".join(f"p[{i}]" for i in range(len(names)))
    source = directory / "probe.cpp"
    source.write_text(
        '#include "generated.hpp"\nextern "C" double point(const double* p) {'
        f"double result=0; generativeqc::cc::triples::generated_df::energy_element({args},result);"
        "return result;}\n"
    )
    obj, library = directory / "probe.o", directory / "probe.so"
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
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(library))
    call = dll.point
    call.argtypes, call.restype = [ct.POINTER(ct.c_double)], ct.c_double
    return call, names


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2), (3, 4)])
def test_occupied_domain_preserves_all_permutations_and_multiplicities(
    scalar_probe: typing.Any, o: int, v: int
) -> None:
    inputs, ovvv = case(o, v)
    _, _, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    call, names = scalar_probe
    energy = 0.0
    for i in range(o):
        for j in range(i + 1):
            for k in range(j + 1):
                occupied = i, j, k
                moments = {}
                for label, order in zip(_LABELS, PERMUTATIONS, strict=True):
                    I, J, K = (occupied[x] for x in order)
                    # Independent direct seeds in the pinned source's layout.
                    moments[label] = (
                        np.einsum("afb,cf->abc", ovvv[I], t2[K, J])
                        - np.einsum("am,mbc->abc", ovoo[I, :, J], t2[:, K]),
                        np.einsum("ab,c->abc", ovov[I, :, J], t1[K])
                        + np.einsum("ab,c->abc", t2[I, J], fov[K]),
                    )
                degeneracy = 6 if i == k else 2 if i == j or j == k else 1
                for abc in itertools.product(range(v), repeat=3):
                    feeds = {
                        "denominator": (eo[i] + eo[j] + eo[k] - sum(ev[a] for a in abc))
                        * degeneracy
                    }
                    for occ in _LABELS:
                        w, vv = moments[occ]
                        feeds[f"v_{occ}"] = vv[abc]
                        for vir in _LABELS:
                            feeds[f"w_{occ}_{vir}"] = w[tuple(abc[x] for x in VP[vir])]
                    values = np.array([feeds[name] for name in names])
                    energy += call(values.ctypes.data_as(ct.POINTER(ct.c_double)))
    np.testing.assert_allclose(energy, reference(inputs, ovvv), atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("o,v,q", [(2, 3, 4), (3, 2, 1)])
def test_panel_and_each_moment_match_source_index_loops(o: int, v: int, q: int) -> None:
    inputs, ovvv = case(o, v, q)
    bov, bvv, ovoo, ovov, fov, t1, t2, _, _ = inputs
    for i, j, k in itertools.product(range(o), repeat=3):
        panel = execute(
            df_panel_program(v, q), {"bov_i": bov[:, i], "bvv": bvv}
        ).outputs["panel"]
        np.testing.assert_allclose(
            panel, ovvv[i].transpose(0, 2, 1), atol=2e-15, rtol=2e-15
        )
        actual = execute(
            moment_program(o, v),
            {
                "panel": panel,
                "t2_kj": t2[k, j],
                "ovoo_ij": ovoo[i, :, j],
                "t2_mk": t2[:, k],
                "ovov_ij": ovov[i, :, j],
                "t1_k": t1[k],
                "t2_ij": t2[i, j],
                "fov_k": fov[k],
            },
        ).outputs
        for a, b, c in itertools.product(range(v), repeat=3):
            w = sum(ovvv[i, a, f, b] * t2[k, j, c, f] for f in range(v))
            w -= sum(ovoo[i, a, j, m] * t2[m, k, b, c] for m in range(o))
            vv = ovov[i, a, j, b] * t1[k, c] + t2[i, j, a, b] * fov[k, c]
            np.testing.assert_allclose(
                [actual["w"][a, b, c], actual["v"][a, b, c]],
                [w, vv],
                atol=2e-15,
                rtol=2e-15,
            )


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if not GPU:
        pytest.skip("requires finite Slurm GPU allocation and built library")
    directory = tmp_path_factory.mktemp("df-triples-native")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj, output = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-c",
            str(ROOT / "tests/native/df_triples_probe.cpp"),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            str(obj),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(output))
    call = dll.df_triples_probe
    call.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.POINTER(ct.c_double)),
        ct.c_double,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run(
    call: typing.Any,
    inputs: list[np.ndarray],
    budget: int = 1 << 30,
    panels: int = 3,
    threshold: float = 1e-10,
) -> tuple:
    q, o, v = inputs[0].shape
    arrays = [np.ascontiguousarray(x) for x in inputs]
    pointers = (ct.POINTER(ct.c_double) * 9)(
        *(x.ctypes.data_as(ct.POINTER(ct.c_double)) for x in arrays)
    )
    values = np.full(3, np.nan)
    counts = np.zeros(14, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = call(
        o,
        v,
        q,
        pointers,
        threshold,
        budget,
        panels,
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, values, counts, error.value.decode()


@pytest.mark.parametrize(
    "o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 1), (3, 5, 7), (4, 3, 2)]
)
def test_native_energy_work_budget_fallback_and_repeatability(
    native_probe: typing.Any, o: int, v: int, q: int, tmp_path: Path
) -> None:
    inputs, ovvv = case(o, v, q)
    want = reference(inputs, ovvv)
    status, values, counts, error = run(native_probe, inputs)
    assert status == 0, error
    np.testing.assert_allclose(values[0], want, atol=3e-12, rtol=3e-12)
    tiles = o * (o + 1) * (o + 2) // 6
    assert counts[0] == v * (v + 1) * (v + 2) // 6
    assert counts[1] == tiles and counts[5] == min(3, o)
    assert counts[7] == 12 * tiles
    assert counts[8] == tiles and counts[9] == tiles + 1
    assert counts[10] == tiles * v**3
    assert counts[11] == counts[6] * q * v**3 + 6 * tiles * (v**4 + o * v**3)
    assert counts[12] == sum(x.nbytes for x in inputs) and counts[13] == 12
    assert counts[4] <= 96 << 20 and counts[2] == counts[3] + (96 << 20)
    assert values[1] == pytest.approx(3 * (inputs[-1].min() - inputs[-2].max()))
    status, small_values, small_counts, error = run(native_probe, inputs, panels=1)
    assert status == 0, error
    assert small_counts[5] == 1 and small_counts[6] >= counts[6]
    np.testing.assert_array_equal(small_values[:2], values[:2])
    budget = int(small_counts[2])
    status, _, fallback, error = run(native_probe, inputs, budget=budget)
    assert status == 0, error
    # At very small shapes the extra panels can occupy alignment padding and
    # require no additional bytes. Fallback is needed only when the plans differ.
    assert fallback[5] == (counts[5] if counts[2] == budget else 1)
    status, unpublished, _, error = run(native_probe, inputs, budget=budget - 1)
    assert status != 0 and "budget" in error
    assert np.isnan(unpublished).all()
    status, repeated, _, error = run(native_probe, inputs)
    assert status == 0, error
    np.testing.assert_array_equal(repeated[:2], values[:2])
    (tmp_path / "record.json").write_text(
        json.dumps(
            {
                "shape": [o, v, q],
                "values": values.tolist(),
                "counts": counts.tolist(),
                "energy_error": abs(values[0] - want),
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("bad", ["nan", "pair", "gap", "threshold", "overflow"])
def test_native_failure_does_not_publish(native_probe: typing.Any, bad: str) -> None:
    inputs, _ = case(2, 3)
    threshold = 1e-10
    if bad == "nan":
        inputs[0][0, 0, 0] = np.nan
    elif bad == "pair":
        inputs[1][0, 1, 0] += 0.1
    elif bad == "gap":
        inputs[-1][0] = inputs[-2].max() + 1e-12
    elif bad == "threshold":
        threshold = np.nan
    else:
        inputs[0].fill(1e300)
        inputs[1].fill(1e300)
    status, values, _, _ = run(native_probe, inputs, threshold=threshold)
    assert status != 0
    assert np.isnan(values).all()


@pytest.mark.parametrize("o,v", [(2, 1 << 16), (256, 1024)])
def test_dimension_and_complete_work_preflight_precedes_input_access(
    native_probe: typing.Any, o: int, v: int
) -> None:
    """Huge logical shapes must fail before dereferencing even null inputs."""
    pointers = (ct.POINTER(ct.c_double) * 9)()
    values = np.full(3, np.nan)
    counts = np.full(14, 17, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = native_probe(
        o,
        v,
        1,
        pointers,
        1e-10,
        ct.c_size_t(-1).value,
        3,
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    assert status != 0
    assert b"BLAS indexing" in error.value or b"overflow" in error.value
    assert np.isnan(values).all()
    assert (counts == 17).all()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("h2o", -6.731393342463869e-05),
        ("nh3", -1.122922812723691e-04),
        ("ch4", -1.555665872715297e-04),
    ],
)
def test_native_pinned_independent_molecular_energies(
    native_probe: typing.Any, name: str, expected: float
) -> None:
    """Exact Gram factors of committed ERIs reproduce pinned PySCF 2.14.0 (T).

    This factorization is test-only: the native production owner receives DF
    factors, and neither imports PySCF nor factors a dense four-index tensor.
    """
    with np.load(ROOT / "tests/reference_data/cc/endpoints" / f"{name}.npz") as z:
        eps, occ, c, f, g, t1, t2 = (
            z[key] for key in ("eps", "occ", "C", "F", "g", "t1", "t2")
        )
    o, n = int(np.sum(occ > 0)), len(eps)
    eigenvalues, vectors = np.linalg.eigh(g.reshape(n * n, n * n))
    assert eigenvalues.min() > -1e-12
    b = (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T.reshape(n * n, n, n)
    b = (b + b.transpose(0, 2, 1)) / 2
    np.testing.assert_allclose(
        np.einsum("Qpq,Qrs->pqrs", b, b), g, atol=3e-13, rtol=3e-13
    )
    inputs = [
        b[:, :o, o:],
        b[:, o:, o:],
        g[:o, o:, :o, :o],
        g[:o, o:, :o, o:],
        (c.T @ f @ c)[:o, o:],
        t1,
        t2,
        eps[:o],
        eps[o:],
    ]
    status, values, _, error = run(native_probe, inputs)
    assert status == 0, error
    np.testing.assert_allclose(values[0], expected, atol=3e-12, rtol=3e-12)
