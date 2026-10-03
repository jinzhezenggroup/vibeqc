"""Occupied-domain reverse composition against independent energy differences."""

from __future__ import annotations

import ctypes as ct
import itertools
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.occupied_triples import PERMUTATIONS
from generativeqc_compiler.cc.occupied_triples_response import (
    energy_scalar_vjp,
    moment_vjp,
    panel_vjp,
)
from generativeqc_compiler.cc.triples import _LABELS, VP, triples_energy
from generativeqc_compiler.tensor import execute
from test_df_cc_factor_response import (
    factor_probe,  # noqa: F401 -- pytest fixture
    run_factor,
)
from test_df_cc_lambda import case as lambda_case
from test_df_cc_lambda import feeds as lambda_feeds
from test_df_cc_lambda import probe as lambda_probe  # noqa: F401 -- pytest fixture
from test_df_cc_lambda import run as run_lambda
from test_df_occupied_triples import case


def reverse(inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Tiny host validation traversal; production must lower these local graphs.

    Explicit source-coordinate scatter here independently checks the intended
    native inverse-permutation gather and strided BLAS output placements.
    """
    bov, bvv, ovoo, ovov, fov, t1, t2, eo, ev = inputs
    o, v, q = len(eo), len(ev), len(bov)
    result = [np.zeros_like(x) for x in inputs]
    dbov, dbvv, dovoo, dovov, dfov, dt1, dt2, deo, dev = result
    scalar, w_reverse, v_reverse, p_reverse = (
        energy_scalar_vjp(),
        moment_vjp(o, v, "w"),
        moment_vjp(o, v, "v"),
        panel_vjp(v, q),
    )
    for i in range(o):
        for j in range(i + 1):
            for k in range(j + 1):
                occupied = i, j, k
                moments = []
                for order in PERMUTATIONS:
                    I, J, K = (occupied[x] for x in order)
                    panel = np.einsum("Qa,Qbf->abf", bov[:, I], bvv)
                    w = np.einsum("abf,cf->abc", panel, t2[K, J])
                    w -= np.einsum("am,mbc->abc", ovoo[I, :, J], t2[:, K])
                    vv = np.einsum("ab,c->abc", ovov[I, :, J], t1[K])
                    vv += np.einsum("ab,c->abc", t2[I, J], fov[K])
                    moments.append((panel, w, vv))
                dw, dv = np.zeros((6, v, v, v)), np.zeros((6, v, v, v))
                degeneracy = 6 if i == k else 2 if i == j or j == k else 1
                for abc in itertools.product(range(v), repeat=3):
                    gap = eo[i] + eo[j] + eo[k] - sum(ev[a] for a in abc)
                    feed = {
                        "denominator": np.array(gap * degeneracy),
                        "bar_energy": np.array(1.0),
                    }
                    for index, occ in enumerate(_LABELS):
                        _, w, vv = moments[index]
                        feed[f"v_{occ}"] = np.array(vv[abc])
                        for vir in _LABELS:
                            feed[f"w_{occ}_{vir}"] = np.array(
                                w[tuple(abc[x] for x in VP[vir])]
                            )
                    bars = execute(scalar, feed).outputs
                    for index, occ in enumerate(_LABELS):
                        dv[(index, *abc)] += bars[f"bar_v_{occ}"]
                        for vir in _LABELS:
                            address = tuple(abc[x] for x in VP[vir])
                            dw[(index, *address)] += bars[f"bar_w_{occ}_{vir}"]
                    bar_gap = float(bars["bar_denominator"]) * degeneracy
                    for I in occupied:
                        deo[I] += bar_gap
                    for a in abc:
                        dev[a] -= bar_gap
                for index, order in enumerate(PERMUTATIONS):
                    I, J, K = (occupied[x] for x in order)
                    wr = execute(
                        w_reverse,
                        {
                            "panel": moments[index][0],
                            "t2_kj": t2[K, J],
                            "ovoo_ij": ovoo[I, :, J],
                            "t2_mk": t2[:, K],
                            "bar_w": dw[index],
                        },
                    ).outputs
                    vr = execute(
                        v_reverse,
                        {
                            "ovov_ij": ovov[I, :, J],
                            "t1_k": t1[K],
                            "t2_ij": t2[I, J],
                            "fov_k": fov[K],
                            "bar_v": dv[index],
                        },
                    ).outputs
                    pr = execute(
                        p_reverse,
                        {"bov_i": bov[:, I], "bvv": bvv, "bar_panel": wr["bar_panel"]},
                    ).outputs
                    dbov[:, I] += pr["bar_bov_i"]
                    dbvv += pr["bar_bvv"]
                    dovoo[I, :, J] += wr["bar_ovoo_ij"]
                    dt2[K, J] += wr["bar_t2_kj"]
                    dt2[:, K] += wr["bar_t2_mk"]
                    dovov[I, :, J] += vr["bar_ovov_ij"]
                    dt1[K] += vr["bar_t1_k"]
                    dt2[I, J] += vr["bar_t2_ij"]
                    dfov[K] += vr["bar_fov_k"]
    return result


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 2)])
def test_complete_reverse_composition_matches_independent_physical_differences(
    o: int, v: int, q: int
) -> None:
    inputs, _ = case(o, v, q)
    result = reverse(inputs)
    rng = np.random.default_rng(1797)
    delta = [rng.normal(scale=0.02, size=x.shape) for x in inputs]
    delta[1] = (delta[1] + delta[1].transpose(0, 2, 1)) / 2
    delta[6] = (delta[6] + delta[6].transpose(1, 0, 3, 2)) / 2

    def energy(values: list[np.ndarray]) -> float:
        bov, bvv, ovoo, ovov, fov, t1, t2, eo, ev = values
        ovvv = np.einsum("Qia,Qfb->iafb", bov, bvv)
        return float(triples_energy(o, v, ovvv, ovoo, ovov, fov, t1, t2, eo, ev))

    step = 1e-4
    for selected in range(len(inputs)):

        def at(t: float, selected: int = selected) -> float:
            values = list(inputs)
            values[selected] = values[selected] + t * delta[selected]
            return energy(values)

        fd = (at(-2 * step) - 8 * at(-step) + 8 * at(step) - at(2 * step)) / (12 * step)
        analytic = float(np.sum(result[selected] * delta[selected]))
        np.testing.assert_allclose(
            analytic, fd, atol=3e-12, rtol=3e-7, err_msg=f"input {selected}"
        )


@pytest.mark.parametrize("o,v,q", [(9, 221, 488), (21, 243, 666)])
def test_large_reverse_graphs_keep_only_local_rank_three_contractions(
    o: int, v: int, q: int
) -> None:
    programs = [panel_vjp(v, q), moment_vjp(o, v, "w"), moment_vjp(o, v, "v")]
    assert [sum(n.op == "einsum" for n in p.live_nodes) for p in programs] == [2, 4, 4]
    for p in programs:
        assert max(len(n.spec.indices) for n in p.live_nodes) <= 3
        assert all(n.op in ("input", "einsum", "add") for n in p.live_nodes)
    # One scalar W derivative is sufficient for each inverse-permutation gather;
    # generating all 43 outputs in each gather would repeat unrelated arithmetic.
    small = energy_scalar_vjp(("w_abc_abc",))
    full = energy_scalar_vjp()
    assert len(small.live_nodes) < len(full.live_nodes)
    assert tuple(small.outputs) == ("bar_w_abc_abc",)


@pytest.fixture(scope="module")
def blas_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    """Compile the actual emitted traversal with a checked generic BLAS emulator."""
    from tools.generate_df_occupied_triples import header

    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-triples-reverse-blas")
    (directory / "generated.hpp").write_text(header())
    source = directory / "probe.cpp"
    source.write_text(r"""
#include <vector>
#include <stdexcept>
#include "generated.hpp"
using namespace generativeqc::cc::triples::generated_df;
extern "C" void run(std::size_t o,std::size_t v,std::size_t q,
  std::size_t i,std::size_t j,std::size_t k,const double* const* in,double* const* out,
  const double* dw,const double* dv) {
  Inputs p{in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],in[8]};
  ResponseOutputs r{out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8]};
  std::vector<double> panel(v*v*v),dp(v*v*v),ovov(v*v),dg(v*v);
  // Column-major BLAS semantics, with explicit physical leading dimensions.
  auto gemm=[](char ta,char tb,std::size_t m,std::size_t n,std::size_t kk,double alpha,
      const double* a,std::size_t lda,const double* b,std::size_t ldb,double beta,
      double* c,std::size_t ldc) {
    for(std::size_t col=0;col<n;++col)for(std::size_t row=0;row<m;++row){
      double sum=0;
      for(std::size_t x=0;x<kk;++x)
        sum+=(ta=='N'?a[row+x*lda]:a[x+row*lda])*(tb=='N'?b[x+col*ldb]:b[col+x*ldb]);
      c[row+col*ldc]=alpha*sum+beta*c[row+col*ldc];
    }
  };
  build_panel(o,v,q,i,p,panel.data(),gemm);
  for(std::size_t a=0;a<v;++a)for(std::size_t b=0;b<v;++b)
    ovov[a*v+b]=p.ovov[((i*v+a)*o+j)*v+b];
  pullback_w(o,v,i,j,k,p,panel.data(),dw,dp.data(),r,gemm);
  pullback_v(o,v,i,j,k,p,ovov.data(),dv,dg.data(),r,gemm);
  for(std::size_t a=0;a<v;++a)for(std::size_t b=0;b<v;++b)
    r.ovov[((i*v+a)*o+j)*v+b]+=dg[a*v+b];
  pullback_panel(o,v,q,i,p,dp.data(),r,gemm);
}
""")
    obj, output = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(output)],
        check=True,
        capture_output=True,
    )
    dll = ct.CDLL(str(output))
    call = dll.run
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [ct.c_size_t] * 6 + [ct.POINTER(dp), ct.POINTER(dp), dp, dp]
    call.restype = None
    return call


@pytest.mark.parametrize(
    "o,v,q,ijk", [(1, 3, 2, (0, 0, 0)), (2, 3, 4, (1, 1, 0)), (3, 2, 2, (2, 1, 0))]
)
def test_emitted_reverse_blas_matches_local_ad_on_strided_and_aliased_views(
    blas_probe: typing.Any, o: int, v: int, q: int, ijk: tuple[int, int, int]
) -> None:
    inputs, _ = case(o, v, q)
    bov, bvv, ovoo, ovov, fov, t1, t2, _, _ = inputs
    i, j, k = ijk
    rng = np.random.default_rng(1797)
    dw, dv = rng.normal(size=(2, v, v, v))
    initial = [rng.normal(scale=0.01, size=x.shape) for x in inputs]
    expected = [x.copy() for x in initial]
    panel = np.einsum("Qa,Qbf->abf", bov[:, i], bvv)
    wr = execute(
        moment_vjp(o, v, "w"),
        {
            "panel": panel,
            "t2_kj": t2[k, j],
            "ovoo_ij": ovoo[i, :, j],
            "t2_mk": t2[:, k],
            "bar_w": dw,
        },
    ).outputs
    vr = execute(
        moment_vjp(o, v, "v"),
        {
            "ovov_ij": ovov[i, :, j],
            "t1_k": t1[k],
            "t2_ij": t2[i, j],
            "fov_k": fov[k],
            "bar_v": dv,
        },
    ).outputs
    pr = execute(
        panel_vjp(v, q), {"bov_i": bov[:, i], "bvv": bvv, "bar_panel": wr["bar_panel"]}
    ).outputs
    expected[0][:, i] += pr["bar_bov_i"]
    expected[1] += pr["bar_bvv"]
    expected[2][i, :, j] += wr["bar_ovoo_ij"]
    expected[3][i, :, j] += vr["bar_ovov_ij"]
    expected[4][k] += vr["bar_fov_k"]
    expected[5][k] += vr["bar_t1_k"]
    expected[6][k, j] += wr["bar_t2_kj"]
    expected[6][:, k] += wr["bar_t2_mk"]
    expected[6][i, j] += vr["bar_t2_ij"]
    dp = ct.POINTER(ct.c_double)
    blas_probe(
        o,
        v,
        q,
        *ijk,
        (dp * 9)(*(a.ctypes.data_as(dp) for a in inputs)),
        (dp * 9)(*(a.ctypes.data_as(dp) for a in initial)),
        dw.ctypes.data_as(dp),
        dv.ctypes.data_as(dp),
    )
    for actual, want in zip(initial, expected, strict=True):
        np.testing.assert_allclose(actual, want, atol=3e-13, rtol=3e-13)


@pytest.fixture(scope="module")
def native_response_probe(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    if os.environ.get("GENERATIVEQC_DF_TRIPLES_CUDA_TEST") != "1":
        pytest.skip("requires finite Slurm real-GPU allocation")
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("requires C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-triples-response-native")
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
            "-I" + str(root / "src"),
            "-c",
            str(root / "tests/native/df_triples_response_probe.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
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
        timeout=60,
    )
    dll = ct.CDLL(str(output))
    call = dll.df_triples_response_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = (
        [ct.c_size_t] * 3
        + [ct.POINTER(dp), ct.c_double]
        + [ct.c_size_t] * 3
        + [ct.POINTER(dp), dp, ct.POINTER(ct.c_size_t), ct.c_void_p, ct.c_size_t]
    )
    call.restype = ct.c_int
    return call


def run_native(
    call: typing.Any,
    inputs: list[np.ndarray],
    *,
    budget: int = 1 << 30,
    caller_bytes: int = 0,
    panels: int = 3,
    threshold: float = 1e-10,
) -> tuple:
    q, o, v = inputs[0].shape
    arrays = [np.ascontiguousarray(x) for x in inputs]
    output = [np.full_like(x, np.nan) for x in arrays]
    values = np.full(3, np.nan)
    counts = np.zeros(17, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    dp = ct.POINTER(ct.c_double)
    status = call(
        o,
        v,
        q,
        (dp * 9)(*(x.ctypes.data_as(dp) for x in arrays)),
        threshold,
        budget,
        caller_bytes,
        panels,
        (dp * 9)(*(x.ctypes.data_as(dp) for x in output)),
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    return status, output, values, counts, error.value.decode()


@pytest.mark.parametrize(
    "o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 1), (3, 5, 7), (4, 3, 2)]
)
def test_native_complete_response_matches_reverse_and_work(
    native_response_probe: typing.Any, o: int, v: int, q: int
) -> None:
    from test_df_occupied_triples import reference

    inputs, ovvv = case(o, v, q)
    status, output, values, counts, error = run_native(native_response_probe, inputs)
    assert status == 0, error
    expected = reverse(inputs)
    for actual, want in zip(output, expected, strict=True):
        np.testing.assert_allclose(actual, want, atol=3e-12, rtol=3e-11)
    np.testing.assert_allclose(
        values[0], reference(inputs, ovvv), atol=3e-12, rtol=3e-12
    )
    tiles = o * (o + 1) * (o + 2) // 6
    groups = sum(
        len({i, j, k}) for i in range(o) for j in range(i + 1) for k in range(j + 1)
    )
    assert counts[7] == 12 * tiles and counts[8] == 48 * tiles + 2 * groups
    assert (
        counts[9]
        == (counts[6] + 2 * groups) * q * v**3
        + 18 * tiles * (v**4 + o * v**3)
        + 24 * tiles * v**3
    )
    assert (
        counts[10] == tiles
        and counts[11] == tiles * v**3
        and counts[12] == 43 * tiles * v**3
    )
    assert counts[14] == counts[6] + counts[7] + counts[8]
    assert counts[1] == counts[15] == sum(x.nbytes for x in inputs)
    assert counts[16] == counts[1] + 12
    assert counts[0] == counts[2] + 2 * counts[1]
    assert counts[2] == counts[3] + (96 << 20) and counts[4] <= 96 << 20


@pytest.mark.parametrize("gap", [0.0, 1e-11])
def test_native_response_handles_same_space_degeneracy_at_fixed_frame(
    native_response_probe: typing.Any, gap: float
) -> None:
    inputs, _ = case(3, 3, 4)
    inputs[7][:] = -0.8 + gap * np.arange(3)
    inputs[8][:] = 0.5 + gap * np.arange(3)
    status, output, _, _, error = run_native(native_response_probe, inputs)
    assert status == 0, error
    for actual, want in zip(output, reverse(inputs), strict=True):
        np.testing.assert_allclose(actual, want, atol=3e-12, rtol=3e-11)


def test_native_response_panel_fallback_exact_budget_and_replay(
    native_response_probe: typing.Any,
) -> None:
    inputs, _ = case(3, 5, 4)
    status, want, values, counts, error = run_native(native_response_probe, inputs)
    assert status == 0, error
    status, small, small_values, small_counts, error = run_native(
        native_response_probe, inputs, panels=1
    )
    assert status == 0, error
    assert small_counts[5] == 1 and small_counts[6] >= counts[6]
    for actual, expected in zip(small, want, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(small_values[:2], values[:2])
    budget = int(small_counts[0])
    status, actual, _, fallback, error = run_native(
        native_response_probe, inputs, budget=budget + 123, caller_bytes=123
    )
    assert status == 0, error
    assert fallback[0] == budget + 123
    assert fallback[5] == 1
    for a, b in zip(actual, want, strict=True):
        np.testing.assert_array_equal(a, b)
    status, outputs, values, _, error = run_native(
        native_response_probe, inputs, budget=budget - 1
    )
    assert status != 0 and "budget" in error
    assert np.isnan(values).all() and all(np.isnan(x).all() for x in outputs)


@pytest.mark.parametrize(
    "bad", ["nan", "pair", "gap", "threshold", "overflow", "masked_overflow"]
)
def test_native_response_failure_is_transactional(
    native_response_probe: typing.Any, bad: str
) -> None:
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
        if bad == "masked_overflow":
            inputs[5].fill(0)
            inputs[6].fill(0)
    status, output, values, _, _ = run_native(
        native_response_probe, inputs, threshold=threshold
    )
    assert status != 0
    assert np.isnan(values).all() and all(np.isnan(x).all() for x in output)


@pytest.mark.parametrize("o,v", [(2, 3), (3, 2)])
def test_native_triples_corrected_lambda_and_factor_chain_matches_resolved_energy(
    native_response_probe: typing.Any,
    lambda_probe: typing.Any,  # noqa: F811 -- pytest fixture
    factor_probe: typing.Any,  # noqa: F811 -- pytest fixture
    o: int,
    v: int,
) -> None:
    """Differentiate the converged CCSD+(T) energy, including amplitude relaxation."""
    b, f, _ = lambda_case(o, v)
    b *= 3
    arrays = lambda_feeds(b, f, o)
    status, cc, _, _, error = run_lambda(lambda_probe, arrays)
    assert status == 0, error
    eps = np.diag(f)

    def triples_inputs(
        a: dict[str, np.ndarray], amplitudes: list[np.ndarray]
    ) -> list[np.ndarray]:
        return [
            a["bov"],
            a["bvv"],
            a["ovoo"],
            a["ovov"],
            a["fov"],
            *amplitudes[:2],
            eps[:o],
            eps[o:],
        ]

    status, triples, _, _, error = run_native(
        native_response_probe, triples_inputs(arrays, cc)
    )
    assert status == 0, error
    dp = ct.POINTER(ct.c_double)
    source = [np.ascontiguousarray(triples[5]), np.ascontiguousarray(triples[6])]

    def corrected(*args: typing.Any) -> int:
        values = list(args)
        values[6], values[7] = (x.ctypes.data_as(dp) for x in source)
        return lambda_probe(*values)

    status, response, values, _, error = run_lambda(corrected, arrays, source=True)
    assert status == 0, error
    assert np.max(values[1:]) < 1e-9

    def factors(r: list[np.ndarray]) -> list[np.ndarray]:
        inputs = {
            "boo": b[:, :o, :o],
            "bov": b[:, :o, o:],
            "bvv": b[:, o:, o:],
            "bar_ovov": r[7] + triples[3],
            "bar_ovvo": r[8],
            "bar_oovv": r[9],
            "bar_ovoo": r[10] + triples[2],
            "bar_oooo": r[11],
            "bar_bov": r[12] + triples[0],
            "bar_bvv": r[13] + triples[1],
        }
        status, output, _, error = run_factor(factor_probe, inputs)
        assert status == 0, error
        return output

    bars = factors(response)
    uncorrected = factors(cc)
    assert (
        max(np.max(np.abs(x - y)) for x, y in zip(bars, uncorrected, strict=True))
        > 1e-9
    )
    delta = np.random.default_rng(1765).normal(scale=0.02, size=b.shape)
    delta = (delta + delta.transpose(0, 2, 1)) / 2
    analytic = sum(
        float(np.sum(bar * d))
        for bar, d in zip(
            bars, (delta[:, :o, :o], delta[:, :o, o:], delta[:, o:, o:]), strict=True
        )
    )
    for step in (1e-4, 5e-5):
        energies = []
        for sign in (-1, 1):
            a = lambda_feeds(b + sign * step * delta, f, o)
            status, amp, value, _, error = run_lambda(lambda_probe, a)
            assert status == 0, error
            # The independent original virtual-triangle algebra evaluates (T)
            # on each reconverged state; the occupied response is not its oracle.
            ovvv = np.einsum("Qia,Qfb->iafb", a["bov"], a["bvv"])
            et = triples_energy(
                o,
                v,
                ovvv,
                a["ovoo"],
                a["ovov"],
                a["fov"],
                amp[0],
                amp[1],
                eps[:o],
                eps[o:],
            )
            energies.append(value[0] + float(et))
        np.testing.assert_allclose(
            analytic, (energies[1] - energies[0]) / (2 * step), atol=3e-10, rtol=3e-7
        )


@pytest.mark.parametrize("o,v,q", [(2, 1 << 16, 1), (256, 1024, 4)])
def test_response_complete_work_preflight_precedes_null_input_access(
    native_response_probe: typing.Any,
    o: int,
    v: int,
    q: int,
) -> None:
    dp = ct.POINTER(ct.c_double)
    nulls = (dp * 9)()
    values = np.full(3, np.nan)
    counts = np.full(17, 19, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    status = native_response_probe(
        o,
        v,
        q,
        nulls,
        1e-10,
        np.iinfo(np.uintp).max,
        0,
        3,
        nulls,
        values.ctypes.data_as(dp),
        counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        error,
        len(error),
    )
    assert status != 0 and any(
        word in error.value.decode() for word in ("indexing", "overflow")
    )
    assert np.isnan(values).all()
    np.testing.assert_array_equal(counts, 19)
