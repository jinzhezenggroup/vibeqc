"""Exercise emitted production BLAS calls with an independent host BLAS stand-in.

This verifies transpose/stride/output-region/error-propagation contracts, not
CUDA compilation, GPU accuracy/performance or the full GenerativeQC runtime.
"""

import ctypes as ct
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
EMITTER = (
    HERE.parents[2] / "python/generativeqc_compiler/method/df_occupied_response_cuda.py"
)

HEADER = r"""
#pragma once
using cublasHandle_t = void*;
using cublasStatus_t = int;
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cublasFillMode_t { CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER };
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_INVALID_VALUE=7;
int cublasDgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,const double*,int,const double*,double*,int);
int cublasDgemmStridedBatched(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,long long,const double*,int,long long,
               const double*,double*,int,long long,int);
int cublasDsyrk(cublasHandle_t,cublasFillMode_t,cublasOperation_t,int,int,
               const double*,const double*,int,const double*,double*,int);
"""
STANDIN = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include "cublas_v2.h"
static int calls, fail_on;
static void product(cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
                    double alpha,const double* a,int lda,const double* b,int ldb,
                    double beta,double* c,int ldc) {
  for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
    double v=0;
    for(int z=0;z<k;++z)
      v+=(ta==CUBLAS_OP_N?a[i+z*lda]:a[z+i*lda])*
         (tb==CUBLAS_OP_N?b[z+j*ldb]:b[j+z*ldb]);
    c[i+j*ldc]=alpha*v+(beta==0?0:beta*c[i+j*ldc]);
  }
}
int cublasDgemm(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
               const double* alpha,const double* a,int lda,const double* b,int ldb,
               const double* beta,double* c,int ldc) {
  if (++calls==fail_on) return 13;
  product(ta,tb,m,n,k,*alpha,a,lda,b,ldb,*beta,c,ldc); return 0;
}
int cublasDgemmStridedBatched(cublasHandle_t,cublasOperation_t ta,cublasOperation_t tb,
               int m,int n,int k,const double* alpha,const double* a,int lda,long long sa,
               const double* b,int ldb,long long sb,const double* beta,double* c,int ldc,
               long long sc,int batches) {
  if (++calls==fail_on) return 13;
  for(int p=0;p<batches;++p)
    product(ta,tb,m,n,k,*alpha,a+p*sa,lda,b+p*sb,ldb,*beta,c+p*sc,ldc);
  return 0;
}
int cublasDsyrk(cublasHandle_t,cublasFillMode_t triangle,cublasOperation_t trans,
               int n,int k,const double* alpha,const double* a,int lda,
               const double* beta,double* c,int ldc) {
  if (++calls==fail_on) return 13;
  for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
    if(triangle==CUBLAS_FILL_MODE_LOWER ? i<j : i>j) continue;
    double value=0;
    for(int z=0;z<k;++z)
      value+=(trans==CUBLAS_OP_T?a[z+i*lda]:a[i+z*lda])*
             (trans==CUBLAS_OP_T?a[z+j*lda]:a[j+z*lda]);
    c[i+j*ldc]=*alpha*value+(*beta==0?0:*beta*c[i+j*ldc]);
  }
  return 0;
}
"""
WRAPPERS = r"""
extern "C" int project(int n,int r,int a,int begin,int count,const double* c,
                         const double* values,double* temp,double* out,int fail) {
  calls=0;fail_on=fail;
  return generativeqc::scf::generated::df_occupied_project_panel(
      nullptr,n,r,a,begin,count,c,values,temp,out);
}
extern "C" int finish_project(int n,int r,int a,const double* c,
                                const double* linear,double* out,int fail) {
  calls=0;fail_on=fail;
  std::vector<double> pair_major(static_cast<std::size_t>(a)*r*r);
  auto result=generativeqc::scf::generated::df_occupied_finish_projection(
      nullptr,n,r,a,c,linear,pair_major.data());
  if(result) return result;
  // Mirror the production gather from [i,j,Q] into [Q,i,j].
  for(int q=0;q<a;++q) for(int i=0;i<r;++i) for(int j=0;j<r;++j)
    out[q*r*r+i+j*r]=pair_major[(i*r+j)*a+q];
  return 0;
}
extern "C" int call_count() { return calls; }
extern "C" std::size_t tile_size(std::size_t n,std::size_t r,std::size_t a,
                         std::size_t capacity,std::size_t request,bool extra) {
  return generativeqc::scf::generated::df_occupied_projection_tile(n,r,a,capacity,request,extra);
}
extern "C" int small_metric(int a,int rr,const double* e,const double* eigenvalues,
                              const double* s,double* temp,double* out,int fail) {
  calls=0;fail_on=fail;
  auto result=generativeqc::scf::generated::df_occupied_to_metric_eigenbasis(
      nullptr,a,rr,e,s,temp);
  if(result) return result;
  for(int j=0;j<rr;++j) for(int q=0;q<a;++q) temp[q+j*a]/=std::sqrt(eigenvalues[q]);
  return generativeqc::scf::generated::df_occupied_from_metric_eigenbasis(
      nullptr,a,rr,e,temp,out);
}
extern "C" int retained_metric(int a,int rr,const double* x,const double* s,
                                 double* out,int fail) {
  calls=0;fail_on=fail;
  return generativeqc::scf::generated::df_occupied_apply_metric_root(nullptr,a,rr,x,s,out);
}
extern "C" std::size_t symmetric_pair(std::size_t r,std::size_t i,std::size_t j) {
  return generativeqc::scf::generated::df_occupied_symmetric_pair(r,i,j);
}
extern "C" int symmetric_gram(int a,int r,double scale,const double* u,double* out,int fail) {
  calls=0;fail_on=fail;
  return generativeqc::scf::generated::df_occupied_symmetric_metric_gram(
      nullptr,a,r,scale,u,out);
}
"""


@pytest.fixture(scope="session", params=[True, False], ids=["syrk", "gemm-provider"])
def native(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    spec = importlib.util.spec_from_file_location(
        "isolated_df_occupied_emitter", EMITTER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path_factory.mktemp("df-projection-host")
    has_syrk = request.param
    header = HEADER if has_syrk else HEADER.replace("cublasDsyrk", "unavailableDsyrk")
    standin = (
        STANDIN if has_syrk else STANDIN.replace("cublasDsyrk", "unavailableDsyrk")
    )
    (folder / "cublas_v2.h").write_text(header)
    code = (
        standin
        + "\nnamespace generativeqc::scf::generated {\n"
        + module.emit_occupied_response_helpers()
        + "\n}\n"
        + WRAPPERS
    )
    (folder / "test.cpp").write_text(code)
    output = folder / "lowering.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-shared",
            "-fPIC",
            str(folder / "test.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    lib = ct.CDLL(str(output))
    lib.has_syrk = has_syrk
    ptr = ct.POINTER(ct.c_double)
    lib.project.argtypes = [ct.c_int] * 5 + [ptr] * 4 + [ct.c_int]
    lib.project.restype = ct.c_int
    lib.finish_project.argtypes = [ct.c_int] * 3 + [ptr] * 3 + [ct.c_int]
    lib.finish_project.restype = ct.c_int
    lib.call_count.restype = ct.c_int
    lib.tile_size.argtypes = [ct.c_size_t] * 5 + [ct.c_bool]
    lib.tile_size.restype = ct.c_size_t
    lib.small_metric.argtypes = [ct.c_int] * 2 + [ptr] * 5 + [ct.c_int]
    lib.small_metric.restype = ct.c_int
    lib.retained_metric.argtypes = [ct.c_int] * 2 + [ptr] * 3 + [ct.c_int]
    lib.retained_metric.restype = ct.c_int
    lib.symmetric_pair.argtypes = [ct.c_size_t] * 3
    lib.symmetric_pair.restype = ct.c_size_t
    lib.symmetric_gram.argtypes = [ct.c_int] * 2 + [ct.c_double, ptr, ptr, ct.c_int]
    lib.symmetric_gram.restype = ct.c_int
    return lib


def pointer(array: np.ndarray) -> ct.POINTER(ct.c_double):
    return array.ctypes.data_as(ct.POINTER(ct.c_double))


@pytest.mark.parametrize(
    "n,r,a", [(1, 1, 1), (4, 1, 7), (7, 3, 9), (12, 5, 17), (16, 8, 13), (31, 7, 11)]
)
@pytest.mark.parametrize("width", [1, 2, 5, 64])
def test_projected_panel_layout_and_tails(
    native: ct.CDLL, n: int, r: int, a: int, width: int
) -> None:
    rng = np.random.default_rng(20260925 + n + r + a)
    c = np.asfortranarray(rng.normal(size=(n, r)))
    values = rng.normal(size=(a, n, n))
    values = np.ascontiguousarray((values + values.transpose(0, 2, 1)) * 0.5)
    out = np.full((a, r, r), np.nan)
    expected = np.einsum("mi,qmn,nj->qij", c, values, c)
    total_calls = 0
    for begin in range(0, a, width):
        count = min(width, a - begin)
        temp = np.full(count * n * r, np.nan)
        assert (
            native.project(
                n,
                r,
                a,
                begin,
                count,
                pointer(c),
                pointer(values[begin:]),
                pointer(temp),
                pointer(out),
                0,
            )
            == 0
        )
        total_calls += native.call_count()
        np.testing.assert_allclose(
            out[: begin + count], expected[: begin + count], atol=3e-12, rtol=3e-13
        )
        assert np.isnan(out[begin + count :]).all()
    assert total_calls == 2 * ((a + width - 1) // width)


@pytest.mark.parametrize("n,r,a", [(4, 1, 3), (7, 3, 9), (12, 5, 13), (31, 7, 11)])
def test_finish_final_k_projection_layout(
    native: ct.CDLL, n: int, r: int, a: int
) -> None:
    """The carried final-K B*C factor reproduces C^T B C without rereading B."""
    rng = np.random.default_rng(20260927 + n + r + a)
    coefficients = np.asfortranarray(rng.normal(size=(n, r)))
    fitted = rng.normal(size=(a, n, n))
    # The packed final-K kernel stores Q fastest within each occupied column.
    linear = np.empty((a * r, n), dtype=np.float64, order="F")
    expected = []
    for q in range(a):
        projection = fitted[q] @ coefficients
        for j in range(r):
            linear[q + a * j, :] = projection[:, j]
        expected.append(coefficients.T @ fitted[q] @ coefficients)
    output = np.full(a * r * r, np.nan)
    assert (
        native.finish_project(
            n,
            r,
            a,
            pointer(coefficients),
            pointer(linear),
            pointer(output),
            0,
        )
        == 0
    )
    for q, reference in enumerate(expected):
        actual = output[q * r * r : (q + 1) * r * r].reshape((r, r), order="F")
        np.testing.assert_allclose(actual, reference, atol=3e-12, rtol=3e-13)
    assert native.call_count() == 1


def test_finish_final_k_projection_rejects_invalid_shape(native: ct.CDLL) -> None:
    data = np.ones(4)
    assert (
        native.finish_project(2, 3, 1, pointer(data), pointer(data), pointer(data), 0)
        == 7
    )


@pytest.mark.parametrize("a,r", [(1, 1), (3, 2), (9, 3), (17, 5)])
@pytest.mark.parametrize("condition", [1.0, 1e3, 1e6, 1e10])
@pytest.mark.parametrize("retained", [False, True])
def test_small_metric_layout(
    native: ct.CDLL, a: int, r: int, condition: float, retained: bool
) -> None:
    rng = np.random.default_rng(71 + a + r)
    eigenvectors = np.asfortranarray(np.linalg.qr(rng.normal(size=(a, a)))[0])
    eigenvalues = np.geomspace(1, condition, a)
    s = np.ascontiguousarray(rng.normal(size=(a, r, r)))
    temp = np.empty(a * r * r)
    out = np.full_like(s, np.nan)
    assert (
        native.small_metric(
            a,
            r * r,
            pointer(eigenvectors),
            pointer(eigenvalues),
            pointer(s),
            pointer(temp),
            pointer(out),
            0,
        )
        == 0
    )
    x = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    if retained:
        assert (
            native.retained_metric(
                a, r * r, pointer(np.asfortranarray(x)), pointer(s), pointer(out), 0
            )
            == 0
        )
    np.testing.assert_allclose(
        out, np.einsum("pq,qij->pij", x, s), atol=5e-14, rtol=2e-12
    )
    assert native.call_count() == (1 if retained else 2)


def test_retained_metric_failure_and_alias_contract(native: ct.CDLL) -> None:
    """A failed BLAS call leaves output untouched; in-place contraction is invalid."""
    x = np.eye(3)
    s = np.arange(12.0)
    out = np.full_like(s, np.nan)
    assert native.retained_metric(3, 4, pointer(x), pointer(s), pointer(out), 1) == 13
    assert np.isnan(out).all()
    assert native.retained_metric(3, 4, pointer(x), pointer(s), pointer(s), 0) == 7


def test_96_atom_capacity_and_overflow(native: ct.CDLL) -> None:
    n, r, a = 768, 160, 3712
    cap = a * r * r
    assert native.tile_size(n, r, a, cap, 64, False) == 64
    assert native.tile_size(n, r, a, cap, 64, True) == 64
    assert native.tile_size(n, r, a, n * n + n * r - 1, 64, False) == 0
    assert native.tile_size(n, r, a, n * n + n * r, 64, False) == 1
    assert native.tile_size(n, r, a, 2 * n * n + n * r - 1, 64, True) == 0


@pytest.mark.parametrize("n,r,a", [(4, 1, 3), (7, 3, 9), (12, 5, 13), (16, 8, 17)])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("condition", [1.0, 1e3, 1e6])
@pytest.mark.parametrize("retained", [False, True])
def test_full_rank_adjoints_using_emitted_helpers(
    native: ct.CDLL,
    n: int,
    r: int,
    a: int,
    scale: float,
    condition: float,
    retained: bool,
) -> None:
    rng = np.random.default_rng(800 + n + r + a)
    c = np.asfortranarray(np.linalg.qr(rng.normal(size=(n, r)))[0])
    raw = rng.normal(size=(a, n, n))
    raw = (raw + raw.transpose(0, 2, 1)) * 0.5
    e = np.asfortranarray(np.linalg.qr(rng.normal(size=(a, a)))[0])
    eigenvalues = np.geomspace(1.0, condition, a)
    x = (e / np.sqrt(eigenvalues)) @ e.T
    b = np.ascontiguousarray(np.einsum("pq,qmn->pmn", x, raw))
    d = scale * c @ c.T
    inverse = (e / eigenvalues) @ e.T
    fitted = np.einsum("pq,qmn->pmn", inverse, raw)
    charge = np.einsum("qmn,mn->q", fitted, d)
    dd = np.einsum("mi,qij,jn->qmn", d, fitted, d)
    ref_a = d[None] * charge[:, None, None] - 0.5 * dd
    ref_m = -0.5 * np.outer(charge, charge) + 0.25 * np.einsum(
        "pmn,qmn->pq", dd, fitted
    )
    s = np.full((a, r, r), np.nan)
    for begin in range(0, a, 5):
        count = min(5, a - begin)
        tmp = np.empty(count * n * r)
        assert (
            native.project(
                n,
                r,
                a,
                begin,
                count,
                pointer(c),
                pointer(b[begin:]),
                pointer(tmp),
                pointer(s),
                0,
            )
            == 0
        )
    u = np.empty_like(s)
    temp = np.empty_like(s)
    assert (
        native.small_metric(
            a,
            r * r,
            pointer(e),
            pointer(eigenvalues),
            pointer(s),
            pointer(temp),
            pointer(u),
            0,
        )
        == 0
    )
    qb = np.ascontiguousarray(np.einsum("qmn,mn->q", b, d))
    if retained:
        assert (
            native.retained_metric(
                a, r * r, pointer(np.asfortranarray(x)), pointer(s), pointer(u), 0
            )
            == 0
        )
    q = np.empty_like(qb)
    qt = np.empty_like(qb)
    assert (
        native.small_metric(
            a,
            1,
            pointer(e),
            pointer(eigenvalues),
            pointer(qb),
            pointer(qt),
            pointer(q),
            0,
        )
        == 0
    )
    actual_a = d[None] * q[:, None, None] - 0.5 * scale**2 * np.einsum(
        "mi,pij,nj->pmn", c, u, c
    )
    actual_m = -0.5 * np.outer(q, q) + 0.25 * scale**2 * np.einsum("pij,qij->pq", u, u)
    np.testing.assert_allclose(actual_a, ref_a, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(actual_m, ref_m, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("symmetric_pairs", [False, True])
def test_rooted_final_projection_weights_match_independent_energy_derivative(
    native: ct.CDLL,
    symmetric_pairs: bool,
) -> None:
    """Emitted finish/root calls feed both response terms and a raw-energy oracle."""
    rng = np.random.default_rng(1694)
    n, r, a, weight, exchange = 7, 3, 9, 2.0, 0.25
    c = np.linalg.qr(rng.normal(size=(n, r)))[0]
    gauge = np.linalg.qr(rng.normal(size=(r, r)))[0]
    density = weight * c @ c.T
    q = np.linalg.qr(rng.normal(size=(a, a)))[0]
    metric = (q * np.linspace(0.8, 3.0, a)) @ q.T
    root = (q / np.sqrt(np.linspace(0.8, 3.0, a))) @ q.T
    raw = rng.normal(size=(a, n, n))
    raw = (raw + raw.transpose(0, 2, 1)) * 0.5
    fitted = np.einsum("pq,qmn->pmn", root, raw)
    delta_raw = rng.normal(size=raw.shape)
    delta_raw = (delta_raw + delta_raw.transpose(0, 2, 1)) * 0.5
    delta_metric = rng.normal(size=metric.shape)
    delta_metric = (delta_metric + delta_metric.T) * 0.5

    def energy(values: np.ndarray, m: np.ndarray) -> float:
        solved = np.linalg.solve(m, values.reshape(a, -1)).reshape(a, n, n)
        charge = np.einsum("mn,qmn->q", density, values)
        return float(
            0.5 * charge @ np.linalg.solve(m, charge)
            - exchange * np.einsum("qij,ki,qkl,lj->", values, density, solved, density)
        )

    analytic = []
    for frame in (c, c @ gauge):
        coefficients = np.asfortranarray(frame)
        linear = np.empty((a * r, n), order="F")
        for p in range(a):
            for j in range(r):
                linear[p + a * j] = (fitted[p] @ coefficients)[:, j]
        projected = np.full(a * r * r, np.nan)
        rooted = np.full_like(projected, np.nan)
        assert (
            native.finish_project(
                n, r, a, pointer(coefficients), pointer(linear), pointer(projected), 0
            )
            == 0
        )
        pair_count = r * (r + 1) // 2 if symmetric_pairs else r * r
        if symmetric_pairs:
            matrices = np.stack(
                [
                    projected[p * r * r : (p + 1) * r * r].reshape((r, r), order="F")
                    for p in range(a)
                ]
            )
            packed = np.empty((a, pair_count))
            for i in range(r):
                for j in range(i + 1):
                    packed[:, native.symmetric_pair(r, i, j)] = 0.5 * (
                        matrices[:, i, j] + matrices[:, j, i]
                    )
            projected = packed
            rooted = np.full_like(packed, np.nan)
        assert (
            native.retained_metric(
                a,
                pair_count,
                pointer(np.asfortranarray(root)),
                pointer(projected),
                pointer(rooted),
                0,
            )
            == 0
        )
        if symmetric_pairs:
            u = np.empty((a, r, r))
            for i in range(r):
                for j in range(r):
                    u[:, i, j] = rooted[:, native.symmetric_pair(r, i, j)]
        else:
            u = np.stack(
                [
                    rooted[p * r * r : (p + 1) * r * r].reshape((r, r), order="F")
                    for p in range(a)
                ]
            )
        potential = weight * np.trace(u, axis1=1, axis2=2)
        raw_weight = density[None] * potential[
            :, None, None
        ] - 2 * exchange * weight**2 * np.einsum(
            "mi,qij,nj->qmn", coefficients, u, coefficients
        )
        metric_weight = np.asfortranarray(-0.5 * np.outer(potential, potential))
        if symmetric_pairs:
            assert (
                native.symmetric_gram(
                    a,
                    r,
                    exchange * weight**2,
                    pointer(rooted),
                    pointer(metric_weight),
                    0,
                )
                == 0
            )
            metric_weight = np.tril(metric_weight) + np.tril(metric_weight, -1).T
        else:
            metric_weight += exchange * weight**2 * np.einsum("pij,qij->pq", u, u)
        analytic.append(
            float(np.vdot(raw_weight, delta_raw) + np.vdot(metric_weight, delta_metric))
        )
    np.testing.assert_allclose(analytic[0], analytic[1], atol=3e-12, rtol=3e-13)
    for step in (1e-4, 3e-5, 1e-5):
        finite = (
            energy(raw + step * delta_raw, metric + step * delta_metric)
            - energy(raw - step * delta_raw, metric - step * delta_metric)
        ) / (2 * step)
        assert abs(finite - analytic[0]) < 2e-6


@pytest.mark.parametrize("a,r", [(1, 1), (5, 1), (7, 2), (11, 5), (17, 9)])
@pytest.mark.parametrize("coefficient", [-0.75, 0.0, 0.25])
def test_symmetric_metric_gram_matches_full_occupied_contraction(
    native: ct.CDLL,
    a: int,
    r: int,
    coefficient: float,
) -> None:
    """Diagonal weight one, off-diagonal weight two, and beta preserve the adjoint."""
    rng = np.random.default_rng(1709 + a + r)
    full = rng.normal(size=(a, r, r))
    full = 0.5 * (full + full.transpose(0, 2, 1))
    pairs = [(i, i) for i in range(r)] + [(i, j) for i in range(r) for j in range(i)]
    assert [native.symmetric_pair(r, i, j) for i, j in pairs] == list(range(len(pairs)))
    packed = np.ascontiguousarray(np.array([full[:, i, j] for i, j in pairs]).T)
    initial = rng.normal(size=(a, a))
    initial = 0.5 * (initial + initial.T)
    actual = np.asfortranarray(initial)
    expected = initial + coefficient * np.einsum("pij,qij->pq", full, full)
    assert (
        native.symmetric_gram(a, r, coefficient, pointer(packed), pointer(actual), 0)
        == 0
    )
    np.testing.assert_allclose(
        np.tril(actual), np.tril(expected), atol=4e-13, rtol=4e-13
    )
    if native.has_syrk:
        np.testing.assert_array_equal(np.triu(actual, 1), np.triu(initial, 1))
    else:
        np.testing.assert_allclose(actual, expected, atol=4e-13, rtol=4e-13)
    assert native.call_count() == (1 if r == 1 else 2)


def test_symmetric_metric_gram_rejects_invalid_shapes_and_propagates_failures(
    native: ct.CDLL,
) -> None:
    """No invalid shape/alias reaches BLAS; either product can propagate its failure."""
    factors = np.ones((3, 6))
    output = np.zeros((3, 3), order="F")
    for a, rank in [(0, 3), (3, 0), (3, -1), (3, 65536), (3, 2**31 - 1)]:
        assert (
            native.symmetric_gram(a, rank, 1.0, pointer(factors), pointer(output), 0)
            == 7
        )
        assert native.call_count() == 0
    assert native.symmetric_gram(3, 3, 1.0, pointer(factors), pointer(factors), 0) == 7
    for failed_product in (1, 2):
        output.fill(0)
        assert (
            native.symmetric_gram(
                3, 3, 1.0, pointer(factors), pointer(output), failed_product
            )
            == 13
        )
        assert native.call_count() == failed_product
