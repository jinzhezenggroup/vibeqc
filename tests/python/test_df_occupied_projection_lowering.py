"""Exercise emitted production BLAS calls with an independent host BLAS stand-in.

This verifies transpose/stride/output-region/error-propagation contracts, not
CUDA compilation, GPU accuracy/performance or the full VibeQC runtime.
"""

import ctypes as ct
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
EMITTER = HERE.parents[2] / "python/vibeqc_compiler/method/df_occupied_response_cuda.py"

HEADER = r"""
#pragma once
using cublasHandle_t = void*;
using cublasStatus_t = int;
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_INVALID_VALUE=7;
int cublasDgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,const double*,int,const double*,double*,int);
int cublasDgemmStridedBatched(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,long long,const double*,int,long long,
               const double*,double*,int,long long,int);
"""
STANDIN = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
"""
WRAPPERS = r"""
extern "C" int project(int n,int r,int a,int begin,int count,const double* c,
                         const double* values,double* temp,double* out,int fail) {
  calls=0;fail_on=fail;
  return vibeqc::scf::generated::df_occupied_project_panel(
      nullptr,n,r,a,begin,count,c,values,temp,out);
}
extern "C" int call_count() { return calls; }
extern "C" std::size_t tile_size(std::size_t n,std::size_t r,std::size_t a,
                         std::size_t capacity,std::size_t request,bool extra) {
  return vibeqc::scf::generated::df_occupied_projection_tile(n,r,a,capacity,request,extra);
}
extern "C" int small_metric(int a,int rr,const double* e,const double* eigenvalues,
                              const double* s,double* temp,double* out,int fail) {
  calls=0;fail_on=fail;
  auto result=vibeqc::scf::generated::df_occupied_to_metric_eigenbasis(
      nullptr,a,rr,e,s,temp);
  if(result) return result;
  for(int j=0;j<rr;++j) for(int q=0;q<a;++q) temp[q+j*a]/=std::sqrt(eigenvalues[q]);
  return vibeqc::scf::generated::df_occupied_from_metric_eigenbasis(
      nullptr,a,rr,e,temp,out);
}
"""


@pytest.fixture(scope="session")
def native(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
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
    (folder / "cublas_v2.h").write_text(HEADER)
    code = (
        STANDIN
        + "\nnamespace vibeqc::scf::generated {\n"
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
    ptr = ct.POINTER(ct.c_double)
    lib.project.argtypes = [ct.c_int] * 5 + [ptr] * 4 + [ct.c_int]
    lib.project.restype = ct.c_int
    lib.call_count.restype = ct.c_int
    lib.tile_size.argtypes = [ct.c_size_t] * 5 + [ct.c_bool]
    lib.tile_size.restype = ct.c_size_t
    lib.small_metric.argtypes = [ct.c_int] * 2 + [ptr] * 5 + [ct.c_int]
    lib.small_metric.restype = ct.c_int
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


@pytest.mark.parametrize("a,r", [(1, 1), (3, 2), (9, 3), (17, 5)])
@pytest.mark.parametrize("condition", [1.0, 1e3, 1e6, 1e10])
def test_small_metric_layout(
    native: ct.CDLL, a: int, r: int, condition: float
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
    np.testing.assert_allclose(
        out, np.einsum("pq,qij->pij", x, s), atol=5e-14, rtol=2e-12
    )
    assert native.call_count() == 2


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
def test_full_rank_adjoints_using_emitted_helpers(
    native: ct.CDLL, n: int, r: int, a: int, scale: float, condition: float
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
