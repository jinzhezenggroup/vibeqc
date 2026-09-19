"""Compile the provider-free wheel ABI without relying on installed cuBLAS."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_provider_free_sgemm_declarations_and_imports(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; enum cudaDataType { CUDA_R_32F=0 };\n"
    )
    (tmp_path / "library_types.h").write_text(
        "#pragma once\nenum libraryPropertyType { MAJOR_VERSION=0 };\n"
    )
    source = tmp_path / "abi.cpp"
    source.write_text(r"""
#include <type_traits>
#include "vibeqc_nvidia_host_api.h"
using Gemm = cublasStatus_t (*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,
 int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
using Batch = cublasStatus_t (*)(cublasHandle_t,cublasOperation_t,cublasOperation_t,
 int,int,int,const float*,const float*,int,long long,const float*,int,long long,
 const float*,float*,int,long long,int);
static_assert(std::is_same_v<decltype(&cublasSgemm), Gemm>);
static_assert(std::is_same_v<decltype(&cublasSgemm_v2), Gemm>);
static_assert(std::is_same_v<decltype(&cublasSgemmStridedBatched), Batch>);
""")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src/runtime/nvidia_host_api"),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    cmake = (ROOT / "cmake/VibeQCCudaImplib.cmake").read_text()
    symbols = cmake.split("set(VIBEQC_CUBLAS_SYMBOLS", 1)[1].split(")", 1)[0].split()
    assert "cublasSgemm_v2" in symbols
    assert "cublasSgemmStridedBatched" in symbols


def test_sgemm_trampolines_link_without_providers_and_forward_abi(tmp_path):
    """Exercise real lazy import stubs, including 64-bit strides, without CUDA."""
    import platform
    import sys

    from tools.generate_cuda_implib import generate

    machine = platform.machine().lower()
    targets = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    if platform.system() != "Linux" or machine not in targets:
        pytest.skip("provider-free wheel trampolines require supported Linux ELF")
    cc, cxx, readelf = (shutil.which(name) for name in ("cc", "c++", "readelf"))
    if not all((cc, cxx, readelf)):
        pytest.skip("C/C++ compiler and ELF inspector required")
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; enum cudaDataType { CUDA_R_32F=0 };\n"
    )
    (tmp_path / "library_types.h").write_text(
        "#pragma once\nenum libraryPropertyType { MAJOR_VERSION=0 };\n"
    )
    includes = ["-I", str(tmp_path), "-I", str(ROOT / "src/runtime/nvidia_host_api")]
    cmake = (ROOT / "cmake/VibeQCCudaImplib.cmake").read_text()
    symbols = cmake.split("set(VIBEQC_CUBLAS_SYMBOLS", 1)[1].split(")", 1)[0].split()
    provider = tmp_path / "mock-cublas.so"
    generate(
        "libcublas.so",
        symbols,
        str(provider),
        targets[machine],
        ROOT / "cmake/3rdparty/implib",
        tmp_path,
    )

    def run(*command):
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    # Compile the actual generated C/assembly separately, just as CMake does.
    objects = []
    for suffix in ("init.c", "tramp.S"):
        source = tmp_path / f"libcublas.so.{suffix}"
        obj = tmp_path / f"{suffix}.o"
        run(cc, "-fPIC", "-c", str(source), "-o", str(obj))
        objects.append(str(obj))
    consumer = tmp_path / "consumer.cpp"
    consumer.write_text(r"""
#include "cublas_v2.h"
extern "C" int probe() {
  float alpha=2, beta=3, a=4, b=5, c=6;
  auto handle=reinterpret_cast<cublasHandle_t>(&a);
  auto status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,7,8,9,
                         &alpha,&a,11,&b,12,&beta,&c,13);
  if(status!=CUBLAS_STATUS_SUCCESS || c!=58) return 1;
  const long long stride=(1LL<<34);
  status=cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,7,8,9,
          &alpha,&a,11,stride+1,&b,12,stride+2,&beta,&c,13,stride+3,17);
  return status==CUBLAS_STATUS_SUCCESS && c==214 ? 0 : 2;
}
""")
    library_path = tmp_path / "consumer.so"
    run(
        cxx,
        "-std=c++17",
        "-shared",
        "-fPIC",
        *includes,
        str(consumer),
        *objects,
        "-Wl,-z,defs",
        "-ldl",
        "-o",
        str(library_path),
    )
    dynamic = run(readelf, "-d", str(library_path)).stdout
    assert not any(
        name in dynamic for name in ("libcublas", "libcudart", "libcusolver")
    )
    # A provider-free wheel must load even before its runtime provider exists.
    assert not provider.exists()

    fake = tmp_path / "provider.cpp"
    fake.write_text(r"""
#include "cublas_v2.h"
static bool arguments(cublasHandle_t h,int m,int n,int k,const float* alpha,
                      const float* a,int lda,const float* b,int ldb,
                      const float* beta,float* c,int ldc) {
  return h==reinterpret_cast<cublasHandle_t>(const_cast<float*>(a)) &&
         m==7 && n==8 && k==9 && lda==11 && ldb==12 && ldc==13 &&
         *alpha==2 && *beta==3 && *a==4 && *b==5 && c;
}
extern "C" cublasStatus_t cublasSgemm_v2(cublasHandle_t h,cublasOperation_t ta,
  cublasOperation_t tb,int m,int n,int k,const float* alpha,const float* a,int lda,
  const float* b,int ldb,const float* beta,float* c,int ldc) {
  if(ta!=CUBLAS_OP_T || tb!=CUBLAS_OP_N ||
     !arguments(h,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  *c=*alpha * *a * *b + *beta * *c;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h,cublasOperation_t ta,
  cublasOperation_t tb,int m,int n,int k,const float* alpha,const float* a,int lda,
  long long sa,const float* b,int ldb,long long sb,const float* beta,float* c,int ldc,
  long long sc,int batches) {
  const long long stride=(1LL<<34);
  if(ta!=CUBLAS_OP_N || tb!=CUBLAS_OP_T || sa!=stride+1 || sb!=stride+2 ||
     sc!=stride+3 || batches!=17 || !arguments(h,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc))
    return CUBLAS_STATUS_INVALID_VALUE;
  *c=*alpha * *a * *b + *beta * *c;
  return CUBLAS_STATUS_SUCCESS;
}
""")
    available_provider = tmp_path / "available-provider.so"
    run(
        cxx,
        "-std=c++17",
        "-shared",
        "-fPIC",
        *includes,
        str(fake),
        "-o",
        str(available_provider),
    )
    # A bad lazy resolver can abort: keep that failure inside a subprocess.
    run(
        sys.executable,
        "-c",
        """
import ctypes
import pathlib
import sys
library_path, available_path, provider_path = map(pathlib.Path, sys.argv[1:])
assert not provider_path.exists()
library = ctypes.CDLL(str(library_path))
library.probe.argtypes = []
library.probe.restype = ctypes.c_int
available_path.rename(provider_path)
assert library.probe() == 0
assert library.probe() == 0
""",
        str(library_path),
        str(available_provider),
        str(provider),
    )
