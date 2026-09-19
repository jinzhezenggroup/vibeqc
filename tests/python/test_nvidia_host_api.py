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
