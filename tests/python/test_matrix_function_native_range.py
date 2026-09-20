"""Native spectral coefficients retain representable FP64 subnormal responses."""

import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_matrix_function_range_uses_no_overflowing_products(
    tmp_path: typing.Any,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "range.cpp"
    source.write_text(r"""
#include "tensor/symmetric_matrix_function.hpp"
#include <array>
#include <cmath>
#include <iostream>
using namespace vibeqc::tensor;
int main() {
  const std::array<double, 1> q{1.0};
  const std::array<std::uint8_t, 1> keep{1};
  for (const auto function : {SymmetricMatrixFunction::pseudoinverse,
                              SymmetricMatrixFunction::inverse_sqrt}) {
    const bool inverse = function == SymmetricMatrixFunction::pseudoinverse;
    const int exponent = inverse ? 520 : 700;
    for (const int seed_exponent : {500, 1023}) {
      const auto actual = symmetric_matrix_function_vjp(
          std::array<double,1>{std::ldexp(1.0, exponent)}, q, keep,
          std::array<double,1>{std::ldexp(1.0, seed_exponent)}, function, 0.0);
      // Powers of two give exactly representable coefficients and responses.
      const auto expected = std::ldexp(-1.0, seed_exponent - (inverse ? 1040 : 1051));
      if (actual[0] != expected) {
        std::cerr << actual[0] << " != " << expected << '\n';
        return 1;
      }
    }
  }
}
""")
    binary = tmp_path / "range"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(source),
            str(ROOT / "src/tensor/symmetric_matrix_function.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


_CUDA_SOURCE = '#include <cuda_runtime.h>\n#include <cmath>\n#include <cstdlib>\n#include <iostream>\n#include "generated_symmetric_matrix_function.cuh"\nint main() {\n  if (!std::getenv("SLURM_JOB_ID")) return 77;\n  double *p=nullptr; if(cudaMalloc(&p,6*sizeof(double))!=cudaSuccess)return 1;\n  for(int e:{500,1023}){\n    double host[6]={1.0,std::ldexp(1.0,520),std::ldexp(1.0,e),0,0,0};\n    if(cudaMemcpy(p,host,sizeof(host),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;\n    vibeqc::tensor::launch_symmetric_pseudoinverse_vjp(1,p,p+1,1e-12,p+2,p+3,p+4,p+5,nullptr);\n    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 3;\n    double result=0;\n    if(cudaMemcpy(&result,p+5,sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;\n    if(result!=std::ldexp(-1.0,e-1040)){std::cerr<<result<<" wrong for "<<e<<\'\\n\';return 5;}\n  }\n  cudaFree(p); std::cout<<"Exact FP64 spectral range and seed symmetrization passed\\n";\n}\n'


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MATRIX_FUNCTION_CUDA_TEST") != "1",
    reason="explicit allocated CUDA tier",
)
def test_generated_cuda_matrix_function_range(tmp_path: typing.Any) -> None:
    from vibeqc_compiler.method.matrix_function_cuda import emit_pseudoinverse_vjp_cuda

    assert os.environ.get("SLURM_JOB_ID"), (
        "run CUDA qualification in a Slurm allocation"
    )
    nvcc = os.environ.get("VIBEQC_NVCC") or shutil.which("nvcc")
    assert nvcc is not None, "explicit CUDA tier requires NVCC"
    header = tmp_path / "generated_symmetric_matrix_function.cuh"
    header.write_text(emit_pseudoinverse_vjp_cuda())
    source = tmp_path / "range.cu"
    source.write_text(_CUDA_SOURCE)
    binary = tmp_path / "range-cuda"
    subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-arch=" + os.environ.get("VIBEQC_MATRIX_FUNCTION_ARCH", "sm_120"),
            "-I" + str(tmp_path),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
