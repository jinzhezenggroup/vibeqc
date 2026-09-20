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


_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "generated_symmetric_matrix_function.cuh"

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  double* p = nullptr;
  if (cudaMalloc(&p, 22 * sizeof(double)) != cudaSuccess) return 1;
  for (int e : {500, 1023}) {
    for (int function = 0; function < 2; ++function) {
      const int exponent = function == 0 ? 700 : 520;
      const int derivative_exponent = function == 0 ? 1051 : 1040;
      double host[6] = {
          1.0, std::ldexp(1.0, exponent), std::ldexp(1.0, e), 0, 0, 0};
      if (cudaMemcpy(p, host, sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess)
        return 2;
      if (function == 0)
        vibeqc::tensor::launch_symmetric_inverse_sqrt_vjp(
            1, p, p + 1, 1e-12, p + 2, p + 3, p + 4, p + 5, nullptr);
      else
        vibeqc::tensor::launch_symmetric_pseudoinverse_vjp(
            1, p, p + 1, 1e-12, p + 2, p + 3, p + 4, p + 5, nullptr);
      if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return 3;
      double result = 0;
      if (cudaMemcpy(&result, p + 5, sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
        return 4;
      if (result != std::ldexp(-1.0, e - derivative_exponent)) {
        std::cerr << result << " wrong for function " << function << " seed " << e
                  << '\n';
        return 5;
      }
    }
  }

  const double cross_fixture[22] = {
      1.0, 0.0, 0.0, 1.0,  // eigenvectors
      1.0, 4.0,            // eigenvalues
      0.0, 1.0, 1.0, 0.0, // symmetric seed
      0.0, 0.0, 0.0, 0.0, // scratch0
      0.0, 0.0, 0.0, 0.0, // scratch1
      0.0, 0.0, 0.0, 0.0  // output
  };
  if (cudaMemcpy(p, cross_fixture, sizeof(cross_fixture), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 6;
  vibeqc::tensor::launch_symmetric_inverse_sqrt_vjp(
      2, p, p + 4, 0.3, p + 6, p + 10, p + 14, p + 18, nullptr);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return 7;
  double cross_result[4] = {};
  if (cudaMemcpy(cross_result, p + 18, sizeof(cross_result), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 8;
  if (std::abs(cross_result[1] - 1.0 / 6.0) > 1e-15 ||
      std::abs(cross_result[2] - 1.0 / 6.0) > 1e-15)
    return 9;

  vibeqc::tensor::launch_symmetric_pseudoinverse_vjp(
      2, p, p + 4, 0.3, p + 6, p + 10, p + 14, p + 18, nullptr);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return 10;
  if (cudaMemcpy(cross_result, p + 18, sizeof(cross_result), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 11;
  if (std::abs(cross_result[1] - 1.0 / 12.0) > 1e-15 ||
      std::abs(cross_result[2] - 1.0 / 12.0) > 1e-15)
    return 12;

  for (int function = 0; function < 2; ++function) {
    for (bool small : {false, true}) {
      const bool inverse = function == 1;
      const int exponent = inverse ? (small ? -600 : 800) : (small ? -700 : 1000);
      const int seed_exponent = inverse ? (small ? -600 : 900) : (small ? -500 : 800);
      const int derivative = inverse ? 2 * exponent : 3 * exponent / 2 + 1;
      double host[6] = {1, std::ldexp(1., exponent), std::ldexp(1., seed_exponent), 0, 0, 0};
      if (cudaMemcpy(p, host, sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess) return 13;
      vibeqc::tensor::launch_symmetric_matrix_function_vjp(
          1, function, p, p + 1, 0, p + 2, p + 3, p + 4, p + 5, nullptr);
      if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 14;
      double result;
      if (cudaMemcpy(&result, p + 5, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 15;
      if (result != std::ldexp(-1., seed_exponent - derivative)) return 16;
      double cross[22] = {1, 0, 0, 1, 0, std::ldexp(1., exponent),
                         0, std::ldexp(1., seed_exponent), std::ldexp(1., seed_exponent), 0};
      if (cudaMemcpy(p, cross, sizeof(cross), cudaMemcpyHostToDevice) != cudaSuccess) return 17;
      vibeqc::tensor::launch_symmetric_matrix_function_vjp(
          2, function, p, p + 4, 0.3, p + 6, p + 10, p + 14, p + 18, nullptr);
      if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 18;
      double out[4];
      if (cudaMemcpy(out, p + 18, sizeof(out), cudaMemcpyDeviceToHost) != cudaSuccess) return 19;
      const double expected = std::ldexp(1., seed_exponent - derivative + (inverse ? 0 : 1));
      if (out[1] != expected || out[2] != expected) return 20;
    }
  }

  cudaFree(p);
  std::cout << "Exact inverse-sqrt/pseudoinverse FP64 response passed\n";
}
"""


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MATRIX_FUNCTION_CUDA_TEST") != "1",
    reason="explicit allocated CUDA tier",
)
def test_generated_cuda_matrix_function_range(tmp_path: typing.Any) -> None:
    from vibeqc_compiler.method.matrix_function_cuda import (
        emit_symmetric_matrix_function_vjp_cuda,
    )

    assert os.environ.get("SLURM_JOB_ID"), (
        "run CUDA qualification in a Slurm allocation"
    )
    nvcc = os.environ.get("VIBEQC_NVCC") or shutil.which("nvcc")
    assert nvcc is not None, "explicit CUDA tier requires NVCC"
    header = tmp_path / "generated_symmetric_matrix_function.cuh"
    header.write_text(emit_symmetric_matrix_function_vjp_cuda())
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


def test_native_weighted_spectral_response_survives_nonrepresentable_coefficient(
    tmp_path: Path,
) -> None:
    """Exact powers-of-two oracle; the final VJP, not its factor, must fit FP64."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path / "weighted-range"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(ROOT / "tests/native/matrix_function_scale_cases.cpp"),
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
