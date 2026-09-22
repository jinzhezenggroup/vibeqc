"""Translation regression for the actual emitted FP32 AO body, without a GPU.

Host execution validates coordinate narrowing and generated scalar algebra, not
CUDA scheduling or device libm. Allocated CUDA endpoint qualification is separate.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.dft.ao_cuda import (
    emit_grid_policy,
    emit_grid_scientific_kernels,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_emitted_fp32_ao_preserves_local_coordinates(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    policy = emit_grid_policy()
    end = policy.index(";", policy.index("__constant__ int derivatives")) + 1
    policy = policy[:end] + "\n}\n"
    kernels = emit_grid_scientific_kernels()
    begin = kernels.index("__global__ void ao_kernel(")
    end = kernels.index("__global__ void feature_kernel(")
    driver = tmp_path / "ao_translation.cpp"
    driver.write_text(
        """
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#define __device__
#define __noinline__
#define __constant__
#define __global__
using I = std::size_t;
struct Index { I x; };
constexpr Index blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};
double finite(double value, int* error, int) {
  if (!std::isfinite(value)) *error = 1;
  return value;
}
"""
        + policy
        + "using namespace vibeqc_grid_policy;\n"
        + kernels[begin:end]
        + """
int main() {
  constexpr I nao = 4, npoint = 2, jets = 20, count = nao*npoint*jets;
  const int powers[4][3] = {{0,0,0},{1,0,0},{0,2,0},{1,1,1}};
  const std::array<std::size_t,4> selected{3,1,0,2};
  for (bool reordered : {false, true}) {
    std::array<double,3+4+16*nao> basis{};
    basis[3] = 0.75; basis[4] = 1.0;
    basis[5] = 1.25; basis[6] = -0.25;
    for (I i=0; i<nao; ++i) {
      double* record = basis.data()+7+16*i;
      record[2]=2; record[3]=1; record[7]=1;
      for (I axis=0; axis<3; ++axis) record[4+axis]=powers[i][axis];
    }
    std::array<double,6> points{0.25,-0.5,0.75,-0.125,0.375,-0.625};
    std::array<double,count> reference{}, strict{}, translated{};
    int error=0;
    const auto* ids = reordered ? selected.data() : nullptr;
    ao_kernel_fp32(basis.data(),1,2,nao,points.data(),npoint,jets,
                   reference.data(),&error,ids);
    ao_kernel(basis.data(),1,2,nao,points.data(),npoint,jets,
              strict.data(),&error,ids);
    if (error) return 2;
    for (I i=0; i<count; ++i)
      if (std::abs(reference[i]-strict[i])>1e-4) return 3;
    for (double shift : {1024.0,67108864.0,-67108864.0}) {
      auto moved_basis=basis;
      auto moved_points=points;
      for (I axis=0; axis<3; ++axis) moved_basis[axis]+=shift;
      for (double& point : moved_points) point+=shift;
      ao_kernel_fp32(moved_basis.data(),1,2,nao,moved_points.data(),npoint,jets,
                     translated.data(),&error,ids);
      if (error) return 4;
      for (I i=0; i<count; ++i) {
        if (translated[i]!=reference[i]) {
          std::cerr << "translation changed AO jet " << i << ": "
                    << reference[i] << " -> " << translated[i] << '\\n';
          return 1;
        }
      }
    }
  }
}
"""
    )
    executable = tmp_path / "ao_translation"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(driver), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
