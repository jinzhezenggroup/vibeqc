// SPDX-License-Identifier: GPL-3.0-or-later
// Synchronous, single-molecule CUDA qualification harness, not production runtime.
#include <cuda_runtime.h>

#include <vector>

#include "d3_bj_reference.hpp"
namespace {
__global__ void run(std::int64_t n, const double* x, const double* p, double* y, double* s,
                    int* ok) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *ok = vibeqc_d3_baseline::evaluate(n, x, p, y, s) ? 0 : 2;
}
struct Allocation {
  void* p = nullptr;
  ~Allocation() {
    if (p) cudaFree(p);
  }
};
}  // namespace
extern "C" int vibeqc_d3_reference(std::int64_t n, const double* input, const double* parameters,
                                   double* output) {
  if (n < 1 || n > 512 || !input || !parameters || !output) return 1;
  const std::int64_t size = 12 * n + 51 * n * (n - 1) / 2;
  Allocation data;
  const std::int64_t total = size + 5 + (1 + 3 * n) + 16 * n;
  if (cudaMalloc(&data.p, total * sizeof(double) + sizeof(int)) != cudaSuccess) return 3;
  auto* x = static_cast<double*>(data.p);
  auto* p = x + size;
  auto* y = p + 5;
  auto* s = y + 1 + 3 * n;
  auto* ok = reinterpret_cast<int*>(s + 16 * n);
  if (cudaMemcpy(x, input, size * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(p, parameters, 5 * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    return 4;
  run<<<1, 1>>>(n, x, p, y, s, ok);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 5;
  int status = 2;
  if (cudaMemcpy(&status, ok, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
  if (status) return status;
  return cudaMemcpy(output, y, (1 + 3 * n) * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess
             ? 0
             : 6;
}

extern "C" int vibeqc_d3_backend() { return 1; }
