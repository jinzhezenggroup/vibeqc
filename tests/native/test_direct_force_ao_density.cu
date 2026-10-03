#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "scf/cuda/direct_screening.cuh"

using namespace generativeqc::scf::cuda_execution;

__global__ void evaluate(const double* density, bool unrestricted, double bound, double tolerance,
                         unsigned offset, int* result) {
  *result = unrestricted ? direct_ao_force_survives_density_products<true>(
                               bound, tolerance, 4, offset, offset, density, 0, 1, 2, 3)
                         : direct_ao_force_survives_density_products<false>(
                               bound, tolerance, 4, offset, offset, density, 0, 1, 2, 3);
}

void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** Explicit predicate edge cases, not a numerical ERI/force oracle. */
int main() try {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  double* device_density = nullptr;
  int* device_result = nullptr;
  checked(cudaMalloc(&device_density, 48 * sizeof(double)));
  checked(cudaMalloc(&device_result, sizeof(int)));
  unsigned passed = 0;
  const auto check_case = [&](const std::array<double, 48>& density, bool unrestricted,
                              bool expected, double bound = 1.0, double tolerance = 1e-12,
                              unsigned offset = 0) {
    checked(cudaMemcpy(device_density, density.data(), sizeof(density), cudaMemcpyHostToDevice));
    evaluate<<<1, 1>>>(device_density, unrestricted, bound, tolerance, offset, device_result);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    int actual = -1;
    checked(cudaMemcpy(&actual, device_result, sizeof(actual), cudaMemcpyDeviceToHost));
    if (actual != int(expected)) throw std::runtime_error("gate case " + std::to_string(passed));
    ++passed;
  };
  std::array<double, 48> density{};
  for (bool unrestricted : {false, true}) {
    check_case(density, unrestricted, false);
    check_case(density, unrestricted, true, 1.0, 0.0);
    check_case(density, unrestricted, true, std::numeric_limits<double>::infinity());
    check_case(density, unrestricted, true, std::numeric_limits<double>::quiet_NaN());
    for (unsigned spin = 0; spin < (unrestricted ? 2U : 1U); ++spin) {
      const unsigned base = 16 * spin;
      density = {};
      density[base + 4] = density[base + 14] = 1.0;
      check_case(density, unrestricted, true);
      density = {};
      density[base + 2] = density[base + 7] = 1.0;
      check_case(density, unrestricted, true);
      density = {};
      density[base + 3] = density[base + 6] = 1.0;
      check_case(density, unrestricted, true);
      density = {};
      density[base + 1] = density[base + 11] = 1e-8;
      check_case(density, unrestricted, false);
      density[base + 1] = density[base + 11] = 1e-6;
      check_case(density, unrestricted, true, 1.0, 1e-3);
      density[base + 1] = std::numeric_limits<double>::quiet_NaN();
      check_case(density, unrestricted, true);
      density = {};
    }
  }
  density[2] = density[16 + 7] = 1.0;
  check_case(density, true, false);
  density[16 + 2] = density[7] = -1.0;
  check_case(density, true, true);
  density = {};
  density[1] = 1.0;
  density[16 + 1] = -1.0;
  density[11] = 1.0;
  check_case(density, true, false);
  density = {};
  density[16 + 1] = density[16 + 11] = 1.0;
  check_case(density, false, true, 1.0, 1e-12, 16);
  density = {};
  density[32 + 2] = density[32 + 7] = 1.0;
  check_case(density, true, true, 1.0, 1e-12, 16);
  checked(cudaFree(device_density));
  checked(cudaFree(device_result));
  std::printf("%u AO density predicate cases PASS\n", passed);
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
