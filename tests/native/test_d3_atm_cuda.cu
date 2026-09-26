#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d3_atm.hpp"

namespace {
using namespace vibeqc::dft::dispersion;

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <class T, std::size_t N>
T* copy_table(const std::array<T, N>& source) {
  T* device = nullptr;
  check(cudaMalloc(&device, sizeof(T) * N), "cudaMalloc table");
  check(cudaMemcpy(device, source.data(), sizeof(T) * N, cudaMemcpyHostToDevice),
        "cudaMemcpy table");
  return device;
}

__global__ void evaluate_kernel(const std::int32_t* z, const double* xyz,
                                D3ATMParameters parameters, D3Tables tables, double* workspace,
                                double* energy, double* gradient, std::int32_t* status) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *status = static_cast<std::int32_t>(evaluate_d3_bj_atm(
        4, z, xyz, parameters, tables, workspace, d3_atm_workspace_elements(4), energy, gradient));
}

}  // namespace

int main() {
  using namespace vibeqc::dft::dispersion;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    constexpr std::array<std::int32_t, 4> numbers{6, 8, 7, 1};
    constexpr std::array<double, 12> coordinates{0.0, 0.0, 0.0, 2.5,  0.1, 0.0,
                                                 0.6, 2.7, 0.2, -1.2, 0.8, 2.4};
    const D3ATMParameters parameters{1.0, 0.0, 4.5, 2.0};

    std::vector<double> host_workspace(d3_atm_workspace_elements(4));
    double cpu_energy = 0.0;
    std::array<double, 12> cpu_gradient{};
    if (evaluate_d3_bj_atm(4, numbers.data(), coordinates.data(), parameters, d3_host_tables(),
                           host_workspace.data(), host_workspace.size(), &cpu_energy,
                           cpu_gradient.data()) != D3Status::success)
      throw std::runtime_error("CPU ATM reference failed");

    std::int32_t *dz = nullptr, *dstatus = nullptr;
    double *dxyz = nullptr, *dworkspace = nullptr, *denergy = nullptr, *dgradient = nullptr;
    check(cudaMalloc(&dz, sizeof(numbers)), "cudaMalloc z");
    check(cudaMalloc(&dxyz, sizeof(coordinates)), "cudaMalloc xyz");
    check(cudaMalloc(&dworkspace, sizeof(double) * host_workspace.size()), "cudaMalloc workspace");
    check(cudaMalloc(&denergy, sizeof(double)), "cudaMalloc energy");
    check(cudaMalloc(&dgradient, sizeof(cpu_gradient)), "cudaMalloc gradient");
    check(cudaMalloc(&dstatus, sizeof(std::int32_t)), "cudaMalloc status");
    check(cudaMemcpy(dz, numbers.data(), sizeof(numbers), cudaMemcpyHostToDevice), "cudaMemcpy z");
    check(cudaMemcpy(dxyz, coordinates.data(), sizeof(coordinates), cudaMemcpyHostToDevice),
          "cudaMemcpy xyz");

    auto* delements = copy_table(d3_data::kElements);
    auto* dpairs = copy_table(d3_data::kPairs);
    auto* dcn = copy_table(d3_data::kReferenceCn);
    auto* dc6 = copy_table(d3_data::kReferenceC6);
    D3Tables device_tables{delements, dpairs, dcn, dc6};

    evaluate_kernel<<<1, 1>>>(dz, dxyz, parameters, device_tables, dworkspace, denergy, dgradient,
                              dstatus);
    check(cudaGetLastError(), "ATM kernel launch");
    check(cudaDeviceSynchronize(), "ATM kernel synchronize");

    double gpu_energy = 0.0;
    std::array<double, 12> gpu_gradient{};
    std::int32_t gpu_status = -1;
    check(cudaMemcpy(&gpu_energy, denergy, sizeof(double), cudaMemcpyDeviceToHost), "copy energy");
    check(cudaMemcpy(gpu_gradient.data(), dgradient, sizeof(gpu_gradient), cudaMemcpyDeviceToHost),
          "copy gradient");
    check(cudaMemcpy(&gpu_status, dstatus, sizeof(gpu_status), cudaMemcpyDeviceToHost),
          "copy status");

    if (gpu_status != static_cast<std::int32_t>(D3Status::success))
      throw std::runtime_error("GPU ATM evaluation failed");
    if (std::abs(gpu_energy - cpu_energy) > 2.0e-18)
      throw std::runtime_error("CPU/CUDA ATM energy parity failed");
    for (std::size_t i = 0; i < gpu_gradient.size(); ++i)
      if (std::abs(gpu_gradient[i] - cpu_gradient[i]) > 2.0e-17)
        throw std::runtime_error("CPU/CUDA ATM gradient parity failed");

    cudaFree(delements);
    cudaFree(dpairs);
    cudaFree(dcn);
    cudaFree(dc6);
    cudaFree(dz);
    cudaFree(dxyz);
    cudaFree(dworkspace);
    cudaFree(denergy);
    cudaFree(dgradient);
    cudaFree(dstatus);
    std::cout << "D3(BJ)-ATM CPU/CUDA parity passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
