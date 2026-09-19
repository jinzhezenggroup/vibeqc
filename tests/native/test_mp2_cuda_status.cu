#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/block_capacity_generated.hpp"
#include "posthf/cuda_derivative.hpp"
#include "posthf/cuda_transform.hpp"
#include "posthf/mp2_force.hpp"
#include "posthf/raw_source.hpp"
#include "scf/rhf.hpp"
#include "tensor/cuda_runtime.cuh"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<vibeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "H2 setup failed");
  return system;
}

void derivative_and_force_parity() {
  const auto system = h2();
  const std::array<std::size_t, 4> shells{0, 1, 0, 1};
  const std::array<double, 1> weights{0.37};
  const auto cpu_shell =
      vibeqc::integrals::contract_weighted_eri_shell_derivative(system, shells, weights);
  std::array<double, 12> cuda_shell{};
  std::string detail;
  const auto shell_status = vibeqc::posthf::contract_weighted_eri_shell_derivative_cuda(
      0, system, shells, weights, 64ULL << 20, cuda_shell, detail);
  require(shell_status == VIBEQC_STATUS_SUCCESS, "CUDA shell derivative failed");
  for (std::size_t i = 0; i < cuda_shell.size(); ++i)
    require(std::abs(cuda_shell[i] - cpu_shell[i]) < 2e-11,
            "CUDA shell derivative differs from CPU oracle");

  auto unchanged = cuda_shell;
  unchanged.fill(123.0);
  require(vibeqc::posthf::contract_weighted_eri_shell_derivative_cuda(
              0, system, shells, weights, 1, unchanged, detail) == VIBEQC_STATUS_OUT_OF_MEMORY,
          "tiny CUDA shell derivative budget was accepted");
  require(
      std::all_of(unchanged.begin(), unchanged.end(), [](double value) { return value == 123.0; }),
      "failed CUDA shell derivative modified caller output");

  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0.0;
  options.energy_tolerance = options.density_tolerance = 1e-12;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "H2 reference did not converge");
  vibeqc::posthf::RawSource source(system);
  vibeqc::response::GmresOptions response;
  response.relative_tolerance = 1e-12;
  response.absolute_tolerance = 1e-13;
  response.restart = 8;
  response.max_iterations = 40;
  response.max_workspace_bytes = 64ULL << 20;
  const auto cpu = vibeqc::mp2::conventional_force_cpu(*hf.reference, source, 256ULL << 20, 1e-10,
                                                       1e-10, response);
  const auto cuda = vibeqc::mp2::conventional_force_cuda(*hf.reference, source, 256ULL << 20, 1e-10,
                                                         1e-10, response, 0);
  require(cuda.forces.size() == cpu.forces.size(), "CUDA force shape differs from CPU");
  for (std::size_t i = 0; i < cuda.forces.size(); ++i)
    require(std::abs(cuda.forces[i] - cpu.forces[i]) < 2e-9,
            "CUDA conventional force differs from CPU");
  require(cuda.planned_endpoint_peak_bytes <= (256ULL << 20),
          "CUDA force exceeded its planned endpoint budget");
}
}  // namespace

int main() {
  if (!std::getenv("VIBEQC_MP2_CUDA_TEST")) return 77;
  try {
    derivative_and_force_parity();
    bool cuda_oom = false, blas_oom = false;
    try {
      vibeqc_tensor::cuda_check(cudaErrorMemoryAllocation);
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      cuda_oom = true;
    }
    try {
      vibeqc_tensor::blas_check(CUBLAS_STATUS_ALLOC_FAILED);
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      blas_oom = true;
    }
    if (!cuda_oom || !blas_oom) throw std::runtime_error("allocation status type lost");
    const std::array<std::size_t, 4> shape{2, 2, 2, 2}, tile{1, 1, 1, 1};
    const auto plan = vibeqc::posthf::numeric_block_plan(INT_MAX, 0, 0, shape, tile, true);
    std::size_t free_bytes = 0, total_bytes = 0;
    vibeqc_tensor::cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (plan.allocation_bytes <= total_bytes)
      throw std::runtime_error("OOM probe requires a capacity larger than this device");
    // Failure happens at allocation, before the coefficient pointer is read.
    // No competing application memory is touched and no stress loop is used.
    void* handle = nullptr;
    double coefficients[8]{};
    char error[2048]{};
    const auto status = posthf_cuda_create_v1(0, INT_MAX, shape.data(), tile.data(), coefficients,
                                              plan.allocation_bytes, &handle, error, sizeof(error));
    if (handle) posthf_cuda_destroy_v1(handle);
    if (status != 2 || handle)
      throw std::runtime_error("CG10 native allocation failure category/rollback lost");
    std::cout << "CUDA and cuBLAS OOM types; native CG10 OOM category and rollback passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
