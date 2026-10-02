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
#include "posthf/ri_mp2_cuda.hpp"
#include "scf/mean_field.hpp"
#include "tensor/cuda_runtime.cuh"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::core::System h2() {
  generativeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<generativeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "H2 setup failed");
  return system;
}

void ri_mp2_block_planner() {
  constexpr std::size_t fixed = 2ULL << 20;
  const auto full =
      generativeqc::mp2::plan_ri_mp2_cuda_blocks(fixed, 64ULL << 20, 120, 20, 100, 180);
  require(full.full_resident && full.virtual_block == 100 && full.peak_bytes <= (64ULL << 20),
          "RI-MP2 planner did not select resident full B");

  const auto blocked =
      generativeqc::mp2::plan_ri_mp2_cuda_blocks(fixed, 8ULL << 20, 120, 20, 100, 180);
  require(!blocked.full_resident && blocked.virtual_block == 27 && blocked.j_batch == 4 &&
              blocked.peak_bytes == 8364608 && blocked.peak_bytes <= (8ULL << 20),
          "RI-MP2 planner did not select the expected bounded B block");

  bool rejected = false;
  try {
    (void)generativeqc::mp2::plan_ri_mp2_cuda_blocks(fixed, fixed, 120, 20, 100, 180);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "RI-MP2 planner accepted a budget without one B block");
}

void batched_eri_derivatives() {
  // Distinct shell slots deliberately share atoms. Full p quartets force
  // primitive-buffer splits; many s quartets force output-tile splits.
  for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
    generativeqc::core::System system;
    system.basis_representation = representation;
    system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {1.2, 0.3, -0.4}}, {1, {-0.2, 0.7, 0.5}}};
    const std::vector<generativeqc::core::Primitive> p{{1.4, .3}, {.7, .4}, {.2, .5}};
    system.shells = {{0, 0, {{1.1, 1.0}}}, {1, 1, p},           {0, 1, p},
                     {2, 2, {{.8, 1.0}}},  {1, 3, {{.6, 1.0}}}, {2, 0, {{.9, 1.0}}}};
    std::string detail;
    require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "batch fixture normalization failed");
    struct Tile {
      std::array<std::size_t, 4> shells;
      std::vector<double> weights;
    };
    std::vector<Tile> tiles;
    for (std::size_t index = 0; index < 310; ++index) {
      std::array<std::size_t, 4> shells;
      if (index < 5)
        shells = {1, 2, 1, 2};
      else if (index == 5)
        shells = {4, 0, 1, 0};
      else if (index == 6)
        shells = {3, 0, 1, 2};
      else if (index == 7)
        shells = {1, 0, 0, 0};
      else
        shells = {0, 5, 0, 5};
      std::size_t count = 1;
      for (const auto shell : shells) {
        const auto l = system.shells[shell].angular_momentum;
        count *= representation == GENERATIVEQC_BASIS_SPHERICAL
                     ? 2 * l + 1
                     : generativeqc::molecule::cartesian_count(l);
      }
      std::vector<double> weights(count);
      for (std::size_t k = 0; k < count; ++k)
        weights[k] = index == 8 ? 0.0 : .03 * std::sin(1.0 + k + 3 * index);
      tiles.push_back({shells, std::move(weights)});
    }
    std::vector<double> expected(9);
    for (const auto& tile : tiles) {
      const auto center = generativeqc::integrals::contract_weighted_eri_shell_derivative(
          system, tile.shells, tile.weights);
      for (std::size_t slot = 0; slot < 4; ++slot)
        for (std::size_t axis = 0; axis < 3; ++axis)
          expected[3 * system.shells[tile.shells[slot]].atom_index + axis] +=
              center[3 * slot + axis];
    }
    std::size_t records = 0;
    for (const auto budget : {8ULL << 20, 32768ULL, 8192ULL}) {
      generativeqc::posthf::CudaEriDerivativeBatch batch(0, system, budget, true);
      std::vector<double> actual(9);
      for (const auto& tile : tiles) batch.append(tile.shells, tile.weights, actual);
      batch.finish(actual);
      require(batch.numeric_capacity_bytes() <= budget, "ERI batch exceeded its numeric cap");
      require(batch.batched() == (budget != 8192), "ERI batch/fallback admission changed");
      if (!records) records = batch.diagnostic().primitive_records;
      require(batch.diagnostic().primitive_records == records,
              "batching repeated or lost primitives");
      if (budget == (8ULL << 20))
        require(batch.diagnostic().consumer_calls < 10, "large batch retained per-shell work");
      for (std::size_t i = 0; i < actual.size(); ++i)
        require(std::abs(actual[i] - expected[i]) < 2e-9,
                "batched derivative differs from independent CPU shell oracle");
      const auto calls = batch.diagnostic().consumer_calls;
      batch.finish(actual);
      require(batch.diagnostic().consumer_calls == calls, "empty finish repeated GPU work");
      bool rejected = false;
      try {
        batch.append(tiles.front().shells, {}, actual);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "batch accepted malformed shell weights");
    }
  }
  const auto system = h2();
  generativeqc::posthf::CudaEriDerivativeBatch tiny(0, system, 1, true);
  std::vector<double> gradient(6);
  bool rejected = false;
  try {
    tiny.append({0, 1, 0, 1}, std::array<double, 1>{.3}, gradient);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "tiny derivative stage budget was accepted");
  std::cout
      << "Batched ERI: Cartesian/spherical s/p/d/f, split quartets, repeated atoms, budgets PASS\n";
}

void derivative_and_force_parity() {
  const auto system = h2();
  const std::array<std::size_t, 4> shells{0, 1, 0, 1};
  const std::array<double, 1> weights{0.37};
  const auto cpu_shell =
      generativeqc::integrals::contract_weighted_eri_shell_derivative(system, shells, weights);
  std::array<double, 12> cuda_shell{};
  std::string detail;
  const auto shell_status = generativeqc::posthf::contract_weighted_eri_shell_derivative_cuda(
      0, system, shells, weights, 64ULL << 20, cuda_shell, detail);
  require(shell_status == GENERATIVEQC_STATUS_SUCCESS, "CUDA shell derivative failed");
  for (std::size_t i = 0; i < cuda_shell.size(); ++i)
    require(std::abs(cuda_shell[i] - cpu_shell[i]) < 2e-11,
            "CUDA shell derivative differs from CPU oracle");

  auto unchanged = cuda_shell;
  unchanged.fill(123.0);
  require(
      generativeqc::posthf::contract_weighted_eri_shell_derivative_cuda(
          0, system, shells, weights, 1, unchanged, detail) == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
      "tiny CUDA shell derivative budget was accepted");
  require(
      std::all_of(unchanged.begin(), unchanged.end(), [](double value) { return value == 123.0; }),
      "failed CUDA shell derivative modified caller output");

  generativeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0.0;
  options.energy_tolerance = options.density_tolerance = 1e-12;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = generativeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "H2 reference did not converge");
  generativeqc::posthf::RawSource source(system);
  generativeqc::response::GmresOptions response;
  response.relative_tolerance = 1e-12;
  response.absolute_tolerance = 1e-13;
  response.restart = 8;
  response.max_iterations = 40;
  response.max_workspace_bytes = 64ULL << 20;
  const auto cpu = generativeqc::mp2::conventional_force_cpu(*hf.reference, source, 256ULL << 20,
                                                             1e-10, 1e-10, response);
  const auto cuda = generativeqc::mp2::conventional_force_cuda(*hf.reference, source, 256ULL << 20,
                                                               1e-10, 1e-10, response, 0);
  require(cuda.forces.size() == cpu.forces.size(), "CUDA force shape differs from CPU");
  for (std::size_t i = 0; i < cuda.forces.size(); ++i)
    require(std::abs(cuda.forces[i] - cpu.forces[i]) < 2e-9,
            "CUDA conventional force differs from CPU");
  require(cuda.planned_endpoint_peak_bytes <= (256ULL << 20),
          "CUDA force exceeded its planned endpoint budget");
}
}  // namespace

int main(int argc, char** argv) {
  if (!std::getenv("GENERATIVEQC_MP2_CUDA_TEST")) return 77;
  try {
    batched_eri_derivatives();
    // Memcheck the scientific batch paths without the intentional cudaMalloc
    // failure below, which compute-sanitizer correctly reports as an API error.
    if (argc == 2 && std::string(argv[1]) == "--eri-batches-only") return 0;
    ri_mp2_block_planner();
    derivative_and_force_parity();
    bool cuda_oom = false, blas_oom = false;
    try {
      generativeqc_tensor::cuda_check(cudaErrorMemoryAllocation);
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      cuda_oom = true;
    }
    try {
      generativeqc_tensor::blas_check(CUBLAS_STATUS_ALLOC_FAILED);
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      blas_oom = true;
    }
    if (!cuda_oom || !blas_oom) throw std::runtime_error("allocation status type lost");
    const std::array<std::size_t, 4> shape{2, 2, 2, 2}, tile{1, 1, 1, 1};
    const auto plan = generativeqc::posthf::numeric_block_plan(INT_MAX, 0, 0, shape, tile, true);
    std::size_t free_bytes = 0, total_bytes = 0;
    generativeqc_tensor::cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
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
