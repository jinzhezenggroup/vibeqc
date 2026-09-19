#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cosx_reference.hpp"
#include "dft/cuda_cosx.hpp"
#include "dft/grid.hpp"
#include "molecule/basis.hpp"

#if defined(VIBEQC_COSX_TEST_INTERPOSE)
// Linker interposition is test-only: the production CUDA translation unit and
// actual device transfers remain unchanged. Inject an error after a queued D2H.
namespace fault_injection {
bool fail_download = false, awaiting_download = false, injected = false;
bool fail_get_device = false, get_device_injected = false;
unsigned downloads = 0, failure_fences = 0;
}  // namespace fault_injection
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                              cudaStream_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __real_cudaGetDevice(int*);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* out, const void* in, std::size_t bytes,
                                              cudaMemcpyKind kind, cudaStream_t stream) {
  using namespace fault_injection;
  if (fail_download && kind == cudaMemcpyDeviceToHost && bytes == 4 * sizeof(double)) {
    if (++downloads == 2) {
      injected = true;
      fail_download = false;
      return cudaErrorInvalidValue;
    }
    awaiting_download = true;
  }
  return __real_cudaMemcpyAsync(out, in, bytes, kind, stream);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  const auto status = __real_cudaStreamSynchronize(stream);
  if (fault_injection::awaiting_download) {
    fault_injection::awaiting_download = false;
    if (fault_injection::injected) ++fault_injection::failure_fences;
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaGetDevice(int* device) {
  if (fault_injection::fail_get_device) {
    fault_injection::fail_get_device = false;
    fault_injection::get_device_injected = true;
    return cudaErrorInvalidValue;
  }
  return __real_cudaGetDevice(device);
}
#endif

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "CUDA COSX H2 normalization failed");
  return system;
}

double max_error(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size() == b.size(), "CUDA COSX comparison size mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}

vibeqc::core::System spherical_sdf() {
  vibeqc::core::System system;
  system.atoms = {{2, {0.2, -0.1, 0.3}}};
  system.shells = {
      {0, 0, {{1.4, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {0, 3, {{0.6, 1.0}}},
  };
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "CUDA COSX spherical s/d/f normalization failed");
  return system;
}

std::vector<double> symmetric_density(std::size_t n) {
  std::vector<double> density(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      const double value =
          i == j ? 0.15 + 0.01 * static_cast<double>(i) : 0.015 / static_cast<double>(1 + i + j);
      density[i * n + j] = density[j * n + i] = value;
    }
  }
  return density;
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const int device = 0;
    const auto system = h2();
    const vibeqc::dft::MolecularGrid grid(system, vibeqc::dft::GridSpec{1, 12, 8, 16, 3, 1.0e-12});
    const std::vector<double> density{0.8, 0.2, 0.2, 0.6};
    const auto cpu =
        vibeqc::dft::build_cosx_reference(system, grid.points(), grid.weights(), density,
                                          vibeqc::dft::CosxDensityConvention::rhf_spin_summed);

    std::vector<double> first_exchange;
    for (std::size_t tile : {std::size_t(1), std::size_t(7), grid.point_count()}) {
      vibeqc::dft::CudaCosxStagingPlan plan(system, grid.points(), grid.weights(), tile, device);
      const auto gpu = plan.build(density, vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
      require(max_error(gpu.raw_exchange, cpu.raw_exchange) < 3.0e-12,
              "bounded CUDA COSX raw K differs from the CPU discrete oracle");
      require(max_error(gpu.exchange, cpu.exchange) < 3.0e-12,
              "bounded CUDA COSX symmetric K differs from the CPU discrete oracle");
      require(std::abs(gpu.exchange_energy - cpu.exchange_energy) < 3.0e-12,
              "bounded CUDA COSX exchange energy differs from the CPU oracle");
      if (first_exchange.empty())
        first_exchange = gpu.exchange;
      else
        require(max_error(gpu.exchange, first_exchange) < 3.0e-12,
                "CUDA COSX tile partition changed the discrete result");

      const auto& info = plan.diagnostic();
      require(info.nbf == 2 && info.npoint == grid.point_count() &&
                  info.tile_points == std::min(tile, grid.point_count()) &&
                  info.device_bytes == info.grid_device_bytes + info.cosx_device_bytes &&
                  info.esp_tile_elements == info.tile_points * info.nbf * info.nbf &&
                  info.ao_tile_elements == info.tile_points * info.nbf && info.ao_on_device &&
                  info.esp_on_device && info.assembly_on_device,
              "CUDA COSX staging diagnostics lost the bounded execution contract");
      if (info.tile_points < info.npoint)
        require(info.esp_tile_elements < info.npoint * info.nbf * info.nbf,
                "CUDA COSX staging allocated a global point-by-AO^2 ESP tensor");
    }

    std::vector<double> half_density = density;
    for (double& value : half_density) value *= 0.5;
    vibeqc::dft::CudaCosxStagingPlan spin_plan(system, grid.points(), grid.weights(), 7, device);
    const auto gpu_spin =
        spin_plan.build(half_density, vibeqc::dft::CosxDensityConvention::spin_resolved);
    const auto cpu_spin =
        vibeqc::dft::build_cosx_reference(system, grid.points(), grid.weights(), half_density,
                                          vibeqc::dft::CosxDensityConvention::spin_resolved);
    require(max_error(gpu_spin.exchange, cpu_spin.exchange) < 3.0e-12 &&
                std::abs(gpu_spin.exchange_energy - cpu_spin.exchange_energy) < 3.0e-12,
            "native CUDA COSX single-spin convention differs from the CPU oracle");

    {
      const auto high = spherical_sdf();
      const vibeqc::dft::MolecularGrid high_grid(high,
                                                 vibeqc::dft::GridSpec{1, 3, 3, 6, 3, 1.0e-12});
      const auto high_density = symmetric_density(vibeqc::molecule::ao_count(high));
      const auto high_cpu = vibeqc::dft::build_cosx_reference(
          high, high_grid.points(), high_grid.weights(), high_density,
          vibeqc::dft::CosxDensityConvention::spin_resolved);
      vibeqc::dft::CudaCosxStagingPlan high_plan(high, high_grid.points(), high_grid.weights(), 5,
                                                 device);
      const auto high_gpu =
          high_plan.build(high_density, vibeqc::dft::CosxDensityConvention::spin_resolved);
      require(max_error(high_gpu.raw_exchange, high_cpu.raw_exchange) < 2.0e-11 &&
                  max_error(high_gpu.exchange, high_cpu.exchange) < 2.0e-11 &&
                  std::abs(high_gpu.exchange_energy - high_cpu.exchange_energy) < 2.0e-11,
              "native CUDA COSX spherical d/f expansion differs from the CPU oracle");
    }

    bool bad_density = false;
    try {
      vibeqc::dft::CudaCosxStagingPlan plan(system, grid.points(), grid.weights(), 7, device);
      (void)plan.build(std::vector<double>{1.0},
                       vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
    } catch (const std::invalid_argument&) {
      bad_density = true;
    }
    require(bad_density, "CUDA COSX staging accepted a malformed density");

#if defined(VIBEQC_COSX_TEST_INTERPOSE)
    {
      using namespace fault_injection;
      auto owner = std::make_unique<vibeqc::dft::CudaCosxStagingPlan>(system, grid.points(),
                                                                      grid.weights(), 7, device);
      fail_download = true;
      bool caught = false;
      try {
        (void)owner->build(density, vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
      } catch (const std::runtime_error&) {
        caught = true;
      }
      require(caught && injected, "COSX did not reach the injected second-download error");
      require(!awaiting_download && failure_fences == 1,
              "COSX download failure released host targets before draining their stream");
      const auto retry = owner->build(density, vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
      require(max_error(retry.exchange, cpu.exchange) < 3.0e-12,
              "COSX transfer failure contaminated the next replay");
      fail_get_device = true;
      owner.reset();
      require(get_device_injected, "COSX teardown device-query failure was not exercised");
    }
#endif

    std::cout << "bounded CUDA COSX AO/device assembly matches the discrete CPU oracle\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
