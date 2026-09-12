#include <stdexcept>
#include <utility>
#include <vector>

#include "scf/cuda_batch.hpp"
#include "scf/rhf.hpp"

namespace vibeqc::scf {

// Host-only single-system adapters share the bucket execution and error
// contract. Keep them outside the kernel translation unit so host changes
// do not require adding more code to the large CUDA implementation.
ScfResult run_rhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  const vibeqc_status status = result.front().status;
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) {
    throw std::invalid_argument("CUDA RHF received invalid arguments");
  }
  if (status != VIBEQC_STATUS_SUCCESS && status != VIBEQC_STATUS_SCF_NOT_CONVERGED) {
    throw std::runtime_error("CUDA RHF execution failed");
  }
  return std::move(result.front().scf);
}

ScfResult run_uhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_uhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA UHF returned no result");
  const vibeqc_status status = result.front().status;
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) {
    throw std::invalid_argument("CUDA UHF received invalid arguments");
  }
  if (status != VIBEQC_STATUS_SUCCESS && status != VIBEQC_STATUS_SCF_NOT_CONVERGED) {
    throw std::runtime_error("CUDA UHF execution failed");
  }
  return std::move(result.front().scf);
}

}  // namespace vibeqc::scf
