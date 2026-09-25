#include "methods/gfn2_runtime_bridge.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gfn2_cpu_execution.hpp"
#if defined(VIBEQC_HAS_GFN2_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif

namespace vibeqc::methods::detail {
namespace {

template <typename T>
vibeqc_xtb_const_buffer_t input_buffer(std::span<const T> values) {
  return {values.empty() ? nullptr : values.data(), values.size_bytes(), VIBEQC_XTB_MEMORY_HOST,
          0u};
}

template <typename T>
vibeqc_xtb_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          VIBEQC_XTB_MEMORY_HOST, 0u};
}

Gfn2RuntimeStatus map_status(vibeqc_xtb_status_t status) noexcept {
  switch (status) {
    case VIBEQC_XTB_STATUS_SUCCESS:
      return Gfn2RuntimeStatus::kSuccess;
    case VIBEQC_XTB_STATUS_INVALID_ARGUMENT:
      return Gfn2RuntimeStatus::kInvalidArgument;
    case VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE:
      return Gfn2RuntimeStatus::kBackendUnavailable;
    case VIBEQC_XTB_STATUS_NOT_SUPPORTED:
      return Gfn2RuntimeStatus::kNotSupported;
    case VIBEQC_XTB_STATUS_NOT_IMPLEMENTED:
      return Gfn2RuntimeStatus::kNotImplemented;
    case VIBEQC_XTB_STATUS_ALLOCATION_FAILED:
      return Gfn2RuntimeStatus::kAllocationFailed;
    case VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED:
      return Gfn2RuntimeStatus::kNotConverged;
    case VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED:
      return Gfn2RuntimeStatus::kEigensolverFailed;
    default:
      return Gfn2RuntimeStatus::kInternalError;
  }
}

}  // namespace

struct Gfn2RuntimeBridge::Impl {
  Impl(Gfn2RuntimeBackend requested_backend, int device_id) : backend(requested_backend) {
#if defined(VIBEQC_HAS_GFN2_CUDA)
    if (backend == Gfn2RuntimeBackend::kCuda)
      cuda_cache =
          std::make_unique<vibeqc::xtb::detail::Gfn2CudaExecutionCache>(device_id, nullptr);
#else
    (void)device_id;
#endif
  }

  Gfn2RuntimeBackend backend;
  vibeqc::xtb::detail::Gfn2CpuExecutionCache cpu_cache;
#if defined(VIBEQC_HAS_GFN2_CUDA)
  std::unique_ptr<vibeqc::xtb::detail::Gfn2CudaExecutionCache> cuda_cache;
#endif
};

Gfn2RuntimeBridge::Gfn2RuntimeBridge(Gfn2RuntimeBackend backend, int device_id)
    : impl_(std::make_unique<Impl>(backend, device_id)) {}

Gfn2RuntimeBridge::~Gfn2RuntimeBridge() = default;
Gfn2RuntimeBridge::Gfn2RuntimeBridge(Gfn2RuntimeBridge&&) noexcept = default;
Gfn2RuntimeBridge& Gfn2RuntimeBridge::operator=(Gfn2RuntimeBridge&&) noexcept = default;

Gfn2RuntimeResult Gfn2RuntimeBridge::execute(const Gfn2RuntimeRequest& request) {
  Gfn2RuntimeResult result;
  if (request.atomic_numbers.empty() ||
      request.positions.size() != 3u * request.atomic_numbers.size()) {
    result.status = Gfn2RuntimeStatus::kInvalidArgument;
    result.detail = "GFN2 runtime request has inconsistent atom/position extents";
    return result;
  }
  if (request.multiplicity == 0u) {
    result.status = Gfn2RuntimeStatus::kInvalidArgument;
    result.detail = "GFN2 runtime multiplicity must be positive";
    return result;
  }

  const auto atom_count = static_cast<std::int64_t>(request.atomic_numbers.size());
  std::vector<std::int64_t> atom_offsets{0, atom_count};
  std::vector<double> molecular_charges{static_cast<double>(request.charge)};
  std::vector<std::int32_t> unpaired_electrons{
      static_cast<std::int32_t>(request.multiplicity - 1u)};
  // Preserve the public GFN2 restricted/shared-orbital open-shell contract.
  // Multiplicity sets the unpaired-electron count; unrestricted GFN2 is a
  // separate capability and must not be inferred at this adapter boundary.
  std::vector<std::int32_t> spin_channels{1};

  vibeqc_xtb_batch_t batch{};
  vibeqc_xtb_compute_options_t options{};
  vibeqc_xtb_batch_result_t output{};
  batch.struct_size = sizeof(batch);
  batch.api_version = VIBEQC_XTB_API_VERSION;
  options.struct_size = sizeof(options);
  options.api_version = VIBEQC_XTB_API_VERSION;
  output.struct_size = sizeof(output);
  output.api_version = VIBEQC_XTB_API_VERSION;

  batch.batch_size = 1;
  batch.total_atoms = atom_count;
  batch.atom_offsets = input_buffer<std::int64_t>(atom_offsets);
  batch.atomic_numbers = input_buffer<std::int32_t>(request.atomic_numbers);
  batch.positions = input_buffer<double>(request.positions);
  batch.molecular_charges = input_buffer<double>(molecular_charges);
  batch.unpaired_electrons = input_buffer<std::int32_t>(unpaired_electrons);
  batch.spin_channels = input_buffer<std::int32_t>(spin_channels);

  options.model = VIBEQC_XTB_MODEL_GFN2_XTB;
  options.flags =
      static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY) |
      (request.compute_forces ? static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES) : 0u) |
      (request.compute_atomic_charges
           ? static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)
           : 0u);
  options.max_scc_iterations = request.maximum_iterations;
  options.charge_tolerance = request.charge_tolerance;
  options.energy_tolerance = request.energy_tolerance;
  options.electronic_temperature = VIBEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.scc_start_mode = VIBEQC_XTB_SCC_START_FRESH;
  options.scc_mixer = VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
  options.scc_mixer_history = request.mixer_history;
  options.scc_mixer_damping = 0.4;

  std::vector<double> energies(1u);
  std::vector<double> forces(request.compute_forces ? 3u * request.atomic_numbers.size() : 0u);
  std::vector<double> atomic_charges(request.compute_atomic_charges ? request.atomic_numbers.size()
                                                                    : 0u);
  std::vector<std::int32_t> iterations(1u);
  std::vector<std::uint8_t> converged(1u);
  std::vector<std::int32_t> statuses(1u);
  output.energies = output_buffer(energies);
  output.forces = output_buffer(forces);
  output.atomic_charges = output_buffer(atomic_charges);
  output.scc_iterations = output_buffer(iterations);
  output.scc_converged = output_buffer(converged);
  output.per_system_status = output_buffer(statuses);

  std::string execution_error;
  vibeqc_xtb_status_t execution_status = VIBEQC_XTB_STATUS_NOT_IMPLEMENTED;
  if (impl_->backend == Gfn2RuntimeBackend::kCpu) {
    execution_status = vibeqc::xtb::detail::execute_restricted_gfn2_cpu(
        impl_->cpu_cache, batch, options, output, execution_error);
#if defined(VIBEQC_HAS_GFN2_CUDA)
  } else if (impl_->backend == Gfn2RuntimeBackend::kCuda && impl_->cuda_cache) {
    execution_status = vibeqc::xtb::detail::execute_restricted_gfn2_cuda(
        *impl_->cuda_cache, batch, options, output, execution_error);
#endif
  }
  if (execution_status != VIBEQC_XTB_STATUS_SUCCESS) {
    result.status = map_status(execution_status);
    result.detail = std::move(execution_error);
    return result;
  }

  const auto system_status = static_cast<vibeqc_xtb_status_t>(statuses[0]);
  if (system_status == VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED) {
    result.status = Gfn2RuntimeStatus::kEigensolverFailed;
    result.detail = "generalized eigensolver failed";
    return result;
  }
  if (system_status != VIBEQC_XTB_STATUS_SUCCESS &&
      system_status != VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED) {
    // Match the former public-method boundary exactly: only the eigensolver
    // status has a dedicated public mapping; every other unexpected terminal
    // per-system status remains an internal runtime failure.
    result.status = Gfn2RuntimeStatus::kInternalError;
    result.detail = "unexpected per-system runtime status";
    return result;
  }

  result.status = Gfn2RuntimeStatus::kSuccess;
  result.energy = energies[0];
  result.forces = std::move(forces);
  result.atomic_charges = std::move(atomic_charges);
  result.iterations = iterations[0] < 0 ? 0u : static_cast<unsigned>(iterations[0]);
  result.converged = system_status == VIBEQC_XTB_STATUS_SUCCESS && converged[0] != 0u;
  return result;
}

const char* gfn2_runtime_status_name(Gfn2RuntimeStatus status) noexcept {
  switch (status) {
    case Gfn2RuntimeStatus::kSuccess:
      return "success";
    case Gfn2RuntimeStatus::kInvalidArgument:
      return "invalid argument";
    case Gfn2RuntimeStatus::kBackendUnavailable:
      return "backend unavailable";
    case Gfn2RuntimeStatus::kNotSupported:
      return "not supported";
    case Gfn2RuntimeStatus::kNotImplemented:
      return "not implemented";
    case Gfn2RuntimeStatus::kAllocationFailed:
      return "allocation failed";
    case Gfn2RuntimeStatus::kNotConverged:
      return "SCC not converged";
    case Gfn2RuntimeStatus::kEigensolverFailed:
      return "eigensolver failed";
    case Gfn2RuntimeStatus::kInternalError:
      return "internal error";
  }
  return "internal error";
}

}  // namespace vibeqc::methods::detail
