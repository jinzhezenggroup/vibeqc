#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/df_source_internal.hpp"
#include "scf/cuda/df_source_kernels.hpp"
#include "scf/cuda/topology.hpp"

namespace vibeqc::scf {

/** Bounded raw/transformed tile replay and public source diagnostics. Geometry/topology allocations
 * remain owned by the source until destruction. */
// Public opaque handle; private source state owns only integral metadata and transforms.
struct CudaDensityFittingIntegralSource {
  void* implementation{};
};

namespace {
using namespace cuda_execution;

void destroy_cuda_density_fitting_integral_source_impl(
    CudaDensityFittingIntegralSourceImpl* source) noexcept {
  delete source;
}

std::size_t cuda_density_fitting_integral_source_device_bytes_impl(
    const CudaDensityFittingIntegralSourceImpl* source) noexcept {
  return source == nullptr ? 0U : source->device_bytes;
}

vibeqc_status generate_cuda_density_fitting_transformed_tile_impl(
    CudaDensityFittingIntegralSourceImpl* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t auxiliary_begin, std::size_t auxiliary_count,
    std::int64_t derivative_coordinate, const double* inverse_square_root, void* stream_handle,
    double* output, std::string& detail, bool apply_metric_transform) {
  detail.clear();
  std::size_t pair_total = 0;
  if (source != nullptr && !checked_multiply(source->public_nbf, source->public_nbf, pair_total)) {
    detail = "bounded DF transformed tile dimensions overflow size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  if (source == nullptr || output == nullptr ||
      (apply_metric_transform && inverse_square_root == nullptr) || stream_handle == nullptr ||
      system >= source->batch_size || pair_begin > pair_total ||
      pair_count > pair_total - pair_begin || auxiliary_begin > source->public_naux ||
      auxiliary_count > source->public_naux - auxiliary_begin) {
    detail = "bounded DF transformed tile dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // Empty blocks at valid offsets are legal no-ops and perform no device work.
  if (pair_count == 0U || auxiliary_count == 0U) return VIBEQC_STATUS_SUCCESS;
  if (derivative_coordinate >= 0) {
    // The public API indexes coordinates relative to the selected system,
    // while the packed recurrence metadata uses fleet-global atom offsets.
    // Validate against this system's atom span before translating below;
    // validating against total_atoms would accept an out-of-range coordinate
    // for every system after the first and could read a neighbor's geometry.
    std::size_t coordinate_count = 0;
    if (system + 1U >= source->host_atom_offsets.size() || source->host_atom_offsets[system] < 0 ||
        source->host_atom_offsets[system + 1U] < source->host_atom_offsets[system] ||
        !checked_multiply(static_cast<std::size_t>(source->host_atom_offsets[system + 1U] -
                                                   source->host_atom_offsets[system]),
                          3U, coordinate_count) ||
        static_cast<std::size_t>(derivative_coordinate) >= coordinate_count) {
      detail = "bounded DF transformed tile derivative coordinate is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  cudaError_t cuda_error = cudaSetDevice(source->device_id);
  if (cuda_error != cudaSuccess) return source_cuda_status(cuda_error);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
  constexpr unsigned source_threads = 128U;
  const unsigned outputs_per_block = derivative_coordinate < 0 && source->value_mapping == 2U
                                         ? source_threads / 32U
                                         : source_threads;
  std::size_t tile_elements = 0;
  if (!checked_multiply(pair_count, auxiliary_count, tile_elements)) {
    detail = "bounded DF transformed tile size overflows size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  if (tile_elements >
      static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) * outputs_per_block) {
    detail = "bounded DF transformed tile launch exceeds CUDA grid limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const unsigned blocks =
      static_cast<unsigned>((tile_elements + outputs_per_block - 1U) / outputs_per_block);
  // Public callers address derivatives relative to one system.  The packed
  // recurrence metadata is fleet-global, so translate the coordinate to the
  // selected system's atom range before evaluating the tile.
  if (derivative_coordinate >= 0 &&
      (source->host_atom_offsets[system] >
       (std::numeric_limits<std::int64_t>::max() - derivative_coordinate) / 3)) {
    detail = "bounded DF transformed tile derivative coordinate overflows";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + source->host_atom_offsets[system] * 3;
  runtime::cuda_trace::trace_tile(system, pair_begin, pair_count, auxiliary_begin, auxiliary_count,
                                  derivative_coordinate, apply_metric_transform);
  runtime::cuda_trace::TraceRegion generation(
      derivative_coordinate >= 0 ? "three_center_derivatives"
      : apply_metric_transform   ? "transformed_three_center_generation"
                                 : "raw_three_center_generation",
      stream);
  if (derivative_coordinate < 0) {
    launch_build_cuda_df_transformed_tile_kernel(
        false, blocks, source_threads, 0, stream, source->batch, source->cartesian_nbf,
        source->cartesian_naux, source->public_nbf, source->public_naux, source->dummy_index,
        system, pair_begin, pair_count, auxiliary_begin, auxiliary_count,
        system_derivative_coordinate, source->orbital_to_cartesian, source->auxiliary_to_cartesian,
        inverse_square_root, apply_metric_transform, output, source->value_mapping);
  } else {
    launch_build_cuda_df_transformed_tile_kernel(
        true, blocks, source_threads, 0, stream, source->batch, source->cartesian_nbf,
        source->cartesian_naux, source->public_nbf, source->public_naux, source->dummy_index,
        system, pair_begin, pair_count, auxiliary_begin, auxiliary_count,
        system_derivative_coordinate, source->orbital_to_cartesian, source->auxiliary_to_cartesian,
        inverse_square_root, apply_metric_transform, output);
  }
  cuda_error = cudaPeekAtLastError();
  return cuda_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS : source_cuda_status(cuda_error);
}

vibeqc_status generate_cuda_density_fitting_metric_derivative_tile_impl(
    CudaDensityFittingIntegralSourceImpl* source, std::size_t system,
    std::size_t auxiliary_row_begin, std::size_t auxiliary_row_count,
    std::int64_t derivative_coordinate, void* stream_handle, double* output, std::string& detail) {
  detail.clear();
  if (source == nullptr || output == nullptr || stream_handle == nullptr ||
      system >= source->batch_size || derivative_coordinate < 0 ||
      system + 1U >= source->host_atom_offsets.size() || source->host_atom_offsets[system] < 0 ||
      source->host_atom_offsets[system + 1U] < source->host_atom_offsets[system] ||
      auxiliary_row_begin > source->public_naux ||
      auxiliary_row_count > source->public_naux - auxiliary_row_begin) {
    detail = "bounded DF metric derivative tile dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(source->host_atom_offsets[system + 1U] -
                                                          source->host_atom_offsets[system]);
  std::size_t coordinate_count = 0;
  if (!checked_multiply(atom_count, 3U, coordinate_count) ||
      static_cast<std::size_t>(derivative_coordinate) >= coordinate_count) {
    detail = "bounded DF metric derivative coordinate is invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::size_t elements = 0;
  if (!checked_multiply(auxiliary_row_count, source->public_naux, elements) || elements == 0U ||
      elements > static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) * 128U) {
    detail = "bounded DF metric derivative tile dimensions overflow";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t cuda_error = cudaSetDevice(source->device_id);
  if (cuda_error != cudaSuccess) return source_cuda_status(cuda_error);
  if (source->host_atom_offsets[system] >
      (std::numeric_limits<std::int64_t>::max() - derivative_coordinate) / 3) {
    detail = "bounded DF metric derivative coordinate overflows";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t global_coordinate =
      derivative_coordinate + source->host_atom_offsets[system] * 3;
  const unsigned blocks = static_cast<unsigned>((elements + 127U) / 128U);
  launch_build_cuda_df_metric_source_kernel(
      true, blocks, 128U, 0, reinterpret_cast<cudaStream_t>(stream_handle), source->batch,
      source->cartesian_nbf, source->cartesian_naux, source->public_naux, source->dummy_index,
      system, auxiliary_row_begin, auxiliary_row_count, global_coordinate,
      source->auxiliary_to_cartesian, output);
  cuda_error = cudaPeekAtLastError();
  return cuda_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS : source_cuda_status(cuda_error);
}

}  // namespace

vibeqc_status create_cuda_density_fitting_integral_source(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems, CudaDensityFittingIntegralSource** source,
    std::vector<double>& metrics, std::size_t& nbf, std::size_t& naux, std::string& detail) {
  if (source == nullptr) {
    detail = "bounded DF source output handle is null";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *source = nullptr;
  CudaDensityFittingIntegralSourceImpl* implementation = nullptr;
  const vibeqc_status status = create_cuda_density_fitting_integral_source_impl(
      device_id, orbital_systems, auxiliary_systems, &implementation, metrics, nbf, naux, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  auto* handle = new (std::nothrow) CudaDensityFittingIntegralSource{};
  if (handle == nullptr) {
    destroy_cuda_density_fitting_integral_source_impl(implementation);
    detail = "bounded DF source handle allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  handle->implementation = implementation;
  *source = handle;
  return VIBEQC_STATUS_SUCCESS;
}

void destroy_cuda_density_fitting_integral_source(
    CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr) return;
  destroy_cuda_density_fitting_integral_source_impl(
      static_cast<CudaDensityFittingIntegralSourceImpl*>(source->implementation));
  delete source;
}

std::size_t cuda_density_fitting_integral_source_device_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr) return 0U;
  return cuda_density_fitting_integral_source_device_bytes_impl(
      static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation));
}

CudaDensityFittingSourceDiagnostic cuda_density_fitting_integral_source_diagnostic(
    const CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr || source->implementation == nullptr) return {};
  const auto& implementation =
      *static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation);
  const char* mapping = implementation.value_mapping == 1U   ? "component"
                        : implementation.value_mapping == 2U ? "primitive"
                                                             : "auxiliary";
  return {"generated_rys", mapping, true, true};
}

std::size_t cuda_density_fitting_integral_source_host_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr || source->implementation == nullptr) return 0U;
  const auto* implementation =
      static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation);
  const std::size_t handle_bytes = sizeof(CudaDensityFittingIntegralSource);
  return implementation->host_bytes > std::numeric_limits<std::size_t>::max() - handle_bytes
             ? std::numeric_limits<std::size_t>::max()
             : implementation->host_bytes + handle_bytes;
}

std::size_t cuda_density_fitting_integral_source_host_peak_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr || source->implementation == nullptr) return 0U;
  const auto* implementation =
      static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation);
  const std::size_t handle_bytes = sizeof(CudaDensityFittingIntegralSource);
  return implementation->host_peak_bytes > std::numeric_limits<std::size_t>::max() - handle_bytes
             ? std::numeric_limits<std::size_t>::max()
             : implementation->host_peak_bytes + handle_bytes;
}

std::size_t cuda_density_fitting_integral_source_coordinate_count(
    const CudaDensityFittingIntegralSource* source) noexcept {
  if (source == nullptr || source->implementation == nullptr) return 0U;
  const auto* implementation =
      static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation);
  if (implementation->host_atom_offsets.size() < 2U) return 0U;
  std::size_t maximum = 0U;
  for (std::size_t system = 0; system + 1U < implementation->host_atom_offsets.size(); ++system) {
    const std::int64_t begin = implementation->host_atom_offsets[system];
    const std::int64_t end = implementation->host_atom_offsets[system + 1U];
    if (begin < 0 || end < begin) return 0U;
    const std::uint64_t atoms = static_cast<std::uint64_t>(end - begin);
    if (atoms > std::numeric_limits<std::size_t>::max() / 3U) {
      return std::numeric_limits<std::size_t>::max();
    }
    maximum = std::max(maximum, static_cast<std::size_t>(atoms) * std::size_t{3});
  }
  return maximum;
}

bool cuda_density_fitting_integral_source_matches(const CudaDensityFittingIntegralSource* source,
                                                  int device_id, std::size_t batch_size,
                                                  std::size_t nbf, std::size_t naux) noexcept {
  if (source == nullptr || source->implementation == nullptr) return false;
  const auto* implementation =
      static_cast<const CudaDensityFittingIntegralSourceImpl*>(source->implementation);
  return implementation->device_id == device_id && implementation->batch_size == batch_size &&
         implementation->public_nbf == nbf && implementation->public_naux == naux;
}

vibeqc_status generate_cuda_density_fitting_transformed_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t auxiliary_begin, std::size_t auxiliary_count,
    std::int64_t derivative_coordinate, const double* inverse_square_root, void* stream_handle,
    double* output, std::string& detail) {
  if (source == nullptr || source->implementation == nullptr) {
    detail = "bounded DF source handle is null";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return generate_cuda_density_fitting_transformed_tile_impl(
      static_cast<CudaDensityFittingIntegralSourceImpl*>(source->implementation), system,
      pair_begin, pair_count, auxiliary_begin, auxiliary_count, derivative_coordinate,
      inverse_square_root, stream_handle, output, detail, true);
}

vibeqc_status generate_cuda_density_fitting_raw_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t auxiliary_begin, std::size_t auxiliary_count,
    std::int64_t derivative_coordinate, void* stream_handle, double* output, std::string& detail) {
  if (source == nullptr || source->implementation == nullptr) {
    detail = "bounded DF source handle is null";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return generate_cuda_density_fitting_transformed_tile_impl(
      static_cast<CudaDensityFittingIntegralSourceImpl*>(source->implementation), system,
      pair_begin, pair_count, auxiliary_begin, auxiliary_count, derivative_coordinate, nullptr,
      stream_handle, output, detail, false);
}

vibeqc_status generate_cuda_density_fitting_metric_derivative_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t auxiliary_row_begin,
    std::size_t auxiliary_row_count, std::int64_t derivative_coordinate, void* stream_handle,
    double* output, std::string& detail) {
  if (source == nullptr || source->implementation == nullptr) {
    detail = "bounded DF source handle is null";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return generate_cuda_density_fitting_metric_derivative_tile_impl(
      static_cast<CudaDensityFittingIntegralSourceImpl*>(source->implementation), system,
      auxiliary_row_begin, auxiliary_row_count, derivative_coordinate, stream_handle, output,
      detail);
}

}  // namespace vibeqc::scf
