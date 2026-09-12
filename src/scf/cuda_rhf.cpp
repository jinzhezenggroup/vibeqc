#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "integrals/ecp_cuda.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/allocation_measurement.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/aot_shell_registry.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/direct_angular_fock.hpp"
#include "scf/cuda/direct_angular_force.hpp"
#include "scf/cuda/direct_bounded_dddd.hpp"
#include "scf/cuda/direct_bounded_exact_force.hpp"
#include "scf/cuda/direct_bounded_fallback.hpp"
#include "scf/cuda/direct_bounded_pages.hpp"
#include "scf/cuda/direct_bounded_tasks.hpp"
#include "scf/cuda/direct_cached_tensor_kernels.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_density_bounds.hpp"
#include "scf/cuda/direct_generated_tasks.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_packed_fock_kernels.hpp"
#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/direct_queue_diagnostics.hpp"
#include "scf/cuda/direct_queue_scan.hpp"
#include "scf/cuda/direct_reference_force.hpp"
#include "scf/cuda/direct_resident_tasks.hpp"
#include "scf/cuda/direct_schwarz_kernels.hpp"
#include "scf/cuda/direct_tile_compaction.hpp"
#include "scf/cuda/direct_tile_validation.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/nuclear_kernels.hpp"
#include "scf/cuda/one_electron_derivatives.cuh"
#include "scf/cuda/one_electron_export_kernels.hpp"
#include "scf/cuda/one_electron_force_reference.hpp"
#include "scf/cuda/one_electron_force_workspace.hpp"
#include "scf/cuda/one_electron_values.cuh"
#include "scf/cuda/one_electron_view.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/queue_plan.hpp"
#include "scf/cuda/reference_export.cuh"
#include "scf/cuda/resources.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/scf_convergence_kernels.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda/scf_state_kernels.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda/weighted_eri_kernels.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/cuda_eigensolver_policy.hpp"
#include "scf/cuda_weighted_eri.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/generated_shell_task.hpp"
#include "scf/mean_field.hpp"
#include "scf/rhf.hpp"
#include "tensor/metrics.hpp"

namespace vibeqc::scf {

namespace {

using namespace cuda_execution;

// Runtime policy parsing lives in a host-only C++ TU. Keep the numerical CUDA
// source's call sites unchanged while making policy-only edits incremental.
// The legacy spellings below remain documented here for source-level tooling;
// their actual ``std::getenv`` calls and thresholds are implemented by
// ``scf/cuda/rhf_policy.cpp`` so editing policy does not rebuild this TU:
//   std::getenv("VIBEQC_FINAL_FOCK_REBUILD")
//   std::getenv("VIBEQC_PPPS_SIGNATURE_BUCKETING")
//   std::getenv("VIBEQC_PPPS_BLOCK_THREADS")
//   std::getenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")
//   std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR")
//   std::getenv("VIBEQC_PSSS_RESIDENT_BRA")
//   scalar_one_electron_force_environment == nullptr
//   kTightConvergedFockReuseDensityRms = 1.0e-12
//   kExpandedConvergedFockReuseDensityTolerance = 1.0e-9
//   kExpandedConvergedFockReuseDensityRms = 2.0e-9
//   kAutoMixedPrecisionErrorBudgetFraction = 6.25e-02
//   kFloat32UnitRoundoff = 5.9604644775390625e-08
using cuda_policy::bounded_direct_aot_only_diagnostic_requested;
using cuda_policy::bounded_direct_count_diagnostic_requested;
using cuda_policy::bounded_direct_fock_only_diagnostic_requested;
using cuda_policy::bounded_direct_streaming_override_requested;
using cuda_policy::bounded_fock_class_timing_requested;
using cuda_policy::configured_mixed_precision_fock_threshold;
using cuda_policy::converged_fock_reuse_density_rms;
using cuda_policy::direct_tile_validation_requested;
using cuda_policy::force_density_product_screening_requested;
using cuda_policy::graph_native_eigensolver_override_requested;
using cuda_policy::MixedPrecisionFockPolicy;
using cuda_policy::MixedPrecisionItemPolicy;
using cuda_policy::one_electron_force_scalar_requested;
using cuda_policy::ppps_resident_block_threads_requested;
using cuda_policy::ppps_signature_bucketing_requested;
using cuda_policy::ppss_signature_bucketing_requested;
using cuda_policy::psps_signature_bucketing_requested;
using cuda_policy::resident_ppps_bra_requested;
using cuda_policy::resident_psss_bra_requested;
using cuda_policy::resolve_mixed_precision_fock_policy;
using cuda_policy::resolve_mixed_precision_item;
using cuda_policy::reuse_converged_fock_requested;
using cuda_policy::xsyev_probe_skip_diagnostic_requested;

/** Prepare the resident ppps histogram, prefix, descriptors, and ket records. */
cudaError_t prepare_ppps_resident_tasks(
    cudaStream_t stream, std::size_t active_tile_capacity, std::size_t total_shell_pairs,
    DeviceBatch batch, const std::uint32_t* active_tile_count,
    const ActiveShellQuartetTile* active_tiles, GeneratedPppsResidentTask* resident_tasks,
    GeneratedShellTask* resident_ket_tasks, std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_bra_offsets, std::uint32_t* resident_bra_write_counts,
    std::uint32_t* resident_signature_counts, std::uint32_t* resident_signature_offsets,
    std::uint32_t* resident_ket_signatures, std::uint64_t enabled_mask) {
  if (active_tile_capacity == 0 || total_shell_pairs == 0 || resident_tasks == nullptr ||
      resident_ket_tasks == nullptr || resident_bra_counts == nullptr ||
      resident_bra_offsets == nullptr || resident_bra_write_counts == nullptr ||
      (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) == 0U) {
    return cudaSuccess;
  }
  cudaError_t error =
      cudaMemsetAsync(resident_bra_counts, 0, total_shell_pairs * sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr) {
    error = cudaMemsetAsync(resident_signature_counts, 0,
                            total_shell_pairs * kPppsSignatureBucketCount * sizeof(std::uint32_t),
                            stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks =
      static_cast<unsigned>((active_tile_capacity + preparation_threads - 1) / preparation_threads);
  launch_count_ppps_resident_bra_tasks_kernel(preparation_blocks, preparation_threads, 0, stream,
                                              batch, active_tile_capacity, active_tile_count,
                                              active_tiles, enabled_mask, resident_bra_counts,
                                              resident_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  launch_prefix_ppps_resident_bra_tasks_kernel(1, 1, 0, stream, total_shell_pairs,
                                               resident_bra_counts, resident_bra_offsets,
                                               resident_bra_write_counts, resident_tasks);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr && resident_signature_offsets != nullptr) {
    launch_prefix_ppps_resident_signature_buckets_kernel(
        static_cast<unsigned>(total_shell_pairs), 1, 0, stream, total_shell_pairs,
        resident_bra_offsets, resident_signature_counts, resident_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  launch_materialize_ppps_resident_bra_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, active_tile_capacity,
      active_tile_count, active_tiles, resident_bra_offsets, resident_bra_write_counts,
      resident_signature_offsets, resident_signature_counts, resident_ket_tasks,
      resident_ket_signatures);
  return cudaPeekAtLastError();
}

/** Prepare compact exact-class slices shared by generated Fock and force code. */
cudaError_t prepare_generated_shell_tasks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    std::uint64_t low_order_signature_mask, std::uint32_t* low_order_signature_counts,
    std::uint32_t* low_order_signature_offsets, std::uint64_t enabled_mask,
    const std::uint64_t* enabled_mask_pointer, bool exclude_resident_ppps) {
  if ((enabled_mask == 0U && enabled_mask_pointer == nullptr) || generated_task_capacity == 0 ||
      total_tile_capacity == 0) {
    return cudaSuccess;
  }
  cudaError_t error =
      cudaMemsetAsync(generated_task_counts, 0,
                      detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr) {
    error = cudaMemsetAsync(low_order_signature_counts, 0,
                            kLowOrderSignatureElementCount * sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks =
      static_cast<unsigned>((total_tile_capacity + preparation_threads - 1) / preparation_threads);
  launch_classify_generated_shell_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, total_tile_capacity,
      active_tile_offsets, active_tile_counts, active_tiles, enabled_mask, enabled_mask_pointer,
      exclude_resident_ppps, generated_task_counts, generated_shell_classes,
      low_order_signature_mask, low_order_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  launch_prefix_generated_shell_task_counts_kernel(
      1, 1, 0, stream, generated_task_counts, generated_task_offsets, generated_task_write_counts,
      generated_task_heads);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr && low_order_signature_offsets != nullptr) {
    launch_prefix_low_order_signature_counts_kernel(
        1, 1, 0, stream, generated_task_offsets, low_order_signature_mask,
        low_order_signature_counts, low_order_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  launch_materialize_generated_shell_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, total_tile_capacity, active_tiles,
      generated_shell_classes, generated_task_offsets, generated_task_write_counts, generated_tasks,
      low_order_signature_mask, low_order_signature_offsets, low_order_signature_counts);
  return cudaPeekAtLastError();
}

/**
 * Bucket all enabled generated classes once, then launch their force slices.
 *
 * Counts, offsets, and worker heads remain device-resident, so adding an AOT
 * class does not add a host synchronization or another scan of every active
 * quartet. The generated persistent kernels apply their class offset when
 * loading tasks from the shared compact allocation.
 */
cudaError_t launch_generated_shell_class_forces(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    std::uint32_t* low_order_signature_counts, std::uint32_t* low_order_signature_offsets,
    GeneratedPppsResidentTask* resident_ppps_tasks, GeneratedShellTask* resident_ppps_ket_tasks,
    std::uint32_t* resident_ppps_bra_counts, std::uint32_t* resident_ppps_bra_offsets,
    std::uint32_t* resident_ppps_bra_write_counts, std::uint32_t* resident_ppps_signature_counts,
    std::uint32_t* resident_ppps_signature_offsets, std::uint32_t* resident_ppps_signatures,
    std::size_t total_shell_pairs, bool resident_ppps_enabled,
    bool resident_ppps_signature_bucketing, bool psps_signature_bucketing,
    bool ppss_signature_bucketing, unsigned resident_ppps_block_threads,
    unsigned persistent_worker_blocks, bool unrestricted, std::uint64_t enabled_mask,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    double* forces) {
  bool use_resident_ppps =
      resident_ppps_enabled && (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) != 0U;
  if (total_tile_capacity == 0 || (enabled_mask == 0U && !use_resident_ppps) ||
      (generated_task_capacity == 0 && !use_resident_ppps)) {
    return cudaSuccess;
  }
  cudaError_t error = cudaSuccess;
  if (use_resident_ppps) {
    std::size_t ppps_tile_offset = 0;
    for (unsigned order = 0; order < kPppsAngularOrder; ++order) {
      ppps_tile_offset += capacities[order];
    }
    // ppps has total angular order three.  Restrict both grouping scans to
    // that fixed partition instead of rereading every active shell quartet.
    error = prepare_ppps_resident_tasks(
        stream, capacities[kPppsAngularOrder], total_shell_pairs, batch,
        active_tile_counts + kPppsAngularOrder, active_tiles + ppps_tile_offset,
        resident_ppps_tasks, resident_ppps_ket_tasks, resident_ppps_bra_counts,
        resident_ppps_bra_offsets, resident_ppps_bra_write_counts,
        resident_ppps_signature_bucketing ? resident_ppps_signature_counts : nullptr,
        resident_ppps_signature_bucketing ? resident_ppps_signature_offsets : nullptr,
        resident_ppps_signatures, enabled_mask);
    if (error != cudaSuccess) return error;
  }
  if (use_resident_ppps) {
    // Probe the selected AOT profile before excluding eligible ppps records
    // from the ordinary queue.  A portable profile may contain the ordinary
    // ppps class without its optional resident route; in that case fall back
    // losslessly instead of dropping the resident-eligible quartets.
    error = generated::launch_ppps_resident(
        stream, unrestricted, resident_ppps_tasks, resident_ppps_ket_tasks,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance, schwarz_bounds, density,
        forces, resident_ppps_block_threads, total_shell_pairs);
    if (error == cudaErrorNotSupported) {
      use_resident_ppps = false;
    } else if (error != cudaSuccess) {
      return error;
    }
  }
  const std::uint64_t low_order_signature_mask =
      (psps_signature_bucketing ? (std::uint64_t{1} << kPspsShellClass) : 0U) |
      (ppss_signature_bucketing ? (std::uint64_t{1} << kPpssShellClass) : 0U);
  error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, low_order_signature_mask,
      low_order_signature_mask != 0U ? low_order_signature_counts : nullptr,
      low_order_signature_mask != 0U ? low_order_signature_offsets : nullptr, enabled_mask, nullptr,
      use_resident_ppps);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels = generated::selected_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((enabled_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, forces,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  // The resident launch precedes ordinary preparation so an unsupported
  // optional route can select the complete fallback queue without dropping
  // any eligible ppps records.
  return cudaSuccess;
}

/**
 * Bucket all enabled generated classes once, then launch their Fock slices.
 *
 * The mask stays device-resident so graph replay can change the environment
 * selection without changing its fixed preparation and worker launch nodes.
 */
cudaError_t launch_generated_shell_class_focks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask, unsigned persistent_worker_blocks, bool unrestricted,
    double screening_tolerance, const double* schwarz_bounds, const double* density, double* fock) {
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

/** Bucket the FP32 queue and launch only generated mixed-Fock capabilities. */
cudaError_t launch_generated_shell_class_mixed_focks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask, unsigned persistent_worker_blocks, bool unrestricted,
    double screening_tolerance, const double* schwarz_bounds, const double* density, double* fock) {
  const std::uint64_t capability_mask = generated::enabled_mixed_fock_shell_class_mask();
  if (capability_mask == 0U) return cudaSuccess;
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((capability_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class_mixed_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

void fill_global_failure(std::vector<RhfBucketItem>& outputs, vibeqc_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
}

}  // namespace

struct CudaRhfBucketPlan {
  CudaResources resources;
  ArenaLayout layout;
  HostBatch topology;
  // Geometry-derived arena state is reusable until coordinates change.
  std::vector<double> cached_positions;
  // The current device density and its associated convergence seed are one
  // cache, while a fixed benchmark dm0 and seed are a separate cache. The
  // distinction matters because finalization advances the returned density
  // after evaluating the final energy, so repeated fixed-dm0 replays cease to
  // be resident hits even though they must retain the original energy seed.
  std::vector<double> resident_warm_positions;
  std::vector<double> resident_warm_density;
  std::vector<double> resident_previous_energy;
  std::vector<double> frozen_warm_positions;
  std::vector<double> frozen_warm_density;
  std::vector<double> frozen_previous_energy;
  std::optional<CudaRhfShellClassProfile> last_shell_class_profile;
  std::optional<CudaPppsQueueProfile> last_ppps_queue_profile;
  std::optional<CudaInactiveEigensolverProfile> last_inactive_eigensolver_profile;
  CudaEigensolverDiagnostic eigensolver_diagnostic;
  ScfOptions options;
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t total_atoms{};
  std::size_t total_shells{};
  std::size_t total_shell_pairs{};
  std::size_t total_shell_quartets{};
  std::size_t total_shell_pair_blocks{};
  std::size_t total_shell_pair_block_quartets{};
  std::size_t total_shell_quartet_tiles{};
  std::vector<std::uint32_t> bounded_direct_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_pair_class_offsets;
  std::size_t bounded_generated_task_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
      bounded_generated_task_offsets{};
  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>
      bounded_generated_task_upper_bounds{};
  std::size_t generated_shell_task_capacity{};
  std::size_t resident_ppps_ket_task_capacity{};
  std::array<std::size_t, detail::kDirectQuartetAngularOrderCount> shell_quartet_tile_capacities{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      shell_quartet_tile_offsets{};
  std::size_t fp32_shell_quartet_tile_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      fp32_shell_quartet_tile_offsets{};
  unsigned persistent_quartet_worker_blocks{};
  std::size_t resident_psss_bra_primitive_pairs{};
  std::size_t resident_psss_task_count{};
  bool generated_psss_weighted{};
  unsigned one_electron_value_mapping{};
  std::size_t primitive_count{};
  std::size_t diis_history{};
  int lwork{};
  bool persistent_eri{};
  bool quartet_direct{};
  bool transformed_direct{};
  bool bounded_direct_streaming{};
  bool unrestricted{};
  bool shell_class_profiling{};
  bool inactive_eigensolver_profiling{};
  bool bounded_fock_class_timing{};
  // These switches change captured work even when topology and arithmetic match.
  bool bounded_streaming_override{};
  bool fock_only_diagnostic{};
  bool graph_native_eigensolver_override{};
  bool reuse_converged_fock{};
  bool mixed_precision_fock{};
  double mixed_precision_fock_threshold{};
  /** Largest item census the batch admission ceiling was bound to. */
  std::size_t mixed_precision_eligible_tile_count{};
  /** Exact per-system mixed-capable tile census the per-item budget divides. */
  std::vector<std::uint32_t> mixed_precision_system_census;
  bool warm_start_updates_enabled{true};
  bool cublas_enabled{true};
  bool retry_without_cublas{};
  bool initialized{};
};

namespace {

bool same_options(const ScfOptions& first, const ScfOptions& second) {
  return first.max_iterations == second.max_iterations &&
         first.diis_history == second.diis_history &&
         first.energy_tolerance == second.energy_tolerance &&
         first.density_tolerance == second.density_tolerance &&
         first.screening_tolerance == second.screening_tolerance &&
         first.compute_forces == second.compute_forces &&
         first.export_physical_reference == second.export_physical_reference &&
         first.reference_memory_budget_bytes == second.reference_memory_budget_bytes &&
         first.precision_mode == second.precision_mode &&
         first.resolved_fock_build == second.resolved_fock_build;
}

std::vector<RhfBucketItem> execute_hf_cuda_bucket(CudaRhfBucketPlan& plan, const HostBatch& host,
                                                  const ScfOptions& options, int device_id,
                                                  bool unrestricted, bool shell_class_profiling,
                                                  bool inactive_eigensolver_profiling) {
  const std::size_t batch_size = host.warm_mask.size();
  std::vector<RhfBucketItem> outputs(batch_size);
  if (options.export_physical_reference &&
      (unrestricted || options.screening_tolerance != 0 || batch_size != 1)) {
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_IMPLEMENTED);
    return outputs;
  }
  plan.last_shell_class_profile.reset();
  plan.last_ppps_queue_profile.reset();
  plan.last_inactive_eigensolver_profile.reset();

  const std::size_t nbf = host.nbf;
  const std::size_t direct_nbf = host.direct_nbf;
  const std::size_t spin_count = host.spin_count;
  std::size_t spin_batch_size = 0;
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrix_elements = 0;
  std::size_t spin_matrix_elements = 0;
  std::size_t eri_elements = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrix_elements = 0;
  std::size_t direct_spin_matrix_elements = 0;
  std::size_t direct_nbf_plus_one = 0;
  std::size_t direct_pair_product = 0;
  std::size_t public_ao_elements = 0;
  std::size_t rectangular_matrix_elements = 0;
  std::size_t spin_rectangular_matrix_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_size) ||
      !checked_multiply(matrix_size, matrix_size, eri_size) ||
      !checked_multiply(batch_size, matrix_size, matrix_elements) ||
      !checked_multiply(batch_size, spin_count, spin_batch_size) ||
      !checked_multiply(matrix_elements, spin_count, spin_matrix_elements) ||
      !checked_multiply(batch_size, eri_size, eri_elements) || !checked_add(nbf, 1, nbf_plus_one) ||
      !checked_multiply(nbf, nbf_plus_one, pair_product) ||
      !checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !checked_multiply(batch_size, direct_matrix_size, direct_matrix_elements) ||
      !checked_multiply(direct_matrix_elements, spin_count, direct_spin_matrix_elements) ||
      !checked_add(direct_nbf, 1, direct_nbf_plus_one) ||
      !checked_multiply(direct_nbf, direct_nbf_plus_one, direct_pair_product) ||
      !checked_multiply(batch_size, nbf, public_ao_elements) ||
      !checked_multiply(public_ao_elements, direct_nbf, rectangular_matrix_elements) ||
      !checked_multiply(rectangular_matrix_elements, spin_count,
                        spin_rectangular_matrix_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t pair_count = pair_product / 2;
  const std::size_t direct_pair_count = direct_pair_product / 2;
  std::size_t pair_elements = 0;
  std::size_t direct_pair_elements = 0;
  if (!checked_multiply(batch_size, pair_count, pair_elements) ||
      !checked_multiply(batch_size, direct_pair_count, direct_pair_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      rectangular_matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_rectangular_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_pair_elements > std::numeric_limits<unsigned>::max() ||
      spin_batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_atoms = host.atomic_numbers.size();
  const std::size_t total_shells = host.shell_atoms.size();
  const std::size_t total_shell_pairs = host.shell_pair_first.size();
  if (host.shell_pair_primitive_offsets.size() != total_shell_pairs + 1 ||
      host.shell_pair_primitive_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_primitives =
      static_cast<std::size_t>(host.shell_pair_primitive_offsets.back());
  if (host.system_shell_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_quartets =
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back());
  if (host.system_shell_pair_block_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_offsets.back() < 0 ||
      host.system_shell_pair_block_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_blocks =
      static_cast<std::size_t>(host.system_shell_pair_block_offsets.back());
  const std::size_t total_shell_pair_block_quartets =
      static_cast<std::size_t>(host.system_shell_pair_block_quartet_offsets.back());
  const bool requested_persistent_eri =
      !options.export_physical_reference && nbf <= kPersistentEriAoLimit;
  const bool requested_quartet_direct =
      !options.export_physical_reference && !requested_persistent_eri &&
      std::all_of(host.shell_angular.begin(), host.shell_angular.end(),
                  [](std::uint8_t angular) { return angular <= 3; });
  const bool requested_transformed_direct = requested_quartet_direct && direct_nbf != nbf;
  // Reference export selects the bounded matrix-direct evaluator; optimized
  // quartet dispatch retains its generated-class coverage gate.
  bool requested_bounded_direct_streaming =
      requested_quartet_direct &&
      (detail::direct_topology_requires_bounded_streaming(total_shell_quartets) ||
       bounded_direct_streaming_override_requested() || options.export_physical_reference);
  const bool cooperative_one_electron_force = one_electron_force_scalar_requested();
  const bool requested_graph_native_eigensolver_override =
      !options.export_physical_reference && graph_native_eigensolver_override_requested();
  const bool xsyev_probe_skip_diagnostic = xsyev_probe_skip_diagnostic_requested();
  // Read this on every cached execution so one prepared batch can provide a
  // fixed-dm0 old/new A/B without rebuilding its immutable topology plan.
  const bool force_density_product_screening = force_density_product_screening_requested();
  const bool bounded_direct_count_diagnostic = bounded_direct_count_diagnostic_requested();
  const bool bounded_direct_aot_only_diagnostic = bounded_direct_aot_only_diagnostic_requested();
  const bool bounded_direct_fock_only_diagnostic = bounded_direct_fock_only_diagnostic_requested();
  const bool bounded_fock_class_timing = bounded_fock_class_timing_requested();
  const bool direct_tile_validation = direct_tile_validation_requested();
  // Read this per execution so one prepared topology can compare the new
  // route with the complete ordinary ppps queue in the same binary.
  const bool resident_ppps_bra = resident_ppps_bra_requested();
  const bool resident_ppps_signature_bucketing = ppps_signature_bucketing_requested();
  const bool psps_signature_bucketing = psps_signature_bucketing_requested();
  const bool ppss_signature_bucketing = ppss_signature_bucketing_requested();
  const unsigned resident_ppps_block_threads = ppps_resident_block_threads_requested();
  const bool first_setup = !plan.initialized;
  detail::DirectQuartetTaskLayout direct_task_layout{};
  std::size_t total_shell_quartet_tiles = 0;
  if (requested_quartet_direct && first_setup && !requested_bounded_direct_streaming) {
    // Exact topology capacities are immutable for a prepared bucket. Their
    // pair-of-pairs enumeration is O(n_shell_pairs^2), so recomputing it on
    // every warm replay adds substantial host latency at large AO counts.
    if (!detail::make_direct_quartet_task_layout(
            host.shell_direct_ao_offsets, host.shell_angular, host.system_shell_pair_offsets,
            host.shell_pair_first, host.shell_pair_second, kMixedFockMinimumAngularOrder,
            direct_task_layout) ||
        direct_task_layout.shell_quartet_count != total_shell_quartets) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
    total_shell_quartet_tiles = direct_task_layout.exact_tile_count;
    if (total_shell_quartet_tiles > detail::kDirectFixedTopologyTileLimit) {
      // A high-angular topology can exceed the fixed grid before its quartet
      // count alone proves that fact. Discard the setup-only exact counts and
      // use the same bounded device enumerator as obviously large systems.
      requested_bounded_direct_streaming = true;
      direct_task_layout = {};
      total_shell_quartet_tiles = 0;
    } else if (direct_task_layout.exact_tile_count >
               kFixedGeneratedTaskArenaMaximumBytes / sizeof(GeneratedShellTask)) {
      // The uint32 grid limit is much larger than a practical descriptor
      // arena on a 32 GiB device.  Route large-but-grid-addressable buckets
      // through bounded streaming before make_layout() reserves the complete
      // generated task array.
      requested_bounded_direct_streaming = true;
      direct_task_layout = {};
      total_shell_quartet_tiles = 0;
    }
  } else if (requested_quartet_direct) {
    requested_bounded_direct_streaming =
        first_setup ? requested_bounded_direct_streaming : plan.bounded_direct_streaming;
    total_shell_quartet_tiles =
        requested_bounded_direct_streaming ? 0 : plan.total_shell_quartet_tiles;
  }
  // Per-item mixed-capable tile census: the FP32-error budget is evaluated for
  // every system on its own count. Bounded streaming keeps zeros, which the
  // policy refuses rather than guesses, and the largest census is the batch
  // ceiling that decides whether the plan allocates the route at all.
  std::vector<std::size_t> mixed_precision_system_census(batch_size, 0U);
  if (requested_quartet_direct && !requested_bounded_direct_streaming) {
    if (first_setup) {
      mixed_precision_system_census = direct_task_layout.system_mixed_capable_tile_counts;
      if (mixed_precision_system_census.size() != batch_size) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    } else if (plan.mixed_precision_system_census.size() == batch_size) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        mixed_precision_system_census[system] = plan.mixed_precision_system_census[system];
      }
    }
  }
  const std::size_t mixed_precision_eligible_tile_count =
      mixed_precision_system_census.empty()
          ? 0U
          : *std::max_element(mixed_precision_system_census.begin(),
                              mixed_precision_system_census.end());
  const MixedPrecisionFockPolicy requested_precision_policy =
      requested_quartet_direct
          ? resolve_mixed_precision_fock_policy(
                options.precision_mode, options.energy_tolerance, options.screening_tolerance,
                static_cast<double>(mixed_precision_eligible_tile_count))
          : MixedPrecisionFockPolicy{};
  const std::optional<double> requested_mixed_precision_fock_threshold =
      requested_precision_policy.threshold;
  const bool requested_mixed_precision_fock = requested_mixed_precision_fock_threshold.has_value();
  // A mixed item is promoted to exact FP64 by the target refinement before any
  // consumer runs, so the matrix it retains is target precision. The density
  // criterion and convergence check below still decide each item's reuse.
  const bool requested_reuse_converged_fock =
      reuse_converged_fock_requested() && !options.export_physical_reference;
  // Direct consumers expand each compact logical tile into one-warp blocks;
  // validate the resulting fixed Graph grid before narrowing it to unsigned.
  if (total_shell_pairs > std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartets > std::numeric_limits<unsigned>::max()) ||
      host.psss_resident_tasks.size() > std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartet_tiles > detail::kDirectFixedTopologyTileLimit)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t force_coordinate_count = 0;
  std::size_t one_electron_force_elements = 0;
  std::size_t force_matrix_elements = 0;
  std::size_t persistent_force_elements = 0;
  std::size_t direct_force_elements = 0;
  if (!checked_multiply(total_atoms, 3, force_coordinate_count) ||
      !checked_multiply(batch_size, pair_count, one_electron_force_elements) ||
      !checked_multiply(force_coordinate_count, matrix_size, force_matrix_elements) ||
      !checked_multiply(force_coordinate_count, eri_size, persistent_force_elements) ||
      !checked_multiply(force_matrix_elements, pair_count, direct_force_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t diis_history = std::max<std::size_t>(1, options.diis_history);
  if (diis_history > 64) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (!first_setup &&
      (plan.resources.device_id_ != device_id || !same_topology(plan.topology, host) ||
       !same_options(plan.options, options) || plan.unrestricted != unrestricted ||
       plan.bounded_direct_streaming != requested_bounded_direct_streaming ||
       plan.shell_class_profiling != shell_class_profiling ||
       plan.inactive_eigensolver_profiling != inactive_eigensolver_profiling ||
       plan.bounded_fock_class_timing != bounded_fock_class_timing ||
       plan.bounded_streaming_override != bounded_direct_streaming_override_requested() ||
       plan.fock_only_diagnostic != bounded_direct_fock_only_diagnostic ||
       plan.graph_native_eigensolver_override != requested_graph_native_eigensolver_override ||
       plan.reuse_converged_fock != requested_reuse_converged_fock ||
       plan.one_electron_value_mapping != cuda_policy::one_electron_value_mapping_requested() ||
       plan.mixed_precision_fock != requested_mixed_precision_fock ||
       plan.mixed_precision_fock_threshold !=
           requested_mixed_precision_fock_threshold.value_or(0.0))) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t generated_shell_task_capacity = first_setup ? 0 : plan.generated_shell_task_capacity;
  std::size_t resident_ppps_ket_task_capacity =
      first_setup ? 0 : plan.resident_ppps_ket_task_capacity;
  std::size_t fp32_shell_quartet_tile_capacity =
      first_setup ? 0 : plan.fp32_shell_quartet_tile_capacity;
  if (first_setup && requested_mixed_precision_fock) {
    for (std::size_t order = kMixedFockMinimumAngularOrder;
         order < detail::kDirectQuartetAngularOrderCount; ++order) {
      if (!checked_add(fp32_shell_quartet_tile_capacity,
                       direct_task_layout.angular_order_tile_counts[order],
                       fp32_shell_quartet_tile_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (fp32_shell_quartet_tile_capacity >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }
  const std::size_t generic_order5_tile_capacity =
      requested_quartet_direct && !requested_bounded_direct_streaming
          ? (first_setup
                 ? direct_task_layout.angular_order_tile_counts[kGenericOrderFiveAngularOrder]
                 : plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder])
          : 0;
  std::size_t bounded_generated_task_capacity =
      first_setup ? 0 : plan.bounded_generated_task_capacity;
  if (first_setup && requested_bounded_direct_streaming) {
    if (!checked_multiply(total_shell_pairs, kBoundedGeneratedTasksPerShellPair,
                          bounded_generated_task_capacity)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    bounded_generated_task_capacity =
        std::min(bounded_generated_task_capacity, kBoundedGeneratedMaximumTaskCapacity);
  }
  if (requested_quartet_direct && first_setup && !requested_bounded_direct_streaming) {
    // The shared generated-task arena serves both exact Fock and force
    // consumers.  Their registries are intentionally not identical: for
    // example, `ssss` has a generated Fock consumer but remains on the
    // handwritten force path.  Build the capacity from their union so a
    // Fock-only class cannot leave its persistent kernel with a zero-sized
    // task arena.
    std::array<bool, detail::kDirectQuartetShellClassCount> generated_task_classes{};
    const auto include_generated_task_classes = [&](const generated::ShellKernelMetadata* kernels,
                                                    std::size_t kernel_count) {
      for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
        const unsigned shell_class = kernels[kernel_index].shell_class;
        if (shell_class < generated_task_classes.size()) {
          generated_task_classes[shell_class] = true;
        }
      }
    };
    std::size_t force_kernel_count = 0;
    const generated::ShellKernelMetadata* force_kernels =
        generated::selected_shell_kernels(force_kernel_count);
    include_generated_task_classes(force_kernels, force_kernel_count);
    std::size_t fock_kernel_count = 0;
    const generated::ShellKernelMetadata* fock_kernels =
        generated::selected_fock_shell_kernels(fock_kernel_count);
    include_generated_task_classes(fock_kernels, fock_kernel_count);
    for (std::size_t shell_class = 0; shell_class < generated_task_classes.size(); ++shell_class) {
      if (!generated_task_classes[shell_class]) continue;
      if (!checked_add(generated_shell_task_capacity,
                       direct_task_layout.shell_class_tile_counts[shell_class],
                       generated_shell_task_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    // Every active canonical ppps shell quartet occupies one tile at angular
    // order three, so this exact class count is the maximum resident ket
    // record count.  Reserve a reusable tail of the existing generated-task
    // arena only when the selected AOT bundle contains a ppps force consumer;
    // portable/stub builds then retain their original arena footprint.
    bool ppps_force_available = false;
    for (std::size_t kernel_index = 0; kernel_index < force_kernel_count; ++kernel_index) {
      if (force_kernels[kernel_index].shell_class == kPppsShellClass) {
        ppps_force_available = true;
        break;
      }
    }
    resident_ppps_ket_task_capacity =
        ppps_force_available ? direct_task_layout.shell_class_tile_counts[kPppsShellClass] : 0;
  }
  if (resident_ppps_ket_task_capacity > generated_shell_task_capacity ||
      resident_ppps_ket_task_capacity >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (first_setup) {
    if (!make_layout(
            batch_size, nbf, direct_nbf, total_atoms, total_shells, total_shell_pairs,
            total_shell_pair_blocks, bounded_generated_task_capacity, total_shell_pair_primitives,
            requested_quartet_direct ? host.psss_resident_tasks.size() : 0,
            requested_quartet_direct ? host.psss_resident_ket_pairs.size() : 0,
            total_shell_quartet_tiles, fp32_shell_quartet_tile_capacity,
            generated_shell_task_capacity, resident_ppps_ket_task_capacity,
            generic_order5_tile_capacity, host.primitive_exponents.size(), diis_history,
            options.max_iterations, host.spin_count, requested_persistent_eri,
            requested_transformed_direct, shell_class_profiling, inactive_eigensolver_profiling,
            bounded_fock_class_timing, requested_bounded_direct_streaming,
            requested_mixed_precision_fock, plan.layout)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    plan.batch_size = batch_size;
    plan.nbf = nbf;
    plan.direct_nbf = direct_nbf;
    plan.total_atoms = total_atoms;
    plan.total_shells = total_shells;
    plan.total_shell_pairs = total_shell_pairs;
    plan.total_shell_quartets = total_shell_quartets;
    plan.total_shell_pair_blocks = total_shell_pair_blocks;
    plan.total_shell_pair_block_quartets = total_shell_pair_block_quartets;
    plan.total_shell_quartet_tiles = total_shell_quartet_tiles;
    plan.bounded_generated_task_capacity = bounded_generated_task_capacity;
    plan.bounded_generated_task_offsets =
        requested_bounded_direct_streaming
            ? make_bounded_generated_task_offsets(host, bounded_generated_task_capacity,
                                                  &plan.bounded_generated_task_upper_bounds)
            : std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>{};
    if (!requested_bounded_direct_streaming) {
      plan.bounded_generated_task_upper_bounds.fill(0U);
    }
    if (requested_bounded_direct_streaming) {
      plan.bounded_direct_shell_pair_order.resize(total_shell_pairs);
      std::iota(plan.bounded_direct_shell_pair_order.begin(),
                plan.bounded_direct_shell_pair_order.end(), 0U);
      if (!make_bounded_stream_shell_pair_order(host, plan.bounded_stream_shell_pair_order,
                                                plan.bounded_stream_pair_class_offsets)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    plan.resident_psss_bra_primitive_pairs = 0;
    plan.generated_psss_weighted = cuda_policy::generated_psss_weighted_requested();
    plan.one_electron_value_mapping = cuda_policy::one_electron_value_mapping_requested();
    const bool resident_psss_enabled = resident_psss_bra_requested();
    // The bounded direct force path has its own exact page consumer for psss.
    // Keep the resident-bra optimization on the fixed-queue path only until
    // its bounded scheduling and force accumulation are independently gated.
    plan.resident_psss_task_count =
        requested_quartet_direct && !requested_bounded_direct_streaming && resident_psss_enabled
            ? host.psss_resident_tasks.size()
            : 0;
    for (std::size_t pair = 0; pair < total_shell_pairs; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      if (host.shell_angular[first_shell] + host.shell_angular[second_shell] != 1U) {
        continue;
      }
      const std::size_t primitive_pairs = static_cast<std::size_t>(
          host.shell_pair_primitive_offsets[pair + 1] - host.shell_pair_primitive_offsets[pair]);
      plan.resident_psss_bra_primitive_pairs =
          std::max(plan.resident_psss_bra_primitive_pairs, primitive_pairs);
    }
    plan.generated_shell_task_capacity = generated_shell_task_capacity;
    plan.resident_ppps_ket_task_capacity = resident_ppps_ket_task_capacity;
    plan.shell_quartet_tile_capacities = direct_task_layout.angular_order_tile_counts;
    for (std::size_t order = 0; order < direct_task_layout.angular_order_tile_offsets.size();
         ++order) {
      plan.shell_quartet_tile_offsets[order] =
          static_cast<std::uint32_t>(direct_task_layout.angular_order_tile_offsets[order]);
    }
    plan.fp32_shell_quartet_tile_capacity = fp32_shell_quartet_tile_capacity;
    std::size_t fp32_tile_offset = 0;
    for (std::size_t order = 0; order < detail::kDirectQuartetAngularOrderCount; ++order) {
      plan.fp32_shell_quartet_tile_offsets[order] = static_cast<std::uint32_t>(fp32_tile_offset);
      if (requested_mixed_precision_fock && order >= kMixedFockMinimumAngularOrder) {
        fp32_tile_offset += direct_task_layout.angular_order_tile_counts[order];
      }
    }
    plan.fp32_shell_quartet_tile_offsets[detail::kDirectQuartetAngularOrderCount] =
        static_cast<std::uint32_t>(fp32_tile_offset);
    plan.primitive_count = host.primitive_exponents.size();
    plan.diis_history = diis_history;
    plan.persistent_eri = requested_persistent_eri;
    plan.quartet_direct = requested_quartet_direct;
    plan.transformed_direct = requested_transformed_direct;
    plan.bounded_direct_streaming = requested_bounded_direct_streaming;
    plan.unrestricted = unrestricted;
    plan.shell_class_profiling = shell_class_profiling;
    plan.inactive_eigensolver_profiling = inactive_eigensolver_profiling;
    plan.bounded_fock_class_timing = bounded_fock_class_timing;
    plan.bounded_streaming_override = bounded_direct_streaming_override_requested();
    plan.fock_only_diagnostic = bounded_direct_fock_only_diagnostic;
    plan.graph_native_eigensolver_override = requested_graph_native_eigensolver_override;
    plan.reuse_converged_fock = requested_reuse_converged_fock;
    plan.mixed_precision_fock = requested_mixed_precision_fock;
    plan.mixed_precision_fock_threshold = requested_mixed_precision_fock_threshold.value_or(0.0);
    plan.mixed_precision_eligible_tile_count = mixed_precision_eligible_tile_count;
    plan.mixed_precision_system_census.assign(mixed_precision_system_census.size(), 0U);
    for (std::size_t system = 0; system < mixed_precision_system_census.size(); ++system) {
      if (mixed_precision_system_census[system] >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
      plan.mixed_precision_system_census[system] =
          static_cast<std::uint32_t>(mixed_precision_system_census[system]);
    }
    plan.options = options;
    plan.topology = host;
    // Positions and warm guesses are dynamic execution inputs, not part of
    // the immutable fixed-topology cache identity.
    plan.topology.positions.clear();
    plan.topology.warm_mask.clear();
    plan.topology.warm_density.clear();
    plan.resources.device_id_ = device_id;
  }
  ArenaLayout& layout = plan.layout;
  CudaResources& resources = plan.resources;
  const bool persistent_eri = plan.persistent_eri;
  const bool quartet_direct = plan.quartet_direct;
  const bool transformed_direct = plan.transformed_direct;
  if (transformed_direct != !host.ao_to_direct_transform.empty()) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const bool bounded_direct_streaming = plan.bounded_direct_streaming;
  const bool reuse_converged_fock = plan.reuse_converged_fock;
  const bool mixed_precision_fock = plan.mixed_precision_fock;
  const double mixed_precision_fock_threshold = plan.mixed_precision_fock_threshold;
  if (first_setup) {
    plan.eigensolver_diagnostic = {};
    plan.eigensolver_diagnostic.matrix_dimension = nbf;
    plan.eigensolver_diagnostic.physical_system_count = batch_size;
    plan.eigensolver_diagnostic.solver_batch_count = spin_batch_size;
    // A forced graph-native selection is an explicit escape hatch for large
    // matrices where cuSOLVER XsyevBatched capture is known to be unusable.
    // Do not probe that provider first: the probe allocates and executes a
    // full-size eigensystem and can itself spend minutes in host-side setup.
    if (requested_graph_native_eigensolver_override) {
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.selection_source =
          CudaEigensolverSelectionSource::benchmark_override;
    } else if (nbf <= static_cast<std::size_t>(kSmallEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::small_native;
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::small_native;
    } else if (nbf <= static_cast<std::size_t>(kBatchedEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::jacobi_batched;
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::jacobi_batched;
    } else if (options.export_physical_reference) {
      // The existing ordinary-stream Xsyevd path avoids an unbudgeted full
      // provider/capture probe. Query and bound its real workspaces below.
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyevd;
      plan.eigensolver_diagnostic.selection_source =
          CudaEigensolverSelectionSource::dimension_policy;
    } else {
      plan.eigensolver_diagnostic.xsyev_probe =
          probe_xsyev_batched_device_launch_graph(device_id, nbf, spin_batch_size);
      // A rejected capture intentionally leaves CUDA's per-thread last-error
      // slot set to cudaErrorStreamCaptureUnsupported (901) on CUDA 12.9.
      // The probe has already recorded that evidence; clear the sticky slot
      // before the real plan allocates, captures, and launches its fallback.
      // Otherwise the unrelated final cudaGetLastError() would report the
      // old probe rejection as a calculation failure.
      (void)cudaGetLastError();
      const XsyevBatchedDispatch dispatch =
          select_xsyev_batched_dispatch(plan.eigensolver_diagnostic.xsyev_probe);
      plan.eigensolver_diagnostic.family = dispatch.device_launch_graph_provider
                                               ? CudaEigensolverFamily::xsyev_batched
                                               : CudaEigensolverFamily::graph_native;
      if (dispatch.device_launch_graph_provider) {
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyev_batched;
      } else if (dispatch.ordinary_stream_provider) {
        // Match GPU4PySCF's robust large-matrix strategy when the generic
        // batched provider works on a stream but rejects Graph capture.
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyevd;
      } else {
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::graph_native;
      }
      plan.eigensolver_diagnostic.selection_source =
          xsyev_probe_skip_diagnostic ? CudaEigensolverSelectionSource::benchmark_override
          : plan.eigensolver_diagnostic.xsyev_probe.graph_eligible
              ? CudaEigensolverSelectionSource::exact_probe
              : CudaEigensolverSelectionSource::exact_probe_fallback;
    }
  }
  const CudaEigensolverFamily graph_eigensolver_family = plan.eigensolver_diagnostic.family;
  const CudaEigensolverFamily ordinary_eigensolver_family =
      plan.eigensolver_diagnostic.ordinary_family;
  const bool use_jacobi = graph_eigensolver_family == CudaEigensolverFamily::jacobi_batched ||
                          ordinary_eigensolver_family == CudaEigensolverFamily::jacobi_batched;
  const bool use_cusolver = use_jacobi ||
                            ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched ||
                            ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd;
  const bool geometry_changed = first_setup || plan.cached_positions != host.positions;
  const bool all_systems_warm = std::all_of(host.warm_mask.begin(), host.warm_mask.end(),
                                            [](std::uint8_t value) { return value != 0; });
  const bool any_system_warm = std::any_of(host.warm_mask.begin(), host.warm_mask.end(),
                                           [](std::uint8_t value) { return value != 0; });
  // Density residency and the previous-energy baseline are deliberately
  // independent. A fixed warm start can reuse its frozen energy seed after a
  // prior replay advanced the device density. Geometry remains part of a
  // resident-density hit because applying an external dm0 also renormalizes
  // its electron trace against the geometry-dependent overlap matrix.
  const bool device_resident_density_hit = all_systems_warm &&
                                           plan.resident_warm_positions == host.positions &&
                                           plan.resident_warm_density == host.warm_density;
  const bool resident_energy_baseline_hit = device_resident_density_hit &&
                                            plan.resident_warm_positions == host.positions &&
                                            plan.resident_previous_energy.size() == batch_size;
  const bool frozen_energy_baseline_hit = all_systems_warm && !plan.warm_start_updates_enabled &&
                                          plan.frozen_warm_density == host.warm_density &&
                                          plan.frozen_warm_positions == host.positions &&
                                          plan.frozen_previous_energy.size() == batch_size;
  const bool cached_energy_baseline_hit =
      frozen_energy_baseline_hit || resident_energy_baseline_hit;
  // Copy the tiny seed vector locally before invalidating residency. Any
  // early CUDA or validation failure below may have partially changed the
  // device density; only a fully successful execution republishes it.
  std::vector<double> host_previous_energy_seed;
  if (frozen_energy_baseline_hit) {
    host_previous_energy_seed = plan.frozen_previous_energy;
  } else if (resident_energy_baseline_hit) {
    host_previous_energy_seed = plan.resident_previous_energy;
  }
  plan.resident_warm_positions.clear();
  plan.resident_warm_density.clear();
  plan.resident_previous_energy.clear();
  const bool use_cublas = plan.cublas_enabled && nbf >= kCublasMatrixProductAoThreshold;
  std::size_t reference_base_bytes = 0;
  const std::size_t reference_provider_allowance =
      (use_cublas ? 96ULL << 20 : 0) + (use_cusolver ? 96ULL << 20 : 0);
  if (options.export_physical_reference) {
    reference_base_bytes = reference_detail::base_capacity(layout.bytes, host, matrix_elements,
                                                           reference_provider_allowance,
                                                           options.reference_memory_budget_bytes);
    resources.reference_peak_bytes_ = reference_base_bytes;
  }
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (quartet_direct || options.export_physical_reference) {
    // High-order direct ERI recurrences use a bounded per-thread local
    // workspace.  CUDA's default stack limit is only 1 KiB, which is enough
    // for s/p/d low-order tiles but lets d/f quartets fault with an apparent
    // local-memory out-of-bounds access.  Reserve a generous fixed ceiling
    // once per device; low-order kernels do not consume it.
    std::size_t stack_limit = 0;
    cuda_error = cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
    if (cuda_error == cudaSuccess && stack_limit < kDirectCudaStackLimitBytes) {
      cuda_error = cudaDeviceSetLimit(cudaLimitStackSize, kDirectCudaStackLimitBytes);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (first_setup && quartet_direct) {
    int multiprocessor_count = 0;
    cuda_error =
        cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id);
    if (cuda_error != cudaSuccess || multiprocessor_count <= 0 ||
        static_cast<unsigned>(multiprocessor_count) >
            std::numeric_limits<unsigned>::max() / kPersistentQuartetWarpsPerMultiprocessor) {
      fill_global_failure(outputs, cuda_error == cudaSuccess ? VIBEQC_STATUS_INVALID_ARGUMENT
                                                             : cuda_status(cuda_error));
      return outputs;
    }
    plan.persistent_quartet_worker_blocks =
        static_cast<unsigned>(multiprocessor_count) * kPersistentQuartetWarpsPerMultiprocessor;
  }
  cublasStatus_t blas_error = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t solver_error = CUSOLVER_STATUS_SUCCESS;
  if (first_setup) {
    std::unique_lock<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
    if ((cuda_error = cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking)) !=
            cudaSuccess ||
        (cuda_error = runtime::resource_cuda_malloc_async(&resources.arena_, layout.bytes,
                                                          resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (quartet_direct &&
        (cuda_error = runtime::resource_cuda_malloc_async(&resources.direct_tile_validation_,
                                                          sizeof(DirectTileValidationRecord),
                                                          resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    runtime::sample_cuda_arena_capacity(layout.bytes);
    const auto provider_before =
        options.export_physical_reference ? reference_detail::free_bytes(resources.stream_) : 0;
    if (use_cublas) {
      blas_error = cublasCreate(&resources.blas_);
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetStream(resources.blas_, resources.stream_);
      }
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetPointerMode(resources.blas_, CUBLAS_POINTER_MODE_HOST);
      }
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
    if (use_cusolver) {
      solver_error = cusolverDnCreate(&resources.solver_);
      if (solver_error != CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnSetStream(resources.solver_, resources.stream_)) !=
              CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
      if (use_jacobi) {
        if ((solver_error = cusolverDnCreateSyevjInfo(&resources.jacobi_)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetTolerance(resources.jacobi_, 1.0e-13)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetMaxSweeps(resources.jacobi_, 100)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetSortEig(resources.jacobi_, 1)) !=
                CUSOLVER_STATUS_SUCCESS) {
          fill_global_failure(outputs, solver_status(solver_error));
          return outputs;
        }
      } else if ((solver_error = cusolverDnCreateParams(&resources.solver_parameters_)) !=
                 CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
    }
    if (options.export_physical_reference) {
      resources.provider_retained_bytes_ = reference_detail::retained_bytes(
          resources.stream_, provider_before, reference_provider_allowance);
    }
  }

  auto atom_offsets = arena_pointer<std::int64_t>(resources.arena_, layout.atom_offsets);
  auto atom_systems = arena_pointer<std::int32_t>(resources.arena_, layout.atom_systems);
  auto atomic_numbers = arena_pointer<std::int32_t>(resources.arena_, layout.atomic_numbers);
  auto positions = arena_pointer<double>(resources.arena_, layout.positions);
  auto system_shell_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_offsets);
  auto shell_atoms = arena_pointer<std::int32_t>(resources.arena_, layout.shell_atoms);
  auto shell_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.shell_angular);
  auto shell_ao_offsets = arena_pointer<std::int64_t>(resources.arena_, layout.shell_ao_offsets);
  auto shell_direct_ao_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_direct_ao_offsets);
  auto shell_primitive_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_primitive_offsets);
  auto system_shell_pair_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_offsets);
  auto system_shell_quartet_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_quartet_offsets);
  auto system_shell_pair_block_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_block_offsets);
  auto system_shell_pair_block_quartet_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_block_quartet_offsets);
  auto shell_pair_systems =
      arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_systems);
  auto shell_pair_first = arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_first);
  auto shell_pair_second = arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_second);
  auto shell_pair_primitive_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_pair_primitive_offsets);
  auto shell_primitive_pairs =
      arena_pointer<PrimitivePairData>(resources.arena_, layout.shell_primitive_pairs);
  auto psss_resident_tasks =
      arena_pointer<PsssResidentTask>(resources.arena_, layout.psss_resident_tasks);
  auto psss_resident_ket_pairs =
      arena_pointer<std::uint32_t>(resources.arena_, layout.psss_resident_ket_pairs);
  auto ao_shells = arena_pointer<std::int32_t>(resources.arena_, layout.ao_shells);
  auto ao_term_counts = arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_counts);
  auto ao_term_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_angular);
  auto ao_term_coefficients = arena_pointer<double>(resources.arena_, layout.ao_term_coefficients);
  auto direct_ao_shells = arena_pointer<std::int32_t>(resources.arena_, layout.direct_ao_shells);
  auto direct_ao_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.direct_ao_angular);
  auto direct_ao_coefficients =
      arena_pointer<double>(resources.arena_, layout.direct_ao_coefficients);
  auto ao_to_direct_transform =
      arena_pointer<double>(resources.arena_, layout.ao_to_direct_transform);
  auto primitive_exponents = arena_pointer<double>(resources.arena_, layout.primitive_exponents);
  auto primitive_coefficients =
      arena_pointer<double>(resources.arena_, layout.primitive_coefficients);
  auto occupied = arena_pointer<std::int32_t>(resources.arena_, layout.occupied);
  auto warm_mask = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_mask);
  // Per-item admission gate consumed by the tile compaction and the target
  // refinement. It is only allocated when the plan may run the mixed route.
  std::uint32_t* mixed_precision_item_census =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.mixed_item_census)
          : nullptr;
  auto warm_density = arena_pointer<double>(resources.arena_, layout.warm_density);
  auto warm_invalid = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_invalid);
  auto overlap = arena_pointer<double>(resources.arena_, layout.overlap);
  auto hcore = arena_pointer<double>(resources.arena_, layout.hcore);
  auto eri = arena_pointer<double>(resources.arena_, layout.eri);
  auto schwarz_bounds = arena_pointer<double>(resources.arena_, layout.schwarz_bounds);
  auto direct_density = arena_pointer<double>(resources.arena_, layout.direct_density);
  auto direct_fock = arena_pointer<double>(resources.arena_, layout.direct_fock);
  auto direct_transform_temporary =
      arena_pointer<double>(resources.arena_, layout.direct_transform_temporary);
  auto shell_pair_bounds = arena_pointer<double>(resources.arena_, layout.shell_pair_bounds);
  auto shell_pair_density_bounds =
      arena_pointer<ShellPairDensityBounds>(resources.arena_, layout.shell_pair_density_bounds);
  auto bounded_direct_shell_pair_order =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_shell_pair_order);
  auto bounded_stream_shell_pair_order =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_stream_shell_pair_order);
  auto bounded_stream_pair_class_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_stream_pair_class_offsets);
  auto bounded_stream_topology =
      arena_pointer<GeneratedShellPairStream>(resources.arena_, layout.bounded_stream_topology);
  auto bounded_direct_shell_pair_block_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_shell_pair_block_bounds);
  auto bounded_direct_system_density_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_system_density_bounds);
  auto bounded_direct_system_pair_density_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_system_pair_density_bounds);
  auto bounded_direct_generated_overflow =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_overflow);
  auto bounded_direct_generated_tasks =
      arena_pointer<GeneratedShellTask>(resources.arena_, layout.bounded_direct_generated_tasks);
  auto bounded_direct_generated_task_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_counts);
  auto bounded_direct_generated_task_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_offsets);
  auto bounded_direct_generated_retry_task_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_retry_task_offsets);
  auto bounded_direct_generated_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_heads);
  auto bounded_direct_generated_retry_mask =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_retry_mask);
  auto bounded_direct_generated_retry_any =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_retry_any);
  auto bounded_force_signature_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_counts);
  auto bounded_force_signature_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_offsets);
  auto bounded_force_signature_block_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_block_offsets);
  std::uint64_t* bounded_fock_class_timer_starts =
      bounded_fock_class_timing
          ? arena_pointer<std::uint64_t>(resources.arena_, layout.bounded_fock_class_timer_starts)
          : nullptr;
  std::uint64_t* bounded_fock_class_timer_elapsed =
      bounded_fock_class_timing
          ? arena_pointer<std::uint64_t>(resources.arena_, layout.bounded_fock_class_timer_elapsed)
          : nullptr;
  std::uint32_t* bounded_fock_class_timer_launches =
      bounded_fock_class_timing
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_fock_class_timer_launches)
          : nullptr;
  unsigned long long* bounded_fock_fp64_work_counts =
      bounded_fock_class_timing ? arena_pointer<unsigned long long>(
                                      resources.arena_, layout.bounded_fock_fp64_work_counts)
                                : nullptr;
  unsigned long long* bounded_fock_fp32_work_counts =
      bounded_fock_class_timing ? arena_pointer<unsigned long long>(
                                      resources.arena_, layout.bounded_fock_fp32_work_counts)
                                : nullptr;
  auto active_shell_quartet_tile_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.active_shell_quartet_tile_offsets);
  auto active_shell_quartet_tile_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.active_shell_quartet_tile_counts);
  auto active_shell_quartet_tiles =
      arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.active_shell_quartet_tiles);
  std::uint32_t* fp32_shell_quartet_tile_offsets =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_shell_quartet_tile_offsets)
          : nullptr;
  std::uint32_t* fp32_shell_quartet_tile_counts =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_shell_quartet_tile_counts)
          : nullptr;
  ActiveShellQuartetTile* fp32_shell_quartet_tiles =
      mixed_precision_fock
          ? arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.fp32_shell_quartet_tiles)
          : nullptr;
  auto shell_class_profile =
      arena_pointer<DeviceShellClassProfileEntry>(resources.arena_, layout.shell_class_profile);
  auto persistent_fock_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.persistent_fock_task_heads);
  std::uint32_t* fp32_persistent_fock_task_heads =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_persistent_fock_task_heads)
          : nullptr;
  auto persistent_force_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.persistent_force_task_heads);
  auto generated_shell_tasks =
      arena_pointer<GeneratedShellTask>(resources.arena_, layout.generated_shell_tasks);
  auto generated_shell_classes =
      arena_pointer<std::uint8_t>(resources.arena_, layout.generated_shell_classes);
  auto generated_shell_task_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_offsets);
  auto generated_shell_task_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_counts);
  auto generated_shell_task_write_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_write_counts);
  auto generated_shell_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_heads);
  auto generated_low_order_signature_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_low_order_signature_counts);
  auto generated_low_order_signature_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_low_order_signature_offsets);
  auto generated_ppps_resident_tasks = arena_pointer<GeneratedPppsResidentTask>(
      resources.arena_, layout.generated_ppps_resident_tasks);
  // Final force preparation is ordered after the last Fock consumer on the
  // same stream.  Reuse the ppps-sized tail of the ordinary generated-task
  // arena for resident ket records, then let ordinary force preparation
  // overwrite it only after the resident launch completes.  This avoids a
  // multi-gigabyte duplicate queue at the 384-AO endpoint.
  GeneratedShellTask* generated_ppps_resident_ket_tasks = nullptr;
  if (resident_ppps_ket_task_capacity != 0) {
    generated_ppps_resident_ket_tasks =
        generated_shell_tasks + (generated_shell_task_capacity - resident_ppps_ket_task_capacity);
  }
  auto generated_ppps_resident_bra_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_ppps_resident_bra_counts);
  auto generated_ppps_resident_bra_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_ppps_resident_bra_offsets);
  auto generated_ppps_resident_bra_write_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_bra_write_counts);
  auto generated_ppps_resident_signature_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_signature_counts);
  auto generated_ppps_resident_signature_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_signature_offsets);
  // A zero-sized arena slice still has an offset, so do not turn it into a
  // writable pointer when profiling did not allocate per-task signatures.
  std::uint32_t* generated_ppps_resident_signatures =
      shell_class_profiling ? arena_pointer<std::uint32_t>(
                                  resources.arena_, layout.generated_ppps_resident_signatures)
                            : nullptr;
  auto generated_fock_shell_class_mask =
      arena_pointer<std::uint64_t>(resources.arena_, layout.generated_fock_shell_class_mask);
  std::uint64_t* generated_mixed_fock_shell_class_mask =
      mixed_precision_fock ? arena_pointer<std::uint64_t>(
                                 resources.arena_, layout.generated_mixed_fock_shell_class_mask)
                           : nullptr;
  auto generic_order5_tiles =
      arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.generic_order5_tiles);
  auto generic_order5_tile_count =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generic_order5_tile_count);
  auto ao_pair_first = arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_first);
  auto ao_pair_second = arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_second);
  auto nuclear_repulsion = arena_pointer<double>(resources.arena_, layout.nuclear_repulsion);
  auto orthogonalizer = arena_pointer<double>(resources.arena_, layout.orthogonalizer);
  auto temporary = arena_pointer<double>(resources.arena_, layout.temporary);
  auto eigensystem = arena_pointer<double>(resources.arena_, layout.eigensystem);
  auto coefficients = arena_pointer<double>(resources.arena_, layout.coefficients);
  auto eigenvalues = arena_pointer<double>(resources.arena_, layout.eigenvalues);
  auto density = arena_pointer<double>(resources.arena_, layout.density);
  auto next_density = arena_pointer<double>(resources.arena_, layout.next_density);
  auto fock = arena_pointer<double>(resources.arena_, layout.fock);
  auto residual = arena_pointer<double>(resources.arena_, layout.residual);
  auto weighted_density = arena_pointer<double>(resources.arena_, layout.weighted_density);
  auto total_density = arena_pointer<double>(resources.arena_, layout.total_density);
  auto total_weighted_density =
      arena_pointer<double>(resources.arena_, layout.total_weighted_density);
  auto fock_history = arena_pointer<double>(resources.arena_, layout.fock_history);
  auto residual_history = arena_pointer<double>(resources.arena_, layout.residual_history);
  auto diis_linear_system = arena_pointer<double>(resources.arena_, layout.diis_linear_system);
  auto diis_coefficients = arena_pointer<double>(resources.arena_, layout.diis_coefficients);
  auto diis_count = arena_pointer<std::uint32_t>(resources.arena_, layout.diis_count);
  auto diis_head = arena_pointer<std::uint32_t>(resources.arena_, layout.diis_head);
  auto energy = arena_pointer<double>(resources.arena_, layout.energy);
  auto previous_energy = arena_pointer<double>(resources.arena_, layout.previous_energy);
  auto energy_change = arena_pointer<double>(resources.arena_, layout.energy_change);
  auto density_rms = arena_pointer<double>(resources.arena_, layout.density_rms);
  auto forces = arena_pointer<double>(resources.arena_, layout.forces);
  auto active = arena_pointer<std::uint8_t>(resources.arena_, layout.active);
  auto converged = arena_pointer<std::uint8_t>(resources.arena_, layout.converged);
  auto failed = arena_pointer<std::uint8_t>(resources.arena_, layout.failed);
  auto final_fock_reuse_mask =
      arena_pointer<std::uint8_t>(resources.arena_, layout.final_fock_reuse_mask);
  auto final_fock_rebuild_count =
      arena_pointer<std::uint32_t>(resources.arena_, layout.final_fock_rebuild_count);
  auto spin_active = arena_pointer<std::uint8_t>(resources.arena_, layout.spin_active);
  auto iterations = arena_pointer<std::uint32_t>(resources.arena_, layout.iterations);
  auto solver_info = arena_pointer<int>(resources.arena_, layout.solver_info);
  std::uint32_t* inactive_eigensolver_profile_count =
      inactive_eigensolver_profiling
          ? arena_pointer<std::uint32_t>(resources.arena_,
                                         layout.inactive_eigensolver_profile_count)
          : nullptr;
  DeviceInactiveEigensolverProfileEntry* inactive_eigensolver_profile =
      inactive_eigensolver_profiling ? arena_pointer<DeviceInactiveEigensolverProfileEntry>(
                                           resources.arena_, layout.inactive_eigensolver_profile)
                                     : nullptr;
  auto bounded_direct_cursor =
      arena_pointer<unsigned long long>(resources.arena_, layout.bounded_direct_cursor);

  std::vector<std::int32_t> host_pair_first;
  std::vector<std::int32_t> host_pair_second;
  if (first_setup) {
    host_pair_first.reserve(pair_count);
    host_pair_second.reserve(pair_count);
    // Canonical lower-triangle order is stable for the lifetime of a topology
    // plan. Upload it once so one-electron and direct-J/K consumers reuse the
    // same device metadata without rebuilding or decoding pair indices.
    for (std::size_t first = 0; first < nbf; ++first) {
      for (std::size_t second = 0; second <= first; ++second) {
        host_pair_first.push_back(static_cast<std::int32_t>(first));
        host_pair_second.push_back(static_cast<std::int32_t>(second));
      }
    }
  }

  const std::size_t shell_quartet_offset_bytes =
      quartet_direct && !bounded_direct_streaming
          ? plan.shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
          : 0;
  const std::size_t fp32_shell_quartet_offset_bytes =
      mixed_precision_fock ? plan.fp32_shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
                           : 0;
  const GeneratedShellPairStream host_bounded_stream_topology{
      static_cast<std::int32_t>(batch_size),
      static_cast<std::uint32_t>(direct_nbf),
      system_shell_offsets,
      system_shell_pair_offsets,
      shell_atoms,
      shell_angular,
      shell_direct_ao_offsets,
      shell_primitive_offsets,
      shell_pair_systems,
      shell_pair_first,
      shell_pair_second,
      bounded_stream_shell_pair_order,
      bounded_stream_pair_class_offsets,
      shell_pair_bounds,
      reinterpret_cast<const detail::GeneratedShellPairDensityBounds*>(shell_pair_density_bounds),
      bounded_direct_system_density_bounds,
      bounded_direct_system_pair_density_bounds,
      bounded_direct_generated_overflow,
      active};
  const std::pair<const void*, std::pair<void*, std::size_t>> static_uploads[] = {
      {host.atom_offsets.data(), {atom_offsets, host.atom_offsets.size() * sizeof(std::int64_t)}},
      {host.atom_systems.data(), {atom_systems, host.atom_systems.size() * sizeof(std::int32_t)}},
      {host.atomic_numbers.data(),
       {atomic_numbers, host.atomic_numbers.size() * sizeof(std::int32_t)}},
      {host.system_shell_offsets.data(),
       {system_shell_offsets, host.system_shell_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_atoms.data(), {shell_atoms, host.shell_atoms.size() * sizeof(std::int32_t)}},
      {host.shell_angular.data(),
       {shell_angular, host.shell_angular.size() * sizeof(std::uint8_t)}},
      {host.shell_ao_offsets.data(),
       {shell_ao_offsets, host.shell_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_direct_ao_offsets.data(),
       {shell_direct_ao_offsets, host.shell_direct_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_primitive_offsets.data(),
       {shell_primitive_offsets, host.shell_primitive_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_offsets.data(),
       {system_shell_pair_offsets, host.system_shell_pair_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_quartet_offsets.data(),
       {system_shell_quartet_offsets,
        host.system_shell_quartet_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_block_offsets.data(),
       {system_shell_pair_block_offsets,
        host.system_shell_pair_block_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_block_quartet_offsets.data(),
       {system_shell_pair_block_quartet_offsets,
        host.system_shell_pair_block_quartet_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_pair_systems.data(),
       {shell_pair_systems, host.shell_pair_systems.size() * sizeof(std::int32_t)}},
      {host.shell_pair_first.data(),
       {shell_pair_first, host.shell_pair_first.size() * sizeof(std::int32_t)}},
      {host.shell_pair_second.data(),
       {shell_pair_second, host.shell_pair_second.size() * sizeof(std::int32_t)}},
      {host.shell_pair_primitive_offsets.data(),
       {shell_pair_primitive_offsets,
        quartet_direct ? host.shell_pair_primitive_offsets.size() * sizeof(std::int64_t) : 0}},
      {host.psss_resident_tasks.data(),
       {psss_resident_tasks, quartet_direct && !bounded_direct_streaming
                                 ? host.psss_resident_tasks.size() * sizeof(PsssResidentTask)
                                 : 0}},
      {host.psss_resident_ket_pairs.data(),
       {psss_resident_ket_pairs, quartet_direct && !bounded_direct_streaming
                                     ? host.psss_resident_ket_pairs.size() * sizeof(std::uint32_t)
                                     : 0}},
      {host.ao_shells.data(), {ao_shells, host.ao_shells.size() * sizeof(std::int32_t)}},
      {host.ao_term_counts.data(),
       {ao_term_counts, host.ao_term_counts.size() * sizeof(std::uint8_t)}},
      {host.ao_term_angular.data(),
       {ao_term_angular, host.ao_term_angular.size() * sizeof(std::uint8_t)}},
      {host.ao_term_coefficients.data(),
       {ao_term_coefficients, host.ao_term_coefficients.size() * sizeof(double)}},
      {host.direct_ao_shells.data(),
       {direct_ao_shells, host.direct_ao_shells.size() * sizeof(std::int32_t)}},
      {host.direct_ao_angular.data(),
       {direct_ao_angular, host.direct_ao_angular.size() * sizeof(std::uint8_t)}},
      {host.direct_ao_coefficients.data(),
       {direct_ao_coefficients, host.direct_ao_coefficients.size() * sizeof(double)}},
      {host.ao_to_direct_transform.data(),
       {ao_to_direct_transform, host.ao_to_direct_transform.size() * sizeof(double)}},
      {host.primitive_exponents.data(),
       {primitive_exponents, host.primitive_exponents.size() * sizeof(double)}},
      {host.primitive_coefficients.data(),
       {primitive_coefficients, host.primitive_coefficients.size() * sizeof(double)}},
      {host.occupied.data(), {occupied, host.occupied.size() * sizeof(std::int32_t)}},
      {plan.shell_quartet_tile_offsets.data(),
       {active_shell_quartet_tile_offsets, shell_quartet_offset_bytes}},
      {plan.fp32_shell_quartet_tile_offsets.data(),
       {fp32_shell_quartet_tile_offsets, fp32_shell_quartet_offset_bytes}},
      {plan.bounded_generated_task_offsets.data(),
       {bounded_direct_generated_task_offsets,
        bounded_direct_streaming
            ? plan.bounded_generated_task_offsets.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_direct_shell_pair_order.data(),
       {bounded_direct_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_direct_shell_pair_order.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_shell_pair_order.data(),
       {bounded_stream_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_stream_shell_pair_order.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_pair_class_offsets.data(),
       {bounded_stream_pair_class_offsets,
        bounded_direct_streaming
            ? plan.bounded_stream_pair_class_offsets.size() * sizeof(std::uint32_t)
            : 0}},
      {&host_bounded_stream_topology,
       {bounded_stream_topology, bounded_direct_streaming ? sizeof(GeneratedShellPairStream) : 0}},
      {host_pair_first.data(), {ao_pair_first, host_pair_first.size() * sizeof(std::int32_t)}},
      {host_pair_second.data(), {ao_pair_second, host_pair_second.size() * sizeof(std::int32_t)}},
  };
  // Registry selections may include f-shell kernels for a batch containing
  // only s/p/d shells.  Intersect with the topology before deciding whether
  // the streaming set is complete; otherwise an impossible class forces an
  // O(N_shell^4) generic bounded scan.
  const std::uint64_t host_present_shell_class_mask =
      quartet_direct ? present_direct_shell_class_mask(host) : 0U;
  const std::uint64_t host_generated_fock_shell_class_mask =
      (generated::enabled_fock_shell_class_mask() & host_present_shell_class_mask) &
      (bounded_direct_streaming ? std::numeric_limits<std::uint64_t>::max()
                                : ~kFixedTopologyGeneratedFockExclusionMask);
  const std::uint64_t host_generated_mixed_fock_shell_class_mask =
      mixed_precision_fock
          ? generated::enabled_mixed_fock_shell_class_mask() &
                host_generated_fock_shell_class_mask & host_present_shell_class_mask
          : 0U;
  const std::uint64_t host_generated_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask & kGeneratedStreamingFockShellClassMask;
  const std::uint64_t host_native_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask & kNativeStreamingFockShellClassMask;
  const std::uint64_t host_uncovered_fock_shell_class_mask =
      host_present_shell_class_mask & ~host_generated_fock_shell_class_mask;
  // Per-item admission: each item divides the certified batch budget with its
  // own mixed-capable census and only enters the mixed route from a validated
  // warm state, because the reserved error bounds the perturbation of a known
  // state rather than of a cold guess. A refused item keeps a zero census, so
  // its tiles stay in the FP64 lists while its neighbors may still use mixed.
  // An explicit diagnostic cutoff stays item agnostic.
  std::vector<std::uint32_t> host_mixed_item_census(batch_size, 0U);
  std::vector<double> host_mixed_item_threshold(batch_size, 0.0);
  if (mixed_precision_fock) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const MixedPrecisionItemPolicy item = resolve_mixed_precision_item(
          requested_precision_policy, host.warm_mask[system] != 0,
          mixed_precision_system_census[system], options.screening_tolerance);
      if (!item.admitted) continue;
      host_mixed_item_census[system] = item.census;
      host_mixed_item_threshold[system] = item.threshold;
    }
  }
  const std::pair<const void*, std::pair<void*, std::size_t>> dynamic_uploads[] = {
      {host_mixed_item_census.data(),
       {mixed_precision_item_census,
        mixed_precision_fock ? host_mixed_item_census.size() * sizeof(std::uint32_t) : 0}},
      {host.warm_mask.data(),
       {warm_mask, device_resident_density_hit ? 0 : host.warm_mask.size() * sizeof(std::uint8_t)}},
      {host.warm_density.data(),
       {warm_density, device_resident_density_hit ? 0 : host.warm_density.size() * sizeof(double)}},
      {&host_generated_fock_shell_class_mask,
       {generated_fock_shell_class_mask, quartet_direct ? sizeof(std::uint64_t) : 0}},
      {&host_generated_mixed_fock_shell_class_mask,
       {generated_mixed_fock_shell_class_mask, mixed_precision_fock ? sizeof(std::uint64_t) : 0}},
  };
  if (first_setup) {
    for (const auto& upload : static_uploads) {
      const vibeqc_status status = copy_to_device(upload.second.first, upload.first,
                                                  upload.second.second, resources.stream_);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
    }
  }
  if (geometry_changed) {
    const vibeqc_status position_status =
        copy_to_device(positions, host.positions.data(), host.positions.size() * sizeof(double),
                       resources.stream_);
    if (position_status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, position_status);
      return outputs;
    }
  }
  for (const auto& upload : dynamic_uploads) {
    const vibeqc_status status =
        copy_to_device(upload.second.first, upload.first, upload.second.second, resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  if (cached_energy_baseline_hit) {
    // The SCF graph initializes previous_energy from this device buffer. A
    // frozen replay therefore restores its original seed explicitly instead
    // of relying on whatever energy the most recent resident density left.
    const vibeqc_status status =
        copy_to_device(energy, host_previous_energy_seed.data(),
                       host_previous_energy_seed.size() * sizeof(double), resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }

  DeviceBatch device_batch{static_cast<std::int32_t>(batch_size),
                           static_cast<std::int32_t>(nbf),
                           static_cast<std::int32_t>(direct_nbf),
                           static_cast<std::int64_t>(total_atoms),
                           static_cast<std::int64_t>(total_shells),
                           static_cast<std::int64_t>(total_shell_pairs),
                           static_cast<std::int64_t>(total_shell_quartets),
                           static_cast<std::int64_t>(total_shell_pair_blocks),
                           static_cast<std::int64_t>(total_shell_pair_block_quartets),
                           atom_offsets,
                           atom_systems,
                           atomic_numbers,
                           positions,
                           system_shell_offsets,
                           shell_atoms,
                           shell_angular,
                           shell_ao_offsets,
                           shell_direct_ao_offsets,
                           shell_primitive_offsets,
                           system_shell_pair_offsets,
                           system_shell_quartet_offsets,
                           system_shell_pair_block_offsets,
                           system_shell_pair_block_quartet_offsets,
                           shell_pair_systems,
                           shell_pair_first,
                           shell_pair_second,
                           shell_pair_primitive_offsets,
                           shell_primitive_pairs,
                           ao_shells,
                           ao_term_counts,
                           ao_term_angular,
                           ao_term_coefficients,
                           direct_ao_shells,
                           direct_ao_angular,
                           direct_ao_coefficients,
                           ao_to_direct_transform,
                           primitive_exponents,
                           primitive_coefficients,
                           occupied};
  device_batch.generated_psss_weighted = plan.generated_psss_weighted;

  if (quartet_direct && geometry_changed) {
    launch_build_shell_primitive_pair_cache_kernel(
        static_cast<unsigned>(total_shell_pairs), detail::kDirectQuartetThreads, 0,
        resources.stream_, device_batch, shell_primitive_pairs);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  if (first_setup && use_cusolver) {
    if (use_jacobi) {
      solver_error = cusolverDnDsyevjBatched_bufferSize(
          resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
          static_cast<int>(nbf), eigensystem, static_cast<int>(nbf), eigenvalues, &plan.lwork,
          resources.jacobi_, static_cast<int>(spin_batch_size));
      resources.solver_workspace_bytes_ = static_cast<std::size_t>(plan.lwork) * sizeof(double);
    } else if (ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd) {
      // Xsyevd is the non-batched counterpart used by GPU4PySCF for large
      // matrices.  Its workspace is independent of the number of systems;
      // launch_solver serializes one call per matrix on the ordinary stream.
      std::size_t device_bytes = 0;
      std::size_t host_bytes = 0;
      solver_error = cusolverDnXsyevd_bufferSize(
          resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(nbf), CUDA_R_64F, eigensystem,
          static_cast<std::int64_t>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F, &device_bytes,
          &host_bytes);
      resources.solver_workspace_bytes_ = device_bytes;
      resources.solver_host_workspace_bytes_ = host_bytes;
    } else {
      // RHF submits batch_size matrices; UHF additionally submits the doubled
      // spin batch. Query both actual capacities because cuSOLVER does not
      // guarantee workspace sizes are monotonic in batch count.
      const std::array<int, 2> capacities{static_cast<int>(batch_size),
                                          static_cast<int>(spin_batch_size)};
      for (const int capacity : capacities) {
        std::size_t device_bytes = 0;
        std::size_t host_bytes = 0;
        solver_error = cusolverDnXsyevBatched_bufferSize(
            resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf), CUDA_R_64F, eigensystem,
            static_cast<int>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F, &device_bytes, &host_bytes,
            capacity);
        if (solver_error != CUSOLVER_STATUS_SUCCESS) break;
        resources.solver_workspace_bytes_ =
            std::max(resources.solver_workspace_bytes_, device_bytes);
        resources.solver_host_workspace_bytes_ =
            std::max(resources.solver_host_workspace_bytes_, host_bytes);
      }
      plan.lwork = 0;
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      fill_global_failure(outputs, solver_status(solver_error));
      return outputs;
    }
    if (plan.lwork < 0 || resources.solver_workspace_bytes_ == 0) {
      fill_global_failure(outputs, VIBEQC_STATUS_CUDA_ERROR);
      return outputs;
    }
    if (options.export_physical_reference) {
      resources.reference_peak_bytes_ = reference_detail::check_capacity(
          reference_base_bytes,
          posthf::checked_add(resources.solver_workspace_bytes_,
                              resources.solver_host_workspace_bytes_),
          options.reference_memory_budget_bytes);
    }
    if ((cuda_error = runtime::resource_cuda_malloc_async(
             &resources.solver_workspace_, resources.solver_workspace_bytes_, resources.stream_)) !=
        cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (resources.solver_host_workspace_bytes_ != 0) {
      resources.solver_host_workspace_ = std::malloc(resources.solver_host_workspace_bytes_);
      if (resources.solver_host_workspace_ == nullptr) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (use_cublas) {
      blas_error = cublasSetWorkspace(resources.blas_, resources.solver_workspace_,
                                      resources.solver_workspace_bytes_);
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
  } else if (first_setup) {
    plan.lwork = 0;
  }
  const int lwork = plan.lwork;

  constexpr unsigned threads = kCaptureSafeKernelThreads;
  constexpr unsigned matrix_reduction_threads = kMatrixReductionThreads;
  const auto blocks_for = [](std::size_t elements) {
    return static_cast<unsigned>((elements + threads - 1) / threads);
  };
  const auto multiply_matrices = [&](const double* left, bool transpose_left, const double* right,
                                     double* output) {
    const vibeqc_status product_status = launch_matrix_product(
        resources.matrix_view(), static_cast<int>(batch_size), static_cast<int>(nbf), left,
        transpose_left, right, active, output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto multiply_spin_matrices = [&](const double* left, bool left_is_spin,
                                          bool transpose_left, const double* right,
                                          bool right_is_spin, double* output) {
    const vibeqc_status product_status = launch_spin_matrix_product(
        resources.matrix_view(), static_cast<int>(batch_size), 2, static_cast<int>(nbf), left,
        left_is_spin, transpose_left, right, right_is_spin, active, output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto build_commutator_residual = [&]() -> vibeqc_status {
    // [F, P]S is evaluated as four O(N^3) products.  `temporary` and
    // `eigensystem` are iteration scratch at this point: the former holds the
    // first product until it is folded into `residual`, while the latter is
    // overwritten before DIIS uses it as its effective-Fock output.  Keeping
    // the products in separate launches also lets the existing cuBLAS
    // strided-batched wrapper handle RHF and interleaved UHF layouts alike.
    vibeqc_status product_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      product_status = multiply_spin_matrices(fock, true, false, density, true, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(temporary, true, false, overlap, false, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(overlap, false, false, density, true, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(eigensystem, true, false, fock, true, temporary);
      }
    } else {
      product_status = multiply_matrices(fock, false, density, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(temporary, false, overlap, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(overlap, false, density, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(eigensystem, false, fock, temporary);
      }
    }
    if (product_status != VIBEQC_STATUS_SUCCESS) return product_status;
    launch_subtract_matrix_batches_kernel(
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
        static_cast<std::int32_t>(nbf), temporary, active, residual);
    return cuda_status(cudaPeekAtLastError());
  };
  const auto launch_direct_quartet_metadata = [&](const double* density_input,
                                                  bool allow_mixed_precision) -> cudaError_t {
    if (!quartet_direct) return cudaSuccess;
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      cudaError_t validation_error =
          cudaMemsetAsync(resources.direct_tile_validation_, 0xff,
                          sizeof(DirectTileValidationRecord), resources.stream_);
      if (validation_error != cudaSuccess) return validation_error;
    }
    const double* quartet_density = density_input;
    if (transformed_direct) {
      launch_transform_density_to_direct_right_kernel(
          blocks_for(spin_rectangular_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, density_input, active, direct_transform_temporary);
      launch_transform_density_to_direct_left_kernel(
          blocks_for(direct_spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_transform_temporary, active, direct_density);
      quartet_density = direct_density;
    }
    if (!bounded_direct_streaming) {
      launch_clear_active_shell_quartet_tile_counts_kernel(
          blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0, resources.stream_,
          active_shell_quartet_tile_counts, persistent_fock_task_heads,
          fp32_shell_quartet_tile_counts, fp32_persistent_fock_task_heads);
    }
    if (unrestricted) {
      launch_reduce_shell_pair_density_bounds_kernel(
          true, static_cast<unsigned>(total_shell_pairs), threads, 3 * threads * sizeof(double),
          resources.stream_, device_batch, quartet_density, active, shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        launch_reduce_bounded_system_density_bounds_kernel(
            static_cast<unsigned>(batch_size), threads,
            detail::kDirectShellPairClassCount * threads * sizeof(double), resources.stream_,
            device_batch, shell_pair_density_bounds, bounded_direct_system_density_bounds,
            bounded_direct_system_pair_density_bounds);
        return cudaPeekAtLastError();
      }
      launch_compact_active_shell_quartet_tiles_kernel(
          true, DirectScreeningPurpose::Fock, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          requested_precision_policy.item_cutoff_ceiling,
          requested_precision_policy.item_budget_error, mixed_precision_item_census,
          fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
          fp32_shell_quartet_tiles);
    } else {
      launch_reduce_shell_pair_density_bounds_kernel(
          false, static_cast<unsigned>(total_shell_pairs), threads, 3 * threads * sizeof(double),
          resources.stream_, device_batch, quartet_density, active, shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        launch_reduce_bounded_system_density_bounds_kernel(
            static_cast<unsigned>(batch_size), threads,
            detail::kDirectShellPairClassCount * threads * sizeof(double), resources.stream_,
            device_batch, shell_pair_density_bounds, bounded_direct_system_density_bounds,
            bounded_direct_system_pair_density_bounds);
        return cudaPeekAtLastError();
      }
      launch_compact_active_shell_quartet_tiles_kernel(
          false, DirectScreeningPurpose::Fock, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          requested_precision_policy.item_cutoff_ceiling,
          requested_precision_policy.item_budget_error, mixed_precision_item_census,
          fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
          fp32_shell_quartet_tiles);
    }
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      launch_validate_direct_tile_descriptors_kernel(
          blocks_for(plan.total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
          active_shell_quartet_tiles, plan.total_shell_quartet_tiles,
          resources.direct_tile_validation_);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_direct_force_compaction = [&]() -> cudaError_t {
    if (!quartet_direct || !force_density_product_screening) {
      return cudaSuccess;
    }
    if (bounded_direct_streaming) {
      // The force kernel reapplies the stronger density-product predicate as
      // it enumerates pair-of-pairs, so no intermediate force queue exists.
      return cudaSuccess;
    }
    // The final Fock/metadata path above has already reduced the selected
    // density in the direct Cartesian AO domain. Reuse those bounds and
    // overwrite the no-longer-needed Fock queue with its force-only subset.
    launch_clear_active_shell_quartet_tile_counts_kernel(
        blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0, resources.stream_,
        active_shell_quartet_tile_counts, persistent_fock_task_heads,
        fp32_shell_quartet_tile_counts, fp32_persistent_fock_task_heads);
    if (unrestricted) {
      launch_compact_active_shell_quartet_tiles_kernel(
          true, DirectScreeningPurpose::Force, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, false, 0.0, 0.0, nullptr,
          nullptr, nullptr, nullptr);
    } else {
      launch_compact_active_shell_quartet_tiles_kernel(
          false, DirectScreeningPurpose::Force, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, false, 0.0, 0.0, nullptr,
          nullptr, nullptr, nullptr);
    }
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      cudaError_t validation_error =
          cudaMemsetAsync(resources.direct_tile_validation_, 0xff,
                          sizeof(DirectTileValidationRecord), resources.stream_);
      if (validation_error != cudaSuccess) return validation_error;
      launch_validate_direct_tile_descriptors_kernel(
          blocks_for(plan.total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
          active_shell_quartet_tiles, plan.total_shell_quartet_tiles,
          resources.direct_tile_validation_);
    }
    return cudaPeekAtLastError();
  };
  std::size_t bounded_fock_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_fock_kernels =
      generated::selected_fock_shell_kernels(bounded_fock_kernel_count);
  const auto launch_bounded_streaming_fock =
      [&](bool is_unrestricted, const double* quartet_density, double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // Every selected class owns one independent queue head.  Reset the
    // complete fixed-size head array in one asynchronous memset before the
    // class-major launches instead of issuing one host API call per class.
    // The heads are disjoint, so this preserves launch ordering and atomic
    // accumulation semantics while removing serial dispatch overhead from
    // the bounded warm path.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_heads, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
      const unsigned shell_class = bounded_fock_kernels[kernel_index].shell_class;
      if ((host_generated_streaming_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) ==
          0U) {
        continue;
      }
      if (bounded_fock_class_timing) {
        launch_start_bounded_fock_class_timer_kernel(1, 1, 0, resources.stream_, shell_class,
                                                     bounded_fock_class_timer_starts);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
      error = generated::launch_shell_class_streaming_fock(
          shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
          bounded_stream_topology, device_batch.shell_pair_primitive_offsets,
          device_batch.shell_primitive_pairs, device_batch.direct_ao_coefficients,
          device_batch.positions, options.screening_tolerance,
          allow_mixed_precision && mixed_precision_fock &&
              (host_generated_mixed_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) !=
                  0U,
          mixed_precision_fock_threshold, schwarz_bounds, quartet_density, quartet_fock,
          bounded_direct_generated_task_heads + shell_class,
          bounded_fock_class_timing ? bounded_fock_fp64_work_counts + shell_class : nullptr,
          bounded_fock_class_timing ? bounded_fock_fp32_work_counts + shell_class : nullptr);
      if (error != cudaSuccess) return error;
      if (bounded_fock_class_timing) {
        launch_finish_bounded_fock_class_timer_kernel(
            1, 1, 0, resources.stream_, shell_class, bounded_fock_class_timer_starts,
            bounded_fock_class_timer_elapsed, bounded_fock_class_timer_launches);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
    }
    if ((host_native_streaming_fock_shell_class_mask & kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    // The complete head-array reset above also covers native DDDD.  Do not
    // issue a second class-specific memset here: the native fallback uses the
    // same disjoint head slot as generated classes.
    if (bounded_fock_class_timing) {
      launch_start_bounded_fock_class_timer_kernel(1, 1, 0, resources.stream_, kDdddShellClass,
                                                   bounded_fock_class_timer_starts);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return error;
    }
    if (is_unrestricted) {
      launch_bounded_direct_dddd_streaming_kernel(
          true, DirectScreeningPurpose::Fock, false, plan.persistent_quartet_worker_blocks,
          detail::kDirectQuartetThreads, 0, resources.stream_, device_batch,
          bounded_stream_topology, options.screening_tolerance, schwarz_bounds, quartet_density,
          active, quartet_fock, bounded_direct_generated_task_heads + kDdddShellClass, nullptr,
          bounded_fock_class_timing ? bounded_fock_fp64_work_counts + kDdddShellClass : nullptr);
    } else {
      launch_bounded_direct_dddd_streaming_kernel(
          false, DirectScreeningPurpose::Fock, false, plan.persistent_quartet_worker_blocks,
          detail::kDirectQuartetThreads, 0, resources.stream_, device_batch,
          bounded_stream_topology, options.screening_tolerance, schwarz_bounds, quartet_density,
          active, quartet_fock, bounded_direct_generated_task_heads + kDdddShellClass, nullptr,
          bounded_fock_class_timing ? bounded_fock_fp64_work_counts + kDdddShellClass : nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_fock_class_timing) {
      launch_finish_bounded_fock_class_timer_kernel(
          1, 1, 0, resources.stream_, kDdddShellClass, bounded_fock_class_timer_starts,
          bounded_fock_class_timer_elapsed, bounded_fock_class_timer_launches);
      error = cudaPeekAtLastError();
    }
    return error;
  };
  const auto launch_bounded_paged_generated_fock = [&](bool is_unrestricted,
                                                       const double* quartet_density,
                                                       double* quartet_fock) -> cudaError_t {
    // Generated classes use a fixed descriptor arena.  Enumerate their full
    // candidate ordinal domain in disjoint pages and consume each page before
    // reusing the arena.  This is both bounded in memory and exact: unlike the
    // old overflow tail, no first-wave task is revisited by a second scanner.
    const std::uint32_t page_capacity =
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
    if (page_capacity == 0U) return cudaErrorInvalidValue;
    for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
      const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
      const unsigned shell_class = kernel.shell_class;
      if ((host_generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U ||
          (host_native_streaming_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        continue;
      }
      unsigned high_pair_class = 0U;
      while ((high_pair_class + 1U) * (high_pair_class + 2U) / 2U <= shell_class) {
        ++high_pair_class;
      }
      const unsigned low_pair_class = shell_class - high_pair_class * (high_pair_class + 1U) / 2U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_counts + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_retry_task_offsets + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error != cudaSuccess) return error;

#define VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(unrestricted_value)                                      \
  launch_compact_bounded_exact_class_force_wave_kernel(                                           \
      unrestricted_value, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,    \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, bounded_stream_topology,         \
      shell_class, high_pair_class, low_pair_class, options.screening_tolerance, page_begin,      \
      page_capacity, page_range.bra_begin, page_range.bra_end, high_pair_class == low_pair_class, \
      bounded_direct_generated_tasks, bounded_direct_generated_task_counts + shell_class,         \
      bounded_direct_generated_task_heads + shell_class, nullptr, true, nullptr, nullptr)
        if (is_unrestricted) {
          VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(true);
        } else {
          VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(false);
        }
#undef VIBEQC_COMPACT_BOUNDED_FOCK_PAGE
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
        if (bounded_fock_class_timing) {
          launch_accumulate_fock_precision_work_kernel(
              1, 1, 0, resources.stream_, bounded_direct_generated_task_counts + shell_class,
              bounded_fock_fp64_work_counts + shell_class);
          error = cudaPeekAtLastError();
          if (error != cudaSuccess) return error;
        }
        // The compactor uses the head as a persistent bra scheduler; generated
        // consumers use the same slot as their task scheduler, so reset it
        // after compaction and before launching the page.
        error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                sizeof(std::uint32_t), resources.stream_);
        if (error != cudaSuccess) return error;
        error = generated::launch_shell_class_fock(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks,
            bounded_direct_generated_retry_task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_generated_fock =
      [&](bool is_unrestricted, const double* quartet_density, double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // The bounded Fock path follows the same hard routing invariant as force:
    // every present class must have a generated or native exact consumer.
    // Missing classes are unsupported instead of silently invoking the
    // whole-topology generic evaluator.
    if (host_uncovered_fock_shell_class_mask != 0U && !bounded_direct_aot_only_diagnostic) {
      return cudaErrorNotSupported;
    }
    if (bounded_direct_fock_only_diagnostic) {
      // The fixed-density measurement uses one uniform streaming schedule.
      // Mark every generated class for that consumer so an all-FP64 page does
      // not hide the arithmetic selected by the diagnostic threshold.
      cudaError_t diagnostic_error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 1,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
      if (diagnostic_error != cudaSuccess) return diagnostic_error;
      return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                           allow_mixed_precision);
    }
    if (!bounded_direct_count_diagnostic) {
      // Normal bounded execution uses disjoint exact pages for every
      // generated class.  Keep the legacy count/first-retry machinery below
      // exclusively for diagnostics, where its device readback is useful but
      // no Fock contribution is consumed.
      cudaError_t reset_error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 0,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
      if (reset_error != cudaSuccess) return reset_error;
      cudaError_t paged_error =
          launch_bounded_paged_generated_fock(is_unrestricted, quartet_density, quartet_fock);
      if (paged_error != cudaSuccess) return paged_error;
      return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                           allow_mixed_precision);
    }
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow, bounded_fock_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess || bounded_fock_kernel_count == 0) return error;
    // Even when every selected generated class has a streaming consumer,
    // compact through the shell-pair block gate first.  Directly scanning all
    // shell-pair products is quadratic in the 73,920-pair 768-AO case; the
    // compact route reduces the candidate domain by the fixed 256-pair block
    // factor and leaves streaming only as an overflow fallback.
    error =
        cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(unrestricted_value, task_offsets, selected_classes, \
                                             selected_any)                                       \
  launch_compact_bounded_generated_tasks_kernel(                                                 \
      unrestricted_value, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,   \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,    \
      shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,             \
      bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, active,      \
      generated_fock_shell_class_mask, 0U, host_native_streaming_fock_shell_class_mask,          \
      selected_classes, selected_any, bounded_direct_cursor, bounded_direct_generated_tasks,     \
      bounded_direct_generated_task_counts, task_offsets, bounded_direct_generated_overflow)
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(true, bounded_direct_generated_task_offsets, nullptr,
                                           nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(false, bounded_direct_generated_task_offsets, nullptr,
                                           nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_prepare_bounded_generated_retry_kernel(
        1, 1, 0, resources.stream_,
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask, bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any, bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) return cudaSuccess;
    const auto consume_generated_wave = [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const unsigned shell_class = bounded_fock_kernels[kernel_index].shell_class;
        if ((host_generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U) {
          continue;
        }
        cudaError_t launch_error = generated::launch_shell_class_fock(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(bounded_direct_generated_task_counts, 0,
                            detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                            resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(bounded_direct_generated_overflow, 0,
                              detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                              resources.stream_);
    }
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(true, bounded_direct_generated_retry_task_offsets,
                                           bounded_direct_generated_retry_mask,
                                           bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(false, bounded_direct_generated_retry_task_offsets,
                                           bounded_direct_generated_retry_mask,
                                           bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_normalize_bounded_generated_task_counts_kernel(
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0, resources.stream_,
        bounded_direct_generated_retry_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    error = consume_generated_wave(bounded_direct_generated_retry_task_offsets);
    if (error != cudaSuccess) return error;

    return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                         allow_mixed_precision);
  };
  // The exact provider is resolved/validated by run_hf_cuda_bucket_cached.
  // Dense, packed, generated and streamed paths below are execution schedules
  // of that same operator; retain their fused standard-HF kernel ownership.
  const auto launch_fock_builder = [&](const double* density_input,
                                       bool allow_mixed_precision) -> cudaError_t {
    const double* quartet_density = transformed_direct ? direct_density : density_input;
    double* quartet_fock = transformed_direct ? direct_fock : fock;
    if (quartet_direct) {
      cudaError_t metadata_error =
          launch_direct_quartet_metadata(density_input, allow_mixed_precision);
      if (metadata_error != cudaSuccess) return metadata_error;
      // Validation mode intentionally stops after compaction.  Continuing
      // into a consumer would turn a descriptor report into a secondary
      // illegal access and would obscure whether the queue itself is valid.
      if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
        return cudaSuccess;
      }
    }
    if (quartet_direct && transformed_direct) {
      launch_clear_active_matrices_kernel(
          blocks_for(direct_spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(direct_nbf), active, direct_fock);
    }
    if (quartet_direct) {
      if (plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder] != 0) {
        cudaError_t compact_error =
            cudaMemsetAsync(generic_order5_tile_count, 0, sizeof(std::uint32_t), resources.stream_);
        if (compact_error != cudaSuccess) return compact_error;
        launch_compact_generic_order5_tiles_kernel(
            blocks_for(plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder]), threads,
            0, resources.stream_, device_batch,
            active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
            active_shell_quartet_tiles +
                plan.shell_quartet_tile_offsets[kGenericOrderFiveAngularOrder],
            0U, generated_fock_shell_class_mask, generic_order5_tile_count, generic_order5_tiles);
        compact_error = cudaPeekAtLastError();
        if (compact_error != cudaSuccess) return compact_error;
      }
    }
    if (unrestricted && persistent_eri) {
      launch_build_uhf_fock_kernel(blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size),
                                   static_cast<std::int32_t>(nbf), hcore, eri, density_input,
                                   active, fock);
    } else if (unrestricted && quartet_direct) {
      if (!transformed_direct) {
        launch_initialize_direct_fock_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                             resources.stream_,
                                             static_cast<std::int32_t>(batch_size), 2,
                                             static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            true, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess || bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
            generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, true,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        dispatch_angular_fock_quartets(
            true, false, resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch, active_shell_quartet_tile_counts,
            active_shell_quartet_tiles, generic_order5_tile_count, generic_order5_tiles,
            persistent_fock_task_heads, plan.persistent_quartet_worker_blocks,
            options.screening_tolerance, schwarz_bounds, quartet_density, active, quartet_fock,
            generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity, plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, true,
              options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          dispatch_angular_fock_quartets(
              true, true, resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
    } else if (unrestricted) {
      launch_build_uhf_fock_direct_packed_kernel(
          static_cast<unsigned>(spin_matrix_elements), threads, threads * sizeof(double),
          resources.stream_, device_batch, options.screening_tolerance, hcore, ao_pair_first,
          ao_pair_second, pair_count, schwarz_bounds, density_input, active, fock);
    } else if (persistent_eri) {
      launch_build_fock_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                               static_cast<std::int32_t>(batch_size),
                               static_cast<std::int32_t>(nbf), hcore, eri, density_input, active,
                               fock);
    } else if (quartet_direct) {
      if (!transformed_direct) {
        launch_initialize_direct_fock_kernel(blocks_for(matrix_elements), threads, 0,
                                             resources.stream_,
                                             static_cast<std::int32_t>(batch_size), 1,
                                             static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            false, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess || bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
            generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, false,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        dispatch_angular_fock_quartets(
            false, false, resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch, active_shell_quartet_tile_counts,
            active_shell_quartet_tiles, generic_order5_tile_count, generic_order5_tiles,
            persistent_fock_task_heads, plan.persistent_quartet_worker_blocks,
            options.screening_tolerance, schwarz_bounds, quartet_density, active, quartet_fock,
            generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity, plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, false,
              options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          dispatch_angular_fock_quartets(
              false, true, resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
    } else {
      launch_build_fock_direct_packed_kernel(
          static_cast<unsigned>(matrix_elements), threads, threads * sizeof(double),
          resources.stream_, device_batch, options.screening_tolerance, hcore, ao_pair_first,
          ao_pair_second, pair_count, schwarz_bounds, density_input, active, fock);
    }
    if (quartet_direct && transformed_direct) {
      launch_transform_direct_fock_left_kernel(
          blocks_for(spin_rectangular_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_fock, active, direct_transform_temporary);
      launch_transform_direct_fock_right_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_transform_temporary, hcore, active, fock);
    }
    return cudaPeekAtLastError();
  };
  launch_initialize_state_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), cached_energy_baseline_hit,
                                 energy, active, converged, failed, iterations, previous_energy,
                                 energy_change, density_rms, diis_count, diis_head);
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (geometry_changed) {
    cuda_error = launch_generated_one_electron_values(
        one_electron_view(device_batch), ao_pair_first, ao_pair_second, pair_count,
        plan.one_electron_value_mapping, overlap, hcore, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    for (std::size_t e = 0; e < host.ecp_systems.size(); ++e) {
      std::string ecp_detail;
      const auto ecp_status =
          integrals::add_ecp_cuda(device_id, host.ecp_systems[e], resources.stream_,
                                  hcore + e * matrix_size, nullptr, nullptr, ecp_detail);
      if (ecp_status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, ecp_status);
        return outputs;
      }
    }
    if (persistent_eri) {
      launch_build_eri_kernel(blocks_for(eri_elements), threads, 0, resources.stream_, device_batch,
                              eri);
    } else {
      if (quartet_direct) {
        // The bounded route needs both AO-pair and shell-pair bounds.  Keep
        // the dense AO-pair grid (rather than one block per shell pair) and
        // reduce shell maxima atomically while each diagonal ERI is live.
        cuda_error = cudaMemsetAsync(shell_pair_bounds, 0, total_shell_pairs * sizeof(double),
                                     resources.stream_);
        if (cuda_error == cudaSuccess) {
          launch_build_schwarz_and_shell_pair_bounds_packed_kernel(
              static_cast<unsigned>(direct_pair_elements), kSchwarzThreads, 0, resources.stream_,
              device_batch, direct_pair_count, schwarz_bounds, shell_pair_bounds);
          cuda_error = cudaPeekAtLastError();
        }
        if (cuda_error != cudaSuccess) {
          fill_global_failure(outputs, cuda_status(cuda_error));
          return outputs;
        }
        if (bounded_direct_streaming) {
          // Keep every system and class segment Schwarz-descending for the
          // current geometry.  Generic blocks then group similar work, while
          // generated resident-bra streams may safely stop at the first full
          // ket chunk below the geometry-only gate.
          std::vector<double> host_shell_pair_bounds(total_shell_pairs);
          cuda_error = cudaMemcpyAsync(host_shell_pair_bounds.data(), shell_pair_bounds,
                                       total_shell_pairs * sizeof(double), cudaMemcpyDeviceToHost,
                                       resources.stream_);
          if (cuda_error == cudaSuccess) {
            cuda_error = cudaStreamSynchronize(resources.stream_);
          }
          if (cuda_error != cudaSuccess) {
            fill_global_failure(outputs, cuda_status(cuda_error));
            return outputs;
          }
          for (std::size_t system = 0; system < batch_size; ++system) {
            const std::size_t pair_begin =
                static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
            const std::size_t pair_end =
                static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1]);
            std::stable_sort(plan.bounded_direct_shell_pair_order.begin() + pair_begin,
                             plan.bounded_direct_shell_pair_order.begin() + pair_end,
                             [&](std::uint32_t first, std::uint32_t second) {
                               return host_shell_pair_bounds[first] >
                                      host_shell_pair_bounds[second];
                             });
          }
          const std::size_t class_stride = batch_size + 1U;
          for (std::size_t pair_class = 0; pair_class < detail::kDirectShellPairClassCount;
               ++pair_class) {
            for (std::size_t system = 0; system < batch_size; ++system) {
              const std::size_t segment_begin =
                  plan.bounded_stream_pair_class_offsets[pair_class * class_stride + system];
              const std::size_t segment_end =
                  plan.bounded_stream_pair_class_offsets[pair_class * class_stride + system + 1U];
              std::stable_sort(plan.bounded_stream_shell_pair_order.begin() + segment_begin,
                               plan.bounded_stream_shell_pair_order.begin() + segment_end,
                               [&](std::uint32_t first, std::uint32_t second) {
                                 return host_shell_pair_bounds[first] >
                                        host_shell_pair_bounds[second];
                               });
            }
          }
          const vibeqc_status order_upload_status = copy_to_device(
              bounded_direct_shell_pair_order, plan.bounded_direct_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, order_upload_status);
            return outputs;
          }
          const vibeqc_status stream_order_upload_status = copy_to_device(
              bounded_stream_shell_pair_order, plan.bounded_stream_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (stream_order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, stream_order_upload_status);
            return outputs;
          }
        }
        if (bounded_direct_streaming) {
          launch_reduce_bounded_shell_pair_block_bounds_kernel(
              static_cast<unsigned>(total_shell_pair_blocks), threads, threads * sizeof(double),
              resources.stream_, device_batch, bounded_direct_shell_pair_order, shell_pair_bounds,
              bounded_direct_shell_pair_block_bounds);
        }
      } else if (options.export_physical_reference) {
        // Every unscreened AO pair satisfies 0*0 >= 0. No Cartesian Schwarz
        // array is needed by this bounded public-AO matrix-direct path.
        cuda_error =
            cudaMemsetAsync(schwarz_bounds, 0, matrix_elements * sizeof(double), resources.stream_);
        if (cuda_error != cudaSuccess) {
          fill_global_failure(outputs, cuda_status(cuda_error));
          return outputs;
        }
      } else {
        launch_build_schwarz_bounds_packed_kernel(blocks_for(direct_pair_elements), threads, 0,
                                                  resources.stream_, device_batch,
                                                  direct_pair_count, schwarz_bounds);
      }
    }
    launch_build_nuclear_repulsion_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                          device_batch, nuclear_repulsion);

    launch_copy_matrix_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                              matrix_elements, overlap, eigensystem);
    status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                           static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                           temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    launch_build_orthogonalizer_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                       static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), eigensystem, eigenvalues,
                                       active, orthogonalizer, failed);
  }

  // A valid warm density supersedes the core-Hamiltonian guess. Homogeneous
  // warm replay can therefore skip its transforms, eigensolve, and density
  // construction without changing mixed warm/cold bucket semantics.
  if (!all_systems_warm) {
    status = multiply_matrices(hcore, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                           static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                           temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    if (unrestricted) {
      status = multiply_matrices(orthogonalizer, false, eigensystem, temporary);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      launch_broadcast_spin_matrix_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          2, static_cast<std::int32_t>(nbf), temporary, active,
                                          coefficients);
      launch_mix_open_shell_guess_kernel(blocks_for(batch_size * nbf), threads, 0,
                                         resources.stream_, static_cast<std::int32_t>(batch_size),
                                         static_cast<std::int32_t>(nbf), occupied, active,
                                         coefficients);
      launch_build_spin_density_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                       resources.stream_, static_cast<std::int32_t>(batch_size), 2,
                                       static_cast<std::int32_t>(nbf), occupied, coefficients,
                                       active, density);
    } else {
      status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      launch_build_density_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                  static_cast<std::int32_t>(batch_size),
                                  static_cast<std::int32_t>(nbf), occupied, coefficients, active,
                                  density);
    }
  }
  std::vector<std::uint8_t> host_warm_invalid;
  if (!device_resident_density_hit && any_system_warm) {
    // The normalization kernel also performs the CPU-equivalent metric trace
    // check.  Fence only this exceptional input-validation path; a resident
    // replay skips both the host upload and this O(N^2) setup scan.
    cuda_error =
        cudaMemsetAsync(warm_invalid, 0, batch_size * sizeof(std::uint8_t), resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (unrestricted) {
      launch_apply_uhf_warm_density_kernel(
          static_cast<unsigned>(batch_size), kWarmDensityThreads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), occupied,
          warm_mask, warm_density, overlap, density, warm_invalid);
    } else {
      launch_apply_warm_density_kernel(static_cast<unsigned>(batch_size), kWarmDensityThreads, 0,
                                       resources.stream_, static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), occupied, warm_mask,
                                       warm_density, overlap, density, warm_invalid);
    }
    host_warm_invalid.resize(batch_size, 0);
    cuda_error =
        cudaMemcpyAsync(host_warm_invalid.data(), warm_invalid, batch_size * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (std::any_of(host_warm_invalid.begin(), host_warm_invalid.end(),
                    [](std::uint8_t value) { return value != 0; })) {
      // The validation kernel has already symmetrized the candidate in the
      // resident density buffer. Residency was invalidated before execution,
      // and a rejected trace must not publish a replacement cache entry. A
      // separately frozen valid dm0/energy pair remains safe to replay.
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }

  const EigensolverProfileLaunch graph_eigensolver_profile{
      static_cast<std::int32_t>(batch_size),
      active,
      use_cublas,
      static_cast<std::uint32_t>(options.max_iterations),
      inactive_eigensolver_profile_count,
      inactive_eigensolver_profile,
  };
  const EigensolverProfileLaunch* graph_eigensolver_profile_pointer =
      inactive_eigensolver_profiling ? &graph_eigensolver_profile : nullptr;

  const bool fock_only_iteration = bounded_direct_fock_only_diagnostic && bounded_direct_streaming;
  // CUDA 12.9 rejects XsyevBatched capture for matrices above 512 AOs.  Keep
  // the expensive Fock and matrix work in two reusable Graphs while the host
  // inserts a GPU4PySCF-style ordinary eigensolver call between them and
  // checks the tiny physical active mask once per SCF iteration.
  const bool split_provider_iteration =
      !fock_only_iteration &&
      (ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched ||
       ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd) &&
      graph_eigensolver_family != ordinary_eigensolver_family;

  const auto launch_iteration_pre_eigensolver = [&](bool allow_mixed_precision) -> vibeqc_status {
    const cudaError_t fock_error = launch_fock_builder(density, allow_mixed_precision);
    if (fock_error != cudaSuccess) return cuda_status(fock_error);
    if (fock_only_iteration) return VIBEQC_STATUS_SUCCESS;

    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      launch_compute_uhf_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads,
                                       0, resources.stream_, static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), density, hcore, fock,
                                       nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    } else {
      launch_compute_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                   resources.stream_, static_cast<std::int32_t>(batch_size),
                                   static_cast<std::int32_t>(nbf), density, hcore, fock,
                                   nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;

    launch_update_diis_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                              resources.stream_, static_cast<std::int32_t>(batch_size),
                              static_cast<std::int32_t>(nbf), unrestricted ? 2 : 1,
                              static_cast<std::uint32_t>(diis_history), fock, residual, active,
                              fock_history, residual_history, diis_linear_system, diis_coefficients,
                              diis_count, diis_head, eigensystem);
    if (unrestricted) {
      iteration_status =
          multiply_spin_matrices(eigensystem, true, false, orthogonalizer, false, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status =
            multiply_spin_matrices(orthogonalizer, false, true, temporary, true, eigensystem);
      }
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_expand_spin_active_kernel(blocks_for(spin_batch_size), threads, 0, resources.stream_,
                                         static_cast<std::int32_t>(batch_size), 2, active,
                                         spin_active);
      }
    } else {
      iteration_status = multiply_matrices(eigensystem, false, orthogonalizer, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    return cuda_status(cudaPeekAtLastError());
  };

  const auto launch_iteration_eigensolver = [&](CudaEigensolverFamily family) -> vibeqc_status {
    return launch_solver(resources.eigensolver_view(), family, static_cast<int>(nbf),
                         static_cast<int>(unrestricted ? spin_batch_size : batch_size), eigensystem,
                         temporary, eigenvalues, lwork, solver_info,
                         unrestricted ? spin_active : active, graph_eigensolver_profile_pointer);
  };

  const auto launch_iteration_post_eigensolver = [&](bool append_device_tail) -> vibeqc_status {
    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      launch_inspect_spin_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                        static_cast<std::int32_t>(batch_size), 2, solver_info,
                                        active, failed, converged);
      iteration_status =
          multiply_spin_matrices(orthogonalizer, false, false, eigensystem, true, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_build_spin_density_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                         resources.stream_, static_cast<std::int32_t>(batch_size),
                                         2, static_cast<std::int32_t>(nbf), occupied, coefficients,
                                         active, next_density);
        if (reuse_converged_fock) {
          launch_update_uhf_convergence_kernel(
              true, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          launch_update_uhf_convergence_kernel(
              false, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    } else {
      launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), solver_info, active,
                                   failed, converged);
      iteration_status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_build_density_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                    static_cast<std::int32_t>(batch_size),
                                    static_cast<std::int32_t>(nbf), occupied, coefficients, active,
                                    next_density);
        if (reuse_converged_fock) {
          launch_update_convergence_kernel(
              true, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          launch_update_convergence_kernel(
              false, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    if (append_device_tail) {
      launch_tail_rhf_loop_kernel(1, 1, 0, resources.stream_, static_cast<std::int32_t>(batch_size),
                                  options.max_iterations, active, iterations);
    }
    return cuda_status(cudaPeekAtLastError());
  };

  if (first_setup) {
    // Graph construction is allocation-permitted setup work. Synchronize once
    // so capture cannot race the initial guess; fixed-topology replays reuse
    // this executable and do not repeat the fence or provider setup.
    cuda_error = cudaStreamSynchronize(resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(resources.stream_, cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    status = launch_iteration_pre_eigensolver(true);
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration && !split_provider_iteration) {
      status = launch_iteration_eigensolver(graph_eigensolver_family);
    }
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration && !split_provider_iteration) {
      status = launch_iteration_post_eigensolver(true);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      cudaGraph_t abandoned_graph = nullptr;
      (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
      if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status);
      return outputs;
    }
    cuda_error = cudaStreamEndCapture(resources.stream_, &resources.iteration_graph_);
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess ||
        resources.iteration_graph_ == nullptr) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
    cuda_error =
        cudaGraphInstantiate(&resources.iteration_graph_exec_, resources.iteration_graph_,
                             split_provider_iteration ? 0U : cudaGraphInstantiateFlagDeviceLaunch);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaGraphUpload(resources.iteration_graph_exec_, resources.stream_);
    }
    if (cuda_error == cudaSuccess && split_provider_iteration) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamBeginCapture(resources.stream_, cudaStreamCaptureModeThreadLocal);
      }
      if (cuda_error == cudaSuccess) {
        status = launch_iteration_post_eigensolver(false);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaStreamEndCapture(resources.stream_, &resources.post_eigensolver_graph_);
      } else {
        cudaGraph_t abandoned_graph = nullptr;
        (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
        if (abandoned_graph != nullptr) {
          (void)cudaGraphDestroy(abandoned_graph);
        }
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS &&
          resources.post_eigensolver_graph_ != nullptr) {
        cuda_error = cudaGraphInstantiate(&resources.post_eigensolver_graph_exec_,
                                          resources.post_eigensolver_graph_, 0U);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaGraphUpload(resources.post_eigensolver_graph_exec_, resources.stream_);
      }
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
    plan.initialized = true;
  }
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemsetAsync(inactive_eigensolver_profile_count, 0, sizeof(std::uint32_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_class_timer_elapsed, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(std::uint64_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_class_timer_launches, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_fp64_work_counts, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(unsigned long long),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_fp32_work_counts, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(unsigned long long),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && split_provider_iteration) {
    std::vector<std::uint8_t> host_active(batch_size, 1U);
    for (std::uint32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
      cuda_error = cudaGraphLaunch(resources.iteration_graph_exec_, resources.stream_);
      if (cuda_error != cudaSuccess) break;
      status = launch_iteration_eigensolver(ordinary_eigensolver_family);
      if (status != VIBEQC_STATUS_SUCCESS) break;
      cuda_error = cudaGraphLaunch(resources.post_eigensolver_graph_exec_, resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpyAsync(host_active.data(), active, batch_size * sizeof(std::uint8_t),
                                     cudaMemcpyDeviceToHost, resources.stream_);
      }
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamSynchronize(resources.stream_);
      }
      if (cuda_error != cudaSuccess ||
          std::none_of(host_active.begin(), host_active.end(),
                       [](std::uint8_t value) { return value != 0; })) {
        break;
      }
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else if (cuda_error == cudaSuccess) {
    cuda_error = cudaGraphLaunch(resources.iteration_graph_exec_, resources.stream_);
  }
  if (cuda_error == cudaSuccess && direct_tile_validation &&
      resources.direct_tile_validation_ != nullptr) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    DirectTileValidationRecord host_validation{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(&host_validation, resources.direct_tile_validation_,
                              sizeof(host_validation), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      if (host_validation.error == kDirectTileValidationNoError) {
        std::fprintf(stderr, "direct-tile-validation error=none\n");
        std::fflush(stderr);
      } else {
        const char* error_name = "none";
        switch (static_cast<DirectTileValidationError>(host_validation.error)) {
          case DirectTileValidationError::count_exceeds_capacity:
            error_name = "count-exceeds-capacity";
            break;
          case DirectTileValidationError::pair_out_of_bounds:
            error_name = "pair-out-of-bounds";
            break;
          case DirectTileValidationError::shell_out_of_bounds:
            error_name = "shell-out-of-bounds";
            break;
          case DirectTileValidationError::tile_out_of_bounds:
            error_name = "tile-out-of-bounds";
            break;
          case DirectTileValidationError::ao_range_invalid:
            error_name = "ao-range-invalid";
            break;
          default:
            break;
        }
        std::fprintf(stderr,
                     "direct-tile-validation error=%s order=%u slot=%u tile=%u "
                     "pairs=(%u,%u) shells=(%d,%d,%d,%d) direct_nbf=%u "
                     "pair_counts=(%u,%u) ao=(%u,%u,%u,%u) count=%u capacity=%u "
                     "partition_begin=%u\n",
                     error_name, host_validation.angular_order, host_validation.slot,
                     host_validation.tile, host_validation.first_pair, host_validation.second_pair,
                     host_validation.shell[0], host_validation.shell[1], host_validation.shell[2],
                     host_validation.shell[3], host_validation.direct_nbf,
                     host_validation.first_pair_count, host_validation.second_pair_count,
                     host_validation.i, host_validation.j, host_validation.k, host_validation.l,
                     host_validation.active_tile_count, host_validation.partition_capacity,
                     host_validation.partition_begin);
        std::fflush(stderr);
      }
    }
  }
  if (cuda_error == cudaSuccess && bounded_direct_count_diagnostic && bounded_direct_streaming) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_counts{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_overflow{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(host_counts.data(), bounded_direct_generated_task_counts,
                              host_counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(host_overflow.data(), bounded_direct_generated_overflow,
                              host_overflow.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      std::fprintf(stderr, "bounded-direct-count purpose=scf-fock capacity=%zu\n",
                   plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(stderr, "  %-4s count=%u capacity=%u overflow=%u\n", kernel.name,
                     host_counts[kernel.shell_class], class_capacity,
                     host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing && bounded_direct_streaming) {
    std::array<std::uint64_t, detail::kDirectQuartetShellClassCount> host_elapsed{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_launches{};
    std::array<unsigned long long, detail::kDirectQuartetShellClassCount> host_fp64_work{};
    std::array<unsigned long long, detail::kDirectQuartetShellClassCount> host_fp32_work{};
    cuda_error = cudaMemcpyAsync(host_elapsed.data(), bounded_fock_class_timer_elapsed,
                                 host_elapsed.size() * sizeof(std::uint64_t),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_launches.data(), bounded_fock_class_timer_launches,
                                   host_launches.size() * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_fp64_work.data(), bounded_fock_fp64_work_counts,
                                   host_fp64_work.size() * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_fp32_work.data(), bounded_fock_fp32_work_counts,
                                   host_fp32_work.size() * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      struct HostFockClassTiming {
        const char* name;
        unsigned shell_class;
        std::uint64_t elapsed_nanoseconds;
        std::uint32_t launches;
      };
      std::vector<HostFockClassTiming> timings;
      std::uint64_t total_elapsed_nanoseconds = 0U;
      timings.reserve(bounded_fock_kernel_count);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const unsigned shell_class = kernel.shell_class;
        if (shell_class >= host_launches.size() || host_launches[shell_class] == 0U) {
          continue;
        }
        timings.push_back(
            {kernel.name, shell_class, host_elapsed[shell_class], host_launches[shell_class]});
        total_elapsed_nanoseconds += host_elapsed[shell_class];
      }
      std::sort(timings.begin(), timings.end(),
                [](const HostFockClassTiming& first, const HostFockClassTiming& second) {
                  return first.elapsed_nanoseconds > second.elapsed_nanoseconds;
                });
      std::fprintf(stderr, "bounded-direct-fock-class-profile total_gpu_ms=%.6f classes=%zu\n",
                   static_cast<double>(total_elapsed_nanoseconds) * 1.0e-6, timings.size());
      for (const HostFockClassTiming& timing : timings) {
        const double share = total_elapsed_nanoseconds == 0U
                                 ? 0.0
                                 : 100.0 * static_cast<double>(timing.elapsed_nanoseconds) /
                                       static_cast<double>(total_elapsed_nanoseconds);
        std::fprintf(stderr, "  %-4s class=%u launches=%u gpu_ms=%.6f share=%.2f%%\n", timing.name,
                     timing.shell_class, timing.launches,
                     static_cast<double>(timing.elapsed_nanoseconds) * 1.0e-6, share);
      }
      std::fprintf(stderr, "bounded-direct-fock-precision-profile enabled=%u threshold=%.17g\n",
                   mixed_precision_fock ? 1U : 0U, mixed_precision_fock_threshold);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const unsigned shell_class = kernel.shell_class;
        if (shell_class >= host_fp64_work.size() ||
            (host_fp64_work[shell_class] == 0ULL && host_fp32_work[shell_class] == 0ULL)) {
          continue;
        }
        const unsigned mixed_capable =
            (host_generated_mixed_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U
                ? 1U
                : 0U;
        std::fprintf(stderr,
                     "  %-4s class=%u fp64_quartets=%llu fp32_quartets=%llu mixed_capable=%u\n",
                     kernel.name, shell_class, host_fp64_work[shell_class],
                     host_fp32_work[shell_class], mixed_capable);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // The isolated profile intentionally captures one Fock construction and
    // omits the eigensolve/convergence tail. Return after device timings are
    // copied so no final-Fock rebuild or analytic-force work contaminates it.
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_CONVERGED);
    for (auto& output : outputs) output.fock_only_diagnostic = true;
    return outputs;
  }

  // ---------------------------------------------------------------------------
  // Target-precision refinement.
  //
  // The mixed Fock is only the iterative operator, so a density converged under
  // it is not a converged solution of the requested FP64 equations. Promote the
  // items that used mixed precision to exact FP64 from their mixed density and
  // continue until the same criteria are met. An item that exhausts the bound
  // stays unconverged, so a noisy state is never reported as a success, and the
  // refinement cost is counted in the reported iterations.
  // ---------------------------------------------------------------------------
  std::vector<std::uint32_t> host_mixed_iterations(batch_size, 0U);
  if (mixed_precision_fock) {
    cuda_error = cudaMemcpyAsync(host_mixed_iterations.data(), iterations,
                                 batch_size * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                 resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    // Only the items that actually ran the mixed operator re-enter the loop.
    launch_enter_target_refinement_kernel(
        blocks_for(batch_size), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), mixed_precision_item_census, active, converged,
        failed, iterations, previous_energy, energy_change, density_rms, diis_count, diis_head);
    std::vector<std::uint8_t> host_refinement_active(batch_size, 0U);
    cuda_error =
        cudaMemcpyAsync(host_refinement_active.data(), active, batch_size * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    // A mixed density within the reserved budget needs only a few exact
    // iterations; the full iteration bound applies so a pathological state
    // reports an honest non-convergence instead of a clamped success. Each item
    // leaves the loop on its own convergence, so a stagnating item is promoted
    // without holding back or dictating the precision of its neighbors.
    for (std::uint32_t refinement = 0; refinement < options.max_iterations; ++refinement) {
      if (std::none_of(host_refinement_active.begin(), host_refinement_active.end(),
                       [](std::uint8_t value) { return value != 0; })) {
        break;
      }
      status = launch_iteration_pre_eigensolver(false);
      if (status == VIBEQC_STATUS_SUCCESS) {
        status = launch_iteration_eigensolver(ordinary_eigensolver_family);
      }
      if (status == VIBEQC_STATUS_SUCCESS) {
        status = launch_iteration_post_eigensolver(false);
      }
      if (status != VIBEQC_STATUS_SUCCESS) break;
      cuda_error =
          cudaMemcpyAsync(host_refinement_active.data(), active, batch_size * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost, resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamSynchronize(resources.stream_);
      }
      if (cuda_error != cudaSuccess) break;
    }
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
  }
  std::uint32_t host_final_fock_rebuild_count = static_cast<std::uint32_t>(batch_size);
  if (reuse_converged_fock) {
    // Partition on the device because density RMS is already per-system. This
    // permits a mixed bucket: tight systems retain P_n/F(P_n), while only
    // looser systems restore P_{n+1} and execute the exact legacy rebuild.
    cuda_error =
        cudaMemsetAsync(final_fock_rebuild_count, 0, sizeof(std::uint32_t), resources.stream_);
    if (cuda_error == cudaSuccess) {
      launch_select_final_fock_rebuild_kernel(
          blocks_for(batch_size), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size),
          converged_fock_reuse_density_rms(options.density_tolerance), density_rms, converged,
          failed, final_fock_reuse_mask, active, final_fock_rebuild_count);
      launch_copy_selected_matrices_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), active, next_density, density);
      cuda_error =
          cudaMemcpyAsync(&host_final_fock_rebuild_count, final_fock_rebuild_count,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      // One post-Graph scalar fence avoids launching the expensive Fock
      // worker family when every system can reuse its retained matrix.
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (host_final_fock_rebuild_count != 0) {
      cuda_error = launch_fock_builder(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    launch_select_converged_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), converged, failed,
                                   active);
    if (quartet_direct && batch_size > 1 && host_final_fock_rebuild_count != batch_size) {
      // Later device-tail launches overwrite the shared compact quartet list
      // after an early peer converges. Recreate only density transforms,
      // shell-pair bounds, and task metadata for all final snapshots; do not
      // evaluate any two-electron integrals or modify retained Fock matrices.
      cuda_error = launch_direct_quartet_metadata(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
  } else {
    launch_select_converged_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), converged, failed,
                                   active);
    cuda_error = launch_fock_builder(density, false);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  // Diagonalize each un-extrapolated final Fock. Tight systems consume their
  // retained P_n/F(P_n); rebuilt systems consume P_{n+1}/F(P_{n+1}). The
  // active mask now contains every converged system for common finalization.
  if (unrestricted) {
    status = multiply_spin_matrices(fock, true, false, orthogonalizer, false, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_spin_matrices(orthogonalizer, false, true, temporary, true, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      launch_expand_spin_active_kernel(blocks_for(spin_batch_size), threads, 0, resources.stream_,
                                       static_cast<std::int32_t>(batch_size), 2, active,
                                       spin_active);
      status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                             static_cast<int>(nbf), static_cast<int>(spin_batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info, spin_active);
    }
  } else {
    status = multiply_matrices(fock, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                             static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info, active);
    }
  }
  if (status != VIBEQC_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  if (unrestricted) {
    launch_inspect_spin_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                      static_cast<std::int32_t>(batch_size), 2, solver_info, active,
                                      failed, converged);
    status = multiply_spin_matrices(orthogonalizer, false, false, eigensystem, true, coefficients);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else {
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  // Keep the converged density paired with the un-extrapolated F(P) that was
  // just diagonalized. The canonical coefficients and eigenvalues are needed
  // for the Pulay weighted density, but replacing P with C_occ C_occ^T would
  // require a second complete J/K rebuild before energy and force evaluation.
  // The accepted density update has already passed the requested SCF density
  // tolerance, so retaining P keeps all final energy/two-electron force terms
  // evaluated consistently at the same P and F(P).
  if (options.compute_forces) {
    cuda_error = launch_direct_force_compaction();
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (shell_class_profiling && quartet_direct) {
      cuda_error = cudaMemsetAsync(
          shell_class_profile, 0,
          detail::kDirectQuartetShellClassCount * sizeof(DeviceShellClassProfileEntry),
          resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    if (shell_class_profiling && quartet_direct && total_shell_quartet_tiles != 0) {
      launch_profile_active_shell_quartet_tiles_kernel(
          blocks_for(total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          total_shell_quartet_tiles, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, shell_class_profile);
    }
  }
  if (unrestricted) {
    launch_compute_uhf_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                     resources.stream_, static_cast<std::int32_t>(batch_size),
                                     static_cast<std::int32_t>(nbf), density, hcore, fock,
                                     nuclear_repulsion, active, energy);
    if (options.compute_forces) {
      launch_build_spin_weighted_density_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), occupied,
          coefficients, eigenvalues, active, weighted_density);
      launch_sum_uhf_spin_matrices_kernel(blocks_for(matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          static_cast<std::int32_t>(nbf), density, active,
                                          total_density);
      launch_sum_uhf_spin_matrices_kernel(blocks_for(matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          static_cast<std::int32_t>(nbf), weighted_density, active,
                                          total_weighted_density);
    }
  } else {
    launch_compute_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                 resources.stream_, static_cast<std::int32_t>(batch_size),
                                 static_cast<std::int32_t>(nbf), density, hcore, fock,
                                 nuclear_repulsion, active, energy);
    if (options.compute_forces) {
      launch_build_weighted_density_kernel(blocks_for(matrix_elements), threads, 0,
                                           resources.stream_, static_cast<std::int32_t>(batch_size),
                                           static_cast<std::int32_t>(nbf), occupied, coefficients,
                                           eigenvalues, active, weighted_density);
    }
  }
  if (options.export_physical_reference) {
    outputs[0].status = reference_detail::download(
        resources.stream_, nbf, host.occupied[0], resources.reference_peak_bytes_,
        {overlap, hcore, fock, coefficients, density}, eigenvalues,
        {energy, energy_change, density_rms}, converged, failed, iterations, outputs[0].scf);
    return outputs;
  }
  if (options.compute_forces) {
    cuda_error = cudaMemsetAsync(forces, 0, total_atoms * 3 * sizeof(double), resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (quartet_direct && !bounded_direct_streaming) {
      cuda_error = cudaMemsetAsync(persistent_force_task_heads, 0,
                                   kPersistentForceAngularOrderCount * sizeof(std::uint32_t),
                                   resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    launch_nuclear_force_kernel(blocks_for(force_coordinate_count), threads, 0, resources.stream_,
                                device_batch, active, forces);
    for (std::size_t e = 0; e < host.ecp_systems.size(); ++e) {
      std::string ecp_detail;
      const auto ecp_status =
          integrals::add_ecp_cuda(device_id, host.ecp_systems[e], resources.stream_, nullptr,
                                  (unrestricted ? total_density : density) + e * matrix_size,
                                  forces + 3 * host.atom_offsets[e], ecp_detail);
      if (ecp_status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, ecp_status);
        return outputs;
      }
    }
    // Derivative selection is read on every force execution; it retains no
    // candidate-specific geometry or plan buffers that could become stale.
    if (cuda_policy::generated_one_electron_derivatives_requested()) {
      const OneElectronWeightView weights{unrestricted ? total_weighted_density : weighted_density,
                                          unrestricted ? total_density : density,
                                          unrestricted ? total_density : density,
                                          -1.0,
                                          1.0,
                                          1.0};
      cuda_error = launch_generated_one_electron_gradient(
          one_electron_view(device_batch), ao_pair_first, ao_pair_second, pair_count, weights,
          active, cuda_policy::one_electron_derivative_mapping_requested(), -1.0, forces,
          resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    } else if (cooperative_one_electron_force) {
      constexpr std::size_t shared_bytes = 3 * sizeof(OneElectronDerivativeHermiteCoefficients);
      launch_one_electron_force_cooperative_kernel(
          static_cast<unsigned>(one_electron_force_elements), threads, shared_bytes,
          resources.stream_, device_batch, ao_pair_first, ao_pair_second, pair_count,
          unrestricted ? total_density : density,
          unrestricted ? total_weighted_density : weighted_density, active, forces);
    } else {
      launch_one_electron_force_scalar_kernel(
          blocks_for(one_electron_force_elements), threads, 0, resources.stream_, device_batch,
          ao_pair_first, ao_pair_second, pair_count, unrestricted ? total_density : density,
          unrestricted ? total_weighted_density : weighted_density, active, forces);
    }
  }
  const std::uint64_t explicit_generated_force_shell_class_mask =
      generated::enabled_shell_class_mask() & host_present_shell_class_mask;
  // Fock-only AOT entries (currently ssss/psss) are deliberately not added
  // to the force queue.  The force dispatcher is a separate registry and
  // returns ``cudaErrorNotSupported`` for classes without a validated force
  // consumer.  Keep these classes on the exact handwritten low-order page
  // kernel below until an independently validated generated force entry is
  // promoted.
  const bool bounded_resident_psss_force_enabled =
      bounded_direct_streaming && plan.resident_psss_task_count != 0U &&
      plan.resident_psss_bra_primitive_pairs != 0U &&
      plan.resident_psss_bra_primitive_pairs <= kResidentPsssMaximumBraPrimitivePairs;
  const std::uint64_t bounded_native_paged_force_shell_class_mask =
      (bounded_direct_streaming
           ? host_present_shell_class_mask & kBoundedNativePagedForceShellClassMask
           : 0U) &
      ~(bounded_resident_psss_force_enabled ? (std::uint64_t{1} << kPsssShellClass) : 0U);
  const std::uint64_t selected_force_shell_class_mask =
      (explicit_generated_force_shell_class_mask |
       (bounded_direct_streaming
            ? generated::enabled_fock_shell_class_mask() & kStreamingFockShellClassMask &
                  ~kBoundedNativePagedForceShellClassMask
            : 0U)) &
      host_present_shell_class_mask;
  const std::uint64_t native_streaming_force_shell_class_mask =
      bounded_direct_streaming ? selected_force_shell_class_mask & kDdddShellClassMask &
                                     ~explicit_generated_force_shell_class_mask
                               : 0U;
  const std::uint64_t generated_shell_class_mask =
      selected_force_shell_class_mask & ~native_streaming_force_shell_class_mask;
  const std::uint64_t generated_queued_force_shell_class_mask = generated_shell_class_mask;
  // Whole-task and subgroup-task workers use page-local primitive signatures
  // before advancing independent quartets in warp lockstep. Keep those exact
  // classes out of the unsorted first/retry arenas and route them through the
  // same bounded page stream used by their generated consumers.
  const std::uint64_t bounded_paged_force_shell_class_mask =
      bounded_direct_streaming
          ? generated_queued_force_shell_class_mask & kBoundedForceSignatureShellClassMask
          : 0U;
  const std::uint64_t bounded_first_wave_force_shell_class_mask =
      generated_queued_force_shell_class_mask & ~bounded_paged_force_shell_class_mask;
  // Count diagnostics intentionally materialize every class through the
  // pre-paging queue. Production keeps only classes outside the page
  // mask there; the remaining classes use disjoint exact pages.
  const std::uint64_t bounded_force_legacy_queue_shell_class_mask =
      bounded_direct_count_diagnostic ? generated_queued_force_shell_class_mask
                                      : bounded_first_wave_force_shell_class_mask;
  const std::uint64_t covered_force_shell_class_mask = generated_shell_class_mask |
                                                       native_streaming_force_shell_class_mask |
                                                       bounded_native_paged_force_shell_class_mask;
  const std::uint64_t uncovered_force_shell_class_mask =
      host_present_shell_class_mask & ~covered_force_shell_class_mask;
  std::size_t bounded_force_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_force_kernels =
      generated::selected_shell_kernels(bounded_force_kernel_count);
  const auto launch_bounded_generated_force = [&](bool is_unrestricted,
                                                  DirectScreeningPurpose purpose,
                                                  const double* quartet_density) -> cudaError_t {
    // The count diagnostic is also used to compare the pre-paging generated
    // queue against the exact page consumer.  In that mode every selected
    // generated class must enter the legacy queue, including classes that
    // production would route through the signature-paged stream; otherwise
    // the diagnostic silently omits the very class under investigation.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow, bounded_force_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess || bounded_force_kernel_count == 0) return error;
    if (bounded_force_legacy_queue_shell_class_mask == 0U) return cudaSuccess;
    error =
        cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(unrestricted_value, purpose_value, task_offsets,     \
                                              selected_classes, selected_any)                      \
  launch_compact_bounded_generated_tasks_kernel(                                                   \
      unrestricted_value, purpose_value, plan.persistent_quartet_worker_blocks,                    \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,      \
      shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,               \
      bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, active,        \
      nullptr, bounded_force_legacy_queue_shell_class_mask, 0U, selected_classes, selected_any,    \
      bounded_direct_cursor, bounded_direct_generated_tasks, bounded_direct_generated_task_counts, \
      task_offsets, bounded_direct_generated_overflow)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(true, DirectScreeningPurpose::Force,
                                              bounded_direct_generated_task_offsets, nullptr,
                                              nullptr);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(true, DirectScreeningPurpose::Fock,
                                              bounded_direct_generated_task_offsets, nullptr,
                                              nullptr);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(false, DirectScreeningPurpose::Force,
                                            bounded_direct_generated_task_offsets, nullptr,
                                            nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(false, DirectScreeningPurpose::Fock,
                                            bounded_direct_generated_task_offsets, nullptr,
                                            nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_prepare_bounded_generated_retry_kernel(
        1, 1, 0, resources.stream_,
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask, bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any, bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) {
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_counts{};
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_overflow{};
      error = cudaMemcpyAsync(host_counts.data(), bounded_direct_generated_task_counts,
                              host_counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                              resources.stream_);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(host_overflow.data(), bounded_direct_generated_overflow,
                                host_overflow.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, resources.stream_);
      }
      if (error == cudaSuccess) {
        error = cudaStreamSynchronize(resources.stream_);
      }
      if (error != cudaSuccess) return error;
      std::fprintf(stderr, "bounded-direct-count purpose=%s capacity=%zu\n",
                   purpose == DirectScreeningPurpose::Force ? "force" : "fock",
                   plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count;
           ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_force_kernels[kernel_index];
        if ((bounded_force_legacy_queue_shell_class_mask &
             (std::uint64_t{1} << kernel.shell_class)) == 0U) {
          continue;
        }
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(stderr, "  %-4s count=%u capacity=%u overflow=%u\n", kernel.name,
                     host_counts[kernel.shell_class], class_capacity,
                     host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
      return cudaSuccess;
    }
    const auto consume_generated_wave = [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count;
           ++kernel_index) {
        const unsigned shell_class = bounded_force_kernels[kernel_index].shell_class;
        if ((bounded_force_legacy_queue_shell_class_mask & (std::uint64_t{1} << shell_class)) ==
            0U) {
          continue;
        }
        if (shell_class_profiling) {
          launch_profile_bounded_generated_tasks_kernel(
              plan.persistent_quartet_worker_blocks, threads, 0, resources.stream_, device_batch,
              bounded_direct_generated_tasks, task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class, shell_class_profile);
          cudaError_t profile_error = cudaPeekAtLastError();
          if (profile_error != cudaSuccess) return profile_error;
        }
        cudaError_t launch_error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, forces,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(bounded_direct_generated_task_counts, 0,
                            detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                            resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(bounded_direct_generated_overflow, 0,
                              detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                              resources.stream_);
    }
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Force, bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Fock, bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Force, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Fock, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_normalize_bounded_generated_task_counts_kernel(
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0, resources.stream_,
        bounded_direct_generated_retry_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    return error == cudaSuccess
               ? consume_generated_wave(bounded_direct_generated_retry_task_offsets)
               : error;
  };
  const auto launch_bounded_overflow_force = [&](bool is_unrestricted,
                                                 DirectScreeningPurpose purpose,
                                                 const double* quartet_density) -> cudaError_t {
    // This routine is the normal force queue, despite the historical
    // ``overflow`` name.  Every generated force class is enumerated exactly
    // once in disjoint candidate pages; the optional signature pass only
    // changes task order within a page for lockstep consumers.
    for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count; ++kernel_index) {
      const unsigned shell_class = bounded_force_kernels[kernel_index].shell_class;
      if ((generated_queued_force_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      if ((bounded_force_legacy_queue_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        // The pre-paging queue already consumed this class; keep the paged
        // stream disjoint so a force quartet is never evaluated twice.
        continue;
      }
      unsigned high_pair_class = 0U;
      while ((high_pair_class + 1U) * (high_pair_class + 2U) / 2U <= shell_class) {
        ++high_pair_class;
      }
      const unsigned low_pair_class = shell_class - high_pair_class * (high_pair_class + 1U) / 2U;
      const std::uint32_t page_capacity =
          static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
      if (page_capacity == 0U) return cudaErrorInvalidValue;
      const bool signature_paged =
          (bounded_paged_force_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_counts + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_retry_task_offsets + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess && signature_paged) {
          error = cudaMemsetAsync(bounded_force_signature_counts, 0,
                                  kBoundedForceSignatureBucketCount * sizeof(std::uint32_t),
                                  resources.stream_);
        }
        if (error != cudaSuccess) return error;
        const auto compact_page = [&](std::uint32_t* signature_counts,
                                      const std::uint32_t* signature_offsets,
                                      bool force_execution) -> cudaError_t {
#define VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(unrestricted_value, purpose_value)                    \
  launch_compact_bounded_exact_class_force_wave_kernel(                                           \
      unrestricted_value, purpose_value, plan.persistent_quartet_worker_blocks,                   \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, bounded_stream_topology,         \
      shell_class, high_pair_class, low_pair_class, options.screening_tolerance, page_begin,      \
      page_capacity, page_range.bra_begin, page_range.bra_end, high_pair_class == low_pair_class, \
      bounded_direct_generated_tasks, bounded_direct_generated_task_counts + shell_class,         \
      bounded_direct_generated_task_heads + shell_class, bounded_direct_generated_overflow,       \
      force_execution, signature_counts, signature_offsets)
          if (is_unrestricted) {
            if (purpose == DirectScreeningPurpose::Force) {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(true, DirectScreeningPurpose::Force);
            } else {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(true, DirectScreeningPurpose::Fock);
            }
          } else if (purpose == DirectScreeningPurpose::Force) {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(false, DirectScreeningPurpose::Force);
          } else {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(false, DirectScreeningPurpose::Fock);
          }
#undef VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE
          return cudaPeekAtLastError();
        };
        if (signature_paged) {
          // Count and scatter the same exact candidate page without host
          // readback. The second scan trades a small compaction cost for
          // primitive-uniform batches across all lockstep force workers.
          error = compact_page(bounded_force_signature_counts, nullptr, true);
          if (error == cudaSuccess) {
            launch_scan_bounded_force_signature_counts_kernel(
                kBoundedForceSignatureScanBlockCount, kBoundedForceSignatureScanThreads, 0,
                resources.stream_, bounded_force_signature_counts, bounded_force_signature_offsets,
                bounded_force_signature_block_offsets);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            launch_prefix_bounded_force_signature_blocks_kernel(
                1, kBoundedForceSignatureScanThreads, 0, resources.stream_,
                bounded_force_signature_offsets, bounded_force_signature_block_offsets,
                bounded_direct_generated_task_counts + shell_class);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                    sizeof(std::uint32_t), resources.stream_);
          }
          if (error == cudaSuccess) {
            error =
                compact_page(bounded_force_signature_counts, bounded_force_signature_offsets, true);
          }
        } else {
          error = compact_page(nullptr, nullptr, false);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error != cudaSuccess) return error;
        if (shell_class_profiling) {
          launch_profile_bounded_generated_tasks_kernel(
              plan.persistent_quartet_worker_blocks, threads, 0, resources.stream_, device_batch,
              bounded_direct_generated_tasks,
              bounded_direct_generated_retry_task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class, shell_class_profile);
          error = cudaPeekAtLastError();
          if (error != cudaSuccess) return error;
        }
        error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks,
            bounded_direct_generated_retry_task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, forces,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_native_force = [&](bool is_unrestricted, DirectScreeningPurpose purpose,
                                               const double* quartet_density) -> cudaError_t {
    if ((native_streaming_force_shell_class_mask & kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_heads + kDdddShellClass, 0,
                                        sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(unrestricted_value, purpose_value)                        \
  launch_bounded_direct_dddd_streaming_kernel(                                                    \
      unrestricted_value, purpose_value, true, plan.persistent_quartet_worker_blocks,             \
      detail::kDirectQuartetThreads, 0, resources.stream_, device_batch, bounded_stream_topology, \
      options.screening_tolerance, schwarz_bounds, quartet_density, active, forces,               \
      bounded_direct_generated_task_heads + kDdddShellClass,                                      \
      shell_class_profiling ? shell_class_profile : nullptr, nullptr)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(true, DirectScreeningPurpose::Force);
      } else {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(true, DirectScreeningPurpose::Fock);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(false, DirectScreeningPurpose::Force);
    } else {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(false, DirectScreeningPurpose::Fock);
    }
#undef VIBEQC_LAUNCH_NATIVE_DDDD_FORCE
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_generic_force = [&](bool is_unrestricted,
                                                DirectScreeningPurpose purpose,
                                                const double* quartet_density) -> cudaError_t {
    if (uncovered_force_shell_class_mask == 0U) return cudaSuccess;
    // The generated/native routes above cover the common classes.  Keep the
    // remaining exact classes correct with the bounded runtime dispatcher,
    // rather than rejecting an otherwise valid large-AO force calculation.
    // Zero the generated-overflow state so an earlier diagnostic page cannot
    // make a covered class look like an uncovered fallback candidate.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_overflow, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted && purpose == DirectScreeningPurpose::Force) {
      launch_bounded_direct_shell_quartet_kernel(
          true, DirectScreeningPurpose::Force, plan.persistent_quartet_worker_blocks,
          kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,
          shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,
          bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
          covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
          quartet_density, active, forces, bounded_direct_cursor,
          shell_class_profiling ? shell_class_profile : nullptr);
    } else if (!is_unrestricted && purpose == DirectScreeningPurpose::Force) {
      launch_bounded_direct_shell_quartet_kernel(
          false, DirectScreeningPurpose::Force, plan.persistent_quartet_worker_blocks,
          kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,
          shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,
          bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
          covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
          quartet_density, active, forces, bounded_direct_cursor,
          shell_class_profiling ? shell_class_profile : nullptr);
    } else if (is_unrestricted) {
      launch_bounded_direct_shell_quartet_kernel(
          true, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,
          kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,
          shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,
          bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
          covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
          quartet_density, active, forces, bounded_direct_cursor,
          shell_class_profiling ? shell_class_profile : nullptr);
    } else {
      launch_bounded_direct_shell_quartet_kernel(
          false, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,
          kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,
          shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,
          bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
          covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
          quartet_density, active, forces, bounded_direct_cursor,
          shell_class_profiling ? shell_class_profile : nullptr);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_paged_native_force = [&](bool is_unrestricted,
                                                     DirectScreeningPurpose purpose,
                                                     const double* quartet_density) -> cudaError_t {
    const std::uint64_t native_mask = bounded_native_paged_force_shell_class_mask;
    if (native_mask == 0U) return cudaSuccess;
    const std::uint32_t page_capacity =
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
    if (page_capacity == 0U) return cudaErrorInvalidValue;
    for (unsigned shell_class = kSsssShellClass; shell_class <= kPsssShellClass; ++shell_class) {
      if ((native_mask & (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      const unsigned high_pair_class = shell_class == kSsssShellClass ? 0U : 1U;
      const unsigned low_pair_class = shell_class == kSsssShellClass ? 0U : 0U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(unrestricted_value, purpose_value)                      \
  launch_contract_bounded_exact_low_order_force_page_kernel(                                      \
      unrestricted_value, purpose_value, plan.persistent_quartet_worker_blocks,                   \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, bounded_stream_topology,         \
      shell_class, high_pair_class, low_pair_class, options.screening_tolerance, page_begin,      \
      page_capacity, page_range.bra_begin, page_range.bra_end, high_pair_class == low_pair_class, \
      schwarz_bounds, quartet_density, forces, bounded_direct_generated_task_heads + shell_class)
        if (purpose == DirectScreeningPurpose::Force) {
          if (is_unrestricted) {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(true, DirectScreeningPurpose::Force);
          } else {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(false, DirectScreeningPurpose::Force);
          }
        } else {
          if (is_unrestricted) {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(true, DirectScreeningPurpose::Fock);
          } else {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(false, DirectScreeningPurpose::Fock);
          }
        }
#undef VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_resident_psss_force =
      [&](bool is_unrestricted, DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    if (!bounded_resident_psss_force_enabled) return cudaSuccess;
    const bool use_force_screening = purpose == DirectScreeningPurpose::Force;
    if (is_unrestricted) {
      launch_two_electron_force_psss_resident_bra_kernel(
          true, static_cast<unsigned>(plan.resident_psss_task_count), kResidentPsssThreads,
          plan.resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData), resources.stream_,
          device_batch, psss_resident_tasks, psss_resident_ket_pairs, plan.resident_psss_task_count,
          options.screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
          use_force_screening, schwarz_bounds, quartet_density, active, forces,
          generated_shell_class_mask);
    } else {
      launch_two_electron_force_psss_resident_bra_kernel(
          false, static_cast<unsigned>(plan.resident_psss_task_count), kResidentPsssThreads,
          plan.resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData), resources.stream_,
          device_batch, psss_resident_tasks, psss_resident_ket_pairs, plan.resident_psss_task_count,
          options.screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
          use_force_screening, schwarz_bounds, quartet_density, active, forces,
          generated_shell_class_mask);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_force = [&](bool is_unrestricted, DirectScreeningPurpose purpose,
                                        const double* quartet_density) -> cudaError_t {
    // Bounded production execution is class-specific by construction. The
    // generated/native routes handle the common classes; any registry gap is
    // sent to the exact bounded runtime dispatcher below instead of rejecting
    // a valid topology or silently scanning an unbounded AO-space fallback.
    // Generated classes selected by the page mask use exact, disjoint
    // pages. The pre-paging queue is retained only for diagnostics and
    // classes that do not use the paged route.
    cudaError_t error = cudaSuccess;
    if (bounded_direct_count_diagnostic) {
      return launch_bounded_generated_force(is_unrestricted, purpose, quartet_density);
    }
    if ((bounded_force_legacy_queue_shell_class_mask & generated_queued_force_shell_class_mask) !=
        0U) {
      // Consume non-paged generated classes first, then keep their exact
      // class mask out of the page stream to avoid duplicate quartets.
      error = launch_bounded_generated_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_overflow_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_resident_psss_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_paged_native_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_native_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      return launch_bounded_generic_force(is_unrestricted, purpose, quartet_density);
    }
    error = launch_bounded_overflow_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_resident_psss_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_paged_native_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_native_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_generic_force(is_unrestricted, purpose, quartet_density);
    return error;
  };
  if (options.compute_forces && quartet_direct &&
      plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder] != 0) {
    cuda_error =
        cudaMemsetAsync(generic_order5_tile_count, 0, sizeof(std::uint32_t), resources.stream_);
    if (cuda_error == cudaSuccess) {
      launch_compact_generic_order5_tiles_kernel(
          blocks_for(plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder]), threads, 0,
          resources.stream_, device_batch,
          active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
          active_shell_quartet_tiles +
              plan.shell_quartet_tile_offsets[kGenericOrderFiveAngularOrder],
          generated_shell_class_mask, nullptr, generic_order5_tile_count, generic_order5_tiles);
      cuda_error = cudaPeekAtLastError();
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (!options.compute_forces) {
    // Energy-only execution intentionally omits every analytic-force kernel.
  } else if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // Nuclear and one-electron forces above remain in the timing so this
    // diagnostic isolates only the bounded two-electron force tail.
  } else if (unrestricted && persistent_eri) {
    launch_two_electron_uhf_force_kernel(blocks_for(persistent_force_elements), threads, 0,
                                         resources.stream_, device_batch, density, active, forces);
  } else if (unrestricted && quartet_direct) {
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose = force_density_product_screening
                                                               ? DirectScreeningPurpose::Force
                                                               : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(true, bounded_force_purpose,
                                        transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
          generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts, generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts, generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts, generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets, generated_ppps_resident_signatures,
          total_shell_pairs, resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing, psps_signature_bucketing, ppss_signature_bucketing,
          resident_ppps_block_threads, plan.persistent_quartet_worker_blocks, true,
          generated_shell_class_mask, options.screening_tolerance, schwarz_bounds,
          transformed_direct ? direct_density : density, forces);
      if (cuda_error == cudaSuccess) {
        dispatch_angular_force_quartets(
            true, resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch, active_shell_quartet_tile_counts,
            active_shell_quartet_tiles, generic_order5_tile_count, generic_order5_tiles,
            persistent_force_task_heads, plan.persistent_quartet_worker_blocks, psss_resident_tasks,
            psss_resident_ket_pairs, plan.resident_psss_task_count,
            plan.resident_psss_bra_primitive_pairs, options.screening_tolerance, shell_pair_bounds,
            shell_pair_density_bounds, force_density_product_screening, schwarz_bounds,
            transformed_direct ? direct_density : density, active, forces,
            generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else if (unrestricted) {
    launch_two_electron_uhf_force_direct_kernel(
        blocks_for(direct_force_elements), threads, 0, resources.stream_, device_batch,
        options.screening_tolerance, ao_pair_first, ao_pair_second, pair_count, schwarz_bounds,
        density, active, forces);
  } else if (persistent_eri) {
    launch_two_electron_force_kernel(blocks_for(persistent_force_elements), threads, 0,
                                     resources.stream_, device_batch, density, active, forces);
  } else if (quartet_direct) {
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose = force_density_product_screening
                                                               ? DirectScreeningPurpose::Force
                                                               : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(false, bounded_force_purpose,
                                        transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
          generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts, generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts, generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts, generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets, generated_ppps_resident_signatures,
          total_shell_pairs, resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing, psps_signature_bucketing, ppss_signature_bucketing,
          resident_ppps_block_threads, plan.persistent_quartet_worker_blocks, false,
          generated_shell_class_mask, options.screening_tolerance, schwarz_bounds,
          transformed_direct ? direct_density : density, forces);
      if (cuda_error == cudaSuccess) {
        dispatch_angular_force_quartets(
            false, resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch, active_shell_quartet_tile_counts,
            active_shell_quartet_tiles, generic_order5_tile_count, generic_order5_tiles,
            persistent_force_task_heads, plan.persistent_quartet_worker_blocks, psss_resident_tasks,
            psss_resident_ket_pairs, plan.resident_psss_task_count,
            plan.resident_psss_bra_primitive_pairs, options.screening_tolerance, shell_pair_bounds,
            shell_pair_density_bounds, force_density_product_screening, schwarz_bounds,
            transformed_direct ? direct_density : density, active, forces,
            generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else {
    launch_two_electron_force_direct_kernel(
        blocks_for(direct_force_elements), threads, 0, resources.stream_, device_batch,
        options.screening_tolerance, ao_pair_first, ao_pair_second, pair_count, schwarz_bounds,
        density, active, forces);
  }

  if (reuse_converged_fock) {
    // The requested outputs above consumed each system's selected consistent
    // snapshot. Advance only reused systems to the already accepted P_{n+1}
    // for their returned warm state; rebuilt systems already contain it.
    launch_copy_selected_matrices_kernel(
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
        static_cast<std::int32_t>(nbf), final_fock_reuse_mask, next_density, density);
  }

  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }

  std::vector<double> host_energy(batch_size);
  std::vector<double> host_energy_change(batch_size);
  std::vector<double> host_density_rms(batch_size);
  std::vector<double> host_density(spin_matrix_elements);
  std::vector<double> host_forces(options.compute_forces ? total_atoms * 3 : 0U);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint8_t> host_failed(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  std::uint32_t host_inactive_eigensolver_profile_count = 0U;
  std::vector<DeviceInactiveEigensolverProfileEntry> host_inactive_eigensolver_profile(
      inactive_eigensolver_profiling ? options.max_iterations : 0U);
  CudaRhfShellClassProfile host_shell_class_profile{};
  const bool collect_ppps_queue_profile =
      shell_class_profiling && quartet_direct && resident_ppps_bra &&
      resident_ppps_ket_task_capacity != 0U &&
      (generated_shell_class_mask & (std::uint64_t{1} << kPppsShellClass)) != 0U;
  std::vector<std::uint32_t> host_ppps_descriptor_counts(
      collect_ppps_queue_profile ? total_shell_pairs : 0U);
  std::vector<std::uint32_t> host_ppps_signatures(
      collect_ppps_queue_profile ? resident_ppps_ket_task_capacity : 0U);
  const struct Download {
    void* host;
    const void* device;
    std::size_t bytes;
  } downloads[] = {
      {host_energy.data(), energy, batch_size * sizeof(double)},
      {host_energy_change.data(), energy_change, batch_size * sizeof(double)},
      {host_density_rms.data(), density_rms, batch_size * sizeof(double)},
      {host_density.data(), density, spin_matrix_elements * sizeof(double)},
      {host_converged.data(), converged, batch_size * sizeof(std::uint8_t)},
      {host_failed.data(), failed, batch_size * sizeof(std::uint8_t)},
      {host_iterations.data(), iterations, batch_size * sizeof(std::uint32_t)},
  };
  for (const Download& download : downloads) {
    cuda_error = cudaMemcpyAsync(download.host, download.device, download.bytes,
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (options.compute_forces) {
    cuda_error = cudaMemcpyAsync(host_forces.data(), forces, total_atoms * 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemcpyAsync(&host_inactive_eigensolver_profile_count,
                                 inactive_eigensolver_profile_count, sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess && !host_inactive_eigensolver_profile.empty()) {
      cuda_error = cudaMemcpyAsync(
          host_inactive_eigensolver_profile.data(), inactive_eigensolver_profile,
          host_inactive_eigensolver_profile.size() * sizeof(DeviceInactiveEigensolverProfileEntry),
          cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (shell_class_profiling && quartet_direct) {
    cuda_error =
        cudaMemcpyAsync(host_shell_class_profile.data(), shell_class_profile,
                        host_shell_class_profile.size() * sizeof(CudaRhfShellClassProfileEntry),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (collect_ppps_queue_profile) {
    cuda_error =
        cudaMemcpyAsync(host_ppps_descriptor_counts.data(), generated_ppps_resident_bra_counts,
                        host_ppps_descriptor_counts.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_ppps_signatures.data(), generated_ppps_resident_signatures,
                                   host_ppps_signatures.size() * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  cuda_error = cudaStreamSynchronize(resources.stream_);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (shell_class_profiling && quartet_direct) {
    plan.last_shell_class_profile = host_shell_class_profile;
  }
  if (inactive_eigensolver_profiling) {
    if (host_inactive_eigensolver_profile_count > host_inactive_eigensolver_profile.size()) {
      fill_global_failure(outputs, VIBEQC_STATUS_INTERNAL_ERROR);
      return outputs;
    }
    CudaInactiveEigensolverProfile profile;
    profile.reserve(host_inactive_eigensolver_profile_count);
    for (std::uint32_t index = 0; index < host_inactive_eigensolver_profile_count; ++index) {
      const DeviceInactiveEigensolverProfileEntry& input = host_inactive_eigensolver_profile[index];
      CudaInactiveEigensolverProfileEntry output;
      output.iteration = input.iteration;
      output.family = static_cast<CudaEigensolverFamily>(input.family);
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.active_physical_count = input.active_physical_count;
      output.active_solver_count = input.active_solver_count;
      output.solver_elapsed_nanoseconds = input.solver_elapsed_nanoseconds;
      output.inactive_input_nonfinite_count = input.inactive_input_nonfinite_count;
      output.inactive_submission_nonfinite_count = input.inactive_submission_nonfinite_count;
      output.inactive_info_nonzero_count = input.inactive_info_nonzero_count;
      output.inactive_touch_flags = input.inactive_touch_flags;
      output.provider_invoked = input.provider_invoked != 0U;
      profile.push_back(output);
    }
    plan.last_inactive_eigensolver_profile = std::move(profile);
  }
  if (collect_ppps_queue_profile) {
    const unsigned multiprocessor_count = std::max(
        1U, plan.persistent_quartet_worker_blocks / kPersistentQuartetWarpsPerMultiprocessor);
    CudaPppsQueueProfile ppps_profile = build_ppps_queue_profile(
        host, host_ppps_descriptor_counts, host_ppps_signatures, multiprocessor_count);
    if (ppps_profile.descriptor_slots != 0U) {
      plan.last_ppps_queue_profile = std::move(ppps_profile);
    }
  }
  const bool no_system_failed = std::none_of(host_failed.begin(), host_failed.end(),
                                             [](std::uint8_t value) { return value != 0; });
  if (no_system_failed) {
    plan.cached_positions = host.positions;
  } else if (geometry_changed) {
    // Never reuse an orthogonalizer from a calculation that reported a
    // numerical failure; retry the full geometry path on the next execution.
    plan.cached_positions.clear();
  }
  const bool all_systems_converged =
      no_system_failed && std::all_of(host_converged.begin(), host_converged.end(),
                                      [](std::uint8_t value) { return value != 0; });
  if (all_systems_converged) {
    plan.resident_warm_positions = host.positions;
    plan.resident_warm_density = host_density;
    plan.resident_previous_energy = host_energy;
  } else {
    // A failed or incomplete execution cannot provide an energy baseline for
    // the next warm density, even if the fleet retains another system's state.
    plan.resident_warm_positions.clear();
    plan.resident_warm_density.clear();
    plan.resident_previous_energy.clear();
  }

  // The mixed-precision Fock decision (already gated on quartet-direct) and its
  // forced final FP64 rebuild are fixed for the plan; report what actually ran.
  const int32_t requested_precision_mode = options.precision_mode.value_or(VIBEQC_PRECISION_FP64);
  const bool precision_route_enabled = plan.mixed_precision_fock;
  for (std::size_t system = 0; system < batch_size; ++system) {
    RhfBucketItem& output = outputs[system];
    ScfResult& result = output.scf;
    // The mixed operator belongs to this item alone: an item that stayed on the
    // exact operator reports it, with its own cutoff, budget and refinement
    // cost, regardless of what its batch neighbors resolved.
    const bool precision_item_mixed =
        precision_route_enabled && host_mixed_item_census[system] != 0U;
    result.energy = host_energy[system];
    // The refinement iterations are reported as part of the complete solve.
    result.iterations = precision_item_mixed
                            ? host_mixed_iterations[system] + host_iterations[system]
                            : host_iterations[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    result.converged = host_converged[system] != 0 && host_failed[system] == 0;
    result.initial_density_used = host.warm_mask[system] != 0;
    result.precision.requested_mode = requested_precision_mode;
    result.precision.effective_bits = precision_item_mixed ? 32U : 64U;
    result.precision.mixed_precision_fock_threshold =
        precision_item_mixed ? host_mixed_item_threshold[system] : 0.0;
    result.precision.strict_refinement_applied = precision_item_mixed;
    result.precision.mixed_precision_reserved_error =
        precision_item_mixed ? requested_precision_policy.item_budget_error : 0.0;
    result.precision.refinement_iterations = precision_item_mixed ? host_iterations[system] : 0U;
    const std::size_t density_stride = spin_count * matrix_size;
    result.density.assign(host_density.begin() + system * density_stride,
                          host_density.begin() + (system + 1) * density_stride);
    if (options.compute_forces) {
      const std::size_t atom_begin = static_cast<std::size_t>(host.atom_offsets[system]);
      const std::size_t atom_end = static_cast<std::size_t>(host.atom_offsets[system + 1]);
      result.forces.assign(host_forces.begin() + atom_begin * 3,
                           host_forces.begin() + atom_end * 3);
    }
    output.status = host_failed[system] != 0 ? VIBEQC_STATUS_NUMERICAL_FAILURE
                                             : (result.converged ? VIBEQC_STATUS_SUCCESS
                                                                 : VIBEQC_STATUS_SCF_NOT_CONVERGED);
  }
  return outputs;
}

}  // namespace

bool small_hf_cuda_resource_layout(std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                                   std::size_t shells, std::size_t primitives,
                                   std::size_t diis_history, std::size_t spins,
                                   std::size_t& arena_bytes, std::size_t& plan_object_bytes) {
  // This contract intentionally covers the provider with no cuBLAS/cuSOLVER
  // workspace or exact-quartet descriptor table. Other routes need their own
  // compact provider-workspace query before they can advertise a global bound.
  if (nbf == 0 || nbf > kPersistentEriAoLimit || nbf > kSmallEigensolverLimit ||
      nbf >= kCublasMatrixProductAoThreshold || direct_nbf < nbf || direct_nbf > 2 * nbf ||
      atoms == 0 || shells == 0 || shells > nbf || primitives == 0 || diis_history > 64 ||
      (spins != 1 && spins != 2))
    return false;
  const std::size_t pairs = shells * (shells + 1) / 2;
  const std::size_t blocks =
      detail::bounded_direct_queue_refill_count(pairs, detail::kBoundedDirectShellPairBlockSize);
  ArenaLayout layout{};
  if (!make_layout(1, nbf, direct_nbf, atoms, shells, pairs, blocks, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   primitives, std::max<std::size_t>(1, diis_history), 0, spins, true, false, false,
                   false, false, false, false, layout))
    return false;
  arena_bytes = layout.bytes;
  plan_object_bytes = sizeof(CudaRhfBucketPlan);
  return true;
}

std::size_t hf_cuda_owned_device_bytes(const CudaRhfBucketPlan* plan) noexcept {
  if (plan == nullptr) return 0;
  const auto& resources = plan->resources;
  auto bytes = resources.arena_ == nullptr ? 0 : plan->layout.bytes;
  if (resources.solver_workspace_ != nullptr)
    bytes = runtime::add_capacity(bytes, resources.solver_workspace_bytes_);
  if (resources.direct_tile_validation_ != nullptr)
    bytes = runtime::add_capacity(bytes, sizeof(DirectTileValidationRecord));
  return bytes;
}

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(const std::vector<core::System>& systems) {
  std::vector<const std::vector<double>*> initial_densities(systems.size(), nullptr);
  HostBatch host;
  if (!pack_host_batch(systems, initial_densities, host)) {
    throw std::invalid_argument("systems cannot be represented by one CUDA RHF bucket");
  }

  std::size_t expanded_primitive_references = 0;
  for (const core::System& system : systems) {
    for (const core::Shell& shell : system.shells) {
      std::size_t shell_references = 0;
      if (!checked_multiply(molecule::cartesian_count(shell.angular_momentum),
                            shell.primitives.size(), shell_references) ||
          !checked_add(expanded_primitive_references, shell_references,
                       expanded_primitive_references)) {
        throw std::overflow_error("expanded CUDA primitive reference count overflowed");
      }
    }
  }

  const std::size_t device_basis_bytes =
      host.system_shell_offsets.size() * sizeof(std::int64_t) +
      host.shell_atoms.size() * sizeof(std::int32_t) +
      host.shell_angular.size() * sizeof(std::uint8_t) +
      host.shell_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_direct_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_primitive_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_pair_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_quartet_offsets.size() * sizeof(std::int64_t) +
      host.shell_pair_systems.size() * sizeof(std::int32_t) +
      host.shell_pair_first.size() * sizeof(std::int32_t) +
      host.shell_pair_second.size() * sizeof(std::int32_t) +
      host.ao_shells.size() * sizeof(std::int32_t) +
      host.ao_term_counts.size() * sizeof(std::uint8_t) +
      host.ao_term_angular.size() * sizeof(std::uint8_t) +
      host.ao_term_coefficients.size() * sizeof(double) +
      host.direct_ao_shells.size() * sizeof(std::int32_t) +
      host.direct_ao_angular.size() * sizeof(std::uint8_t) +
      host.direct_ao_coefficients.size() * sizeof(double) +
      host.ao_to_direct_transform.size() * sizeof(double) +
      host.primitive_exponents.size() * sizeof(double) +
      host.primitive_coefficients.size() * sizeof(double);
  return {
      systems.size(),
      host.shell_atoms.size(),
      host.shell_pair_first.size(),
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back()),
      host.ao_shells.size(),
      host.primitive_exponents.size(),
      expanded_primitive_references,
      device_basis_bytes,
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(host.system_shell_quartet_offsets.back())),
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(host.system_shell_quartet_offsets.back()))
          ? detail::kBoundedDirectQueueCapacity
          : 0,
  };
}

namespace {

std::vector<RhfBucketItem> run_hf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems,
    const ScfOptions& requested_options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool unrestricted, bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  if (requested_options.hooks || requested_options.strict_initial_density) {
    // Host callbacks are an explicit CPU capability, never a device fallback.
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_IMPLEMENTED);
    return outputs;
  }

  // Resolve legacy internal callers once per prepared execution, before any
  // device setup. Fock kernels and final exact force assembly share this guard.
  ScfOptions execution_options = requested_options;
  try {
    const FockSpin spin = unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted;
    if (!execution_options.resolved_fock_build.has_value()) {
      execution_options.resolved_fock_build = resolve_fock_build(
          make_hf_fock_spec(spin), FockBackend::Cuda, execution_options.screening_tolerance);
    }
    require_exact_direct_strategy(*execution_options.resolved_fock_build, spin, FockBackend::Cuda);
    if (execution_options.resolved_fock_build->screening_tolerance !=
        execution_options.screening_tolerance) {
      throw std::invalid_argument("CUDA screening differs from its resolved Fock strategy");
    }
  } catch (const std::invalid_argument&) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const ScfOptions& options = execution_options;

  if (plan == nullptr) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  HostBatch candidate;
  if (!pack_host_batch(systems, initial_densities, candidate, unrestricted,
                       options.export_physical_reference)) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::optional<double> mixed_precision_fock_threshold =
      *plan != nullptr && (*plan)->quartet_direct
          ? resolve_mixed_precision_fock_policy(
                options.precision_mode, options.energy_tolerance, options.screening_tolerance,
                static_cast<double>((*plan)->mixed_precision_eligible_tile_count))
                .threshold
          : std::nullopt;
  const bool mixed_precision_fock = mixed_precision_fock_threshold.has_value();
  const bool reuse_converged_fock = reuse_converged_fock_requested();
  const bool graph_native_eigensolver_override = graph_native_eigensolver_override_requested();
  if (*plan != nullptr && (*plan)->initialized &&
      ((*plan)->resources.device_id_ != device_id || !same_topology((*plan)->topology, candidate) ||
       !same_options((*plan)->options, options) || (*plan)->unrestricted != unrestricted ||
       (*plan)->shell_class_profiling != shell_class_profiling ||
       (*plan)->inactive_eigensolver_profiling != inactive_eigensolver_profiling ||
       (*plan)->bounded_fock_class_timing != bounded_fock_class_timing_requested() ||
       (*plan)->bounded_streaming_override != bounded_direct_streaming_override_requested() ||
       (*plan)->fock_only_diagnostic != bounded_direct_fock_only_diagnostic_requested() ||
       (*plan)->graph_native_eigensolver_override != graph_native_eigensolver_override ||
       (*plan)->reuse_converged_fock != reuse_converged_fock ||
       (*plan)->one_electron_value_mapping != cuda_policy::one_electron_value_mapping_requested() ||
       (*plan)->mixed_precision_fock != mixed_precision_fock ||
       (*plan)->mixed_precision_fock_threshold != mixed_precision_fock_threshold.value_or(0.0))) {
    delete *plan;
    *plan = nullptr;
  }
  if (*plan == nullptr) {
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      std::vector<RhfBucketItem> outputs(systems.size());
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
  }
  std::vector<RhfBucketItem> outputs =
      execute_hf_cuda_bucket(**plan, candidate, options, device_id, unrestricted,
                             shell_class_profiling, inactive_eigensolver_profiling);
  const bool retry_without_cublas = !(*plan)->initialized && (*plan)->retry_without_cublas;
  if (!(*plan)->initialized) {
    delete *plan;
    *plan = nullptr;
  }
  if (retry_without_cublas) {
    // Provider setup or graph capture can reject a cuBLAS implementation on a
    // particular CUDA release. Rebuild once with the numerically identical
    // native kernel so public CUDA execution remains available.
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    (*plan)->cublas_enabled = false;
    outputs = execute_hf_cuda_bucket(**plan, candidate, options, device_id, unrestricted,
                                     shell_class_profiling, inactive_eigensolver_profiling);
    if (!(*plan)->initialized) {
      delete *plan;
      *plan = nullptr;
    }
  }
  return outputs;
}

}  // namespace

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(plan, systems, options, initial_densities, device_id, false,
                                   shell_class_profiling, inactive_eigensolver_profiling);
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(plan, systems, options, initial_densities, device_id, true,
                                   shell_class_profiling, inactive_eigensolver_profiling);
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept { delete plan; }

void set_rhf_cuda_bucket_warm_start_updates(CudaRhfBucketPlan* plan, bool enabled) noexcept {
  if (plan == nullptr || plan->warm_start_updates_enabled == enabled) return;
  if (!enabled) {
    // Freeze only on the policy transition. Repeating the setter while fixed
    // must not replace the original post-cold dm0/seed with a later replay's
    // advanced resident state.
    plan->frozen_warm_positions = plan->resident_warm_positions;
    plan->frozen_warm_density = plan->resident_warm_density;
    plan->frozen_previous_energy = plan->resident_previous_energy;
  } else {
    plan->frozen_warm_positions.clear();
    plan->frozen_warm_density.clear();
    plan->frozen_previous_energy.clear();
  }
  plan->warm_start_updates_enabled = enabled;
}

void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan* plan) noexcept {
  if (plan == nullptr) return;
  plan->resident_warm_positions.clear();
  plan->resident_warm_density.clear();
  plan->resident_previous_energy.clear();
  plan->frozen_warm_positions.clear();
  plan->frozen_warm_density.clear();
  plan->frozen_previous_energy.clear();
}

bool get_rhf_cuda_shell_class_profile(const CudaRhfBucketPlan* plan,
                                      CudaRhfShellClassProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_shell_class_profile.has_value()) {
    return false;
  }
  profile = *plan->last_shell_class_profile;
  return true;
}

bool get_rhf_cuda_ppps_queue_profile(const CudaRhfBucketPlan* plan,
                                     CudaPppsQueueProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_ppps_queue_profile.has_value()) {
    return false;
  }
  profile = *plan->last_ppps_queue_profile;
  return true;
}

bool get_rhf_cuda_eigensolver_diagnostic(const CudaRhfBucketPlan* plan,
                                         CudaEigensolverDiagnostic& diagnostic) noexcept {
  if (plan == nullptr || !plan->initialized) return false;
  diagnostic = plan->eigensolver_diagnostic;
  return true;
}

bool get_rhf_cuda_inactive_eigensolver_profile(const CudaRhfBucketPlan* plan,
                                               CudaInactiveEigensolverProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_inactive_eigensolver_profile.has_value()) {
    return false;
  }
  profile = *plan->last_inactive_eigensolver_profile;
  return true;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  try {
    auto outputs =
        run_rhf_cuda_bucket_cached(&plan, systems, options, initial_densities, device_id,
                                   shell_class_profiling, inactive_eigensolver_profiling);
    destroy_rhf_cuda_bucket_plan(plan);
    return outputs;
  } catch (...) {
    destroy_rhf_cuda_bucket_plan(plan);
    throw;
  }
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  std::vector<RhfBucketItem> outputs =
      run_uhf_cuda_bucket_cached(&plan, systems, options, initial_densities, device_id,
                                 shell_class_profiling, inactive_eigensolver_profiling);
  destroy_rhf_cuda_bucket_plan(plan);
  return outputs;
}

vibeqc_status contract_cuda_weighted_eri_primitives(
    int device_id, const CudaWeightedEriPrimitive* records, std::size_t record_count,
    std::size_t tile_count, std::size_t memory_budget_bytes, bool generated,
    std::vector<CudaWeightedEriResult>& output, CudaWeightedEriDiagnostic& diagnostic,
    std::string& detail) {
  // Discard previous output capacity so a small-budget call cannot retain an
  // old larger allocation while reporting only its new logical result size.
  std::vector<CudaWeightedEriResult>{}.swap(output);
  diagnostic = {};
  detail.clear();
  std::size_t output_bytes = 0, result_peak = 0, input_bytes = 0;
  if ((record_count != 0U && records == nullptr) ||
      tile_count > std::numeric_limits<std::uint32_t>::max() ||
      !checked_multiply(record_count, sizeof(CudaWeightedEriPrimitive), input_bytes) ||
      !checked_multiply(tile_count, sizeof(CudaWeightedEriResult), output_bytes) ||
      !checked_multiply(output_bytes, 2U, result_peak) || result_peak > memory_budget_bytes) {
    detail = "weighted ERI dimensions or numeric memory budget are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t index = 0; index < record_count; ++index) {
    const auto& record = records[index];
    if (record.kind > 1U || record.output_tile >= tile_count) {
      detail = "weighted ERI record kind or output tile is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (!(record.exponents[slot] > 0.0) || !std::isfinite(record.exponents[slot])) {
        detail = "weighted ERI primitive exponents must be finite and positive";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      unsigned total = 0;
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto angular = record.angular[slot][axis];
        if (angular > 3U || !std::isfinite(record.centers[slot][axis]) ||
            (record.kind == 1U && angular != static_cast<unsigned>(slot == 0U && axis == 0U))) {
          detail = "weighted ERI angular components or positions are invalid";
          return VIBEQC_STATUS_INVALID_ARGUMENT;
        }
        total += angular;
      }
      if (total > 3U) {
        detail = "weighted ERI primitive shells beyond f are unsupported";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    for (double weight : record.weights) {
      if (!std::isfinite(weight)) {
        detail = "external ERI weights must be finite";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    if (generated && record.kind == 1U)
      ++diagnostic.generated_records;
    else
      ++diagnostic.reference_records;
  }
  try {
    if (record_count == 0U) {
      output.resize(tile_count);
      diagnostic.host_peak_bytes = output_bytes;
      return VIBEQC_STATUS_SUCCESS;
    }
    const std::size_t capacity =
        std::min({record_count, std::size_t{65536},
                  (memory_budget_bytes - result_peak) / sizeof(CudaWeightedEriPrimitive)});
    if (capacity == 0U) {
      detail = "weighted ERI budget cannot hold one primitive and its outputs";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    diagnostic.primitive_capacity = capacity;
    diagnostic.host_peak_bytes = output_bytes;
    diagnostic.device_peak_bytes = output_bytes + capacity * sizeof(CudaWeightedEriPrimitive);
    auto error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
      detail = "weighted ERI CUDA device selection failed";
      return cuda_status(error);
    }
    struct Buffers {
      CudaWeightedEriPrimitive* records{};
      CudaWeightedEriResult* results{};
      cudaStream_t stream{};
      ~Buffers() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (records) (void)runtime::resource_cuda_free(records);
        if (results) (void)runtime::resource_cuda_free(results);
        if (stream) (void)cudaStreamDestroy(stream);
      }
    } buffers;
    error = cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking);
    if (error == cudaSuccess)
      error = runtime::resource_cuda_malloc(&buffers.records, capacity * sizeof(*buffers.records));
    if (error == cudaSuccess) error = runtime::resource_cuda_malloc(&buffers.results, output_bytes);
    if (error == cudaSuccess)
      error = cudaMemsetAsync(buffers.results, 0, output_bytes, buffers.stream);
    constexpr unsigned threads = 64U;
    for (std::size_t begin = 0; begin < record_count && error == cudaSuccess;) {
      const std::size_t count = std::min(capacity, record_count - begin);
      bool has_generated = false, has_reference = false;
      for (std::size_t i = begin; i < begin + count; ++i) {
        if (generated && records[i].kind == 1U)
          has_generated = true;
        else
          has_reference = true;
      }
      error = cudaMemcpyAsync(buffers.records, records + begin, count * sizeof(*records),
                              cudaMemcpyHostToDevice, buffers.stream);
      const unsigned blocks = static_cast<unsigned>((count + threads - 1U) / threads);
      if (error == cudaSuccess && has_reference) {
        launch_weighted_eri_reference_kernel(blocks, threads, 0, buffers.stream, buffers.records,
                                             count, generated, buffers.results);
        error = cudaGetLastError();
      }
      if (error == cudaSuccess && has_generated) {
        launch_weighted_eri_generated_psss_kernel(blocks, threads, 0, buffers.stream,
                                                  buffers.records, count, buffers.results);
        error = cudaGetLastError();
      }
      begin += count;
    }
    if (error == cudaSuccess) {
      output.resize(tile_count);
      error = cudaMemcpyAsync(output.data(), buffers.results, output_bytes, cudaMemcpyDeviceToHost,
                              buffers.stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(buffers.stream);
    if (error != cudaSuccess) {
      output.clear();
      detail = "CUDA external-weight ERI contraction failed";
      return cuda_status(error);
    }
    for (const auto& result : output) {
      bool finite = std::isfinite(result.value);
      for (const auto& center : result.center) {
        for (double value : center) finite = finite && std::isfinite(value);
      }
      if (!finite) {
        output.clear();
        detail = "weighted ERI contraction overflowed or produced nonfinite values";
        return VIBEQC_STATUS_NUMERICAL_FAILURE;
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    output.clear();
    detail = "weighted ERI host allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    output.clear();
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}

}  // namespace vibeqc::scf
