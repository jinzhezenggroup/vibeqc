#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scf/cuda/arena.hpp"
#include "scf/cuda/resources.hpp"
#include "scf/cuda/rhf_graph.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"

namespace vibeqc::scf {

struct CudaRhfBucketPlan {
  cuda_execution::CudaResources resources;
  cuda_execution::RhfIterationGraphs graphs;
  cuda_execution::ArenaLayout layout;
  cuda_execution::HostBatch topology;
  // Generation counters are meaningful only together with this plan's
  // value-checked immutable topology/options identity.
  std::uint64_t execution_generation{};
  std::uint64_t geometry_generation{};
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
  CudaDirectFinalStateAudit last_direct_final_state;
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

/** Synchronous force-ready Direct-HF state consumed before the owning plan or
 * stream can be reused.
 *
 * The owner pointer is not an identity proof by itself. Its immutable
 * topology/options have already passed value equality in the bucket admission
 * path; the generation counters then bind geometry/overlap/operator changes
 * within that owner. Basis, spin and occupations are immutable for an admitted
 * plan.
 */
struct CudaDirectFinalSCFState {
  const CudaRhfBucketPlan* owner{};
  std::uint64_t execution_generation{};
  std::uint64_t state_generation{};
  std::uint64_t geometry_generation{};
  std::uint64_t basis_generation{1};
  std::uint64_t overlap_generation{};
  std::uint64_t density_generation{};
  std::uint64_t operator_generation{};
  std::uint64_t orbital_generation{};
  std::uint64_t energy_generation{};
  std::uint64_t screening_generation{1};
  std::uint64_t precision_generation{1};
  std::uint64_t spin_generation{1};
  std::uint64_t occupation_generation{1};
  double screening_tolerance{};
  int precision_mode{};
  bool unrestricted{};
  const double* density{};
  const double* physical_fock{};
  const double* coefficients{};
  const double* orbital_energies{};
  const std::int32_t* occupations{};
  const double* physical_residual{};
  double* energy{};
  const std::uint8_t* active{};
  cudaStream_t stream{};
  CudaDirectFinalStateAudit proof{};

  bool force_consumable() const noexcept {
    const bool publication_bound =
        state_generation != 0 && state_generation == density_generation &&
        state_generation == operator_generation && state_generation == orbital_generation &&
        state_generation == energy_generation;
    const bool owner_bound = owner != nullptr && stream != nullptr &&
                             owner->resources.stream_ == stream &&
                             execution_generation == owner->execution_generation &&
                             geometry_generation == owner->geometry_generation;
    const bool buffers_bound = density != nullptr && physical_fock != nullptr &&
                               coefficients != nullptr && orbital_energies != nullptr &&
                               occupations != nullptr && physical_residual != nullptr &&
                               energy != nullptr && active != nullptr;
    const bool proof_complete = proof.route != CudaDirectFinalStateRoute::none &&
                                proof.physical_residual_validated && proof.target_precision &&
                                proof.orbital_frame_bound && proof.physical_orbital_energies &&
                                proof.restart_same_density_generation;
    const bool fast_route_valid =
        proof.route != CudaDirectFinalStateRoute::scf_force_ready ||
        (proof.seed_provenance &&
         proof.fallback_reason == CudaDirectFinalStateFallbackReason::none &&
         proof.additional_physical_fock_builds == 0 && proof.additional_final_eigen_solves == 0);
    const bool fallback_route_valid =
        proof.route != CudaDirectFinalStateRoute::canonical_fallback ||
        proof.fallback_reason != CudaDirectFinalStateFallbackReason::none;
    return publication_bound && owner_bound && buffers_bound && proof_complete &&
           fast_route_valid && fallback_route_valid;
  }
};

/** Exact captured-option identity shared by admission and driver invariants. */
inline bool same_hf_bucket_options(const ScfOptions& first, const ScfOptions& second) {
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

/** Internal direct-HF numerical driver consumed by the bucket lifecycle owner. */
std::vector<RhfBucketItem> execute_hf_cuda_bucket_driver(CudaRhfBucketPlan& plan,
                                                         const cuda_execution::HostBatch& host,
                                                         const ScfOptions& options, int device_id,
                                                         bool unrestricted,
                                                         bool shell_class_profiling,
                                                         bool inactive_eigensolver_profiling);

}  // namespace vibeqc::scf
