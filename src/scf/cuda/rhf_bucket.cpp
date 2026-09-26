#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

#include "generated_direct_resident_psss_schedule.cuh"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_tile_validation.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/fock_build.hpp"

namespace vibeqc::scf {

namespace {

using namespace cuda_execution;
using cuda_policy::bounded_direct_fock_only_diagnostic_requested;
using cuda_policy::bounded_direct_primary_streaming_fock_mask_requested;
using cuda_policy::bounded_direct_streaming_override_requested;
using cuda_policy::bounded_fock_class_timing_requested;
using cuda_policy::graph_native_eigensolver_override_requested;
using cuda_policy::resolve_mixed_precision_fock_policy;
using cuda_policy::reuse_converged_fock_requested;

void fill_global_failure(std::vector<RhfBucketItem>& outputs, vibeqc_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
}

bool checked_size_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (second > std::numeric_limits<std::size_t>::max() - first) return false;
  result = first + second;
  return true;
}

bool checked_size_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) return false;
  result = first * second;
  return true;
}

}  // namespace

bool small_hf_cuda_resource_layout(std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                                   std::size_t shells, std::size_t primitives,
                                   std::size_t diis_history, std::size_t spins,
                                   std::size_t& arena_bytes, std::size_t& plan_object_bytes) {
  // This contract intentionally covers the provider with no cuBLAS/cuSOLVER
  // workspace or exact-quartet descriptor table. Other routes need their own
  // compact provider-workspace query before they can advertise a global bound.
  const cuda_policy::SmallHfWorkload small_hf_workload{nbf, spins, 1U, spins};
  const auto small_hf_profitability =
      cuda_policy::resolve_small_hf_profitability(runtime::CudaTargetInfo{}, small_hf_workload);
  if (nbf == 0 || nbf > kPersistentEriAoLimit || nbf > kSmallEigensolverLimit ||
      small_hf_profitability.use_cublas || direct_nbf < nbf || direct_nbf > 2 * nbf || atoms == 0 ||
      shells == 0 || shells > nbf || primitives == 0 || diis_history > 64 ||
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

bool small_hf_cuda_resource_layout_v2(std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                                      const std::uint8_t* shell_angular_values,
                                      const std::size_t* shell_primitive_counts, std::size_t shells,
                                      std::size_t diis_history, std::size_t spins,
                                      int precision_mode, double energy_tolerance,
                                      double screening_tolerance, std::size_t& arena_bytes,
                                      std::size_t& plan_object_bytes) {
  using namespace cuda_execution;
  if (shell_angular_values == nullptr || shell_primitive_counts == nullptr || nbf == 0 ||
      nbf > kPersistentEriAoLimit || nbf > kSmallEigensolverLimit || direct_nbf < nbf ||
      direct_nbf > 2 * nbf || atoms == 0 || shells == 0 || shells > nbf || diis_history > 64 ||
      (spins != 1 && spins != 2) ||
      (precision_mode != VIBEQC_PRECISION_FP64 && precision_mode != VIBEQC_PRECISION_AUTO) ||
      !std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
      !std::isfinite(screening_tolerance) || screening_tolerance <= 0.0) {
    return false;
  }

  // The current dry-run contract is for ordinary production execution. These
  // diagnostics either select a different topology schedule or own additional
  // allocations outside the normal arena, so fail closed instead of guessing.
  if (cuda_policy::bounded_direct_streaming_override_requested() ||
      cuda_policy::bounded_fock_class_timing_requested() ||
      cuda_policy::resolve_direct_tile_validation_policy().requested ||
      cuda_policy::graph_native_eigensolver_override_requested()) {
    return false;
  }

  const cuda_policy::SmallHfWorkload small_hf_workload{nbf, spins, 1U, spins};
  const auto small_hf_profitability =
      cuda_policy::resolve_small_hf_profitability(runtime::CudaTargetInfo{}, small_hf_workload);
  if (small_hf_profitability.use_cublas) return false;

  std::vector<std::uint8_t> shell_angular(shell_angular_values, shell_angular_values + shells);
  std::vector<std::int64_t> shell_direct_ao_offsets(shells + 1, 0);
  std::vector<std::int32_t> shell_pair_first;
  std::vector<std::int32_t> shell_pair_second;
  std::vector<std::int64_t> system_shell_pair_offsets{0};
  shell_pair_first.reserve(shells * (shells + 1) / 2);
  shell_pair_second.reserve(shell_pair_first.capacity());

  std::size_t primitive_count = 0;
  std::size_t direct_ao_count = 0;
  for (std::size_t shell = 0; shell < shells; ++shell) {
    if (shell_angular[shell] > kMaximumAngularMomentum || shell_primitive_counts[shell] == 0)
      return false;
    const std::size_t shell_direct_aos = molecule::cartesian_count(shell_angular[shell]);
    if (!checked_size_add(direct_ao_count, shell_direct_aos, direct_ao_count) ||
        direct_ao_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        !checked_size_add(primitive_count, shell_primitive_counts[shell], primitive_count)) {
      return false;
    }
    shell_direct_ao_offsets[shell + 1] = static_cast<std::int64_t>(direct_ao_count);
  }
  if (direct_ao_count != direct_nbf || primitive_count == 0) return false;

  std::size_t shell_pair_primitive_count = 0;
  std::size_t psss_bra_pair_count = 0;
  std::size_t psss_resident_ket_pair_count = 0;
  for (std::size_t first = 0; first < shells; ++first) {
    for (std::size_t second = 0; second <= first; ++second) {
      if (first > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          second > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return false;
      }
      shell_pair_first.push_back(static_cast<std::int32_t>(first));
      shell_pair_second.push_back(static_cast<std::int32_t>(second));
      std::size_t pair_primitives = 0;
      if (!checked_size_multiply(shell_primitive_counts[first], shell_primitive_counts[second],
                                 pair_primitives) ||
          !checked_size_add(shell_pair_primitive_count, pair_primitives,
                            shell_pair_primitive_count)) {
        return false;
      }
      const unsigned pair_order = static_cast<unsigned>(shell_angular[first]) +
                                  static_cast<unsigned>(shell_angular[second]);
      if (pair_order == 1U) ++psss_bra_pair_count;
      if (shell_angular[first] == 0U && shell_angular[second] == 0U) ++psss_resident_ket_pair_count;
    }
  }
  const std::size_t shell_pair_count = shell_pair_first.size();
  if (shell_pair_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return false;
  system_shell_pair_offsets.push_back(static_cast<std::int64_t>(shell_pair_count));
  const std::size_t shell_pair_block_count = detail::bounded_direct_queue_refill_count(
      shell_pair_count, detail::kBoundedDirectShellPairBlockSize);

  std::size_t psss_chunks_per_bra = 0;
  if (psss_resident_ket_pair_count != 0) {
    psss_chunks_per_bra =
        (psss_resident_ket_pair_count + kResidentPsssThreads - 1) / kResidentPsssThreads;
  }
  std::size_t psss_resident_task_count = 0;
  if (!checked_size_multiply(psss_bra_pair_count, psss_chunks_per_bra, psss_resident_task_count)) {
    return false;
  }

  detail::DirectQuartetTaskLayout direct_task_layout{};
  if (!detail::make_direct_quartet_task_layout(
          shell_direct_ao_offsets, shell_angular, system_shell_pair_offsets, shell_pair_first,
          shell_pair_second, kMixedFockMinimumAngularOrder, direct_task_layout)) {
    return false;
  }

  // <=16-AO production shapes are fixed-topology. If that invariant ever
  // changes, refuse this v2 contract until the bounded schedule has its own
  // device-free capacity query rather than silently undercounting it.
  if (detail::direct_topology_requires_bounded_streaming(direct_task_layout.shell_quartet_count) ||
      direct_task_layout.exact_tile_count > detail::kDirectFixedTopologyTileLimit) {
    return false;
  }

  ArenaLayout energy_layout{};
  if (!make_layout(1, nbf, direct_nbf, atoms, shells, shell_pair_count, shell_pair_block_count, 0,
                   shell_pair_primitive_count, 0, 0, 0, 0, 0, 0, 0, primitive_count,
                   std::max<std::size_t>(1, diis_history), 0, spins, true, false, false, false,
                   false, false, false, energy_layout)) {
    return false;
  }

  const auto mixed_policy = cuda_policy::resolve_mixed_precision_fock_policy(
      static_cast<vibeqc_precision_mode>(precision_mode), energy_tolerance, screening_tolerance,
      direct_task_layout.system_mixed_capable_tile_counts.empty()
          ? 0.0
          : static_cast<double>(direct_task_layout.system_mixed_capable_tile_counts.front()));
  const bool mixed_precision_fock = mixed_policy.threshold.has_value();
  std::size_t fp32_shell_quartet_tile_count = 0;
  if (mixed_precision_fock) {
    for (std::size_t order = kMixedFockMinimumAngularOrder;
         order < detail::kDirectQuartetAngularOrderCount; ++order) {
      if (!checked_size_add(fp32_shell_quartet_tile_count,
                            direct_task_layout.angular_order_tile_counts[order],
                            fp32_shell_quartet_tile_count)) {
        return false;
      }
    }
  }

  // The runtime-selected AOT profile consumes a subset of these exact tiles.
  // Charging every compiler-valid tile is the profile-independent structural
  // upper bound, so the dry-run can stay device-free without undercounting a
  // future compatible profile.
  const std::size_t generated_shell_task_capacity = direct_task_layout.exact_tile_count;
  const std::size_t ppps_resident_ket_task_capacity =
      direct_task_layout.shell_class_tile_counts[kPppsShellClass];
  constexpr std::size_t kGenericOrderFive = 5;
  const std::size_t generic_order5_tile_capacity =
      direct_task_layout.angular_order_tile_counts[kGenericOrderFive];

  ArenaLayout force_layout{};
  if (!make_layout(1, nbf, direct_nbf, atoms, shells, shell_pair_count, shell_pair_block_count, 0,
                   shell_pair_primitive_count, psss_resident_task_count,
                   psss_resident_ket_pair_count, direct_task_layout.exact_tile_count,
                   fp32_shell_quartet_tile_count, generated_shell_task_capacity,
                   ppps_resident_ket_task_capacity, generic_order5_tile_capacity, primitive_count,
                   std::max<std::size_t>(1, diis_history), 0, spins, false, direct_nbf != nbf,
                   false, false, false, false, mixed_precision_fock, force_layout)) {
    return false;
  }

  arena_bytes = std::max(energy_layout.bytes, force_layout.bytes);
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

  const auto expanded_primitive_references = checked_expanded_primitive_references(systems);

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
                       options.export_physical_reference, options.compute_forces)) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  double mixed_precision_fock_threshold = 0.0;
  bool mixed_precision_fock = false;
  if (*plan != nullptr && (*plan)->quartet_direct) {
    const auto mixed_precision_policy = resolve_mixed_precision_fock_policy(
        options.precision_mode, options.energy_tolerance, options.screening_tolerance,
        static_cast<double>((*plan)->mixed_precision_eligible_tile_count));
    if (mixed_precision_policy.threshold.has_value()) {
      mixed_precision_fock = true;
      mixed_precision_fock_threshold = *mixed_precision_policy.threshold;
    }
  }
  const bool reuse_converged_fock = reuse_converged_fock_requested();
  const bool graph_native_eigensolver_override = graph_native_eigensolver_override_requested();
  if (*plan != nullptr && (*plan)->initialized &&
      ((*plan)->resources.device_id_ != device_id || !same_topology((*plan)->topology, candidate) ||
       !same_hf_bucket_options((*plan)->options, options) ||
       (*plan)->unrestricted != unrestricted ||
       (*plan)->shell_class_profiling != shell_class_profiling ||
       (*plan)->inactive_eigensolver_profiling != inactive_eigensolver_profiling ||
       (*plan)->bounded_fock_class_timing != bounded_fock_class_timing_requested() ||
       (*plan)->bounded_streaming_override != bounded_direct_streaming_override_requested() ||
       (*plan)->fock_only_diagnostic != bounded_direct_fock_only_diagnostic_requested() ||
       (*plan)->primary_streaming_fock_mask !=
           bounded_direct_primary_streaming_fock_mask_requested().value_or(0U) ||
       (*plan)->graph_native_eigensolver_override != graph_native_eigensolver_override ||
       (*plan)->reuse_converged_fock != reuse_converged_fock ||
       (*plan)->one_electron_value_mapping != cuda_policy::one_electron_value_mapping_requested() ||
       (*plan)->mixed_precision_fock != mixed_precision_fock ||
       (*plan)->mixed_precision_fock_threshold != mixed_precision_fock_threshold)) {
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
      execute_hf_cuda_bucket_driver(**plan, candidate, options, device_id, unrestricted,
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
    outputs = execute_hf_cuda_bucket_driver(**plan, candidate, options, device_id, unrestricted,
                                            shell_class_profiling, inactive_eigensolver_profiling);
    for (auto& output : outputs) ++output.scf.precision.execution_retries;
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

bool get_rhf_cuda_final_state_audit(const CudaRhfBucketPlan* plan,
                                    CudaDirectFinalStateAudit& audit) noexcept {
  if (plan == nullptr || !plan->initialized ||
      plan->last_direct_final_state.route == CudaDirectFinalStateRoute::none) {
    return false;
  }
  audit = plan->last_direct_final_state;
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
  try {
    auto outputs =
        run_uhf_cuda_bucket_cached(&plan, systems, options, initial_densities, device_id,
                                   shell_class_profiling, inactive_eigensolver_profiling);
    destroy_rhf_cuda_bucket_plan(plan);
    return outputs;
  } catch (...) {
    destroy_rhf_cuda_bucket_plan(plan);
    throw;
  }
}

}  // namespace vibeqc::scf
