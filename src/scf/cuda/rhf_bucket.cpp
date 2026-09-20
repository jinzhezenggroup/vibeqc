#include <algorithm>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

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
using cuda_policy::bounded_direct_streaming_override_requested;
using cuda_policy::bounded_fock_class_timing_requested;
using cuda_policy::graph_native_eigensolver_override_requested;
using cuda_policy::resolve_mixed_precision_fock_policy;
using cuda_policy::reuse_converged_fock_requested;

void fill_global_failure(std::vector<RhfBucketItem>& outputs, vibeqc_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
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
                       options.export_physical_reference)) {
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
