#include "methods/hf_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#include "api/handles.hpp"
#include "scf/fleet.hpp"
#include "scf/mean_field.hpp"
#include "scf/types.hpp"

namespace vibeqc::methods::detail {
namespace {

bool method_field_present(const vibeqc_method_descriptor& descriptor, std::size_t offset,
                          std::size_t width) noexcept {
  return descriptor.struct_size >= offset && descriptor.struct_size - offset >= width;
}

vibeqc_density_fitting_mode density_fitting_mode(const vibeqc_method_descriptor& descriptor) {
  if (!method_field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_mode),
                            sizeof(descriptor.density_fitting_mode))) {
    return VIBEQC_DENSITY_FITTING_NONE;
  }
  const auto mode = descriptor.density_fitting_mode;
  if (mode != VIBEQC_DENSITY_FITTING_NONE && mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
      mode != VIBEQC_DENSITY_FITTING_CUDA && mode != VIBEQC_DENSITY_FITTING_AUTO) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown density-fitting execution mode");
  }
  return mode;
}

double density_fitting_threshold(const vibeqc_method_descriptor& descriptor) {
  if (!method_field_present(descriptor,
                            offsetof(vibeqc_method_descriptor, density_fitting_relative_threshold),
                            sizeof(descriptor.density_fitting_relative_threshold)) ||
      descriptor.density_fitting_relative_threshold == 0.0) {
    return 1.0e-10;
  }
  return descriptor.density_fitting_relative_threshold;
}

std::size_t density_fitting_memory_budget(const vibeqc_method_descriptor& descriptor) {
  if (!method_field_present(descriptor,
                            offsetof(vibeqc_method_descriptor, density_fitting_memory_budget_bytes),
                            sizeof(descriptor.density_fitting_memory_budget_bytes))) {
    return 0;
  }
  return static_cast<std::size_t>(descriptor.density_fitting_memory_budget_bytes);
}

std::optional<core::System> density_fitting_auxiliary_template(
    const vibeqc_method_descriptor& descriptor) {
  if (!method_field_present(descriptor,
                            offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis),
                            sizeof(descriptor.density_fitting_auxiliary_basis)) ||
      descriptor.density_fitting_auxiliary_basis == nullptr) {
    return std::nullopt;
  }
  return descriptor.density_fitting_auxiliary_basis->data;
}

scf::ScfOptions scf_options(const vibeqc_method_descriptor& descriptor) {
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;
  options.density_fitting_mode = density_fitting_mode(descriptor);
  options.density_fitting_relative_threshold = density_fitting_threshold(descriptor);
  options.density_fitting_memory_budget_bytes = density_fitting_memory_budget(descriptor);
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE &&
      (!(options.density_fitting_relative_threshold > 0.0) ||
       !(options.density_fitting_relative_threshold < 1.0) ||
       !std::isfinite(options.density_fitting_relative_threshold))) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "density-fitting threshold must lie strictly between zero and one");
  }
  return options;
}

void validate_density_fitting_auxiliary(const core::System& orbital,
                                        const std::optional<core::System>& auxiliary) {
  if (!auxiliary.has_value()) return;
  if (auxiliary->atoms.size() != orbital.atoms.size()) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "density-fitting auxiliary basis must contain the same atoms");
  }
  for (std::size_t atom = 0; atom < orbital.atoms.size(); ++atom) {
    if (auxiliary->atoms[atom].atomic_number != orbital.atoms[atom].atomic_number ||
        auxiliary->atoms[atom].position != orbital.atoms[atom].position) {
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "density-fitting auxiliary basis must share the system geometry");
    }
  }
  for (const core::Shell& shell : auxiliary->shells) {
    if (shell.atom_index >= orbital.atoms.size()) {
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "density-fitting auxiliary shell atom is out of range");
    }
  }
}

std::optional<core::System> normalized_auxiliary_template(const core::System& orbital,
                                                          std::optional<core::System> auxiliary) {
  if (!auxiliary.has_value()) return std::nullopt;
  validate_density_fitting_auxiliary(orbital, auxiliary);
  return auxiliary;
}

Result adapt_result(scf::ScfResult native, vibeqc_backend backend) {
  Result result;
  result.energy = native.energy;
  result.forces = std::move(native.forces);
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  return result;
}

std::uint32_t histogram_quantile(const std::vector<std::uint64_t>& histogram,
                                 std::uint64_t numerator, std::uint64_t denominator) {
  const std::uint64_t count = std::accumulate(histogram.begin(), histogram.end(), std::uint64_t{0});
  if (count == 0U) return 0U;
  const std::uint64_t rank = (count * numerator + denominator - 1U) / denominator;
  std::uint64_t cumulative = 0U;
  for (std::size_t value = 0; value < histogram.size(); ++value) {
    cumulative += histogram[value];
    if (cumulative >= rank) return static_cast<std::uint32_t>(value);
  }
  return static_cast<std::uint32_t>(histogram.size() - 1U);
}

DirectPppsQueueProfile adapt_ppps_queue_profile(const scf::CudaPppsQueueProfile& native) {
  DirectPppsQueueProfile profile;
  profile.descriptor_slots = native.descriptor_slots;
  profile.non_empty_descriptors = native.non_empty_descriptors;
  profile.empty_descriptors = native.descriptor_slots - native.non_empty_descriptors;
  profile.tasks = native.tasks;
  profile.primitive_work = native.primitive_work;
  profile.ket_count_min =
      histogram_quantile(native.ket_count_histogram, 1U, native.non_empty_descriptors);
  profile.ket_count_median = histogram_quantile(native.ket_count_histogram, 1U, 2U);
  profile.ket_count_p90 = histogram_quantile(native.ket_count_histogram, 9U, 10U);
  profile.ket_count_p99 = histogram_quantile(native.ket_count_histogram, 99U, 100U);
  profile.ket_count_max = histogram_quantile(
      native.ket_count_histogram, native.non_empty_descriptors, native.non_empty_descriptors);
  for (std::size_t index = 0; index < scf::kPppsProfileBlockThreads.size(); ++index) {
    profile.lane_efficiency[index] =
        native.lane_slots[index] == 0U
            ? 0.0
            : static_cast<double>(native.tasks) / static_cast<double>(native.lane_slots[index]);
    profile.task_tail_imbalance[index] =
        native.task_schedule_ideal[index] == 0.0
            ? 0.0
            : native.task_schedule_makespan[index] / native.task_schedule_ideal[index] - 1.0;
    profile.primitive_tail_imbalance[index] =
        native.primitive_schedule_ideal[index] == 0.0
            ? 0.0
            : native.primitive_schedule_makespan[index] / native.primitive_schedule_ideal[index] -
                  1.0;
  }
  profile.primitive_warp_efficiency = native.primitive_warp_slots == 0U
                                          ? 0.0
                                          : static_cast<double>(native.primitive_work) /
                                                static_cast<double>(native.primitive_warp_slots);
  profile.orientation_tasks = native.orientation_tasks;
  profile.orientation_primitive_work = native.orientation_primitive_work;
  profile.bra_primitive_tasks = native.bra_primitive_tasks;
  profile.bra_primitive_work = native.bra_primitive_work;
  profile.ket_primitive_tasks = native.ket_primitive_tasks;
  profile.ket_primitive_work = native.ket_primitive_work;
  return profile;
}

EigensolverDiagnostic adapt_eigensolver_diagnostic(const scf::CudaEigensolverDiagnostic& native) {
  const scf::XsyevBatchedGraphProbeResult& probe = native.xsyev_probe;
  EigensolverDiagnostic diagnostic;
  diagnostic.bucket_id = static_cast<std::uint32_t>(native.bucket_id);
  diagnostic.ordinary_family = static_cast<std::uint32_t>(native.ordinary_family);
  diagnostic.graph_family = static_cast<std::uint32_t>(native.family);
  diagnostic.selection_source = static_cast<std::uint32_t>(native.selection_source);
  diagnostic.matrix_dimension = native.matrix_dimension;
  diagnostic.physical_system_count = native.physical_system_count;
  diagnostic.solver_batch_count = native.solver_batch_count;
  diagnostic.api_eligible = probe.api.eligible;
  diagnostic.api_reason = static_cast<std::uint32_t>(probe.api.reason);
  diagnostic.matrix_batch_product = probe.api.matrix_batch_product;
  diagnostic.probe_failure_stage = static_cast<std::uint32_t>(probe.failure_stage);
  diagnostic.device_workspace_bytes = probe.device_workspace_bytes;
  diagnostic.host_workspace_bytes = probe.host_workspace_bytes;
  diagnostic.available_device_bytes = probe.available_device_bytes;
  diagnostic.device_id = probe.device_id;
  diagnostic.device_uuid = probe.device_uuid;
  diagnostic.device_name = probe.device_name;
  diagnostic.compute_capability_major = probe.compute_capability_major;
  diagnostic.compute_capability_minor = probe.compute_capability_minor;
  diagnostic.cuda_runtime_version = probe.cuda_runtime_version;
  diagnostic.cuda_driver_version = probe.cuda_driver_version;
  diagnostic.cusolver_version = probe.cusolver_version;
  diagnostic.cuda_error = probe.cuda_error;
  diagnostic.cusolver_error = probe.cusolver_error;
  diagnostic.ordinary_execution_passed = probe.ordinary_execution_passed;
  diagnostic.graph_capture_passed = probe.graph_capture_passed;
  diagnostic.host_graph_replay_passed = probe.host_graph_replay_passed;
  diagnostic.device_tail_replay_passed = probe.device_tail_replay_passed;
  diagnostic.graph_eligible = probe.graph_eligible;
  diagnostic.maximum_eigenvalue_error = probe.maximum_eigenvalue_error;
  diagnostic.maximum_residual = probe.maximum_residual;
  diagnostic.maximum_orthogonality_error = probe.maximum_orthogonality_error;
  return diagnostic;
}

InactiveEigensolverProfileEntry adapt_inactive_eigensolver_profile_entry(
    const scf::CudaInactiveEigensolverProfileEntry& native) {
  InactiveEigensolverProfileEntry entry;
  entry.bucket_id = static_cast<std::uint32_t>(native.bucket_id);
  entry.iteration = native.iteration;
  entry.family = static_cast<std::uint32_t>(native.family);
  entry.physical_system_count = native.physical_system_count;
  entry.solver_batch_count = native.solver_batch_count;
  entry.active_physical_count = native.active_physical_count;
  entry.active_solver_count = native.active_solver_count;
  entry.solver_elapsed_nanoseconds = native.solver_elapsed_nanoseconds;
  entry.inactive_input_nonfinite_count = native.inactive_input_nonfinite_count;
  entry.inactive_submission_nonfinite_count = native.inactive_submission_nonfinite_count;
  entry.inactive_info_nonzero_count = native.inactive_info_nonzero_count;
  entry.inactive_touch_flags = native.inactive_touch_flags;
  entry.provider_invoked = native.provider_invoked;
  return entry;
}

class HfPreparedCalculation final : public PreparedCalculation {
 public:
  HfPreparedCalculation(Capabilities capabilities, core::ContextState& context, core::System system,
                        scf::ScfOptions options, std::optional<core::System> auxiliary_template)
      : capabilities_(capabilities),
        context_(&context),
        system_(std::move(system)),
        options_(options),
        auxiliary_template_(std::move(auxiliary_template)) {}

  [[nodiscard]] std::size_t atom_count() const noexcept override { return system_.atoms.size(); }

  [[nodiscard]] const Capabilities& capabilities() const noexcept override { return capabilities_; }

  Result execute() override {
    const bool unrestricted = capabilities_.method == VIBEQC_METHOD_UHF;
    const bool use_cuda = context_->requested_backend == VIBEQC_BACKEND_CUDA &&
                          (options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_NONE ||
                           options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA ||
                           options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_AUTO);
    scf::ScfResult native;
    if (options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_CPU_REFERENCE ||
        (options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_AUTO &&
         context_->requested_backend == VIBEQC_BACKEND_CPU_REFERENCE)) {
      const core::System& auxiliary =
          auxiliary_template_.has_value() ? *auxiliary_template_ : system_;
      native = unrestricted ? scf::run_uhf_density_fitting(system_, auxiliary, options_)
                            : scf::run_rhf_density_fitting(system_, auxiliary, options_);
    } else if (options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA ||
               options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_AUTO) {
      const core::System& auxiliary =
          auxiliary_template_.has_value() ? *auxiliary_template_ : system_;
      native = unrestricted ? scf::run_uhf_density_fitting_cuda(system_, auxiliary, options_,
                                                                context_->device_id)
                            : scf::run_rhf_density_fitting_cuda(system_, auxiliary, options_,
                                                                context_->device_id);
    } else if (context_->requested_backend == VIBEQC_BACKEND_CUDA) {
      native = unrestricted ? scf::run_uhf_cuda(system_, options_, context_->device_id)
                            : scf::run_rhf_cuda(system_, options_, context_->device_id);
    } else {
      native = unrestricted ? scf::run_uhf(system_, options_) : scf::run_rhf(system_, options_);
    }
    return adapt_result(std::move(native),
                        use_cuda ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE);
  }

 private:
  Capabilities capabilities_;
  core::ContextState* context_{};
  core::System system_;
  scf::ScfOptions options_;
  std::optional<core::System> auxiliary_template_;
};

class HfPreparedBatch final : public PreparedBatch {
 public:
  HfPreparedBatch(Capabilities capabilities, core::ContextState& context,
                  std::vector<core::System> systems, scf::ScfOptions options,
                  vibeqc_batch_flags flags, std::optional<core::System> auxiliary_template)
      : plan_(std::move(systems), capabilities.method, options,
              (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0,
              context.requested_backend == VIBEQC_BACKEND_CUDA &&
                  options.density_fitting_mode == VIBEQC_DENSITY_FITTING_NONE,
              (flags & VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING) != 0,
              (flags & VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING) != 0, context.device_id,
              std::move(auxiliary_template),
              context.requested_backend == VIBEQC_BACKEND_CUDA &&
                  (options.density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA ||
                   options.density_fitting_mode == VIBEQC_DENSITY_FITTING_AUTO)) {}

  [[nodiscard]] std::size_t size() const noexcept override { return plan_.size(); }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates) override {
    std::vector<scf::FleetItemResult> native = plan_.execute(coordinates);
    std::vector<BatchItemResult> results;
    results.reserve(native.size());
    for (scf::FleetItemResult& item : native) {
      BatchItemResult result;
      result.status = item.status;
      result.calculation = adapt_result(std::move(item.scf), item.executed_backend);
      result.bucket_id = item.bucket_id;
      result.warm_start_used = item.warm_start_used;
      result.warm_start_fallback = item.warm_start_fallback;
      results.push_back(std::move(result));
    }
    return results;
  }

  void clear_warm_starts() override { plan_.clear_warm_starts(); }

  void set_warm_start_updates(bool enabled) override { plan_.set_warm_start_updates(enabled); }

  [[nodiscard]] std::optional<std::vector<DirectShellClassProfileEntry>>
  last_direct_shell_class_profile() const override {
    const auto& native = plan_.last_shell_class_profile();
    if (!native.has_value()) return std::nullopt;
    std::vector<DirectShellClassProfileEntry> profile;
    profile.reserve(native->size());
    for (const scf::CudaRhfShellClassProfileEntry& entry : *native) {
      profile.push_back(
          {entry.shell_quartets, entry.tiles, entry.ao_quartets, entry.primitive_quartets});
    }
    return profile;
  }

  [[nodiscard]] std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile()
      const override {
    const auto& native = plan_.last_ppps_queue_profile();
    if (!native.has_value()) return std::nullopt;
    return adapt_ppps_queue_profile(*native);
  }

  [[nodiscard]] std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override {
    const auto& native = plan_.last_eigensolver_diagnostics();
    std::vector<EigensolverDiagnostic> diagnostics;
    diagnostics.reserve(native.size());
    std::transform(native.begin(), native.end(), std::back_inserter(diagnostics),
                   adapt_eigensolver_diagnostic);
    return diagnostics;
  }

  [[nodiscard]] std::vector<scf::CudaDensityFittingMetricDiagnostic>
  last_density_fitting_metric_diagnostics() const override {
    return plan_.last_density_fitting_metric_diagnostics();
  }

  [[nodiscard]] std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile()
      const override {
    const auto& native = plan_.last_inactive_eigensolver_profile();
    std::vector<InactiveEigensolverProfileEntry> profile;
    profile.reserve(native.size());
    std::transform(native.begin(), native.end(), std::back_inserter(profile),
                   adapt_inactive_eigensolver_profile_entry);
    return profile;
  }

 private:
  scf::FleetPlan plan_;
};

}  // namespace

vibeqc_status validate_hf_system(vibeqc_method method, const core::System& system,
                                 std::string& detail) {
  if (method == VIBEQC_METHOD_RHF) {
    if (system.electron_count % 2 == 0 && system.multiplicity == 1) {
      return VIBEQC_STATUS_SUCCESS;
    }
    detail = "RHF requires an even electron count and spin multiplicity 1";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (spin_excess >= 0 && spin_excess <= system.electron_count &&
      ((system.electron_count + spin_excess) & 1) == 0) {
    return VIBEQC_STATUS_SUCCESS;
  }
  detail = "UHF requires electron count and multiplicity to define integral spin occupations";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_hf_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  const scf::ScfOptions options = scf_options(descriptor);
  const auto auxiliary = density_fitting_auxiliary_template(descriptor);
  if (options.density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA &&
      context.requested_backend != VIBEQC_BACKEND_CUDA) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "CUDA density-fitting mode requires a CUDA execution context");
  }
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE) {
    validate_density_fitting_auxiliary(system, auxiliary);
  }
  return std::make_unique<HfPreparedCalculation>(capabilities, context, system, options,
                                                 normalized_auxiliary_template(system, auxiliary));
}

std::unique_ptr<PreparedBatch> prepare_hf_batch(const Capabilities& capabilities,
                                                core::ContextState& context,
                                                std::vector<core::System> systems,
                                                const vibeqc_method_descriptor& descriptor,
                                                vibeqc_batch_flags flags) {
  constexpr vibeqc_batch_flags supported_flags = VIBEQC_BATCH_ENABLE_WARM_STARTS |
                                                 VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING |
                                                 VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING;
  if ((flags & ~supported_flags) != 0) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unsupported Hartree-Fock batch flag");
  }
  const scf::ScfOptions options = scf_options(descriptor);
  const auto auxiliary = density_fitting_auxiliary_template(descriptor);
  if (options.density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA &&
      context.requested_backend != VIBEQC_BACKEND_CUDA) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "CUDA density-fitting mode requires a CUDA execution context");
  }
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE && auxiliary.has_value()) {
    if (auxiliary->atoms.size() != systems.front().atoms.size()) {
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "density-fitting auxiliary basis must match every batch topology");
    }
    for (const core::System& system : systems) {
      if (auxiliary->atoms.size() != system.atoms.size()) {
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "density-fitting auxiliary basis must match every batch atom count");
      }
      for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
        if (auxiliary->atoms[atom].atomic_number != system.atoms[atom].atomic_number) {
          throw MethodError(
              VIBEQC_STATUS_INVALID_ARGUMENT,
              "density-fitting auxiliary basis atomic topology differs from a batch system");
        }
      }
    }
    for (const core::Shell& shell : auxiliary->shells) {
      if (shell.atom_index >= systems.front().atoms.size()) {
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "density-fitting auxiliary shell atom is out of range");
      }
    }
  }
  return std::make_unique<HfPreparedBatch>(capabilities, context, std::move(systems), options,
                                           flags, auxiliary);
}

}  // namespace vibeqc::methods::detail
