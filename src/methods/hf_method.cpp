#include "methods/hf_method.hpp"

#include "scf/fleet.hpp"
#include "scf/mean_field.hpp"
#include "scf/types.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

namespace vibeqc::methods::detail {
namespace {

scf::ScfOptions scf_options(const vibeqc_method_descriptor& descriptor) {
  scf::ScfOptions options;
  options.max_iterations =
      descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance = descriptor.energy_tolerance > 0.0
      ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance = descriptor.density_tolerance > 0.0
      ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance = descriptor.screening_tolerance > 0.0
      ? descriptor.screening_tolerance : 1.0e-12;
  return options;
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

std::uint32_t histogram_quantile(
    const std::vector<std::uint64_t>& histogram,
    std::uint64_t numerator,
    std::uint64_t denominator) {
  const std::uint64_t count =
      std::accumulate(histogram.begin(), histogram.end(), std::uint64_t{0});
  if (count == 0U) return 0U;
  const std::uint64_t rank =
      (count * numerator + denominator - 1U) / denominator;
  std::uint64_t cumulative = 0U;
  for (std::size_t value = 0; value < histogram.size(); ++value) {
    cumulative += histogram[value];
    if (cumulative >= rank) return static_cast<std::uint32_t>(value);
  }
  return static_cast<std::uint32_t>(histogram.size() - 1U);
}

DirectPppsQueueProfile adapt_ppps_queue_profile(
    const scf::CudaPppsQueueProfile& native) {
  DirectPppsQueueProfile profile;
  profile.descriptor_slots = native.descriptor_slots;
  profile.non_empty_descriptors = native.non_empty_descriptors;
  profile.empty_descriptors =
      native.descriptor_slots - native.non_empty_descriptors;
  profile.tasks = native.tasks;
  profile.primitive_work = native.primitive_work;
  profile.ket_count_min = histogram_quantile(
      native.ket_count_histogram, 1U, native.non_empty_descriptors);
  profile.ket_count_median =
      histogram_quantile(native.ket_count_histogram, 1U, 2U);
  profile.ket_count_p90 =
      histogram_quantile(native.ket_count_histogram, 9U, 10U);
  profile.ket_count_p99 =
      histogram_quantile(native.ket_count_histogram, 99U, 100U);
  profile.ket_count_max = histogram_quantile(
      native.ket_count_histogram, native.non_empty_descriptors,
      native.non_empty_descriptors);
  for (std::size_t index = 0;
       index < scf::kPppsProfileBlockThreads.size(); ++index) {
    profile.lane_efficiency[index] = native.lane_slots[index] == 0U
        ? 0.0
        : static_cast<double>(native.tasks) /
              static_cast<double>(native.lane_slots[index]);
    profile.task_tail_imbalance[index] =
        native.task_schedule_ideal[index] == 0.0
        ? 0.0
        : native.task_schedule_makespan[index] /
                  native.task_schedule_ideal[index] -
              1.0;
    profile.primitive_tail_imbalance[index] =
        native.primitive_schedule_ideal[index] == 0.0
        ? 0.0
        : native.primitive_schedule_makespan[index] /
                  native.primitive_schedule_ideal[index] -
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

class HfPreparedCalculation final : public PreparedCalculation {
 public:
  HfPreparedCalculation(Capabilities capabilities,
                        core::ContextState& context,
                        core::System system,
                        scf::ScfOptions options)
      : capabilities_(capabilities),
        context_(&context),
        system_(std::move(system)),
        options_(options) {}

  [[nodiscard]] std::size_t atom_count() const noexcept override {
    return system_.atoms.size();
  }

  [[nodiscard]] const Capabilities& capabilities() const noexcept override {
    return capabilities_;
  }

  Result execute() override {
    const bool use_cuda = context_->requested_backend == VIBEQC_BACKEND_CUDA;
    const bool unrestricted = capabilities_.method == VIBEQC_METHOD_UHF;
    scf::ScfResult native;
    if (use_cuda) {
      native = unrestricted
          ? scf::run_uhf_cuda(system_, options_, context_->device_id)
          : scf::run_rhf_cuda(system_, options_, context_->device_id);
    } else {
      native = unrestricted ? scf::run_uhf(system_, options_)
                            : scf::run_rhf(system_, options_);
    }
    return adapt_result(std::move(native),
                        use_cuda ? VIBEQC_BACKEND_CUDA
                                 : VIBEQC_BACKEND_CPU_REFERENCE);
  }

 private:
  Capabilities capabilities_;
  core::ContextState* context_{};
  core::System system_;
  scf::ScfOptions options_;
};

class HfPreparedBatch final : public PreparedBatch {
 public:
  HfPreparedBatch(Capabilities capabilities,
                  core::ContextState& context,
                  std::vector<core::System> systems,
                  scf::ScfOptions options,
                  vibeqc_batch_flags flags)
      : plan_(std::move(systems), capabilities.method, options,
              (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0,
              context.requested_backend == VIBEQC_BACKEND_CUDA,
              (flags & VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING) != 0,
              context.device_id) {}

  [[nodiscard]] std::size_t size() const noexcept override {
    return plan_.size();
  }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates) override {
    std::vector<scf::FleetItemResult> native = plan_.execute(coordinates);
    std::vector<BatchItemResult> results;
    results.reserve(native.size());
    for (scf::FleetItemResult& item : native) {
      BatchItemResult result;
      result.status = item.status;
      result.calculation =
          adapt_result(std::move(item.scf), item.executed_backend);
      result.bucket_id = item.bucket_id;
      result.warm_start_used = item.warm_start_used;
      result.warm_start_fallback = item.warm_start_fallback;
      results.push_back(std::move(result));
    }
    return results;
  }

  void clear_warm_starts() override { plan_.clear_warm_starts(); }

  void set_warm_start_updates(bool enabled) override {
    plan_.set_warm_start_updates(enabled);
  }

  [[nodiscard]] std::optional<std::vector<DirectShellClassProfileEntry>>
  last_direct_shell_class_profile() const override {
    const auto& native = plan_.last_shell_class_profile();
    if (!native.has_value()) return std::nullopt;
    std::vector<DirectShellClassProfileEntry> profile;
    profile.reserve(native->size());
    for (const scf::CudaRhfShellClassProfileEntry& entry : *native) {
      profile.push_back({entry.shell_quartets, entry.tiles, entry.ao_quartets,
                         entry.primitive_quartets});
    }
    return profile;
  }

  [[nodiscard]] std::optional<DirectPppsQueueProfile>
  last_direct_ppps_queue_profile() const override {
    const auto& native = plan_.last_ppps_queue_profile();
    if (!native.has_value()) return std::nullopt;
    return adapt_ppps_queue_profile(*native);
  }

 private:
  scf::FleetPlan plan_;
};

}  // namespace

vibeqc_status validate_hf_system(vibeqc_method method,
                                 const core::System& system,
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
  detail =
      "UHF requires electron count and multiplicity to define integral spin occupations";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_hf_calculation(
    const Capabilities& capabilities,
    core::ContextState& context,
    const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  return std::make_unique<HfPreparedCalculation>(
      capabilities, context, system, scf_options(descriptor));
}

std::unique_ptr<PreparedBatch> prepare_hf_batch(
    const Capabilities& capabilities,
    core::ContextState& context,
    std::vector<core::System> systems,
    const vibeqc_method_descriptor& descriptor,
    vibeqc_batch_flags flags) {
  constexpr vibeqc_batch_flags supported_flags =
      VIBEQC_BATCH_ENABLE_WARM_STARTS |
      VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING;
  if ((flags & ~supported_flags) != 0) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "unsupported Hartree-Fock batch flag");
  }
  return std::make_unique<HfPreparedBatch>(
      capabilities, context, std::move(systems), scf_options(descriptor), flags);
}

}  // namespace vibeqc::methods::detail
