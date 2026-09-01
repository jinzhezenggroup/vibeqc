#ifndef VIBEQC_VIBEQC_HPP
#define VIBEQC_VIBEQC_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc {

class Error : public std::runtime_error {
 public:
  Error(vibeqc_status status, const std::string& message)
      : std::runtime_error(message), status_(status) {}
  [[nodiscard]] vibeqc_status status() const noexcept { return status_; }

 private:
  vibeqc_status status_;
};

inline void check(vibeqc_status status) {
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw Error(status, vibeqc_status_message(status));
  }
}

struct MethodCapabilities {
  vibeqc_method method{};
  vibeqc_method_family family{};
  vibeqc_property_flags supported_properties{};
  bool available{};
  bool supports_batch{};
};

/** Query the native registry without preparing a system or calculation. */
inline MethodCapabilities method_capabilities(vibeqc_method method) {
  vibeqc_method_capabilities_descriptor native{
      sizeof(vibeqc_method_capabilities_descriptor), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0};
  check(vibeqc_method_get_capabilities(method, &native));
  return {native.method, native.family, native.supported_properties, native.available != 0,
          native.supports_batch != 0};
}

class Context {
 public:
  explicit Context(const vibeqc_context_descriptor& descriptor) {
    check(vibeqc_context_create(&descriptor, &handle_));
  }
  ~Context() { vibeqc_context_destroy(handle_); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  [[nodiscard]] vibeqc_context* get() const noexcept { return handle_; }

 private:
  vibeqc_context* handle_ = nullptr;
};

class System {
 public:
  System(Context& context, const vibeqc_system_descriptor& descriptor)
      : atom_count_(descriptor.atom_count) {
    check(vibeqc_system_create(context.get(), &descriptor, &handle_));
  }
  ~System() { vibeqc_system_destroy(handle_); }
  System(const System&) = delete;
  System& operator=(const System&) = delete;
  System(System&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)), atom_count_(other.atom_count_) {}
  [[nodiscard]] vibeqc_system* get() const noexcept { return handle_; }
  [[nodiscard]] std::uint32_t atom_count() const noexcept { return atom_count_; }

 private:
  vibeqc_system* handle_ = nullptr;
  std::uint32_t atom_count_{};
};

struct BatchItemResult {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  double energy{};
  std::vector<double> forces;
  std::uint32_t iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  vibeqc_backend executed_backend{VIBEQC_BACKEND_CPU_REFERENCE};
  std::uint32_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
};

/** CUDA density-fitting metric conditioning and allocation evidence. */
struct DensityFittingMetricDiagnostic {
  std::uint32_t bucket_id{};
  std::uint32_t system_index{};
  std::uint64_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
  std::uint64_t solver_device_workspace_bytes{};
  std::uint64_t solver_host_workspace_bytes{};
  std::uint64_t device_resident_bytes{};
  std::uint64_t peak_device_bytes{};
  std::uint64_t host_resident_bytes{};
  std::uint64_t peak_host_bytes{};
  std::uint64_t auxiliary_tile{};
  bool streamed{};
};

class Batch {
 public:
  Batch(Context& context, std::span<const System* const> systems,
        const vibeqc_method_descriptor& method,
        vibeqc_batch_flags flags = VIBEQC_BATCH_ENABLE_WARM_STARTS)
      : atom_counts_(systems.size()) {
    std::vector<const vibeqc_system*> handles(systems.size());
    for (std::size_t i = 0; i < systems.size(); ++i) {
      if (systems[i] == nullptr) {
        throw Error(VIBEQC_STATUS_INVALID_ARGUMENT, "null system in batch");
      }
      handles[i] = systems[i]->get();
      atom_counts_[i] = systems[i]->atom_count();
    }
    check(vibeqc_batch_prepare(context.get(), handles.data(),
                               static_cast<std::uint32_t>(handles.size()), &method, flags,
                               &handle_));
  }
  ~Batch() { vibeqc_batch_destroy(handle_); }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  Batch(Batch&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        atom_counts_(std::move(other.atom_counts_)) {}

  [[nodiscard]] std::size_t size() const noexcept { return atom_counts_.size(); }

  std::vector<BatchItemResult> execute(
      const std::vector<std::optional<std::vector<double>>>& coordinates = {}) {
    if (!coordinates.empty() && coordinates.size() != size()) {
      throw Error(VIBEQC_STATUS_INVALID_ARGUMENT, "coordinate list does not match batch size");
    }
    std::vector<vibeqc_batch_input_descriptor> inputs;
    if (!coordinates.empty()) {
      inputs.resize(size());
      for (std::size_t i = 0; i < size(); ++i) {
        inputs[i] = {sizeof(vibeqc_batch_input_descriptor), VIBEQC_ABI_VERSION,
                     coordinates[i] ? coordinates[i]->data() : nullptr,
                     coordinates[i] ? static_cast<std::uint32_t>(coordinates[i]->size()) : 0};
      }
    }

    std::vector<BatchItemResult> results(size());
    std::vector<vibeqc_batch_item_result_descriptor> native(size());
    for (std::size_t i = 0; i < size(); ++i) {
      results[i].forces.resize(static_cast<std::size_t>(atom_counts_[i]) * 3);
      native[i] = {sizeof(vibeqc_batch_item_result_descriptor),
                   VIBEQC_ABI_VERSION,
                   VIBEQC_STATUS_INTERNAL_ERROR,
                   0.0,
                   results[i].forces.data(),
                   static_cast<std::uint32_t>(results[i].forces.size()),
                   0,
                   0.0,
                   0.0,
                   0,
                   VIBEQC_BACKEND_CPU_REFERENCE,
                   0,
                   0,
                   0};
    }
    check(vibeqc_batch_execute(handle_, inputs.empty() ? nullptr : inputs.data(),
                               static_cast<std::uint32_t>(inputs.size()), native.data(),
                               static_cast<std::uint32_t>(native.size())));
    for (std::size_t i = 0; i < size(); ++i) {
      results[i].status = native[i].status;
      results[i].energy = native[i].energy;
      results[i].iterations = native[i].iterations;
      results[i].energy_change = native[i].energy_change;
      results[i].density_rms = native[i].density_rms;
      results[i].converged = native[i].converged != 0;
      results[i].executed_backend = native[i].executed_backend;
      results[i].bucket_id = native[i].bucket_id;
      results[i].warm_start_used = native[i].warm_start_used != 0;
      results[i].warm_start_fallback = native[i].warm_start_fallback != 0;
      if (native[i].status != VIBEQC_STATUS_SUCCESS) results[i].forces.clear();
    }
    return results;
  }

  void clear_warm_starts() { check(vibeqc_batch_clear_warm_starts(handle_)); }

  void set_warm_start_updates(bool enabled) {
    check(vibeqc_batch_set_warm_start_updates(handle_, enabled ? 1 : 0));
  }

  /** Return CUDA DF metric/allocation records from the most recent execution. */
  std::vector<DensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics() const {
    std::uint32_t count = 0;
    check(vibeqc_batch_get_last_density_fitting_metric_diagnostics(handle_, nullptr, 0, &count));
    std::vector<vibeqc_density_fitting_metric_diagnostic> native(count);
    std::uint32_t written = 0;
    check(vibeqc_batch_get_last_density_fitting_metric_diagnostics(handle_, native.data(), count,
                                                                   &written));
    if (written != count) {
      throw Error(VIBEQC_STATUS_INTERNAL_ERROR,
                  "CUDA DF metric diagnostic count changed during copy");
    }
    std::vector<DensityFittingMetricDiagnostic> result;
    result.reserve(count);
    for (const auto& input : native) {
      result.push_back({input.bucket_id, input.system_index, input.effective_rank,
                        input.absolute_threshold, input.condition_number,
                        input.solver_device_workspace_bytes, input.solver_host_workspace_bytes,
                        input.device_resident_bytes, input.peak_device_bytes,
                        input.host_resident_bytes, input.peak_host_bytes, input.auxiliary_tile,
                        input.streamed != 0});
    }
    return result;
  }

 private:
  vibeqc_batch* handle_ = nullptr;
  std::vector<std::uint32_t> atom_counts_;
};

}  // namespace vibeqc

#endif
