#ifndef QCE_QCE_HPP
#define QCE_QCE_HPP

#include "qce/qce.h"

#include <cstdint>
#include <stdexcept>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qce {

class Error : public std::runtime_error {
 public:
  Error(qce_status status, const std::string& message)
      : std::runtime_error(message), status_(status) {}
  [[nodiscard]] qce_status status() const noexcept { return status_; }

 private:
  qce_status status_;
};

inline void check(qce_status status) {
  if (status != QCE_STATUS_SUCCESS) {
    throw Error(status, qce_status_message(status));
  }
}

class Context {
 public:
  explicit Context(const qce_context_descriptor& descriptor) {
    check(qce_context_create(&descriptor, &handle_));
  }
  ~Context() { qce_context_destroy(handle_); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  [[nodiscard]] qce_context* get() const noexcept { return handle_; }

 private:
  qce_context* handle_ = nullptr;
};

class System {
 public:
  System(Context& context, const qce_system_descriptor& descriptor)
      : atom_count_(descriptor.atom_count) {
    check(qce_system_create(context.get(), &descriptor, &handle_));
  }
  ~System() { qce_system_destroy(handle_); }
  System(const System&) = delete;
  System& operator=(const System&) = delete;
  System(System&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        atom_count_(other.atom_count_) {}
  [[nodiscard]] qce_system* get() const noexcept { return handle_; }
  [[nodiscard]] std::uint32_t atom_count() const noexcept { return atom_count_; }

 private:
  qce_system* handle_ = nullptr;
  std::uint32_t atom_count_{};
};

struct BatchItemResult {
  qce_status status{QCE_STATUS_INTERNAL_ERROR};
  double energy{};
  std::vector<double> forces;
  std::uint32_t iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  qce_backend executed_backend{QCE_BACKEND_CPU_REFERENCE};
  std::uint32_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
};

class Batch {
 public:
  Batch(Context& context,
        std::span<const System* const> systems,
        const qce_method_descriptor& method,
        qce_batch_flags flags = QCE_BATCH_ENABLE_WARM_STARTS)
      : atom_counts_(systems.size()) {
    std::vector<const qce_system*> handles(systems.size());
    for (std::size_t i = 0; i < systems.size(); ++i) {
      if (systems[i] == nullptr) {
        throw Error(QCE_STATUS_INVALID_ARGUMENT, "null system in batch");
      }
      handles[i] = systems[i]->get();
      atom_counts_[i] = systems[i]->atom_count();
    }
    check(qce_batch_prepare(context.get(), handles.data(),
                            static_cast<std::uint32_t>(handles.size()),
                            &method, flags, &handle_));
  }
  ~Batch() { qce_batch_destroy(handle_); }
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  Batch(Batch&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        atom_counts_(std::move(other.atom_counts_)) {}

  [[nodiscard]] std::size_t size() const noexcept { return atom_counts_.size(); }

  std::vector<BatchItemResult> execute(
      const std::vector<std::optional<std::vector<double>>>& coordinates = {}) {
    if (!coordinates.empty() && coordinates.size() != size()) {
      throw Error(QCE_STATUS_INVALID_ARGUMENT,
                  "coordinate list does not match batch size");
    }
    std::vector<qce_batch_input_descriptor> inputs;
    if (!coordinates.empty()) {
      inputs.resize(size());
      for (std::size_t i = 0; i < size(); ++i) {
        inputs[i] = {sizeof(qce_batch_input_descriptor), QCE_ABI_VERSION,
                     coordinates[i] ? coordinates[i]->data() : nullptr,
                     coordinates[i]
                         ? static_cast<std::uint32_t>(coordinates[i]->size())
                         : 0};
      }
    }

    std::vector<BatchItemResult> results(size());
    std::vector<qce_batch_item_result_descriptor> native(size());
    for (std::size_t i = 0; i < size(); ++i) {
      results[i].forces.resize(static_cast<std::size_t>(atom_counts_[i]) * 3);
      native[i] = {sizeof(qce_batch_item_result_descriptor), QCE_ABI_VERSION,
                   QCE_STATUS_INTERNAL_ERROR, 0.0, results[i].forces.data(),
                   static_cast<std::uint32_t>(results[i].forces.size()), 0,
                   0.0, 0.0, 0, QCE_BACKEND_CPU_REFERENCE, 0, 0, 0};
    }
    check(qce_batch_execute(handle_, inputs.empty() ? nullptr : inputs.data(),
                            static_cast<std::uint32_t>(inputs.size()),
                            native.data(), static_cast<std::uint32_t>(native.size())));
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
      if (native[i].status != QCE_STATUS_SUCCESS) results[i].forces.clear();
    }
    return results;
  }

  void clear_warm_starts() { check(qce_batch_clear_warm_starts(handle_)); }

 private:
  qce_batch* handle_ = nullptr;
  std::vector<std::uint32_t> atom_counts_;
};

}  // namespace qce

#endif
