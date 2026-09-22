// Shared lifecycle for compiler-qualified prepared execution regions.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace vibeqc::runtime {

struct CompiledExecutionBinding {
  std::string qualification;
  int device = 0;
  void* stream = nullptr;
  const void* arena = nullptr;
  const void* library = nullptr;

  bool operator==(const CompiledExecutionBinding& other) const {
    return qualification == other.qualification && device == other.device &&
           stream == other.stream && arena == other.arena && library == other.library;
  }
};

struct CompiledExecutionMetrics {
  std::uint64_t bindings = 0;
  std::uint64_t invalidations = 0;
  std::uint64_t executions = 0;
  std::uint64_t failures = 0;
  std::uint64_t recoveries = 0;
};

class CompiledExecutionRegion {
 public:
  const CompiledExecutionMetrics& metrics() const noexcept { return metrics_; }
  const std::string& reason() const noexcept { return reason_; }
  bool bound() const noexcept { return bound_; }
  bool warmed() const noexcept { return warmed_; }
  bool failed() const noexcept { return failed_; }

  bool matches(const CompiledExecutionBinding& binding) const noexcept {
    return bound_ && binding_ == binding;
  }

  bool bind(CompiledExecutionBinding binding) {
    if (binding.qualification.empty())
      throw std::invalid_argument("compiled execution qualification is empty");
    if (matches(binding)) return false;
    if (bound_) ++metrics_.invalidations;
    binding_ = std::move(binding);
    bound_ = true;
    warmed_ = failed_ = false;
    ++metrics_.bindings;
    reason_ = "compiled execution region bound";
    return true;
  }

  void invalidate() noexcept {
    if (bound_ || warmed_ || failed_) ++metrics_.invalidations;
    binding_ = {};
    bound_ = warmed_ = failed_ = false;
    reason_ = "compiled execution region invalidated";
  }

  void mark_success() {
    require_bound();
    if (failed_)
      throw std::logic_error("failed compiled execution region must recover before success");
    warmed_ = true;
    failed_ = false;
    ++metrics_.executions;
    reason_ = "compiled execution region completed";
  }

  void mark_failure(std::string reason) {
    require_bound();
    failed_ = true;
    ++metrics_.failures;
    reason_ = reason.empty() ? "compiled execution region failed" : std::move(reason);
  }

  void recover() {
    require_bound();
    if (!failed_) return;
    failed_ = false;
    warmed_ = false;
    ++metrics_.recoveries;
    reason_ = "compiled execution region recovered; warmup required";
  }

 private:
  void require_bound() const {
    if (!bound_) throw std::logic_error("compiled execution region is not bound");
  }

  CompiledExecutionBinding binding_;
  CompiledExecutionMetrics metrics_;
  std::string reason_ = "compiled execution region not bound";
  bool bound_ = false;
  bool warmed_ = false;
  bool failed_ = false;
};

}  // namespace vibeqc::runtime
