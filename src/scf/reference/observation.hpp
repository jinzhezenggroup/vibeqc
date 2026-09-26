#pragma once

#include <cstddef>
#include <exception>
#include <limits>

namespace vibeqc::scf::reference::observation {

/** Optional observer of actual reference work. The oracle knows no runtime,
 * file format, CUDA API or timer; its caller owns the thread-local observer.
 * Callbacks must not throw or alter the numerical inputs/outputs. */
struct Observer {
  std::size_t (*begin)(const char*, std::size_t) noexcept;
  void (*end)(std::size_t, int) noexcept;
};
inline thread_local const Observer* active{};
enum class EigenReason {
  unspecified,
  overlap,
  core_guess,
  final_fock,
  reference_export,
  fallback,
  iteration,
  seed_validation
};
inline thread_local EigenReason active_reason = EigenReason::unspecified;

class Reason {
 public:
  explicit Reason(EigenReason value) noexcept : previous_(active_reason) { active_reason = value; }
  ~Reason() { active_reason = previous_; }
  Reason(const Reason&) = delete;
  Reason& operator=(const Reason&) = delete;

 private:
  EigenReason previous_;
};

/** One observed call, including an exceptional exit; a null observer costs
 * only the branch. A sink can reject a sample using the sentinel token. */
class Scope {
 public:
  Scope(const char* name, std::size_t n) noexcept : observer_(active) {
    if (observer_) {
      exceptions_ = std::uncaught_exceptions();
      token_ = observer_->begin(name, n);
    }
  }
  ~Scope() {
    if (observer_ && token_ != std::numeric_limits<std::size_t>::max())
      observer_->end(token_, exceptions_);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  const Observer* observer_{};
  std::size_t token_{};
  int exceptions_{};
};
}  // namespace vibeqc::scf::reference::observation
