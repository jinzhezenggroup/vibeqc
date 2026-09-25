#ifndef VIBEQC_METHODS_GFN2_RUNTIME_BRIDGE_HPP
#define VIBEQC_METHODS_GFN2_RUNTIME_BRIDGE_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vibeqc::methods::detail {

enum class Gfn2RuntimeBackend : std::uint8_t { kCpu = 1, kCuda = 2 };

enum class Gfn2RuntimeStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kBackendUnavailable,
  kNotSupported,
  kNotImplemented,
  kAllocationFailed,
  kNotConverged,
  kEigensolverFailed,
  kInternalError,
};

struct Gfn2RuntimeRequest {
  std::span<const std::int32_t> atomic_numbers;
  std::span<const double> positions;
  int charge = 0;
  unsigned multiplicity = 1u;
  bool compute_forces = false;
  bool compute_atomic_charges = false;
  std::int32_t maximum_iterations = 0;
  std::int32_t mixer_history = 0;
  double energy_tolerance = 0.0;
  double charge_tolerance = 0.0;
};

struct Gfn2RuntimeResult {
  Gfn2RuntimeStatus status = Gfn2RuntimeStatus::kInternalError;
  std::string detail;
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  unsigned iterations = 0u;
  bool converged = false;
};

class Gfn2RuntimeBridge {
 public:
  Gfn2RuntimeBridge(Gfn2RuntimeBackend backend, int device_id);
  ~Gfn2RuntimeBridge();

  Gfn2RuntimeBridge(const Gfn2RuntimeBridge&) = delete;
  Gfn2RuntimeBridge& operator=(const Gfn2RuntimeBridge&) = delete;
  Gfn2RuntimeBridge(Gfn2RuntimeBridge&&) noexcept;
  Gfn2RuntimeBridge& operator=(Gfn2RuntimeBridge&&) noexcept;

  [[nodiscard]] Gfn2RuntimeResult execute(const Gfn2RuntimeRequest& request);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* gfn2_runtime_status_name(Gfn2RuntimeStatus status) noexcept;

}  // namespace vibeqc::methods::detail

#endif
