#include "methods/xtb_method.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "methods/gfn2_runtime_bridge.hpp"

namespace vibeqc::methods::detail {
namespace {

[[noreturn]] void throw_gfn2_runtime(Gfn2RuntimeStatus status, const char* stage,
                                     const std::string& detail = {}) {
  vibeqc_status mapped = VIBEQC_STATUS_INTERNAL_ERROR;
  switch (status) {
    case Gfn2RuntimeStatus::kInvalidArgument:
      mapped = VIBEQC_STATUS_INVALID_ARGUMENT;
      break;
    case Gfn2RuntimeStatus::kBackendUnavailable:
    case Gfn2RuntimeStatus::kNotSupported:
    case Gfn2RuntimeStatus::kNotImplemented:
      mapped = VIBEQC_STATUS_NOT_IMPLEMENTED;
      break;
    case Gfn2RuntimeStatus::kAllocationFailed:
      mapped = VIBEQC_STATUS_OUT_OF_MEMORY;
      break;
    case Gfn2RuntimeStatus::kNotConverged:
      mapped = VIBEQC_STATUS_NOT_CONVERGED;
      break;
    case Gfn2RuntimeStatus::kEigensolverFailed:
      mapped = VIBEQC_STATUS_NUMERICAL_FAILURE;
      break;
    case Gfn2RuntimeStatus::kSuccess:
    case Gfn2RuntimeStatus::kInternalError:
      mapped = VIBEQC_STATUS_INTERNAL_ERROR;
      break;
  }
  std::string message = std::string("GFN2-xTB ") + stage + " failed: ";
  message += detail.empty() ? gfn2_runtime_status_name(status) : detail;
  throw MethodError(mapped, message);
}

Gfn2RuntimeBackend runtime_backend(vibeqc_backend backend) {
  if (backend == VIBEQC_BACKEND_CPU_REFERENCE) return Gfn2RuntimeBackend::kCpu;
  if (backend == VIBEQC_BACKEND_CUDA) return Gfn2RuntimeBackend::kCuda;
  throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unsupported GFN2-xTB execution backend");
}

class Gfn2PreparedCalculation final : public PreparedCalculation {
 public:
  Gfn2PreparedCalculation(const Capabilities& capabilities, const core::System& system,
                          const vibeqc_method_descriptor& descriptor, vibeqc_backend backend,
                          int device_id)
      : capabilities_(capabilities),
        atoms_(system.atoms),
        charge_(system.charge),
        multiplicity_(system.multiplicity),
        backend_(backend),
        maximum_iterations_(descriptor.max_iterations),
        mixer_history_(descriptor.diis_history),
        energy_tolerance_(descriptor.energy_tolerance),
        charge_tolerance_(descriptor.density_tolerance) {
    if (maximum_iterations_ == 0u || maximum_iterations_ > std::numeric_limits<std::int32_t>::max())
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "GFN2-xTB maximum iterations must fit positive int32");
    if (mixer_history_ == 0u || mixer_history_ > 64u)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "GFN2-xTB mixer history must lie in [1, 64]");
    if (!(energy_tolerance_ > 0.0) || !std::isfinite(energy_tolerance_) ||
        !(charge_tolerance_ > 0.0) || !std::isfinite(charge_tolerance_))
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "GFN2-xTB SCC tolerances must be positive finite values");
    runtime_ = std::make_unique<Gfn2RuntimeBridge>(runtime_backend(backend_), device_id);
  }

  [[nodiscard]] std::size_t atom_count() const noexcept override { return atoms_.size(); }
  [[nodiscard]] const Capabilities& capabilities() const noexcept override { return capabilities_; }

  Result execute(bool compute_forces) override {
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> positions;
    atomic_numbers.reserve(atoms_.size());
    positions.reserve(3u * atoms_.size());
    for (const core::Atom& atom : atoms_) {
      atomic_numbers.push_back(atom.atomic_number);
      positions.insert(positions.end(), atom.position.begin(), atom.position.end());
    }

    Gfn2RuntimeRequest request;
    request.atomic_numbers = atomic_numbers;
    request.positions = positions;
    request.charge = charge_;
    request.multiplicity = multiplicity_;
    request.compute_forces = compute_forces;
    request.maximum_iterations = static_cast<std::int32_t>(maximum_iterations_);
    request.mixer_history = static_cast<std::int32_t>(mixer_history_);
    request.energy_tolerance = energy_tolerance_;
    request.charge_tolerance = charge_tolerance_;

    Gfn2RuntimeResult runtime_result = runtime_->execute(request);
    if (runtime_result.status != Gfn2RuntimeStatus::kSuccess)
      throw_gfn2_runtime(runtime_result.status, "execution", runtime_result.detail);

    Result result;
    result.energy = runtime_result.energy;
    result.forces = std::move(runtime_result.forces);
    result.convergence.iterations = runtime_result.iterations;
    result.convergence.energy_change = std::numeric_limits<double>::quiet_NaN();
    result.convergence.residual_rms = std::numeric_limits<double>::quiet_NaN();
    result.convergence.converged = runtime_result.converged;
    result.executed_backend = backend_;
    return result;
  }

 private:
  const Capabilities& capabilities_;
  std::vector<core::Atom> atoms_;
  int charge_{};
  unsigned multiplicity_{1u};
  vibeqc_backend backend_{VIBEQC_BACKEND_CPU_REFERENCE};
  unsigned maximum_iterations_{};
  unsigned mixer_history_{};
  double energy_tolerance_{};
  double charge_tolerance_{};
  std::unique_ptr<Gfn2RuntimeBridge> runtime_;
};

}  // namespace

vibeqc_status validate_xtb_system(vibeqc_method method, const core::System& system,
                                  std::string& detail) {
  if (method != VIBEQC_METHOD_GFN2_XTB) {
    detail = "unsupported semiempirical method identifier";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!system.shells.empty()) {
    detail = "GFN2-xTB uses its intrinsic minimal basis; Gaussian shells must be omitted";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!system.ecp_terms.empty()) {
    detail = "GFN2-xTB does not accept Gaussian ECP descriptors";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (const core::Atom& atom : system.atoms) {
    if (atom.atomic_number < 1 || atom.atomic_number > 86) {
      detail = "GFN2-xTB supports atomic numbers 1 through 86";
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    }
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (spin_excess < 0 || spin_excess > system.electron_count ||
      ((system.electron_count + spin_excess) & 1) != 0) {
    detail = "GFN2-xTB charge and multiplicity do not define integral spin occupations";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_xtb_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE &&
      context.requested_backend != VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unsupported GFN2-xTB execution backend");
#if !defined(VIBEQC_HAS_GFN2_CUDA)
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "GFN2-xTB native CUDA execution is not included in this build");
#endif
  if (descriptor.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "GFN2-xTB does not use Gaussian density fitting");
  if (descriptor.precision_mode != VIBEQC_PRECISION_FP64)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "GFN2-xTB currently exposes FP64 execution only");
  return std::make_unique<Gfn2PreparedCalculation>(capabilities, system, descriptor,
                                                   context.requested_backend, context.device_id);
}

}  // namespace vibeqc::methods::detail
