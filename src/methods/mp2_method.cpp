#include "methods/mp2_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <optional>

#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_energy.hpp"
#include "scf/mean_field.hpp"
#if VIBEQC_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vibeqc::methods::detail {
namespace {
#if VIBEQC_HAS_CUDA
struct DeviceScope {
  int previous{};
  explicit DeviceScope(int device) {
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
      throw MethodError(VIBEQC_STATUS_CUDA_ERROR, "cannot select MP2 CUDA device");
  }
  ~DeviceScope() { cudaSetDevice(previous); }
};
#endif
class Mp2Prepared final : public PreparedCalculation {
 public:
  Mp2Prepared(Capabilities caps, core::ContextState& context, core::System system,
              std::optional<core::System> auxiliary, scf::ScfOptions options, std::size_t budget,
              double threshold, bool density_fitted, bool fitted_cuda)
      : caps_(caps),
        context_(context),
        system_(std::move(system)),
        auxiliary_(std::move(auxiliary)),
        options_(options),
        budget_(budget),
        threshold_(threshold),
        density_fitted_(density_fitted),
        fitted_cuda_(fitted_cuda) {}
  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return caps_; }
  std::optional<vibeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_;
  }
  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
  }
  Result execute(bool compute_forces) override {
    if (compute_forces) {
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "canonical MP2 implements energy only");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    try {
      const bool cuda = context_.requested_backend == VIBEQC_BACKEND_CUDA;
      const bool execution_cuda = density_fitted_ ? fitted_cuda_ : cuda;
      if (!cuda && context_.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE)
        throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                          "MP2 requires an explicit CPU or CUDA backend");
#if VIBEQC_HAS_CUDA
      std::unique_ptr<DeviceScope> device_scope;
      if (execution_cuda) device_scope = std::make_unique<DeviceScope>(context_.device_id);
#else
      if (execution_cuda)
        throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "CUDA MP2 is not compiled");
#endif
      auto hf = density_fitted_
                    ? (fitted_cuda_ ? scf::run_rhf_density_fitting_cuda(
                                          system_, *auxiliary_, options_, context_.device_id)
                                    : scf::run_rhf_density_fitting(system_, *auxiliary_, options_))
                    : (cuda ? scf::run_rhf_cuda(system_, options_, context_.device_id)
                            : scf::run_rhf(system_, options_));
      if (!hf.converged || !hf.reference)
        throw MethodError(VIBEQC_STATUS_NOT_CONVERGED,
                          "HF did not converge; no MP2 energy evaluated");
      const auto& ref = *hf.reference;
      // The HF source/iteration work has been released. Only its owned
      // physical reference enters the correlation phase.
      hf.density.clear();
      hf.density.shrink_to_fit();
      posthf::RawSource source(system_, auxiliary_ ? &*auxiliary_ : nullptr);
      const auto corr =
          density_fitted_ ? mp2::density_fitted_energy(ref, source, budget_, threshold_,
                                                       options_.density_fitting_relative_threshold,
                                                       8, fitted_cuda_, context_.device_id)
                          : mp2::conventional_energy(ref, source, budget_, threshold_, 8, cuda,
                                                     context_.device_id);
      Result result;
      result.energy = ref.energy + corr.opposite_spin + corr.same_spin;
      if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite MP2 total energy");
      result.convergence = {hf.iterations, hf.energy_change, ref.commutator_residual, true};
      const bool executed_cuda = execution_cuda;
      result.executed_backend = executed_cuda ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE;
      last_ = vibeqc_correlation_diagnostic{
          sizeof(vibeqc_correlation_diagnostic),
          VIBEQC_ABI_VERSION,
          ref.energy,
          corr.opposite_spin,
          corr.same_spin,
          corr.minimum_denominator,
          ref.commutator_residual,
          std::max(ref.numeric_capacity_bytes, corr.numeric_capacity_bytes),
          corr.tiles,
          executed_cuda ? 1 : 0};
      last_->correlation_owned_device_bytes = corr.metrics.owned_device_bytes;
      last_->correlation_provider_retained_bytes = corr.metrics.provider_retained_bytes;
      last_->mo_transfer_bytes = corr.mo_transfer_bytes;
      last_->host_to_device_ms = corr.metrics.input_ms;
      last_->device_to_host_ms = corr.metrics.output_ms;
      last_->transform_library_ms = corr.metrics.library_ms;
      last_->tensor_kernel_ms = corr.metrics.kernel_ms;
      std::copy_n(corr.equation_hash, 64, last_->equation_hash);
      return result;
    } catch (const std::length_error& e) {
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY, e.what());
    }
  }

 private:
  Capabilities caps_;
  core::ContextState& context_;
  core::System system_;
  std::optional<core::System> auxiliary_;
  scf::ScfOptions options_;
  std::size_t budget_;
  double threshold_;
  bool density_fitted_{};
  bool fitted_cuda_{};
  std::optional<vibeqc_correlation_diagnostic> last_;
  mutable std::mutex mutex_;
};
}  // namespace

vibeqc_status validate_mp2_system(vibeqc_method, const core::System& system, std::string& detail) {
  // The canonical reference/provider gates cover all-electron systems only.
  // Enabling ECP HF must not silently extend that correlated-method domain.
  if (!system.ecp_terms.empty()) {
    detail = "canonical MP2 with ECP is not implemented";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    detail = "canonical MP2 reference/provider validation supports shells through f";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (system.multiplicity != 1 || system.electron_count <= 0 || system.electron_count % 2) {
    detail = "MP2 supports real closed-shell all-electron RHF only";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (static_cast<std::size_t>(system.electron_count / 2) >= molecule::ao_count(system)) {
    detail = "MP2 reference requires a nonempty virtual space";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_mp2_calculation(const Capabilities& caps,
                                                             core::ContextState& context,
                                                             const core::System& system,
                                                             const vibeqc_method_descriptor& d) {
  auto present = [&](std::size_t end) { return d.struct_size >= end; };
  const auto density_fitting_mode =
      present(offsetof(vibeqc_method_descriptor, density_fitting_mode) +
              sizeof(d.density_fitting_mode))
          ? d.density_fitting_mode
          : VIBEQC_DENSITY_FITTING_NONE;
  if (density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE &&
      density_fitting_mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
      density_fitting_mode != VIBEQC_DENSITY_FITTING_CUDA &&
      density_fitting_mode != VIBEQC_DENSITY_FITTING_AUTO)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown RI-MP2 execution mode");
  const bool density_fitted = density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE;
  const bool context_cuda = context.requested_backend == VIBEQC_BACKEND_CUDA;
  if (density_fitting_mode == VIBEQC_DENSITY_FITTING_CUDA && !context_cuda)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "CUDA RI-MP2 requires a CUDA execution context");
  const bool fitted_cuda = density_fitted && context_cuda &&
                           density_fitting_mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE;
  if (d.screening_tolerance != 0)
    throw std::invalid_argument(
        "canonical MP2 requires unscreened integrals (screening_tolerance=0)");
  if (present(offsetof(vibeqc_method_descriptor, precision_mode) + sizeof(d.precision_mode))) {
    if (d.precision_mode != VIBEQC_PRECISION_FP64 && d.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (d.precision_mode != VIBEQC_PRECISION_FP64)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "canonical MP2 requires FP64 precision");
  }
  std::optional<core::System> auxiliary;
  if (density_fitted) {
    auxiliary = present(offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis) +
                        sizeof(d.density_fitting_auxiliary_basis)) &&
                        d.density_fitting_auxiliary_basis
                    ? d.density_fitting_auxiliary_basis->data
                    : system;
    if (auxiliary->atoms.size() != system.atoms.size())
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "RI-MP2 auxiliary basis must contain the same atoms");
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
      if (auxiliary->atoms[atom].atomic_number != system.atoms[atom].atomic_number ||
          auxiliary->atoms[atom].position != system.atoms[atom].position)
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "RI-MP2 auxiliary basis must share the orbital geometry");
    }
    for (const auto& shell : auxiliary->shells) {
      if (shell.atom_index >= system.atoms.size())
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "RI-MP2 auxiliary shell atom is out of range");
    }
    auxiliary->atoms = system.atoms;
    auxiliary->charge = system.charge;
    auxiliary->multiplicity = system.multiplicity;
    auxiliary->electron_count = system.electron_count;
  } else if (present(offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis) +
                     sizeof(d.density_fitting_auxiliary_basis)) &&
             d.density_fitting_auxiliary_basis) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "an auxiliary basis requires an explicit RI-MP2 mode");
  }
  std::size_t budget = 256ULL << 20;
  if (present(offsetof(vibeqc_method_descriptor, correlation_memory_budget_bytes) +
              sizeof(d.correlation_memory_budget_bytes)) &&
      d.correlation_memory_budget_bytes)
    budget = d.correlation_memory_budget_bytes;
  if (budget > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("MP2 budget exceeds signed-64-bit numeric capacity");
  double threshold = 1e-10;
  if (present(offsetof(vibeqc_method_descriptor, mp2_denominator_threshold) +
              sizeof(d.mp2_denominator_threshold)) &&
      d.mp2_denominator_threshold != 0)
    threshold = d.mp2_denominator_threshold;
  if (!std::isfinite(threshold) || threshold <= 0)
    throw std::invalid_argument("invalid MP2 denominator threshold");
  if (!std::isfinite(d.energy_tolerance) || d.energy_tolerance < 0 ||
      !std::isfinite(d.density_tolerance) || d.density_tolerance < 0)
    throw std::invalid_argument("invalid MP2 reference convergence threshold");
  scf::ScfOptions options;
  options.max_iterations = d.max_iterations ? d.max_iterations : 100;
  options.diis_history = d.diis_history ? d.diis_history : 8;
  options.energy_tolerance = d.energy_tolerance > 0 ? std::min(d.energy_tolerance, 1e-11) : 1e-11;
  options.density_tolerance =
      d.density_tolerance > 0 ? std::min(d.density_tolerance, 1e-11) : 1e-11;
  options.screening_tolerance = 0;
  options.compute_forces = false;
  options.export_physical_reference = true;
  options.reference_memory_budget_bytes = budget;
  options.density_fitting_mode = density_fitting_mode;
  options.density_fitting_relative_threshold =
      present(offsetof(vibeqc_method_descriptor, density_fitting_relative_threshold) +
              sizeof(d.density_fitting_relative_threshold)) &&
              d.density_fitting_relative_threshold != 0
          ? d.density_fitting_relative_threshold
          : 1e-10;
  const std::size_t requested_density_fitting_budget =
      present(offsetof(vibeqc_method_descriptor, density_fitting_memory_budget_bytes) +
              sizeof(d.density_fitting_memory_budget_bytes))
          ? d.density_fitting_memory_budget_bytes
          : 0;
  options.density_fitting_memory_budget_bytes =
      requested_density_fitting_budget == 0 ? budget
                                            : std::min(requested_density_fitting_budget, budget);
  if (!(options.density_fitting_relative_threshold > 0.0) ||
      !(options.density_fitting_relative_threshold < 1.0) ||
      !std::isfinite(options.density_fitting_relative_threshold))
    throw std::invalid_argument("RI-MP2 metric threshold must lie in (0,1)");
  const bool cpu_conventional_reference =
      !density_fitted && context.requested_backend == VIBEQC_BACKEND_CPU_REFERENCE;
  if (posthf::rhf_reference_capacity(system, options.diis_history, cpu_conventional_reference) >
      budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "MP2 bounded reference exceeds numeric memory budget");
  if (density_fitted &&
      posthf::ri_mp2_capacity(system, *auxiliary,
                              static_cast<std::size_t>(system.electron_count / 2)) > budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "RI-MP2 reference and correlation exceed numeric memory budget");
  return std::make_unique<Mp2Prepared>(caps, context, system, std::move(auxiliary), options, budget,
                                       threshold, density_fitted, fitted_cuda);
}
}  // namespace vibeqc::methods::detail
