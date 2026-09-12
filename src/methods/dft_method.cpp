#include "methods/dft_method.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/types.hpp"

namespace vibeqc::methods::detail {
namespace {

bool field_present(const vibeqc_method_descriptor& descriptor, std::size_t offset,
                   std::size_t width) noexcept {
  return descriptor.struct_size >= offset && descriptor.struct_size - offset >= width;
}

scf::ScfOptions dft_options(const vibeqc_method_descriptor& descriptor) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_mode),
                    sizeof(descriptor.density_fitting_mode))) {
    const auto mode = descriptor.density_fitting_mode;
    if (mode != VIBEQC_DENSITY_FITTING_NONE && mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
        mode != VIBEQC_DENSITY_FITTING_CUDA && mode != VIBEQC_DENSITY_FITTING_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown density-fitting execution mode");
    if (mode != VIBEQC_DENSITY_FITTING_NONE)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT RKS supports conventional Coulomb only");
  }
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis),
                    sizeof(descriptor.density_fitting_auxiliary_basis)) &&
      descriptor.density_fitting_auxiliary_basis != nullptr)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT RKS does not accept an unused auxiliary basis");
  if (field_present(descriptor, offsetof(vibeqc_method_descriptor, precision_mode),
                    sizeof(descriptor.precision_mode))) {
    if (descriptor.precision_mode != VIBEQC_PRECISION_FP64 &&
        descriptor.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT RKS supports explicit FP64 precision only");
  }

  scf::FockBuildSpec fock;
  fock.spin = scf::FockSpin::Restricted;
  fock.derivative_order = 0;
  fock.exchange.present = false;
  options.resolved_fock_build =
      scf::resolve_fock_build(fock, scf::FockBackend::Cpu, options.screening_tolerance);
  options.compute_forces = false;
  return options;
}

Result adapt_result(scf::ScfResult native) {
  Result result;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = VIBEQC_BACKEND_CPU_REFERENCE;
  result.fock_builds = native.fock_builds;
  return result;
}

class RksPreparedCalculation final : public PreparedCalculation {
 public:
  RksPreparedCalculation(Capabilities capabilities, core::System system, vibeqc_method method,
                         scf::ScfOptions options)
      : capabilities_(capabilities),
        system_(std::move(system)),
        method_(method),
        options_(std::move(options)),
        fock_(system_, nullptr, *options_.resolved_fock_build),
        basis_(system_),
        grid_(system_) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }

  Result execute(bool compute_forces) override {
    const char* method_name = method_ == VIBEQC_METHOD_PBE_RKS ? "PBE" : "LDA";
    if (compute_forces) {
      throw MethodError(
          VIBEQC_STATUS_NOT_IMPLEMENTED,
          std::string(method_name) + " RKS nuclear gradients are tracked separately in issue #163");
    }
    if (method_ == VIBEQC_METHOD_PBE_RKS)
      return adapt_result(scf::run_pbe_rks(fock_, basis_, grid_, options_));
    return adapt_result(scf::run_lda_rks(fock_, basis_, grid_, options_));
  }

 private:
  Capabilities capabilities_;
  core::System system_;
  vibeqc_method method_{};
  scf::ScfOptions options_;
  scf::PreparedFockPlan fock_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
};

}  // namespace

vibeqc_status validate_dft_system(vibeqc_method method, const core::System& system,
                                  std::string& detail) {
  if (method != VIBEQC_METHOD_LDA_RKS && method != VIBEQC_METHOD_PBE_RKS) {
    detail = "requested DFT method is reserved but not implemented";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (system.electron_count > 0 && system.electron_count % 2 == 0 && system.multiplicity == 1)
    return VIBEQC_STATUS_SUCCESS;
  detail = std::string(method == VIBEQC_METHOD_PBE_RKS ? "PBE" : "LDA") +
           " RKS requires a positive even electron count and spin multiplicity 1";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "DFT RKS is available on the CPU backend only");
  if (descriptor.method != VIBEQC_METHOD_LDA_RKS && descriptor.method != VIBEQC_METHOD_PBE_RKS)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested DFT method is reserved but not implemented");
  return std::make_unique<RksPreparedCalculation>(capabilities, system, descriptor.method,
                                                  dft_options(descriptor));
}

}  // namespace vibeqc::methods::detail
