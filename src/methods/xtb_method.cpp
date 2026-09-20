#include "methods/xtb_method.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gfn2_cpu_execution.hpp"

namespace vibeqc::methods::detail {
namespace {

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

const char* xtb_status_name(xtbloom_status_t status) noexcept {
  switch (status) {
    case XTBLOOM_STATUS_SUCCESS:
      return "success";
    case XTBLOOM_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case XTBLOOM_STATUS_BACKEND_UNAVAILABLE:
      return "backend unavailable";
    case XTBLOOM_STATUS_NOT_SUPPORTED:
      return "not supported";
    case XTBLOOM_STATUS_NOT_IMPLEMENTED:
      return "not implemented";
    case XTBLOOM_STATUS_ALLOCATION_FAILED:
      return "allocation failed";
    case XTBLOOM_STATUS_SCC_NOT_CONVERGED:
      return "SCC not converged";
    case XTBLOOM_STATUS_EIGENSOLVER_FAILED:
      return "eigensolver failed";
    default:
      return "internal error";
  }
}

[[noreturn]] void throw_xtbloom(xtbloom_status_t status, const char* stage,
                                const std::string& detail = {}) {
  vibeqc_status mapped = VIBEQC_STATUS_INTERNAL_ERROR;
  switch (status) {
    case XTBLOOM_STATUS_INVALID_ARGUMENT:
      mapped = VIBEQC_STATUS_INVALID_ARGUMENT;
      break;
    case XTBLOOM_STATUS_BACKEND_UNAVAILABLE:
    case XTBLOOM_STATUS_NOT_SUPPORTED:
    case XTBLOOM_STATUS_NOT_IMPLEMENTED:
      mapped = VIBEQC_STATUS_NOT_IMPLEMENTED;
      break;
    case XTBLOOM_STATUS_ALLOCATION_FAILED:
      mapped = VIBEQC_STATUS_OUT_OF_MEMORY;
      break;
    case XTBLOOM_STATUS_SCC_NOT_CONVERGED:
      mapped = VIBEQC_STATUS_NOT_CONVERGED;
      break;
    case XTBLOOM_STATUS_EIGENSOLVER_FAILED:
      mapped = VIBEQC_STATUS_NUMERICAL_FAILURE;
      break;
    default:
      mapped = VIBEQC_STATUS_INTERNAL_ERROR;
      break;
  }
  std::string message = std::string("GFN2-xTB ") + stage + " failed: ";
  message += detail.empty() ? xtb_status_name(status) : detail;
  throw MethodError(mapped, message);
}

class Gfn2PreparedCalculation final : public PreparedCalculation {
 public:
  Gfn2PreparedCalculation(const Capabilities& capabilities, const core::System& system,
                          const vibeqc_method_descriptor& descriptor)
      : capabilities_(capabilities),
        atoms_(system.atoms),
        charge_(system.charge),
        multiplicity_(system.multiplicity),
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
  }

  [[nodiscard]] std::size_t atom_count() const noexcept override { return atoms_.size(); }
  [[nodiscard]] const Capabilities& capabilities() const noexcept override { return capabilities_; }

  Result execute(bool compute_forces) override {
    std::vector<std::int64_t> atom_offsets{0, static_cast<std::int64_t>(atoms_.size())};
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> positions;
    atomic_numbers.reserve(atoms_.size());
    positions.reserve(3u * atoms_.size());
    for (const core::Atom& atom : atoms_) {
      atomic_numbers.push_back(atom.atomic_number);
      positions.insert(positions.end(), atom.position.begin(), atom.position.end());
    }

    std::vector<double> molecular_charges{static_cast<double>(charge_)};
    std::vector<std::int32_t> unpaired_electrons{static_cast<std::int32_t>(multiplicity_ - 1u)};
    // Public VibeQC GFN2 currently follows xTB/tblite's standard
    // shared-orbital (restricted) open-shell semantics.  Multiplicity controls
    // the unpaired-electron count; two-channel unrestricted GFN2 is a separate
    // capability gate under #560-B and must not be inferred silently.
    std::vector<std::int32_t> spin_channels{1};

    xtbloom_batch_t batch{};
    xtbloom_compute_options_t options{};
    xtbloom_batch_result_t output{};
    batch.struct_size = sizeof(batch);
    batch.api_version = XTBLOOM_API_VERSION;
    options.struct_size = sizeof(options);
    options.api_version = XTBLOOM_API_VERSION;
    output.struct_size = sizeof(output);
    output.api_version = XTBLOOM_API_VERSION;

    batch.batch_size = 1;
    batch.total_atoms = static_cast<std::int64_t>(atoms_.size());
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(molecular_charges);
    batch.unpaired_electrons = input_buffer(unpaired_electrons);
    batch.spin_channels = input_buffer(spin_channels);

    options.model = XTBLOOM_MODEL_GFN2_XTB;
    options.flags = static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY) |
                    (compute_forces ? static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES) : 0u);
    options.max_scc_iterations = static_cast<std::int32_t>(maximum_iterations_);
    options.charge_tolerance = charge_tolerance_;
    options.energy_tolerance = energy_tolerance_;
    options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
    options.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options.scc_mixer_history = static_cast<std::int32_t>(mixer_history_);
    options.scc_mixer_damping = 0.4;

    std::vector<double> energies(1u);
    std::vector<double> forces(compute_forces ? 3u * atoms_.size() : 0u);
    std::vector<std::int32_t> iterations(1u);
    std::vector<std::uint8_t> converged(1u);
    std::vector<std::int32_t> statuses(1u);
    output.energies = output_buffer(energies);
    output.forces = output_buffer(forces);
    output.scc_iterations = output_buffer(iterations);
    output.scc_converged = output_buffer(converged);
    output.per_system_status = output_buffer(statuses);

    std::string execution_error;
    const xtbloom_status_t execution_status = xtbloom::detail::execute_restricted_gfn2_cpu(
        cache_, batch, options, output, execution_error);
    if (execution_status != XTBLOOM_STATUS_SUCCESS)
      throw_xtbloom(execution_status, "execution", execution_error);

    if (statuses[0] == XTBLOOM_STATUS_EIGENSOLVER_FAILED)
      throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, "GFN2-xTB generalized eigensolver failed");
    if (statuses[0] != XTBLOOM_STATUS_SUCCESS && statuses[0] != XTBLOOM_STATUS_SCC_NOT_CONVERGED)
      throw MethodError(VIBEQC_STATUS_INTERNAL_ERROR,
                        "GFN2-xTB returned an unexpected per-system status");

    Result result;
    result.energy = energies[0];
    result.forces = std::move(forces);
    result.convergence.iterations = iterations[0] < 0 ? 0u : static_cast<unsigned>(iterations[0]);
    result.convergence.energy_change = std::numeric_limits<double>::quiet_NaN();
    result.convergence.residual_rms = std::numeric_limits<double>::quiet_NaN();
    result.convergence.converged = statuses[0] == XTBLOOM_STATUS_SUCCESS && converged[0] != 0u;
    result.executed_backend = VIBEQC_BACKEND_CPU_REFERENCE;
    return result;
  }

 private:
  const Capabilities& capabilities_;
  std::vector<core::Atom> atoms_;
  int charge_{};
  unsigned multiplicity_{1u};
  unsigned maximum_iterations_{};
  unsigned mixer_history_{};
  double energy_tolerance_{};
  double charge_tolerance_{};
  xtbloom::detail::Gfn2CpuExecutionCache cache_{1};
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
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "GFN2-xTB CUDA execution is not admitted yet; use device='cpu'");
  if (descriptor.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "GFN2-xTB does not use Gaussian density fitting");
  if (descriptor.precision_mode != VIBEQC_PRECISION_FP64)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "GFN2-xTB currently exposes FP64 execution only");
  return std::make_unique<Gfn2PreparedCalculation>(capabilities, system, descriptor);
}

}  // namespace vibeqc::methods::detail
