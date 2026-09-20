#include "methods/mp2_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>

#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_energy.hpp"
#include "posthf/mp2_force.hpp"
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
              std::size_t reference_capacity, double threshold, bool density_fitted,
              bool fitted_cuda)
      : caps_(caps),
        context_(context),
        system_(std::move(system)),
        auxiliary_(std::move(auxiliary)),
        options_(options),
        budget_(budget),
        reference_capacity_(reference_capacity),
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
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    try {
      const bool cuda = context_.requested_backend == VIBEQC_BACKEND_CUDA;
      const bool execution_cuda = density_fitted_ ? fitted_cuda_ : cuda;
      if (!cuda && context_.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE)
        throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                          "MP2 requires an explicit CPU or CUDA backend");
      if (compute_forces && density_fitted_)
        throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                          "RI-MP2 analytic force is C2 work and is not implemented");
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
      std::optional<mp2::ConventionalForceResult> force_diagnostic;
      if (compute_forces) {
        response::GmresOptions response_options;
        response_options.relative_tolerance = 1e-10;
        response_options.absolute_tolerance = 1e-12;
        response_options.restart = 30;
        response_options.max_iterations = 200;
        response_options.max_workspace_bytes = budget_;
        force_diagnostic =
            cuda ? mp2::conventional_force_cuda(ref, source, budget_, threshold_, 1e-10,
                                                response_options, context_.device_id)
                 : mp2::conventional_force_cpu(ref, source, budget_, threshold_, 1e-10,
                                               response_options);
        result.forces = force_diagnostic->forces;
      }
      result.convergence = {hf.iterations, hf.energy_change, ref.commutator_residual, true};
      const bool executed_cuda = execution_cuda;
      result.executed_backend = executed_cuda ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE;
      vibeqc_correlation_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(vibeqc_correlation_diagnostic);
      diagnostic.abi_version = VIBEQC_ABI_VERSION;
      diagnostic.reference_energy = ref.energy;
      diagnostic.opposite_spin_energy = corr.opposite_spin;
      diagnostic.same_spin_energy = corr.same_spin;
      diagnostic.minimum_absolute_denominator = corr.minimum_denominator;
      diagnostic.reference_residual = ref.commutator_residual;
      diagnostic.numeric_capacity_bytes =
          std::max(reference_capacity_, corr.numeric_capacity_bytes);
      diagnostic.energy_tile_count = corr.tiles;
      diagnostic.mo_host_staging = executed_cuda ? 1 : 0;
      last_ = diagnostic;
      last_->correlation_owned_device_bytes = corr.metrics.owned_device_bytes;
      last_->correlation_provider_retained_bytes = corr.metrics.provider_retained_bytes;
      last_->mo_transfer_bytes = corr.mo_transfer_bytes;
      last_->host_to_device_ms = corr.metrics.input_ms;
      last_->device_to_host_ms = corr.metrics.output_ms;
      last_->transform_library_ms = corr.metrics.library_ms;
      last_->tensor_kernel_ms = corr.metrics.kernel_ms;
      std::copy_n(corr.equation_hash, 64, last_->equation_hash);
      if (force_diagnostic) {
        last_->response_iterations = force_diagnostic->response.iterations;
        last_->response_restarts = force_diagnostic->response.restarts;
        last_->response_absolute_residual = force_diagnostic->response.residual_norm;
        last_->response_relative_residual = force_diagnostic->response.relative_residual;
        last_->response_workspace_bytes = force_diagnostic->response.workspace_bytes;
        last_->measured_response_workspace_peak_bytes =
            force_diagnostic->response.measured_workspace_peak_bytes;
        last_->response_workspace_allocation_count =
            force_diagnostic->response.workspace_allocation_count;
        last_->derivative_workspace_bytes = force_diagnostic->derivative_workspace_bytes;
        last_->planned_endpoint_peak_bytes =
            std::max(reference_capacity_, force_diagnostic->planned_endpoint_peak_bytes);
        // Preserve the producer's unavailable-measurement sentinel. A planned
        // reference capacity cannot turn an unmeasured endpoint into an observation.
        last_->measured_endpoint_peak_bytes = force_diagnostic->measured_endpoint_peak_bytes;
        last_->numeric_capacity_bytes =
            std::max(last_->numeric_capacity_bytes, last_->planned_endpoint_peak_bytes);
        last_->force_provenance_flags = 0x7;
        constexpr char response_hash[] = "rhf-canonical-response-v1";
        std::copy_n(response_hash, sizeof(response_hash), last_->response_operator_hash);
      }
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
  std::size_t reference_capacity_;
  double threshold_;
  bool density_fitted_{};
  bool fitted_cuda_{};
  std::optional<vibeqc_correlation_diagnostic> last_;
  mutable std::mutex mutex_;
};

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<double> positions(const core::System& system) {
  std::vector<double> result;
  result.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    result.insert(result.end(), atom.position.begin(), atom.position.end());
  return result;
}

void set_positions(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    std::copy_n(coordinates.begin() + 3 * atom, 3, system.atoms[atom].position.begin());
}

vibeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

class Mp2PreparedBatch final : public PreparedBatch {
 public:
  Mp2PreparedBatch(Capabilities capabilities, core::ContextState& context,
                   std::vector<core::System> systems, const vibeqc_method_descriptor& descriptor)
      : capabilities_(capabilities), context_(&context), systems_(std::move(systems)) {
    const auto bytes = std::min<std::size_t>(descriptor.struct_size, sizeof(descriptor_));
    std::memcpy(&descriptor_, &descriptor, bytes);
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_mp2_calculation(capabilities_, *context_, system, descriptor_));
      owner_coordinates_.push_back(positions(system));
    }
  }

  [[nodiscard]] std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("MP2 batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& result = results[index];
      result.bucket_id = index;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      result.calculation.executed_backend = context_->requested_backend;
      try {
        auto target = systems_[index];
        auto target_coordinates = positions(target);
        if (!coordinates.empty() && coordinates[index]) {
          if (!valid_positions(*coordinates[index], target))
            throw std::invalid_argument("invalid MP2 batch item coordinates");
          target_coordinates = *coordinates[index];
          set_positions(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          auto candidate = prepare_mp2_calculation(capabilities_, *context_, target, descriptor_);
          owners_[index] = std::move(candidate);
          owner_coordinates_[index] = std::move(target_coordinates);
        }
        result.calculation = owners_[index]->execute(compute_forces);
        result.status = VIBEQC_STATUS_SUCCESS;
      } catch (...) {
        result.status = item_exception_status();
      }
    }
    return results;
  }

  std::optional<vibeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const override {
    if (index >= owners_.size())
      throw std::invalid_argument("correlation diagnostic batch index is out of range");
    return owners_[index]->correlation_diagnostic();
  }

  void clear_warm_starts() override {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "MP2 batch does not support warm starts");
  }
  [[nodiscard]] std::size_t warm_density_size(std::size_t) const override { return 0; }
  [[nodiscard]] const std::optional<scf::HfWarmState>& warm_state(std::size_t) const override {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "MP2 batch does not support warm starts");
  }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>>) override {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "MP2 batch does not support warm starts");
  }
  void set_warm_start_updates(bool) override {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "MP2 batch does not support warm starts");
  }
  [[nodiscard]] std::optional<std::vector<DirectShellClassProfileEntry>>
  last_direct_shell_class_profile() const override {
    return std::nullopt;
  }
  [[nodiscard]] std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile()
      const override {
    return std::nullopt;
  }
  [[nodiscard]] std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override {
    return {};
  }
  [[nodiscard]] std::vector<scf::CudaDensityFittingMetricDiagnostic>
  last_density_fitting_metric_diagnostics() const override {
    return {};
  }
  [[nodiscard]] std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile()
      const override {
    return {};
  }

 private:
  Capabilities capabilities_;
  core::ContextState* context_{};
  std::vector<core::System> systems_;
  vibeqc_method_descriptor descriptor_{};
  std::vector<std::unique_ptr<PreparedCalculation>> owners_;
  std::vector<std::vector<double>> owner_coordinates_;
};
}  // namespace

vibeqc_status validate_mp2_system(vibeqc_method, const core::System& system, std::string& detail) {
  if (system.shells.empty()) {
    detail = "canonical MP2 requires an explicit Gaussian orbital basis";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
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
  if (!(options.density_fitting_relative_threshold > 0.0) ||
      !(options.density_fitting_relative_threshold < 1.0) ||
      !std::isfinite(options.density_fitting_relative_threshold))
    throw std::invalid_argument("RI-MP2 metric threshold must lie in (0,1)");
  const bool cpu_conventional_reference =
      !density_fitted && context.requested_backend == VIBEQC_BACKEND_CPU_REFERENCE;
  const std::size_t standalone_reference_capacity =
      posthf::rhf_reference_capacity(system, options.diis_history, cpu_conventional_reference);
  if (standalone_reference_capacity > budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "MP2 bounded reference exceeds numeric memory budget");
  if (fitted_cuda) {
    if (standalone_reference_capacity == budget)
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                        "RI-MP2 CUDA reference leaves no density-fitting plan budget");
    const std::size_t remaining = budget - standalone_reference_capacity;
    options.density_fitting_memory_budget_bytes =
        requested_density_fitting_budget == 0
            ? remaining
            : std::min(requested_density_fitting_budget, remaining);
  } else {
    options.density_fitting_memory_budget_bytes =
        requested_density_fitting_budget == 0 ? budget
                                              : std::min(requested_density_fitting_budget, budget);
  }
  if (density_fitted &&
      posthf::ri_mp2_capacity(system, *auxiliary,
                              static_cast<std::size_t>(system.electron_count / 2)) > budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "RI-MP2 reference and correlation exceed numeric memory budget");
  const std::size_t reference_capacity =
      density_fitted && !fitted_cuda
          ? posthf::ri_mp2_reference_capacity(system, *auxiliary, options.diis_history)
      : fitted_cuda ? posthf::checked_add(standalone_reference_capacity,
                                          options.density_fitting_memory_budget_bytes)
                    : standalone_reference_capacity;
  if (reference_capacity > budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "RI-MP2 DF reference state exceeds numeric memory budget");
  return std::make_unique<Mp2Prepared>(caps, context, system, std::move(auxiliary), options, budget,
                                       reference_capacity, threshold, density_fitted, fitted_cuda);
}

std::unique_ptr<PreparedBatch> prepare_mp2_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags) {
  if ((flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "MP2 batch does not support warm starts");
  constexpr vibeqc_batch_flags profiling_flags = VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING |
                                                 VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING;
  if ((flags & profiling_flags) != 0)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "MP2 batch does not support profiling");
  if ((flags & ~(VIBEQC_BATCH_ENABLE_WARM_STARTS | profiling_flags)) != 0)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unsupported MP2 batch flag");
  const auto present = [&](std::size_t end) { return descriptor.struct_size >= end; };
  const auto density_fitting_mode =
      present(offsetof(vibeqc_method_descriptor, density_fitting_mode) +
              sizeof(descriptor.density_fitting_mode))
          ? descriptor.density_fitting_mode
          : VIBEQC_DENSITY_FITTING_NONE;
  if (density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "MP2 batch supports conventional correlation only");
  if (present(offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis) +
              sizeof(descriptor.density_fitting_auxiliary_basis)) &&
      descriptor.density_fitting_auxiliary_basis)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "conventional MP2 batch does not accept an auxiliary basis");
  return std::make_unique<Mp2PreparedBatch>(capabilities, context, std::move(systems), descriptor);
}
}  // namespace vibeqc::methods::detail
