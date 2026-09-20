#include "methods/rccsd_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "scf/mean_field.hpp"

namespace vibeqc::methods::detail {
namespace {

bool present(const vibeqc_method_descriptor& d, std::size_t end) { return d.struct_size >= end; }

std::vector<std::size_t> range(std::size_t begin, std::size_t end) {
  std::vector<std::size_t> result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}

std::vector<double> positions(const core::System& system) {
  std::vector<double> result;
  result.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    result.insert(result.end(), atom.position.begin(), atom.position.end());
  return result;
}

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
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
  } catch (const std::length_error&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

cc::SolverOptions cc_options(const vibeqc_method_descriptor& d, std::size_t budget) {
  cc::SolverOptions options;
  options.max_bytes = budget;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_max_iterations) +
                     sizeof(d.ccsd_max_iterations)) &&
      d.ccsd_max_iterations)
    options.max_iterations = d.ccsd_max_iterations;
  if (present(d,
              offsetof(vibeqc_method_descriptor, ccsd_diis_history) + sizeof(d.ccsd_diis_history)))
    options.diis_size = d.ccsd_diis_history;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_energy_tolerance) +
                     sizeof(d.ccsd_energy_tolerance)) &&
      d.ccsd_energy_tolerance)
    options.energy_tolerance = d.ccsd_energy_tolerance;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_residual_tolerance) +
                     sizeof(d.ccsd_residual_tolerance)) &&
      d.ccsd_residual_tolerance)
    options.residual_tolerance = d.ccsd_residual_tolerance;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_denominator_threshold) +
                     sizeof(d.ccsd_denominator_threshold)) &&
      d.ccsd_denominator_threshold)
    options.denominator_threshold = d.ccsd_denominator_threshold;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_damping) + sizeof(d.ccsd_damping)))
    options.damping = d.ccsd_damping;
  if (present(d, offsetof(vibeqc_method_descriptor, ccsd_level_shift) + sizeof(d.ccsd_level_shift)))
    options.level_shift = d.ccsd_level_shift;
  cc::validate_options(options);
  return options;
}

scf::ScfOptions reference_options(const vibeqc_method_descriptor& d, std::size_t budget) {
  if (!std::isfinite(d.energy_tolerance) || d.energy_tolerance < 0 ||
      !std::isfinite(d.density_tolerance) || d.density_tolerance < 0)
    throw std::invalid_argument("invalid RCCSD reference convergence threshold");
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
  options.density_fitting_mode = VIBEQC_DENSITY_FITTING_NONE;
  options.precision_mode = VIBEQC_PRECISION_FP64;
  return options;
}

std::size_t correlation_budget(const vibeqc_method_descriptor& d) {
  std::size_t budget = 256ULL << 20;
  if (present(d, offsetof(vibeqc_method_descriptor, correlation_memory_budget_bytes) +
                     sizeof(d.correlation_memory_budget_bytes))) {
    const auto requested = d.correlation_memory_budget_bytes;
    if (requested > static_cast<std::uint64_t>(INT64_MAX) ||
        requested > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument("RCCSD budget exceeds numeric capacity");
    if (requested) budget = static_cast<std::size_t>(requested);
  }
  return budget;
}

void validate_descriptor(const vibeqc_method_descriptor& d, const core::ContextState& context) {
  if (d.screening_tolerance != 0)
    throw std::invalid_argument(
        "canonical RCCSD requires unscreened integrals (screening_tolerance=0)");
  if (present(d, offsetof(vibeqc_method_descriptor, density_fitting_mode) +
                     sizeof(d.density_fitting_mode)) &&
      d.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD density-fitted reference/integrals are not implemented");
  if (present(d, offsetof(vibeqc_method_descriptor, density_fitting_auxiliary_basis) +
                     sizeof(d.density_fitting_auxiliary_basis)) &&
      d.density_fitting_auxiliary_basis)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "conventional RCCSD does not accept an auxiliary basis");
  if (present(d, offsetof(vibeqc_method_descriptor, precision_mode) + sizeof(d.precision_mode))) {
    if (d.precision_mode != VIBEQC_PRECISION_FP64 && d.precision_mode != VIBEQC_PRECISION_AUTO)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
    if (d.precision_mode != VIBEQC_PRECISION_FP64)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "RCCSD requires FP64 precision");
  }
  if (present(d,
              offsetof(vibeqc_method_descriptor, ccsd_frozen_core) + sizeof(d.ccsd_frozen_core)) &&
      d.ccsd_frozen_core != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD frozen-core references are not implemented");
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE &&
      context.requested_backend != VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD requires an explicit CPU or CUDA backend");
}

std::vector<double> fock_mo(const scf::PhysicalReference& ref) {
  const auto n = ref.nbf;
  std::vector<double> scratch(n * n), result(n * n);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu)
        value += ref.fock[mu * n + nu] * ref.coefficients[nu * n + q];
      scratch[mu * n + q] = value;
    }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < n; ++mu)
        value += ref.coefficients[mu * n + p] * scratch[mu * n + q];
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD MO Fock matrix");
      result[p * n + q] = value;
    }
  return result;
}

std::size_t retained_reference_bytes(const scf::PhysicalReference& ref) {
  std::size_t result = 0;
  const std::vector<double>* arrays[] = {
      &ref.overlap, &ref.hcore,           &ref.fock, &ref.coefficients, &ref.orbital_energies,
      &ref.density, &ref.weighted_density};
  for (const auto* values : arrays)
    result = posthf::checked_add(result, posthf::checked_mul(values->capacity(), sizeof(double)));
  return result;
}

cc::Problem build_problem(const core::System& system, const scf::PhysicalReference& ref,
                          const cc::SolverOptions& options, bool cuda, int device) {
  cc::Problem p;
  p.nocc = ref.nocc;
  p.reference_retained_bytes = retained_reference_bytes(ref);
  p.nvir = ref.nbf - ref.nocc;
  p.reference_energy = ref.energy;
  const auto o = p.nocc, v = p.nvir, n = ref.nbf;
  if (!o || !v || ref.orbital_energies.size() != n)
    throw std::invalid_argument("invalid RCCSD canonical reference dimensions");

  const auto fmo = fock_mo(ref);
  p.foo.resize(o * o);
  p.fov.resize(o * v);
  p.fvv.resize(v * v);
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = 0; j < o; ++j) p.foo[i * o + j] = fmo[i * n + j];
    for (std::size_t a = 0; a < v; ++a) p.fov[i * v + a] = fmo[i * n + o + a];
  }
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = 0; b < v; ++b) p.fvv[a * v + b] = fmo[(o + a) * n + o + b];

  p.d1.resize(o * v);
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      const double physical = ref.orbital_energies[i] - ref.orbital_energies[o + a];
      minimum = std::min(minimum, std::abs(physical));
      if (!std::isfinite(physical) || physical >= 0.0 ||
          std::abs(physical) <= options.denominator_threshold)
        throw std::invalid_argument("near-zero or nonnegative physical RCCSD denominator");
      p.d1[i * v + a] = physical - options.level_shift;
    }
  const auto n2 = o * o * v * v;
  p.d2.resize(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const double physical = (ref.orbital_energies[i] - ref.orbital_energies[o + a]) +
                                  (ref.orbital_energies[j] - ref.orbital_energies[o + b]);
          minimum = std::min(minimum, std::abs(physical));
          if (!std::isfinite(physical) || physical >= 0.0 ||
              std::abs(physical) <= options.denominator_threshold)
            throw std::invalid_argument(
                "near-zero or nonnegative physical RCCSD doubles denominator");
          p.d2[((i * o + j) * v + a) * v + b] = physical - 2.0 * options.level_shift;
        }

  p.minimum_absolute_denominator = minimum;
  posthf::RawSource source(system);
  posthf::NativeBlockProvider provider(source, ref, options.max_bytes, 2);
  const auto occ = range(0, o), vir = range(o, n);
  std::size_t retained = 0, peak = p.reference_retained_bytes;
  auto fetch = [&](posthf::MOSlots slots, std::array<std::size_t, 4> shape) {
    const auto plan = provider.plan(shape, cuda);
    auto live = posthf::checked_add(
        p.reference_retained_bytes,
        posthf::checked_add(retained, posthf::checked_add(plan.host_bytes, plan.device_bytes)));
    peak = std::max(peak, live);
    if (peak > options.max_bytes)
      throw std::length_error(
          "RCCSD reference, MO provider, and retained blocks exceed memory budget");
    auto values = provider.get(slots, cuda, device);
    retained = posthf::checked_add(retained, posthf::checked_mul(values.size(), sizeof(double)));
    peak = std::max(peak, posthf::checked_add(p.reference_retained_bytes, retained));
    return values;
  };
  p.ovov = fetch({occ, vir, occ, vir}, {o, v, o, v});
  p.ovvo = fetch({occ, vir, vir, occ}, {o, v, v, o});
  p.oovv = fetch({occ, occ, vir, vir}, {o, o, v, v});
  p.ovvv = fetch({occ, vir, vir, vir}, {o, v, v, v});
  p.ovoo = fetch({occ, vir, occ, occ}, {o, v, o, o});
  p.oooo = fetch({occ, occ, occ, occ}, {o, o, o, o});
  p.vvvv = fetch({vir, vir, vir, vir}, {v, v, v, v});
  p.provider_peak_bytes = peak;
  p.provider_host_bytes = retained;

  p.initial_t1.assign(o * v, 0.0);
  p.initial_t2.resize(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto t = ((i * o + j) * v + a) * v + b;
          const auto g = ((i * v + a) * o + j) * v + b;
          p.initial_t2[t] = p.ovov[g] / p.d2[t];
          if (!std::isfinite(p.initial_t2[t]))
            throw std::runtime_error("nonfinite RCCSD MP2-like initial amplitude");
        }
  return p;
}

class RccsdPrepared final : public PreparedCalculation {
 public:
  RccsdPrepared(Capabilities capabilities, core::ContextState& context, core::System system,
                scf::ScfOptions reference_options, cc::SolverOptions solver_options,
                std::size_t reference_capacity)
      : capabilities_(capabilities),
        context_(context),
        system_(std::move(system)),
        reference_options_(reference_options),
        solver_options_(solver_options),
        reference_capacity_(reference_capacity) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
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
    if (compute_forces)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "RCCSD exposes energy only; analytic forces are not implemented");
    const char* allocation_stage = "HF reference";
    try {
      const bool cuda = context_.requested_backend == VIBEQC_BACKEND_CUDA;
      auto hf = cuda ? scf::run_rhf_cuda(system_, reference_options_, context_.device_id)
                     : scf::run_rhf(system_, reference_options_);
      if (!hf.converged || !hf.reference)
        throw MethodError(VIBEQC_STATUS_NOT_CONVERGED,
                          "HF did not converge; no RCCSD energy evaluated");
      const auto reference = hf.reference;
      hf.density.clear();
      hf.density.shrink_to_fit();
      allocation_stage = "MO provider/problem";
      const auto problem =
          build_problem(system_, *reference, solver_options_, cuda, context_.device_id);
      allocation_stage = "CC resident solve";
      auto solved = cuda ? cc::solve_cuda(problem, solver_options_, context_.device_id)
                         : cc::solve_cpu(problem, solver_options_);

      vibeqc_correlation_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(diagnostic);
      diagnostic.abi_version = VIBEQC_ABI_VERSION;
      diagnostic.reference_energy = reference->energy;
      diagnostic.reference_residual = reference->commutator_residual;
      diagnostic.minimum_absolute_denominator = problem.minimum_absolute_denominator;
      diagnostic.numeric_capacity_bytes =
          std::max(reference_capacity_, solved.diagnostic.numeric_capacity_bytes);
      diagnostic.mo_host_staging = cuda ? 1 : 0;
      diagnostic.correlation_owned_device_bytes = solved.diagnostic.owned_device_bytes;
      diagnostic.correlation_provider_retained_bytes = problem.provider_host_bytes;
      // The generic MO-transfer field is reserved for provider telemetry.  The
      // native RCCSD owner reports its solver traffic explicitly below instead
      // of conflating setup/final-amplitude copies with AO->MO staging.
      diagnostic.mo_transfer_bytes = 0;
      diagnostic.tensor_kernel_ms = 0.0;
      std::copy_n(cc::generated::iteration_equation_hash,
                  std::min<std::size_t>(64, std::strlen(cc::generated::iteration_equation_hash)),
                  diagnostic.equation_hash);
      diagnostic.ccsd_iterations = solved.diagnostic.iterations;
      diagnostic.ccsd_diis_restarts = solved.diagnostic.diis_restarts;
      diagnostic.ccsd_correlation_energy = solved.correlation_energy;
      diagnostic.ccsd_energy_change = solved.diagnostic.energy_change;
      diagnostic.ccsd_singles_residual_max = solved.diagnostic.r1_max;
      diagnostic.ccsd_doubles_residual_max = solved.diagnostic.r2_max;
      diagnostic.ccsd_replay_singles_residual_max = solved.diagnostic.replay_r1_max;
      diagnostic.ccsd_replay_doubles_residual_max = solved.diagnostic.replay_r2_max;
      diagnostic.ccsd_setup_h2d_bytes = solved.diagnostic.setup_h2d_bytes;
      diagnostic.ccsd_scalar_d2h_bytes = solved.diagnostic.scalar_d2h_bytes;
      diagnostic.ccsd_amplitude_d2h_bytes = solved.diagnostic.amplitude_d2h_bytes;
      diagnostic.ccsd_synchronizations = solved.diagnostic.synchronizations;
      std::copy_n(cc::generated::replay_equation_hash,
                  std::min<std::size_t>(64, std::strlen(cc::generated::replay_equation_hash)),
                  diagnostic.ccsd_replay_equation_hash);
      last_ = diagnostic;

      if (solved.status == cc::SolveStatus::NumericalFailure)
        throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, solved.reason);

      Result result;
      result.energy = solved.total_energy;
      result.convergence = {solved.diagnostic.iterations, solved.diagnostic.energy_change,
                            std::max(solved.diagnostic.r1_max, solved.diagnostic.r2_max),
                            solved.converged()};
      result.executed_backend = cuda ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE;
      return result;
    } catch (const std::length_error& error) {
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY, error.what());
    } catch (const std::bad_alloc&) {
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                        std::string("RCCSD ") + allocation_stage + " allocation failed");
    }
  }

 private:
  Capabilities capabilities_;
  core::ContextState& context_;
  core::System system_;
  scf::ScfOptions reference_options_;
  cc::SolverOptions solver_options_;
  std::size_t reference_capacity_{};
  std::optional<vibeqc_correlation_diagnostic> last_;
  mutable std::mutex mutex_;
};

class RccsdPreparedBatch final : public PreparedBatch {
 public:
  RccsdPreparedBatch(Capabilities capabilities, core::ContextState& context,
                     std::vector<core::System> systems, const vibeqc_method_descriptor& descriptor)
      : capabilities_(capabilities), context_(&context), systems_(std::move(systems)) {
    const auto bytes = std::min<std::size_t>(descriptor.struct_size, sizeof(descriptor_));
    std::memcpy(&descriptor_, &descriptor, bytes);
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_rccsd_calculation(capabilities_, *context_, system, descriptor_));
      owner_coordinates_.push_back(positions(system));
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }
  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }
  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (compute_forces)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "RCCSD batch exposes energy only; forces are unsupported");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("RCCSD batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& result = results[index];
      result.bucket_id = 0;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      result.calculation.executed_backend = context_->requested_backend;
      try {
        auto target = systems_[index];
        auto target_coordinates = positions(target);
        if (!coordinates.empty() && coordinates[index]) {
          if (!valid_positions(*coordinates[index], target))
            throw std::invalid_argument("invalid RCCSD batch item coordinates");
          target_coordinates = *coordinates[index];
          set_positions(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          auto candidate = prepare_rccsd_calculation(capabilities_, *context_, target, descriptor_);
          owners_[index] = std::move(candidate);
          owner_coordinates_[index] = std::move(target_coordinates);
        }
        result.calculation = owners_[index]->execute(false);
        result.status = result.calculation.convergence.converged ? VIBEQC_STATUS_SUCCESS
                                                                 : VIBEQC_STATUS_NOT_CONVERGED;
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

  void clear_warm_starts() override {}
  std::size_t warm_density_size(std::size_t) const override { return 0; }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t) const override {
    static const std::optional<scf::HfWarmState> empty;
    return empty;
  }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>>) override {}
  void set_warm_start_updates(bool) override {}
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    return {};
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
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

vibeqc_status validate_rccsd_system(vibeqc_method, const core::System& system,
                                    std::string& detail) {
  if (!system.ecp_terms.empty()) {
    detail = "RCCSD with ECP is not implemented";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    detail = "RCCSD reference/provider validation supports shells through f";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (system.multiplicity != 1 || system.electron_count <= 0 || system.electron_count % 2) {
    detail = "RCCSD supports real closed-shell all-electron RHF references only";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (static_cast<std::size_t>(system.electron_count / 2) >= molecule::ao_count(system)) {
    detail = "RCCSD requires a nonempty virtual orbital space";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  validate_descriptor(descriptor, context);
  const auto budget = correlation_budget(descriptor);
  auto solver_options = cc_options(descriptor, budget);
  auto reference = reference_options(descriptor, budget);
  const auto reference_capacity = posthf::rhf_reference_capacity(
      system, reference.diis_history, context.requested_backend == VIBEQC_BACKEND_CPU_REFERENCE);
  if (reference_capacity > budget)
    throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                      "RCCSD bounded RHF reference exceeds correlation memory budget");
  return std::make_unique<RccsdPrepared>(capabilities, context, system, reference, solver_options,
                                         reference_capacity);
}

std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities& capabilities,
                                                   core::ContextState& context,
                                                   std::vector<core::System> systems,
                                                   const vibeqc_method_descriptor& descriptor,
                                                   vibeqc_batch_flags flags) {
  constexpr auto supported_flags = static_cast<vibeqc_batch_flags>(VIBEQC_BATCH_ENABLE_WARM_STARTS);
  if ((flags & ~supported_flags) != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD prepared batches support the warm-start compatibility flag only; "
                      "shell/eigensolver profiling is unavailable");
  // The generic Python/C++ batch API enables its warm-start compatibility bit
  // by default.  RCCSD accepts that ABI contract but deliberately starts every
  // solve from the deterministic MP2-like amplitudes: dimensions alone do not
  // establish orbital compatibility, so geometry changes never reuse T1/T2.
  if (systems.empty()) throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "RCCSD batch is empty");
  const auto nbf = molecule::ao_count(systems.front());
  const auto nocc = static_cast<std::size_t>(systems.front().electron_count / 2);
  for (const auto& system : systems) {
    if (molecule::ao_count(system) != nbf ||
        static_cast<std::size_t>(system.electron_count / 2) != nocc)
      throw MethodError(
          VIBEQC_STATUS_NOT_IMPLEMENTED,
          "RCCSD prepared batch requires one homogeneous (nocc,nvir) shape; split ragged groups");
  }
  return std::make_unique<RccsdPreparedBatch>(capabilities, context, std::move(systems),
                                              descriptor);
}

}  // namespace vibeqc::methods::detail
