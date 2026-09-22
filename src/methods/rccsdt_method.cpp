#include "methods/rccsdt_method.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "generated_rccsdt_cpu.hpp"
#include "methods/rccsd_method.hpp"
#include "molecule/basis.hpp"

namespace vibeqc::methods::detail {
namespace {

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

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("RCCSD(T) size overflow");
  return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD(T) size overflow");
  return a * b;
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

class RccsdtPrepared final : public PreparedCalculation {
 public:
  RccsdtPrepared(Capabilities capabilities, core::ContextState& context, core::System system,
                 const vibeqc_method_descriptor& descriptor)
      : capabilities_(capabilities), context_(&context), system_(std::move(system)) {
    const auto bytes = std::min<std::size_t>(descriptor.struct_size, sizeof(descriptor_));
    std::memcpy(&descriptor_, &descriptor, bytes);
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
  }

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
                        "native RCCSD(T) analytic forces are not promoted in this owner yet");

    auto state = run_rccsd_native_state(*context_, system_, descriptor_);
    last_ = state.diagnostic;
    if (state.solved.status == cc::SolveStatus::NumericalFailure)
      throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, state.solved.reason);
    if (!state.solved.converged()) return state.result;

    try {
      auto retained = cc::problem_host_bytes(state.problem);
      retained = checked_add(
          retained, checked_mul(state.eps_o.capacity() + state.eps_v.capacity(), sizeof(double)));
      retained = checked_add(
          retained,
          checked_mul(state.solved.t1.capacity() + state.solved.t2.capacity(), sizeof(double)));
      if (retained >= state.budget)
        throw std::length_error(
            "RCCSD(T) retained CC state exhausts the correlation memory budget");

      const auto triples = cc::triples::generated::evaluate(
          state.problem.nocc, state.problem.nvir, state.problem.ovvv.data(),
          state.problem.ovoo.data(), state.problem.ovov.data(), state.problem.fov.data(),
          state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(), state.eps_v.data(),
          descriptor_.ccsd_denominator_threshold ? descriptor_.ccsd_denominator_threshold : 1e-10,
          state.budget - retained);

      auto diagnostic = state.diagnostic;
      diagnostic.minimum_absolute_denominator =
          std::min(diagnostic.minimum_absolute_denominator, triples.minimum_absolute_denominator);
      diagnostic.numeric_capacity_bytes = std::max<std::uint64_t>(
          diagnostic.numeric_capacity_bytes, checked_add(retained, triples.workspace_bytes));
      diagnostic.ccsd_t_triples_energy = triples.energy;
      diagnostic.ccsd_t_virtual_triples = triples.virtual_triples;
      diagnostic.ccsd_t_workspace_bytes = triples.workspace_bytes;
      std::copy_n(cc::triples::generated::inventory_hash,
                  std::min<std::size_t>(64, std::strlen(cc::triples::generated::inventory_hash)),
                  diagnostic.ccsd_t_equation_hash);
      last_ = diagnostic;

      state.result.energy = state.solved.total_energy + triples.energy;
      return state.result;
    } catch (const std::length_error& error) {
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY, error.what());
    } catch (const std::bad_alloc&) {
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                        "RCCSD(T) triples workspace allocation failed");
    }
  }

 private:
  Capabilities capabilities_;
  core::ContextState* context_{};
  core::System system_;
  vibeqc_method_descriptor descriptor_{};
  std::optional<vibeqc_correlation_diagnostic> last_;
  mutable std::mutex mutex_;
};

class RccsdtPreparedBatch final : public PreparedBatch {
 public:
  RccsdtPreparedBatch(Capabilities capabilities, core::ContextState& context,
                      std::vector<core::System> systems, const vibeqc_method_descriptor& descriptor)
      : capabilities_(capabilities), context_(&context), systems_(std::move(systems)) {
    const auto bytes = std::min<std::size_t>(descriptor.struct_size, sizeof(descriptor_));
    std::memcpy(&descriptor_, &descriptor, bytes);
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_rccsdt_calculation(capabilities_, *context_, system, descriptor_));
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
                        "native RCCSD(T) prepared batch exposes energy only in this owner");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("RCCSD(T) batch coordinates do not match system count");

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
            throw std::invalid_argument("invalid RCCSD(T) batch item coordinates");
          target_coordinates = *coordinates[index];
          set_positions(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          owners_[index] =
              prepare_rccsdt_calculation(capabilities_, *context_, target, descriptor_);
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

vibeqc_status validate_rccsdt_system(vibeqc_method method, const core::System& system,
                                     std::string& detail) {
  return validate_rccsd_system(method, system, detail);
}

std::unique_ptr<PreparedCalculation> prepare_rccsdt_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  if (context.requested_backend != VIBEQC_BACKEND_CPU_REFERENCE)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "native RCCSD(T) CUDA owner is not promoted yet; use the CPU backend");
  return std::make_unique<RccsdtPrepared>(capabilities, context, system, descriptor);
}

std::unique_ptr<PreparedBatch> prepare_rccsdt_batch(const Capabilities& capabilities,
                                                    core::ContextState& context,
                                                    std::vector<core::System> systems,
                                                    const vibeqc_method_descriptor& descriptor,
                                                    vibeqc_batch_flags flags) {
  constexpr auto supported_flags = static_cast<vibeqc_batch_flags>(VIBEQC_BATCH_ENABLE_WARM_STARTS);
  if ((flags & ~supported_flags) != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD(T) prepared batches support only the warm-start compatibility flag");
  if (systems.empty()) throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "RCCSD(T) batch is empty");

  const auto nbf = molecule::ao_count(systems.front());
  const auto nocc = static_cast<std::size_t>(systems.front().electron_count / 2);
  for (const auto& system : systems) {
    if (molecule::ao_count(system) != nbf ||
        static_cast<std::size_t>(system.electron_count / 2) != nocc)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "RCCSD(T) prepared batch requires one homogeneous (nocc,nvir) shape; split "
                        "ragged groups");
  }
  return std::make_unique<RccsdtPreparedBatch>(capabilities, context, std::move(systems),
                                               descriptor);
}

}  // namespace vibeqc::methods::detail
