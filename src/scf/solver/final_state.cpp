#include "scf/solver/final_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "runtime/host_component_trace.hpp"
#include "scf/reference/mean_field.hpp"

namespace vibeqc::scf::solver {
namespace {
using reference::Matrix;
bool finite(const Matrix& values) {
  return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
bool symmetric(const Matrix& values, std::size_t n) {
  if (values.size() != n * n || !finite(values)) return false;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      if (std::abs(values[i * n + j] - values[j * n + i]) >
          1e-12 * std::max({1.0, std::abs(values[i * n + j]), std::abs(values[j * n + i])}))
        return false;
  return true;
}
std::size_t dimension(const Matrix& overlap) {
  const auto n = static_cast<std::size_t>(std::sqrt(overlap.size()));
  return n && n <= std::numeric_limits<std::size_t>::max() / n && n * n == overlap.size() ? n : 0;
}
bool valid_input(const FinalStateIdentity& current, const Matrix& overlap, const Matrix& hcore,
                 double nuclear, const std::vector<Matrix>& density,
                 const FinalStateLimits& limits) {
  const auto n = dimension(overlap);
  const auto& id = current.factor;
  const auto spins = current.model.spec.spin == FockSpin::Restricted ? 1U : 2U;
  if (!n || !id.basis || !id.reference || !id.orbital_generation || !id.density_generation ||
      !current.solve_epoch || current.occupied.size() != spins || density.size() != spins ||
      !std::isfinite(nuclear) || !std::isfinite(limits.density_tolerance) ||
      limits.density_tolerance <= 0 || !std::isfinite(limits.energy_tolerance) ||
      limits.energy_tolerance <= 0 || limits.maximum_corrections > 64 || !symmetric(overlap, n) ||
      !symmetric(hcore, n))
    return false;
  try {
    validate_resolved_fock_build(current.model);
  } catch (const std::invalid_argument&) {
    return false;
  }
  for (std::size_t spin = 0; spin < spins; ++spin)
    if (current.occupied[spin] > n || !symmetric(density[spin], n)) return false;
  return true;
}
bool valid_fock(const FinalStateIdentity& current, const PhysicalFockFrame& fock, std::size_t n,
                bool device = false) {
  return fock.physical && fock.identity == current &&
         fock.spins.size() == current.occupied.size() &&
         std::all_of(fock.spins.begin(), fock.spins.end(), [n, device](const auto& f) {
           return (device && f.empty()) || symmetric(f, n);
         });
}
double physical_energy(const Matrix& hcore, double nuclear, const std::vector<Matrix>& density,
                       const PhysicalFockFrame& fock) {
  long double energy = nuclear;
  for (std::size_t spin = 0; spin < density.size(); ++spin)
    for (std::size_t k = 0; k < hcore.size(); ++k)
      energy += .5L * density[spin][k] * (static_cast<long double>(hcore[k]) + fock.spins[spin][k]);
  return static_cast<double>(energy);
}
}  // namespace

bool validate_final_state(const FinalStateIdentity& current, const Matrix& overlap,
                          const Matrix& hcore, double nuclear_energy,
                          const std::vector<Matrix>& density, const PhysicalFockFrame& fock,
                          const FinalFrameCandidate& orbitals, const FinalStateLimits& limits,
                          FinalStateDiagnostic& diagnostic, std::string& detail,
                          const FinalStateOperations* operations) {
  diagnostic = {};
  detail.clear();
  runtime::host_trace::Region trace("final_state_validation", dimension(overlap));
  if (!valid_input(current, overlap, hcore, nuclear_energy, density, limits)) {
    detail = "invalid final-state identity, model, occupations, matrices or tolerances";
    return false;
  }
  const auto n = dimension(overlap);
  if (!valid_fock(current, fock, n, operations != nullptr) || orbitals.identity != current ||
      !orbitals.physical_origin || !orbitals.fock_density_generation ||
      orbitals.fock_density_generation > current.factor.density_generation ||
      orbitals.spins.size() != density.size()) {
    detail =
        "final-state frame has a stale generation, source/model, occupation or nonphysical Fock";
    return false;
  }
  const double weight = current.model.spec.spin == FockSpin::Restricted ? 2.0 : 1.0;
  const double tolerance = std::min(1e-8, limits.density_tolerance);
  diagnostic.eigenframes.resize(density.size());
  if (operations) {
    if (!operations->products(current, overlap, hcore, nuclear_energy, density, fock, orbitals,
                              limits, diagnostic, detail))
      return false;
    if (diagnostic.eigenframes.size() != density.size()) {
      detail = "final-state provider returned incomplete spin diagnostics";
      return false;
    }
    for (const auto& frame : diagnostic.eigenframes)
      if (!accept_eigen_frame(frame, detail)) return false;
  } else {
    for (std::size_t spin = 0; spin < density.size(); ++spin) {
      const auto& frame = orbitals.spins[spin];
      if (!validate_eigen_frame(fock.spins[spin], &overlap, frame.values, frame.vectors, n,
                                diagnostic.eigenframes[spin], detail))
        return false;
      const auto reconstructed =
          reference::density_from_orbitals(frame.vectors, n, current.occupied[spin], weight);
      const auto ds = reference::multiply(density[spin], overlap, n);
      const auto dsd = reference::multiply(ds, density[spin], n);
      const auto residual =
          reference::commutator_residual(fock.spins[spin], density[spin], overlap, n);
      if (!finite(reconstructed) || !finite(ds) || !finite(dsd) || !finite(residual)) {
        detail = "nonfinite final-state validation products";
        return false;
      }
      long double electrons = 0;
      double drift = 0;
      for (std::size_t row = 0; row < n; ++row) electrons += ds[row * n + row];
      for (std::size_t k = 0; k < n * n; ++k) {
        drift = std::hypot(drift, reconstructed[k] - density[spin][k]);
        diagnostic.maximum_density_error = std::max(diagnostic.maximum_density_error,
                                                    std::abs(reconstructed[k] - density[spin][k]));
        diagnostic.maximum_commutator =
            std::max(diagnostic.maximum_commutator, std::abs(residual[k]));
        diagnostic.maximum_idempotency_error = std::max(
            diagnostic.maximum_idempotency_error, std::abs(dsd[k] - weight * density[spin][k]));
      }
      diagnostic.density_rms = std::max(diagnostic.density_rms, drift / n);
      diagnostic.maximum_trace_error =
          std::max(diagnostic.maximum_trace_error,
                   std::abs(static_cast<double>(electrons - static_cast<long double>(weight) *
                                                                current.occupied[spin])));
      if (limits.require_canonicality) {
        // Export's absolute canonicality gate can be stricter than the scaled
        // eigen residual in an ill-conditioned AO metric. Include it in state
        // selection so rejection enters correction before W/reference output.
        const auto fc = reference::multiply(fock.spins[spin], frame.vectors, n);
        const auto cfc = reference::multiply(reference::transpose(frame.vectors, n), fc, n);
        if (!finite(cfc)) {
          detail = "nonfinite final-state canonicality products";
          return false;
        }
        for (std::size_t row = 0; row < n; ++row)
          for (std::size_t column = 0; column < n; ++column)
            diagnostic.maximum_canonical_error = std::max(
                diagnostic.maximum_canonical_error,
                std::abs(cfc[row * n + column] - (row == column ? frame.values[row] : 0.0)));
      }
    }
    diagnostic.energy = physical_energy(hcore, nuclear_energy, density, fock);
  }
  for (double value : {diagnostic.maximum_commutator, diagnostic.density_rms,
                       diagnostic.maximum_trace_error, diagnostic.maximum_idempotency_error,
                       diagnostic.maximum_density_error, diagnostic.maximum_canonical_error}) {
    if (!std::isfinite(value) || value < 0) {
      detail = "nonfinite or invalid final-state validation diagnostics";
      return false;
    }
  }
  if (!std::isfinite(diagnostic.energy) || !std::isfinite(diagnostic.density_rms) ||
      !std::isfinite(diagnostic.maximum_trace_error) || diagnostic.maximum_commutator > tolerance ||
      diagnostic.density_rms > tolerance || diagnostic.maximum_trace_error > 1e-8 ||
      diagnostic.maximum_idempotency_error > 1e-8 ||
      (limits.require_canonicality &&
       (diagnostic.maximum_density_error > 1e-8 || diagnostic.maximum_canonical_error > 1e-8))) {
    detail =
        "final state failed physical commutator, density, electron trace, idempotency, energy "
        "or requested canonicality checks";
    return false;
  }
  return true;
}

FinalStateSelection select_final_state(
    FinalStateIdentity current, const Matrix& overlap, const Matrix& hcore,
    const Matrix& orthogonalizer, double nuclear_energy, std::vector<Matrix> density,
    const FinalFrameCandidate* candidate, const PhysicalFockOperation& evaluate,
    const initial_guess::EigenOperation& eigen, const FinalStateLimits& limits,
    bool compute_weighted_density, bool force_rebuild, const FinalStateOperations* operations) {
  FinalStateSelection result;
  try {
    const auto n = dimension(overlap);
    if (!evaluate || !eigen ||
        (operations && (!operations->products || !operations->eigen || !operations->project ||
                        !operations->weighted)) ||
        !valid_input(current, overlap, hcore, nuclear_energy, density, limits) ||
        !symmetric(orthogonalizer, n)) {
      result.status = FinalStateStatus::InvalidInput;
      result.detail = "invalid strict final-state request";
      return result;
    }
    std::optional<FinalFrameCandidate> corrected;
    const FinalFrameCandidate* frame = candidate;
    double previous_energy = std::numeric_limits<double>::quiet_NaN();
    for (unsigned step = 0;; ++step) {
      ++result.fock_evaluations;
      auto physical = runtime::host_trace::call("final_state_fock_build",
                                                [&] { return evaluate(current, density); });
      if (!valid_fock(current, physical, n, operations != nullptr)) {
        result.status = FinalStateStatus::ProviderFailure;
        result.detail = "physical provider did not evaluate the current tagged density";
        return result;
      }
      FinalStateDiagnostic diagnostic;
      const bool valid =
          frame && validate_final_state(current, overlap, hcore, nuclear_energy, density, physical,
                                        *frame, limits, diagnostic, result.detail, operations);
      const auto materialize = [&] {
        if (std::any_of(physical.spins.begin(), physical.spins.end(),
                        [](const auto& f) { return f.empty(); })) {
          if (!operations || !operations->materialize_fock)
            throw std::runtime_error("missing physical Fock materialization provider");
          physical.spins = operations->materialize_fock(physical);
          if (!valid_fock(current, physical, n))
            throw std::runtime_error("invalid materialized physical Fock");
        }
      };
      if (!valid) materialize();
      const double energy =
          valid ? diagnostic.energy : physical_energy(hcore, nuclear_energy, density, physical);
      if (!std::isfinite(energy)) {
        result.detail = "nonfinite strict final-state energy";
        return result;
      }
      diagnostic.energy_change = step ? std::abs(energy - previous_energy) : 0;
      std::optional<FinalFrameCandidate> evaluated;
      std::vector<Matrix> projected;
      const auto solve_physical = [&](bool fixed_point) {
        materialize();
        evaluated.emplace();
        projected.clear();
        for (std::size_t spin = 0; spin < density.size(); ++spin) {
          ++result.eigen_solves;
          if (fixed_point) ++result.fixed_point_eigen_solves;
          auto c = runtime::host_trace::with_reason(
              runtime::host_trace::EigenReason::final_fock,
              [&] { return eigen(physical.spins[spin], &overlap, &orthogonalizer, n); });
          EigenFrameDiagnostic checked;
          const bool valid_eigen =
              operations
                  ? operations->eigen(physical.spins[spin], overlap, c, checked, result.detail) &&
                        accept_eigen_frame(checked, result.detail)
                  : validate_eigen_frame(physical.spins[spin], &overlap, c.values, c.vectors, n,
                                         checked, result.detail);
          if (!valid_eigen) {
            result.status = FinalStateStatus::ProviderFailure;
            return false;
          }
          const double weight = current.model.spec.spin == FockSpin::Restricted ? 2.0 : 1.0;
          projected.push_back(operations ? operations->project(c, current.occupied[spin], weight)
                                         : reference::density_from_orbitals(
                                               c.vectors, n, current.occupied[spin], weight));
          if (!symmetric(projected.back(), n)) {
            result.status = FinalStateStatus::ProviderFailure;
            result.detail = "invalid projected final-state density";
            return false;
          }
          evaluated->spins.push_back(std::move(c));
        }
        return true;
      };
      bool accepted = valid && !(force_rebuild && step == 0) &&
                      diagnostic.energy_change <= limits.energy_tolerance;
      if (accepted && compute_weighted_density) {
        runtime::host_trace::Region check("final_state_fixed_point", n);
        ++result.fixed_point_checks;
        if (!solve_physical(true)) return result;
        // Reconstructing a retained frame can give exactly zero drift even
        // when its orbitals came from an older Fock. This projector instead
        // tests the fixed point of the current physical F[D], before force
        // weights or occupied-response leases are authorized.
        for (std::size_t spin = 0; spin < density.size(); ++spin) {
          double norm = 0;
          for (std::size_t k = 0; k < n * n; ++k) {
            const double delta = projected[spin][k] - density[spin][k];
            norm = std::hypot(norm, delta);
            diagnostic.maximum_fixed_point_density_error =
                std::max(diagnostic.maximum_fixed_point_density_error, std::abs(delta));
          }
          diagnostic.fixed_point_density_rms =
              std::max(diagnostic.fixed_point_density_rms, norm / n);
        }
        const double tolerance = std::min(1e-8, limits.density_tolerance);
        accepted = std::isfinite(diagnostic.maximum_fixed_point_density_error) &&
                   std::isfinite(diagnostic.fixed_point_density_rms) &&
                   diagnostic.maximum_fixed_point_density_error <= tolerance &&
                   diagnostic.fixed_point_density_rms <= tolerance;
        if (!accepted) ++result.fixed_point_rejections;
      }
      if (accepted) {
        VerifiedFinalState state{current, std::move(density), physical.spins,
                                 {},      frame->spins,       std::move(diagnostic)};
        if (compute_weighted_density) {
          runtime::host_trace::Region weighted("final_state_weighted_density", n);
          if (operations) {
            state.weighted_density = operations->weighted(current, state.density, physical);
            if (state.weighted_density.size() != current.occupied.size() ||
                !std::all_of(state.weighted_density.begin(), state.weighted_density.end(),
                             [n](const auto& w) { return w.size() == n * n && finite(w); })) {
              result.detail = "invalid device energy-weighted density";
              return result;
            }
          } else {
            const double weight = current.model.spec.spin == FockSpin::Restricted ? 2.0 : 1.0;
            for (std::size_t spin = 0; spin < state.density.size(); ++spin) {
              auto w = reference::multiply(
                  reference::multiply(state.density[spin], physical.spins[spin], n),
                  state.density[spin], n);
              for (auto& value : w) value /= weight;
              if (!finite(w)) {
                result.detail = "nonfinite validated energy-weighted density";
                return result;
              }
              state.weighted_density.push_back(std::move(w));
            }
          }
        }
        result.state = std::move(state);
        result.status = FinalStateStatus::Success;
        result.reused = step == 0;
        result.detail.clear();
        return result;
      }
      if (frame && !valid) ++result.candidate_rejections;
      if (step == limits.maximum_corrections ||
          current.factor.density_generation == std::numeric_limits<std::uint64_t>::max() ||
          current.factor.orbital_generation == std::numeric_limits<std::uint64_t>::max()) {
        result.detail = "strict final-state correction exhausted without a consistent state";
        return result;
      }
      runtime::host_trace::Region correction("strict_final_correction", n);
      const auto origin = current.factor.density_generation;
      // A failed fixed-point probe already solved this exact physical Fock.
      // Promote its checked frame/projector instead of repeating that work.
      runtime::host_trace::Region projection(
          evaluated ? "final_state_fixed_point_promotion" : "final_state_correction_solve", n);
      if (!evaluated && !solve_physical(false)) return result;
      corrected = std::move(evaluated);
      density = std::move(projected);
      ++current.factor.orbital_generation;
      ++current.factor.density_generation;
      ++result.density_updates;
      corrected->identity = current;
      corrected->fock_density_generation = origin;
      corrected->physical_origin = true;
      frame = &*corrected;
      previous_energy = energy;
    }
  } catch (const std::bad_alloc&) {
    result.status = FinalStateStatus::OutOfMemory;
    result.detail = "strict final-state allocation failed";
  } catch (const std::exception& error) {
    result.status = FinalStateStatus::ProviderFailure;
    result.detail = error.what();
  } catch (...) {
    result.status = FinalStateStatus::ProviderFailure;
    result.detail = "strict final-state provider raised an unknown exception";
  }
  return result;
}
}  // namespace vibeqc::scf::solver
