#include "scf/solver/proposal_control.hpp"

#include <atomic>
#include <stdexcept>

#include "scf/reference/observation.hpp"

namespace vibeqc::scf::solver {
using reference::index;
using reference::multiply;
using reference::symmetric_eigen;
using reference::transpose;
double seconds_since(ScfClock::time_point started) {
  return std::chrono::duration<double>(ScfClock::now() - started).count();
}

std::uint64_t new_scf_generation(const ScfOptions& options) {
  static std::atomic<std::uint64_t> next{1};
  return options.hooks ? next.fetch_add(1) : 0;
}

/** Check physical ensemble representability in the actual AO metric. Trace
 * rescaling would conceal a wrong charge/spin proposal, so validation never
 * repairs an input. Determinant proposals also require integer occupations.
 */
std::string invalid_proposal(const ScfSnapshot& state, const ScfProposal& proposal,
                             const initial_guess::EigenOperation& eigen) {
  if (proposal.generation != state.generation || proposal.iteration != state.iteration)
    return "stale_state";
  if (proposal.representation != ProposalRepresentation::ensemble_density &&
      proposal.representation != ProposalRepresentation::determinant_density)
    return "unknown_representation";
  const Matrix& candidate = proposal.density;
  if (candidate.size() != state.density.size()) return "shape";
  if (!std::all_of(candidate.begin(), candidate.end(), [](double x) { return std::isfinite(x); }))
    return "nonfinite";
  const auto n = state.nbf;
  const auto diagonalize = [&](const Matrix& matrix) {
    // The legacy seed contract permits 1e-7 asymmetry; the ordinary device
    // frame requires 1e-12 relative symmetry. Keep that acceptance boundary
    // by choosing the old validator for such inputs, never by symmetrizing D.
    bool device_symmetric = true;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j)
        if (std::abs(matrix[index(i, j, n)] - matrix[index(j, i, n)]) >
            1e-12 *
                std::max({1.0, std::abs(matrix[index(i, j, n)]), std::abs(matrix[index(j, i, n)])}))
          device_symmetric = false;
    if (eigen && device_symmetric) return eigen(matrix, nullptr, nullptr, n);
    reference::observation::Reason reason(eigen ? reference::observation::EigenReason::fallback
                                                : reference::observation::active_reason);
    return symmetric_eigen(matrix, n);
  };
  const auto se = diagonalize(state.overlap);
  Matrix scaled = se.vectors;
  for (std::size_t j = 0; j < n; ++j) {
    if (se.values[j] < 1e-10) return "singular_metric";
    for (std::size_t i = 0; i < n; ++i) scaled[index(i, j, n)] *= std::sqrt(se.values[j]);
  }
  const Matrix root = multiply(scaled, transpose(se.vectors, n), n);
  constexpr double tolerance = 1e-7;
  for (std::size_t spin = 0; spin < state.electrons.size(); ++spin) {
    Matrix p(candidate.begin() + spin * n * n, candidate.begin() + (spin + 1) * n * n);
    double trace = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (std::abs(p[index(i, j, n)] - p[index(j, i, n)]) > tolerance) return "symmetry";
        trace += p[index(i, j, n)] * state.overlap[index(j, i, n)];
      }
    }
    if (!std::isfinite(trace) || std::abs(trace - state.electrons[spin]) > tolerance)
      return "electron_count";
    const auto occupations = diagonalize(multiply(root, multiply(p, root, n), n)).values;
    for (double value : occupations) {
      if (!std::isfinite(value) || value < -tolerance ||
          value > state.occupation_weight + tolerance)
        return "occupation_bounds";
      if (proposal.representation == ProposalRepresentation::determinant_density &&
          std::min(std::abs(value), std::abs(value - state.occupation_weight)) > tolerance)
        return "nonidempotent";
    }
  }
  return {};
}

void validate_seed(const Matrix& overlap, const Matrix& seed, std::size_t n,
                   const std::vector<unsigned>& electrons, double weight,
                   const initial_guess::EigenOperation& eigen) {
  reference::observation::Reason scope(reference::observation::EigenReason::seed_validation);
  ScfSnapshot state;
  state.nbf = n;
  state.overlap = overlap;
  state.electrons = electrons;
  state.occupation_weight = weight;
  state.density.resize(electrons.size() * n * n);
  const auto reason =
      invalid_proposal(state, {ProposalRepresentation::ensemble_density, 0, 0, seed}, eigen);
  if (!reason.empty()) throw std::invalid_argument("invalid initial proposal: " + reason);
}

}  // namespace vibeqc::scf::solver
