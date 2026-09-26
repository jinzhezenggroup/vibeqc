#ifndef VIBEQC_SCF_SOLVER_PROPOSAL_CONTROL_HPP
#define VIBEQC_SCF_SOLVER_PROPOSAL_CONTROL_HPP
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/proposals.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/types.hpp"

namespace vibeqc::scf::solver {
using reference::Matrix;
using reference::residual_rms;
using ScfClock = std::chrono::steady_clock;
double seconds_since(ScfClock::time_point started);
/** Assign one generation per hooked solve; ordinary solves do not consume IDs. */
std::uint64_t new_scf_generation(const ScfOptions& options);
/** Validate generation, spin trace, symmetry and AO-metric occupations without repair. */
std::string invalid_proposal(const ScfSnapshot& state, const ScfProposal& proposal,
                             const initial_guess::EigenOperation& eigen = {});
/** Apply the same ensemble-domain checks to an explicit strict warm seed.
 * Symmetric inputs may borrow a qualified device provider. Legacy tolerated
 * asymmetry selects the reference check before submission, without repairing
 * the seed or retrying a failed device solve. */
void validate_seed(const Matrix& overlap, const Matrix& seed, std::size_t n,
                   const std::vector<unsigned>& electrons, double weight,
                   const initial_guess::EigenOperation& eigen = {});
/** Evaluate every trial with the same unscreened or fitted target operator as
 * the main loop. Rejected trials count as work. Convex damping preserves the
 * validated ensemble domain but is explicitly no longer a determinant. The
 * traditional next density remains the convergence comparator: a proposal
 * returning the current density must never manufacture zero iteration change.
 */
template <class Evaluate>
Matrix safeguarded_update(const ScfOptions& options, std::uint64_t generation, unsigned iteration,
                          const Matrix& overlap, const Matrix& density, const Matrix& fock,
                          const Matrix& residual, Matrix baseline,
                          const std::vector<unsigned>& electrons, double weight, ScfResult& result,
                          Diis& diis, unsigned& proposal_failures, bool terminal,
                          Evaluate&& evaluate) {
  if (!options.hooks) return baseline;
  ScfSnapshot state{generation,
                    iteration,
                    overlap.empty() ? 0U : static_cast<std::size_t>(std::sqrt(overlap.size())),
                    electrons,
                    weight,
                    density,
                    fock,
                    residual,
                    overlap,
                    baseline,
                    result.energy,
                    residual_rms(residual),
                    result.fock_builds};
  ProposalDecision decision;
  decision.energy = state.energy;
  decision.residual_rms = state.residual_rms;
  decision.reason = terminal ? "traditional_convergence" : "no_proposal";
  if (!terminal && proposal_failures >= 3) decision.reason = "traditional_fallback";
  if (options.hooks->propose && !terminal && proposal_failures < 3) {
    ScfProposal proposal;
    const auto started = ScfClock::now();
    try {
      proposal = options.hooks->propose(state);
    } catch (...) {
      proposal.representation = ProposalRepresentation::reset;
      decision.reason = "proposal_exception";
    }
    decision.inference_seconds = seconds_since(started);
    if (proposal.representation == ProposalRepresentation::reset) {
      diis.clear();
      decision.action = ProposalAction::reset;
      if (decision.reason != "proposal_exception") decision.reason = "requested_reset";
    } else if (proposal.representation != ProposalRepresentation::none) {
      const auto validating = ScfClock::now();
      decision.reason = invalid_proposal(state, proposal);
      decision.validation_seconds = seconds_since(validating);
      decision.action = ProposalAction::rejected;
      if (decision.reason.empty()) {
        decision.reason = "target_operator_no_descent";
        const auto evaluating = ScfClock::now();
        // The fixed schedule is reproducible in replay; it is not a learned
        // trust radius and cannot silently clip invalid proposal eigenvalues.
        for (double fraction : {1.0, 0.5, 0.25, 0.125}) {
          Matrix trial(density.size());
          for (std::size_t i = 0; i < trial.size(); ++i)
            trial[i] = (1 - fraction) * density[i] + fraction * proposal.density[i];
          ++decision.trials;
          ++result.fock_builds;
          const auto [energy, trial_residual] = evaluate(trial);
          const double norm = residual_rms(trial_residual);
          if (std::isfinite(energy) && std::isfinite(norm) && energy <= state.energy + 1e-9 &&
              norm <= std::max(1e-12, state.residual_rms * (1 - 1e-4 * fraction))) {
            decision.action = fraction == 1 ? ProposalAction::accepted : ProposalAction::damped;
            decision.reason = "target_operator_descent";
            decision.fraction = fraction;
            decision.energy = energy;
            decision.residual_rms = norm;
            baseline = std::move(trial);
            break;
          }
        }
        decision.operator_seconds = seconds_since(evaluating);
      }
      // Histories belong to the traditional trajectory. Rebuild them after a
      // proposal or rejection; retain the already computed baseline this step.
      diis.clear();
    }
    if (decision.action == ProposalAction::rejected || decision.action == ProposalAction::reset) {
      // A persistently bad model must not clear DIIS forever. The failure
      // budget belongs to this solve, never to the caller or another item.
      ++proposal_failures;
    }
  }
  if (options.hooks->observe) options.hooks->observe(state, decision);
  return baseline;
}

}  // namespace vibeqc::scf::solver
#endif
