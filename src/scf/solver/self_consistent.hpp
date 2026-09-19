#ifndef VIBEQC_SCF_SOLVER_SELF_CONSISTENT_HPP
#define VIBEQC_SCF_SOLVER_SELF_CONSISTENT_HPP

#include <cmath>
#include <limits>
#include <utility>

namespace vibeqc::scf::solver {

/** Method-neutral fixed-point convergence policy.
 *
 * The driver deliberately knows nothing about densities, charges, orbitals,
 * Hamiltonians, DIIS/Broyden, occupations, or finalization.  A method adapter
 * owns those scientific details and supplies one evaluated proposal per
 * iteration.
 */
struct SelfConsistentPolicy {
  unsigned max_iterations{};
  double energy_tolerance{};
  double state_tolerance{};
  double residual_tolerance{};
  bool require_residual{};
};

/** Convergence bookkeeping visible to method adapters and diagnostics. */
struct SelfConsistentProgress {
  unsigned iteration{};
  double energy{};
  double energy_change{std::numeric_limits<double>::infinity()};
  double state_rms{};
  double residual_rms{};
  bool converged{};
};

template <class State>
struct SelfConsistentOutcome {
  State state;
  SelfConsistentProgress progress{};
  bool converged{};
};

/** Run a generic self-consistent fixed-point iteration.
 *
 * evaluate(state, iteration) must return an object exposing:
 *   energy, state_rms, residual_rms
 * and any method-owned proposal payload required by accept().
 *
 * record(progress, evaluation) runs before accept(), matching SCF diagnostic
 * ordering: proposal hooks may inspect the current iteration diagnostics.
 *
 * accept(current_state, evaluation, progress) returns the accepted next state.
 * It may apply damping, DIIS/Broyden policy, external safeguarded proposals, or
 * other method-specific update rules.  Convergence is evaluated from the
 * physical proposal metrics before that acceptance step.
 */
template <class State, class Evaluate, class Accept, class Record>
SelfConsistentOutcome<State> run_self_consistent(State initial_state,
                                                 const SelfConsistentPolicy& policy,
                                                 Evaluate&& evaluate, Accept&& accept,
                                                 Record&& record) {
  State state = std::move(initial_state);
  SelfConsistentProgress latest;
  double previous_energy = std::numeric_limits<double>::infinity();

  for (unsigned iteration = 1; iteration <= policy.max_iterations; ++iteration) {
    auto evaluation = evaluate(state, iteration);

    latest.iteration = iteration;
    latest.energy = evaluation.energy;
    latest.energy_change = std::isfinite(previous_energy)
                               ? std::abs(evaluation.energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    latest.state_rms = evaluation.state_rms;
    latest.residual_rms = evaluation.residual_rms;
    latest.converged =
        iteration > 1 && latest.energy_change < policy.energy_tolerance &&
        latest.state_rms < policy.state_tolerance &&
        (!policy.require_residual || latest.residual_rms < policy.residual_tolerance);

    record(latest, evaluation);
    State next_state = accept(state, std::move(evaluation), latest);
    if (latest.converged) {
      return {std::move(next_state), latest, true};
    }

    previous_energy = latest.energy;
    state = std::move(next_state);
  }

  return {std::move(state), latest, false};
}

}  // namespace vibeqc::scf::solver

#endif
