#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "scf/solver/self_consistent.hpp"

namespace {

using vibeqc::scf::solver::run_self_consistent;
using vibeqc::scf::solver::SelfConsistentPolicy;
using vibeqc::scf::solver::SelfConsistentProgress;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct ScalarEvaluation {
  double proposed_state{};
  double energy{};
  double state_rms{};
  double residual_rms{};
};

void verify_basic_convergence() {
  const SelfConsistentPolicy policy{8, 1.0e-12, 1.0e-12, 1.0e-12, false};
  unsigned records = 0;
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned) {
        const double next = 1.0;
        return ScalarEvaluation{next, (state - 1.0) * (state - 1.0), std::abs(next - state), 99.0};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [&](const SelfConsistentProgress& progress, const ScalarEvaluation&) {
        ++records;
        require(progress.iteration == records, "progress iteration order changed");
      });

  require(outcome.converged, "fixed-point driver failed a convergent scalar problem");
  require(outcome.progress.iteration == 3, "fixed-point convergence iteration changed");
  require(outcome.state == 1.0, "fixed-point driver did not retain the accepted state");
  require(records == 3, "fixed-point recorder call count changed");
}

void verify_residual_gate() {
  const SelfConsistentPolicy policy{8, 1.0e-12, 1.0e-12, 1.0e-6, true};
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned iteration) {
        const double next = 1.0;
        const double residual = iteration < 4 ? 1.0 : 0.0;
        return ScalarEvaluation{next, (state - 1.0) * (state - 1.0), std::abs(next - state),
                                residual};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(outcome.converged, "residual-gated fixed point did not converge");
  require(outcome.progress.iteration == 4, "residual gate was not enforced");
}

void verify_nonconverged_state_retention() {
  const SelfConsistentPolicy policy{2, 1.0e-12, 1.0e-12, 1.0e-12, false};
  const auto outcome = run_self_consistent(
      0.0, policy,
      [](double state, unsigned) {
        const double next = state + 1.0;
        return ScalarEvaluation{next, 1.0, 1.0, 0.0};
      },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(!outcome.converged, "nonconvergent fixed point was marked converged");
  require(outcome.progress.iteration == 2, "max-iteration bookkeeping changed");
  require(outcome.state == 2.0, "last accepted nonconverged state was not retained");
}

void verify_accept_owns_update_policy() {
  const SelfConsistentPolicy policy{1, 0.0, 0.0, 0.0, false};
  const auto outcome = run_self_consistent(
      0.0, policy, [](double, unsigned) { return ScalarEvaluation{1.0, 0.0, 1.0, 0.0}; },
      [](double, ScalarEvaluation evaluation, const SelfConsistentProgress&) {
        return evaluation.proposed_state + 0.5;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(!outcome.converged, "one-step acceptance unexpectedly converged");
  require(outcome.state == 1.5, "driver bypassed method-owned acceptance policy");
}

void verify_terminal_accept_can_keep_current_state() {
  const SelfConsistentPolicy policy{4, 1.0e-12, 1.0e-12, 1.0e-12, false};
  const auto outcome = run_self_consistent(
      0.0, policy, [](double, unsigned) { return ScalarEvaluation{42.0, 0.0, 0.0, 0.0}; },
      [](double& current, ScalarEvaluation evaluation, const SelfConsistentProgress& progress) {
        if (progress.converged) return std::move(current);
        return current + 1.0 + 0.0 * evaluation.proposed_state;
      },
      [](const SelfConsistentProgress&, const ScalarEvaluation&) {});

  require(outcome.converged, "terminal-retention problem did not converge");
  require(outcome.progress.iteration == 2, "terminal-retention convergence iteration changed");
  require(outcome.state == 1.0, "driver replaced a method-retained terminal state");
}

}  // namespace

int main() {
  try {
    verify_basic_convergence();
    verify_residual_gate();
    verify_nonconverged_state_retention();
    verify_accept_owns_update_policy();
    verify_terminal_accept_can_keep_current_state();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
