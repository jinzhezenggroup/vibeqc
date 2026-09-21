#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "scf/solver/self_consistent.hpp"
#include "solver/diis.hpp"

namespace {

using vibeqc::scf::solver::run_bounded_iterations;
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

void verify_bounded_iteration_control() {
  unsigned calls = 0;
  const auto stopped = run_bounded_iterations(8, [&](unsigned iteration) {
    ++calls;
    require(iteration == calls, "bounded iteration numbering changed");
    return iteration < 3;
  });
  require(stopped.completed_iterations == 3, "bounded early-stop count changed");
  require(stopped.stopped_by_callback, "bounded callback stop was not reported");
  require(calls == 3, "bounded driver executed after stop");

  calls = 0;
  const auto exhausted = run_bounded_iterations(2, [&](unsigned) {
    ++calls;
    return true;
  });
  require(exhausted.completed_iterations == 2, "bounded maximum was not enforced");
  require(!exhausted.stopped_by_callback, "bounded exhaustion was marked callback-stopped");
  require(calls == 2, "bounded driver call count changed");

  calls = 0;
  const auto empty = run_bounded_iterations(0, [&](unsigned) {
    ++calls;
    return true;
  });
  require(empty.completed_iterations == 0, "zero iteration budget changed");
  require(!empty.stopped_by_callback, "zero iteration budget was marked callback-stopped");
  require(calls == 0, "zero iteration budget invoked the callback");
}

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

void verify_diis_shape_rejection_preserves_history() {
  vibeqc::solver::Diis diis(3, 2);
  diis.update({0.0, 0.0}, {1.0, 0.0});
  for (const auto& sizes :
       {std::pair{1U, 2U}, std::pair{3U, 2U}, std::pair{2U, 1U}, std::pair{2U, 3U}}) {
    bool rejected = false;
    try {
      diis.update(std::vector<double>(sizes.first, 1.0), std::vector<double>(sizes.second, 1.0));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "shared DIIS accepted inconsistent vector/error dimensions");
  }
  const auto next = diis.update({2.0, 4.0}, {0.0, 1.0});
  require(next.size() == 2 && std::abs(next[0] - 1.0) < 1e-14 && std::abs(next[1] - 2.0) < 1e-14 &&
              diis.restarts() == 0,
          "invalid DIIS update poisoned the retained history");
}

void verify_method_neutral_diis() {
  vibeqc::solver::Diis disabled(0, 2);
  const std::vector<double> original{2.0, 4.0};
  require(disabled.update(original, {1.0, 0.0}) == original,
          "disabled shared DIIS changed the input state");
  require(disabled.restarts() == 0, "disabled shared DIIS reported a restart");

  vibeqc::solver::Diis diis(2, 2);
  require(diis.update({0.0, 0.0}, {1.0, 0.0}) == std::vector<double>({0.0, 0.0}),
          "first shared DIIS state changed");
  const auto extrapolated = diis.update({2.0, 4.0}, {0.0, 1.0});
  require(std::abs(extrapolated[0] - 1.0) < 1.0e-14 && std::abs(extrapolated[1] - 2.0) < 1.0e-14,
          "shared DIIS Pulay extrapolation changed");

  vibeqc::solver::Diis dependent(2, 2);
  dependent.update({0.0, 0.0}, {1.0, 1.0});
  const auto retained = dependent.update({3.0, 5.0}, {1.0, 1.0});
  require(retained == std::vector<double>({3.0, 5.0}),
          "shared DIIS did not retain the latest state after a singular history");
  require(dependent.restarts() == 1, "shared DIIS did not count dependent-history retirement");
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
    verify_bounded_iteration_control();
    verify_basic_convergence();
    verify_residual_gate();
    verify_nonconverged_state_retention();
    verify_accept_owns_update_policy();
    verify_terminal_accept_can_keep_current_state();
    verify_method_neutral_diis();
    verify_diis_shape_rejection_preserves_history();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
