#ifndef VIBEQC_SOLVER_ITERATION_CONTROL_HPP
#define VIBEQC_SOLVER_ITERATION_CONTROL_HPP

namespace vibeqc::solver {

/** Result of a bounded host-controlled iterative loop. */
struct BoundedIterationResult {
  unsigned completed_iterations{};
  bool stopped_by_callback{};
};

/** Run at most max_iterations host-controlled iterations.
 *
 * The callback receives a one-based iteration number and returns true to
 * continue or false to stop. Scientific state, convergence metrics, update
 * policy, provider calls and device synchronization remain owned by the
 * caller.
 *
 * This primitive is intentionally method-neutral: SCF, KS, correlation and
 * response solvers can share iteration-budget/early-stop semantics without
 * importing one another's scientific state.
 */
template <class Step>
BoundedIterationResult run_bounded_iterations(unsigned max_iterations, Step&& step) {
  BoundedIterationResult result;
  for (unsigned completed = 0; completed < max_iterations; ++completed) {
    const unsigned iteration = completed + 1;
    result.completed_iterations = iteration;
    if (!step(iteration)) {
      result.stopped_by_callback = true;
      break;
    }
  }
  return result;
}

}  // namespace vibeqc::solver

#endif
