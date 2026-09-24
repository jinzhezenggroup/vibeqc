#ifndef VIBEQC_SCF_SOLVER_SELF_CONSISTENT_HPP
#define VIBEQC_SCF_SOLVER_SELF_CONSISTENT_HPP

#include "solver/self_consistent.hpp"

namespace vibeqc::scf::solver {

// Compatibility aliases for existing SCF/KS callers. New method-neutral
// consumers should include solver/self_consistent.hpp directly.
using ::vibeqc::solver::BoundedIterationResult;
using ::vibeqc::solver::run_bounded_iterations;
using ::vibeqc::solver::run_self_consistent;
using ::vibeqc::solver::SelfConsistentOutcome;
using ::vibeqc::solver::SelfConsistentPolicy;
using ::vibeqc::solver::SelfConsistentProgress;

}  // namespace vibeqc::scf::solver

#endif
