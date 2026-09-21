#ifndef VIBEQC_SCF_SOLVER_ITERATION_CONTROL_HPP
#define VIBEQC_SCF_SOLVER_ITERATION_CONTROL_HPP

#include "solver/iteration_control.hpp"

namespace vibeqc::scf::solver {

// Compatibility aliases for existing SCF/KS callers. New method-neutral
// consumers should include solver/iteration_control.hpp directly.
using ::vibeqc::solver::BoundedIterationResult;
using ::vibeqc::solver::run_bounded_iterations;

}  // namespace vibeqc::scf::solver

#endif
