#ifndef VIBEQC_SCF_TYPES_HPP
#define VIBEQC_SCF_TYPES_HPP

#include <vector>

namespace vibeqc::scf {

/** Numerical controls shared by the implemented mean-field solvers. */
struct ScfOptions {
  unsigned max_iterations{100};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-10};
  double density_tolerance{1.0e-8};
  double screening_tolerance{1.0e-12};
  /** Select the DF solver; direct four-center remains the default. */
  vibeqc_density_fitting_mode density_fitting_mode{VIBEQC_DENSITY_FITTING_NONE};
  /** Relative cutoff used when factoring the auxiliary Coulomb metric. */
  double density_fitting_relative_threshold{1.0e-10};
  /** Byte budget for bounded DF plan/integral work; zero means implementation default. */
  std::size_t density_fitting_memory_budget_bytes{};
};

/** Internal mean-field result, including state retained for warm starts. */
struct ScfResult {
  double energy{};
  std::vector<double> forces;
  // RHF stores one N x N AO density; UHF stores alpha then beta matrices.
  // The state is explicit and remains private to prepared execution plans.
  std::vector<double> density;
  unsigned iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  bool initial_density_used{};
};

}  // namespace vibeqc::scf

#endif
