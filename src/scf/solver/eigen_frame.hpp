#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vibeqc::scf::solver {
/** Backend-neutral evidence for one real symmetric/generalized eigenframe. */
struct EigenFrameDiagnostic {
  double maximum_eigen_residual{}, scaled_eigen_residual{}, maximum_metric_error{};
  int solver_info{};
};
/** Apply the common absolute/scaled gates to finite backend-produced evidence.
 * Product overflow must be reported by the producer before calling this gate. */
bool accept_eigen_frame(const EigenFrameDiagnostic& diagnostic, std::string& detail);

/** Validate the returned frame independently of the production eigen backend.
 * A null overlap means I. The existing physical-reference absolute 1e-8
 * eigen/metric gates also apply here, alongside a scaled 1e-12 residual.
 * Eigenvector signs and rotations inside degenerate subspaces are immaterial.
 * Nonzero solver info, missing outputs and nonfinite data always fail. */
bool validate_eigen_frame(const std::vector<double>& matrix, const std::vector<double>* overlap,
                          const std::vector<double>& eigenvalues,
                          const std::vector<double>& coefficients, std::size_t n,
                          EigenFrameDiagnostic& diagnostic, std::string& detail);
}  // namespace vibeqc::scf::solver
