#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/eigensolver_workspace.hpp"
#include "scf/solver/eigen_frame.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {
struct CudaDensityFittingJkPlan;

/** Bounded workspace allowance for the existing ordinary Xsyevd provider.
 * Its actual query is checked before allocation; a stack requiring more is
 * rejected explicitly. One serialized AO frame serves all items and spins. */
inline std::size_t df_eigen_workspace_allowance(std::size_t n) {
  return ordinary_eigensolver_workspace_allowance(n);
}

/** Compact batched solves own one workspace, independent of the metric solve.
 * Reserve the provider's fixed floor and matrix storage for every item;
 * both batched providers check their actual query before allocating. */
inline std::size_t df_scf_workspace_allowance(std::size_t n, std::size_t batch) {
  const auto per_item = df_eigen_workspace_allowance(n);
  if (!batch || batch > std::numeric_limits<std::size_t>::max() / per_item)
    throw std::overflow_error("DF batched eigensolver workspace size overflows");
  return batch * per_item;
}

inline std::size_t df_eigen_device_reservation(std::size_t n) {
  const auto workspace = df_eigen_workspace_allowance(n);
  const long double bytes = static_cast<long double>(workspace) +
                            (3.0L * n * n + n) * sizeof(double) + sizeof(int) +
                            sizeof(unsigned char);
  if (bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("DF eigensystem storage size overflows");
  return static_cast<std::size_t>(bytes);
}

/** Per-operation numerical and allocation evidence; not a convergence claim.
 * Eigen residuals refer to the supplied matrix; the adapter never substitutes
 * another Fock. A caller solving a DIIS matrix must not interpret this as a
 * physical-state validation. All matrices use detached row-major layout. */
struct CudaDfEigenDiagnostic : solver::EigenFrameDiagnostic {
  std::size_t device_bytes{}, host_workspace_bytes{}, solver_calls{};
  bool workspace_reused{};
};

/** Detect invalid ownership before clearing outputs, preserving aliased inputs. */
inline bool df_eigen_outputs_alias(const std::vector<double>& matrix,
                                   const std::vector<double>* overlap,
                                   const std::vector<double>* orthogonalizer,
                                   const std::vector<double>& eigenvalues,
                                   const std::vector<double>& coefficients) {
  return &eigenvalues == &coefficients || &matrix == &eigenvalues || &matrix == &coefficients ||
         overlap == &eigenvalues || overlap == &coefficients || orthogonalizer == &eigenvalues ||
         orthogonalizer == &coefficients;
}

/** Use the prepared DF plan's existing stream/handles for a checked FP64 solve.
 * Null S/X requests an ordinary symmetric eigensystem; otherwise both must
 * be present and X must be the caller's validated symmetric S^(-1/2).
 * Eigenvalues are ascending, coefficient columns satisfy F C = S C epsilon.
 * Calls are serialized by the plan owner and require a noncapturing stream.
 * Ordinary execution is independent of Graph eligibility. Inputs and the two
 * outputs must be distinct vectors: alias rejection leaves them unchanged.
 * Other failures clear outputs; no CPU fallback or other item's SCF buffers
 * are used. */
vibeqc_status solve_cuda_density_fitting_eigen(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& matrix,
    const std::vector<double>* overlap, const std::vector<double>* orthogonalizer,
    std::vector<double>& eigenvalues, std::vector<double>& coefficients,
    CudaDfEigenDiagnostic& diagnostic, std::string& detail, std::size_t system_index = 0);

}  // namespace vibeqc::scf
