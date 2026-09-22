#pragma once

#include <cstddef>
#include <vector>

#include "tensor/symmetric_matrix_function.hpp"

namespace vibeqc::integrals {
/** Conditioning diagnostics and symmetric inverse square root of (P|Q). */
struct DensityFittingMetricFactor {
  std::size_t dimension{};
  std::size_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
  std::vector<double> inverse_square_root;
};

/** Factor a Coulomb metric using its fixed relative spectral cutoff.
 * Shared by SCF and correlated response; no SCF iteration policy is owned here. */
DensityFittingMetricFactor factor_density_fitting_metric(const std::vector<double>& metric,
                                                         std::size_t dimension,
                                                         double relative_threshold = 1.0e-10);

/** Validate a supplied spectral value and apply its fixed-branch response.
 * Retained/discarded cross terms and unresolved rank crossings follow the
 * shared symmetric-matrix VJP contract. The eigensolver stays deterministic. */
std::vector<double> density_fitting_metric_response(
    const std::vector<double>& metric, const std::vector<double>& function_value_matrix,
    const std::vector<double>& response, std::size_t dimension, double relative_threshold,
    tensor::SymmetricMatrixFunction function);
}  // namespace vibeqc::integrals
