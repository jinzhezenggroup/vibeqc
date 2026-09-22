#include "scf/solver/warm_subspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::scf::solver {
namespace {
bool finite_vector(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

WarmSubspaceResidual nonfinite_residual() {
  const double infinity = std::numeric_limits<double>::infinity();
  return {infinity, infinity, infinity, false};
}
}  // namespace

WarmSubspaceResidual inspect_warm_occupied_subspace(const std::vector<double>& orthonormal_fock,
                                                    const std::vector<double>& previous_orbitals,
                                                    std::size_t n, std::size_t occupied) {
  if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n ||
      orthonormal_fock.size() != n * n || previous_orbitals.size() != n * n || occupied > n) {
    throw std::invalid_argument("warm occupied-subspace input has inconsistent shape");
  }
  if (!finite_vector(orthonormal_fock) || !finite_vector(previous_orbitals)) {
    return nonfinite_residual();
  }

  // Store only F*C_occ and the occupied Rayleigh matrix.  The diagnostic is
  // deliberately O(n^2*nocc), matching the two-GEMM structure intended for
  // the CUDA path without manufacturing an n-by-n dense eigensolve.
  std::vector<double> fc(n * occupied, 0.0);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < occupied; ++column) {
      double value = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        value += orthonormal_fock[row * n + k] * previous_orbitals[k * n + column];
      }
      fc[row * occupied + column] = value;
    }
  }

  std::vector<double> rayleigh(occupied * occupied, 0.0);
  for (std::size_t row = 0; row < occupied; ++row) {
    for (std::size_t column = 0; column < occupied; ++column) {
      double value = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        value += previous_orbitals[k * n + row] * fc[k * occupied + column];
      }
      rayleigh[row * occupied + column] = value;
    }
  }

  WarmSubspaceResidual diagnostic;
  double fock_norm = 0.0;
  double occupied_norm = 0.0;
  double fc_norm = 0.0;
  for (double value : orthonormal_fock) fock_norm = std::hypot(fock_norm, value);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < occupied; ++column)
      occupied_norm = std::hypot(occupied_norm, previous_orbitals[row * n + column]);

  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < occupied; ++column) {
      const double fc_value = fc[row * occupied + column];
      fc_norm = std::hypot(fc_norm, fc_value);
      double projected = 0.0;
      for (std::size_t k = 0; k < occupied; ++k) {
        projected += previous_orbitals[row * n + k] * rayleigh[k * occupied + column];
      }
      const double residual = fc_value - projected;
      diagnostic.maximum_residual = std::max(diagnostic.maximum_residual, std::abs(residual));
      diagnostic.frobenius_residual = std::hypot(diagnostic.frobenius_residual, residual);
    }
  }

  const double scale = fock_norm * occupied_norm + fc_norm;
  // Finite input entries do not guarantee representable normalization norms.
  // Reject an overflowed scale rather than reporting a spurious zero residual.
  if (!std::isfinite(fock_norm) || !std::isfinite(occupied_norm) || !std::isfinite(fc_norm) ||
      !std::isfinite(scale))
    return nonfinite_residual();
  diagnostic.scaled_residual =
      scale == 0.0 ? diagnostic.frobenius_residual : diagnostic.frobenius_residual / scale;
  diagnostic.finite = std::isfinite(diagnostic.maximum_residual) &&
                      std::isfinite(diagnostic.frobenius_residual) &&
                      std::isfinite(diagnostic.scaled_residual);
  if (!diagnostic.finite) return nonfinite_residual();
  return diagnostic;
}

bool accept_warm_occupied_subspace(const WarmSubspaceResidual& diagnostic, double maximum_tolerance,
                                   double scaled_tolerance, std::string& detail) {
  if (!diagnostic.finite || !std::isfinite(maximum_tolerance) || !std::isfinite(scaled_tolerance) ||
      maximum_tolerance < 0.0 || scaled_tolerance < 0.0 ||
      !std::isfinite(diagnostic.maximum_residual) ||
      !std::isfinite(diagnostic.frobenius_residual) || !std::isfinite(diagnostic.scaled_residual) ||
      diagnostic.maximum_residual < 0.0 || diagnostic.frobenius_residual < 0.0 ||
      diagnostic.scaled_residual < 0.0 || diagnostic.maximum_residual > maximum_tolerance ||
      diagnostic.scaled_residual > scaled_tolerance) {
    detail = "warm occupied subspace failed residual gate; use dense eigensolver";
    return false;
  }
  detail.clear();
  return true;
}

}  // namespace vibeqc::scf::solver
