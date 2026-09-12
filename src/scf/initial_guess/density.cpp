#include "scf/initial_guess/density.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/reference/mean_field.hpp"

namespace vibeqc::scf::initial_guess {

using reference::density_from_orbitals;
using reference::generalized_eigen;
using reference::index;

void mix_open_shell_frontier_orbitals(Matrix& beta_coefficients, std::size_t n,
                                      std::size_t alpha_occupied, std::size_t beta_occupied) {
  if (alpha_occupied == beta_occupied || beta_occupied == 0 || beta_occupied >= n) {
    return;
  }
  // Exact molecular symmetry can make a core-Hamiltonian UHF guess an
  // excited-state fixed point (for example, the sigma-hole state of linear
  // OH). A 45-degree orthogonal HOMO/LUMO rotation preserves electron count
  // and S-orthonormality while moving the seed outside that excited state's
  // basin. The converged orbitals, not this seed angle, define the result.
  constexpr double cosine = 0.7071067811865476;
  constexpr double sine = 0.7071067811865476;
  const std::size_t occupied_orbital = beta_occupied - 1;
  const std::size_t virtual_orbital = beta_occupied;
  for (std::size_t row = 0; row < n; ++row) {
    const double occupied_value = beta_coefficients[index(row, occupied_orbital, n)];
    const double virtual_value = beta_coefficients[index(row, virtual_orbital, n)];
    beta_coefficients[index(row, occupied_orbital, n)] =
        cosine * occupied_value + sine * virtual_value;
    beta_coefficients[index(row, virtual_orbital, n)] =
        -sine * occupied_value + cosine * virtual_value;
  }
}

std::pair<std::size_t, std::size_t> spin_occupations(const core::System& system) {
  const std::size_t electrons = static_cast<std::size_t>(system.electron_count);
  const std::size_t spin_excess = static_cast<std::size_t>(system.multiplicity - 1);
  if (spin_excess > electrons || ((electrons + spin_excess) & 1U) != 0U) {
    throw std::invalid_argument(
        "electron count and multiplicity do not define integral UHF occupations");
  }
  const std::size_t alpha = (electrons + spin_excess) / 2;
  return {alpha, electrons - alpha};
}

void normalize_spin_density(Matrix& density, const Matrix& overlap, std::size_t n,
                            std::size_t target_electrons) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric = 0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
      density[index(i, j, n)] = symmetric;
      density[index(j, i, n)] = symmetric;
    }
  }
  if (target_electrons == 0) {
    std::fill(density.begin(), density.end(), 0.0);
    return;
  }
  double electron_trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      electron_trace += density[index(i, j, n)] * overlap[index(j, i, n)];
    }
  }
  if (!(electron_trace > 0.0) || !std::isfinite(electron_trace)) {
    throw std::invalid_argument("initial spin density has an invalid electron trace");
  }
  const double scale = static_cast<double>(target_electrons) / electron_trace;
  for (double& value : density) value *= scale;
}

std::pair<Matrix, Matrix> prepare_initial_uhf_density(
    const integrals::IntegralData& ints, const Matrix& orthogonalizer, std::size_t alpha_occupied,
    std::size_t beta_occupied, const std::vector<double>* initial_density,
    EigenResult& alpha_orbitals, EigenResult& beta_orbitals) {
  const std::size_t n = ints.nbf;
  const std::size_t matrix_size = n * n;
  alpha_orbitals = generalized_eigen(ints.hcore, orthogonalizer, n);
  beta_orbitals = alpha_orbitals;
  if (initial_density == nullptr) {
    mix_open_shell_frontier_orbitals(beta_orbitals.vectors, n, alpha_occupied, beta_occupied);
    return {
        density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0),
        density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0),
    };
  }
  if (initial_density->size() != 2 * matrix_size ||
      !std::all_of(initial_density->begin(), initial_density->end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "initial UHF density must contain finite alpha and beta AO matrices");
  }
  Matrix alpha(initial_density->begin(), initial_density->begin() + matrix_size);
  Matrix beta(initial_density->begin() + matrix_size, initial_density->end());
  normalize_spin_density(alpha, ints.overlap, n, alpha_occupied);
  normalize_spin_density(beta, ints.overlap, n, beta_occupied);
  return {std::move(alpha), std::move(beta)};
}

Matrix prepare_initial_density(const core::System& system, const integrals::IntegralData& ints,
                               const Matrix& orthogonalizer, std::size_t occupied,
                               const std::vector<double>* initial_density, EigenResult& orbitals) {
  const std::size_t n = ints.nbf;
  orbitals = generalized_eigen(ints.hcore, orthogonalizer, n);
  if (initial_density == nullptr) {
    return density_from_orbitals(orbitals.vectors, n, occupied);
  }
  if (initial_density->size() != n * n ||
      !std::all_of(initial_density->begin(), initial_density->end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("initial density does not match the finite AO matrix topology");
  }
  Matrix density = *initial_density;
  // A density from the same AO topology but a different geometry is a useful
  // guess, although its electron trace changes with the new overlap. Restore
  // symmetry and electron count before entering SCF so warm starts do not
  // introduce a geometry-dependent charge error.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric = 0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
      density[index(i, j, n)] = symmetric;
      density[index(j, i, n)] = symmetric;
    }
  }
  double electron_trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      electron_trace += density[index(i, j, n)] * ints.overlap[index(j, i, n)];
    }
  }
  if (!(electron_trace > 0.0) || !std::isfinite(electron_trace)) {
    throw std::invalid_argument("initial density has an invalid electron trace");
  }
  const double trace_scale = static_cast<double>(system.electron_count) / electron_trace;
  for (double& value : density) value *= trace_scale;
  return density;
}

}  // namespace vibeqc::scf::initial_guess
