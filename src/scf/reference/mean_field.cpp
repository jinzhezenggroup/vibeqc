#include "scf/reference/mean_field.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc::scf::reference {

Matrix density_from_orbitals(const Matrix& coefficients, std::size_t n, std::size_t occupied,
                             double occupation_weight) {
  Matrix density(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        density[index(mu, nu, n)] += occupation_weight * coefficients[index(mu, orbital, n)] *
                                     coefficients[index(nu, orbital, n)];
      }
    }
  }
  return density;
}

Matrix energy_weighted_density(const Matrix& coefficients, const std::vector<double>& energies,
                               std::size_t n, std::size_t occupied, double occupation_weight) {
  Matrix weighted(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        weighted[index(mu, nu, n)] += occupation_weight * energies[orbital] *
                                      coefficients[index(mu, orbital, n)] *
                                      coefficients[index(nu, orbital, n)];
      }
    }
  }
  return weighted;
}

double uhf_electronic_energy(const Matrix& alpha_density, const Matrix& beta_density,
                             const Matrix& hcore, const Matrix& alpha_fock,
                             const Matrix& beta_fock) {
  double energy = 0.0;
  for (std::size_t element = 0; element < hcore.size(); ++element) {
    energy += 0.5 * alpha_density[element] * (hcore[element] + alpha_fock[element]);
    energy += 0.5 * beta_density[element] * (hcore[element] + beta_fock[element]);
  }
  return energy;
}

Matrix concatenate(const Matrix& first, const Matrix& second) {
  Matrix joined;
  joined.reserve(first.size() + second.size());
  joined.insert(joined.end(), first.begin(), first.end());
  joined.insert(joined.end(), second.begin(), second.end());
  return joined;
}

std::pair<Matrix, Matrix> split_spin_matrices(const Matrix& joined, std::size_t matrix_size) {
  if (joined.size() != 2 * matrix_size) {
    throw std::invalid_argument("joined UHF matrix has an invalid size");
  }
  return {
      Matrix(joined.begin(), joined.begin() + matrix_size),
      Matrix(joined.begin() + matrix_size, joined.end()),
  };
}

double electronic_energy(const Matrix& density, const Matrix& hcore, const Matrix& fock) {
  double energy = 0.0;
  for (std::size_t i = 0; i < density.size(); ++i) {
    energy += 0.5 * density[i] * (hcore[i] + fock[i]);
  }
  return energy;
}

Matrix commutator_residual(const Matrix& fock, const Matrix& density, const Matrix& overlap,
                           std::size_t n) {
  const Matrix fps = multiply(multiply(fock, density, n), overlap, n);
  const Matrix spf = multiply(multiply(overlap, density, n), fock, n);
  Matrix residual(n * n);
  for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = fps[i] - spf[i];
  return residual;
}

double density_rms(const Matrix& a, const Matrix& b) {
  double square = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double delta = a[i] - b[i];
    square += delta * delta;
  }
  return std::sqrt(square / static_cast<double>(a.size()));
}

double residual_rms(const Matrix& residual) {
  return std::sqrt(dot(residual, residual) / static_cast<double>(residual.size()));
}

}  // namespace vibeqc::scf::reference
