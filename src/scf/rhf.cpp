#include "scf/rhf.hpp"

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/density_fitting.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace vibeqc::scf {
namespace {

using Matrix = std::vector<double>;

std::size_t index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

std::size_t eri_index(std::size_t i,
                      std::size_t j,
                      std::size_t k,
                      std::size_t l,
                      std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

Matrix identity(std::size_t n) {
  Matrix result(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) result[index(i, i, n)] = 1.0;
  return result;
}

Matrix multiply(const Matrix& a, const Matrix& b, std::size_t n) {
  Matrix out(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = a[index(i, k, n)];
      for (std::size_t j = 0; j < n; ++j) {
        out[index(i, j, n)] += aik * b[index(k, j, n)];
      }
    }
  }
  return out;
}

Matrix transpose(const Matrix& a, std::size_t n) {
  Matrix out(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[index(j, i, n)] = a[index(i, j, n)];
    }
  }
  return out;
}

struct EigenResult {
  std::vector<double> values;
  Matrix vectors;  // eigenvectors are columns
};

EigenResult symmetric_eigen(Matrix matrix, std::size_t n) {
  Matrix vectors = identity(n);
  const std::size_t max_sweeps = std::max<std::size_t>(50, 20 * n * n);
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 0;
    double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double candidate = std::abs(matrix[index(i, j, n)]);
        if (candidate > largest) {
          largest = candidate;
          p = i;
          q = j;
        }
      }
    }
    if (largest < 1.0e-14) break;

    const double app = matrix[index(p, p, n)];
    const double aqq = matrix[index(q, q, n)];
    const double apq = matrix[index(p, q, n)];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double mkp = matrix[index(k, p, n)];
      const double mkq = matrix[index(k, q, n)];
      matrix[index(k, p, n)] = matrix[index(p, k, n)] = c * mkp - s * mkq;
      matrix[index(k, q, n)] = matrix[index(q, k, n)] = s * mkp + c * mkq;
    }
    matrix[index(p, p, n)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[index(q, q, n)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[index(p, q, n)] = matrix[index(q, p, n)] = 0.0;

    for (std::size_t k = 0; k < n; ++k) {
      const double vkp = vectors[index(k, p, n)];
      const double vkq = vectors[index(k, q, n)];
      vectors[index(k, p, n)] = c * vkp - s * vkq;
      vectors[index(k, q, n)] = s * vkp + c * vkq;
    }
  }

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return matrix[index(a, a, n)] < matrix[index(b, b, n)];
  });
  EigenResult result;
  result.values.resize(n);
  result.vectors.resize(n * n);
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source = order[column];
    result.values[column] = matrix[index(source, source, n)];
    for (std::size_t row = 0; row < n; ++row) {
      result.vectors[index(row, column, n)] = vectors[index(row, source, n)];
    }
  }
  return result;
}

Matrix symmetric_orthogonalizer(const Matrix& overlap, std::size_t n) {
  const EigenResult eigen = symmetric_eigen(overlap, n);
  Matrix scaled = eigen.vectors;
  for (std::size_t column = 0; column < n; ++column) {
    if (eigen.values[column] < 1.0e-10) {
      throw std::runtime_error("overlap matrix is singular or severely linearly dependent");
    }
    const double factor = 1.0 / std::sqrt(eigen.values[column]);
    for (std::size_t row = 0; row < n; ++row) {
      scaled[index(row, column, n)] *= factor;
    }
  }
  return multiply(scaled, transpose(eigen.vectors, n), n);
}

EigenResult generalized_eigen(const Matrix& fock,
                              const Matrix& orthogonalizer,
                              std::size_t n) {
  const Matrix transformed = multiply(
      transpose(orthogonalizer, n), multiply(fock, orthogonalizer, n), n);
  EigenResult result = symmetric_eigen(transformed, n);
  result.vectors = multiply(orthogonalizer, result.vectors, n);
  return result;
}

Matrix density_from_orbitals(const Matrix& coefficients,
                             std::size_t n,
                             std::size_t occupied,
                             double occupation_weight = 2.0) {
  Matrix density(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        density[index(mu, nu, n)] +=
            occupation_weight * coefficients[index(mu, orbital, n)] *
            coefficients[index(nu, orbital, n)];
      }
    }
  }
  return density;
}

void mix_open_shell_frontier_orbitals(Matrix& beta_coefficients,
                                      std::size_t n,
                                      std::size_t alpha_occupied,
                                      std::size_t beta_occupied) {
  if (alpha_occupied == beta_occupied || beta_occupied == 0 ||
      beta_occupied >= n) {
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
    const double occupied_value =
        beta_coefficients[index(row, occupied_orbital, n)];
    const double virtual_value =
        beta_coefficients[index(row, virtual_orbital, n)];
    beta_coefficients[index(row, occupied_orbital, n)] =
        cosine * occupied_value + sine * virtual_value;
    beta_coefficients[index(row, virtual_orbital, n)] =
        -sine * occupied_value + cosine * virtual_value;
  }
}

std::pair<std::size_t, std::size_t> spin_occupations(
    const core::System& system) {
  const std::size_t electrons = static_cast<std::size_t>(system.electron_count);
  const std::size_t spin_excess =
      static_cast<std::size_t>(system.multiplicity - 1);
  if (spin_excess > electrons || ((electrons + spin_excess) & 1U) != 0U) {
    throw std::invalid_argument(
        "electron count and multiplicity do not define integral UHF occupations");
  }
  const std::size_t alpha = (electrons + spin_excess) / 2;
  return {alpha, electrons - alpha};
}

void normalize_spin_density(Matrix& density,
                            const Matrix& overlap,
                            std::size_t n,
                            std::size_t target_electrons) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric =
          0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
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
    const integrals::IntegralData& ints,
    const Matrix& orthogonalizer,
    std::size_t alpha_occupied,
    std::size_t beta_occupied,
    const std::vector<double>* initial_density,
    EigenResult& alpha_orbitals,
    EigenResult& beta_orbitals) {
  const std::size_t n = ints.nbf;
  const std::size_t matrix_size = n * n;
  alpha_orbitals = generalized_eigen(ints.hcore, orthogonalizer, n);
  beta_orbitals = alpha_orbitals;
  if (initial_density == nullptr) {
    mix_open_shell_frontier_orbitals(
        beta_orbitals.vectors, n, alpha_occupied, beta_occupied);
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
  Matrix alpha(initial_density->begin(),
               initial_density->begin() + matrix_size);
  Matrix beta(initial_density->begin() + matrix_size, initial_density->end());
  normalize_spin_density(alpha, ints.overlap, n, alpha_occupied);
  normalize_spin_density(beta, ints.overlap, n, beta_occupied);
  return {std::move(alpha), std::move(beta)};
}

Matrix prepare_initial_density(const core::System& system,
                               const integrals::IntegralData& ints,
                               const Matrix& orthogonalizer,
                               std::size_t occupied,
                               const std::vector<double>* initial_density,
                               EigenResult& orbitals) {
  const std::size_t n = ints.nbf;
  orbitals = generalized_eigen(ints.hcore, orthogonalizer, n);
  if (initial_density == nullptr) {
    return density_from_orbitals(orbitals.vectors, n, occupied);
  }
  if (initial_density->size() != n * n ||
      !std::all_of(initial_density->begin(), initial_density->end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "initial density does not match the finite AO matrix topology");
  }
  Matrix density = *initial_density;
  // A density from the same AO topology but a different geometry is a useful
  // guess, although its electron trace changes with the new overlap. Restore
  // symmetry and electron count before entering SCF so warm starts do not
  // introduce a geometry-dependent charge error.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric =
          0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
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
  const double trace_scale =
      static_cast<double>(system.electron_count) / electron_trace;
  for (double& value : density) value *= trace_scale;
  return density;
}

Matrix energy_weighted_density(const Matrix& coefficients,
                               const std::vector<double>& energies,
                               std::size_t n,
                               std::size_t occupied,
                               double occupation_weight = 2.0) {
  Matrix weighted(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        weighted[index(mu, nu, n)] +=
            occupation_weight * energies[orbital] *
            coefficients[index(mu, orbital, n)] *
            coefficients[index(nu, orbital, n)];
      }
    }
  }
  return weighted;
}

std::pair<Matrix, Matrix> build_uhf_focks(
    const Matrix& hcore,
    const std::vector<double>& eri,
    const Matrix& alpha_density,
    const Matrix& beta_density,
    std::size_t n) {
  Matrix alpha_fock = hcore;
  Matrix beta_fock = hcore;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double coulomb = 0.0;
      double alpha_exchange = 0.0;
      double beta_exchange = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          const double alpha = alpha_density[index(k, l, n)];
          const double beta = beta_density[index(k, l, n)];
          coulomb += (alpha + beta) * eri[eri_index(i, j, k, l, n)];
          alpha_exchange += alpha * eri[eri_index(i, k, j, l, n)];
          beta_exchange += beta * eri[eri_index(i, k, j, l, n)];
        }
      }
      alpha_fock[index(i, j, n)] += coulomb - alpha_exchange;
      beta_fock[index(i, j, n)] += coulomb - beta_exchange;
    }
  }
  return {std::move(alpha_fock), std::move(beta_fock)};
}

double uhf_electronic_energy(const Matrix& alpha_density,
                             const Matrix& beta_density,
                             const Matrix& hcore,
                             const Matrix& alpha_fock,
                             const Matrix& beta_fock) {
  double energy = 0.0;
  for (std::size_t element = 0; element < hcore.size(); ++element) {
    energy += 0.5 * alpha_density[element] *
              (hcore[element] + alpha_fock[element]);
    energy += 0.5 * beta_density[element] *
              (hcore[element] + beta_fock[element]);
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

std::pair<Matrix, Matrix> split_spin_matrices(const Matrix& joined,
                                               std::size_t matrix_size) {
  if (joined.size() != 2 * matrix_size) {
    throw std::invalid_argument("joined UHF matrix has an invalid size");
  }
  return {
      Matrix(joined.begin(), joined.begin() + matrix_size),
      Matrix(joined.begin() + matrix_size, joined.end()),
  };
}

Matrix build_fock(const Matrix& hcore,
                  const std::vector<double>& eri,
                  const Matrix& density,
                  std::size_t n) {
  Matrix fock = hcore;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double coulomb = 0.0;
      double exchange = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          const double pkl = density[index(k, l, n)];
          coulomb += pkl * eri[eri_index(i, j, k, l, n)];
          exchange += pkl * eri[eri_index(i, k, j, l, n)];
        }
      }
      fock[index(i, j, n)] += coulomb - 0.5 * exchange;
    }
  }
  return fock;
}

double electronic_energy(const Matrix& density,
                         const Matrix& hcore,
                         const Matrix& fock) {
  double energy = 0.0;
  for (std::size_t i = 0; i < density.size(); ++i) {
    energy += 0.5 * density[i] * (hcore[i] + fock[i]);
  }
  return energy;
}

Matrix commutator_residual(const Matrix& fock,
                           const Matrix& density,
                           const Matrix& overlap,
                           std::size_t n) {
  const Matrix fps = multiply(multiply(fock, density, n), overlap, n);
  const Matrix spf = multiply(multiply(overlap, density, n), fock, n);
  Matrix residual(n * n);
  for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = fps[i] - spf[i];
  return residual;
}

double dot(const Matrix& a, const Matrix& b) {
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
  return result;
}

bool solve_linear(Matrix a, std::vector<double> b, std::vector<double>& x, std::size_t n) {
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row) {
      if (std::abs(a[index(row, column, n)]) >
          std::abs(a[index(pivot, column, n)])) {
        pivot = row;
      }
    }
    if (std::abs(a[index(pivot, column, n)]) < 1.0e-14) return false;
    if (pivot != column) {
      for (std::size_t j = 0; j < n; ++j) {
        std::swap(a[index(column, j, n)], a[index(pivot, j, n)]);
      }
      std::swap(b[column], b[pivot]);
    }
    const double diagonal = a[index(column, column, n)];
    for (std::size_t j = column; j < n; ++j) a[index(column, j, n)] /= diagonal;
    b[column] /= diagonal;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == column) continue;
      const double factor = a[index(row, column, n)];
      for (std::size_t j = column; j < n; ++j) {
        a[index(row, j, n)] -= factor * a[index(column, j, n)];
      }
      b[row] -= factor * b[column];
    }
  }
  x = std::move(b);
  return true;
}

class Diis {
 public:
  explicit Diis(std::size_t capacity) : capacity_(capacity) {}

  Matrix update(const Matrix& fock, const Matrix& residual) {
    if (capacity_ < 2) return fock;
    focks_.push_back(fock);
    residuals_.push_back(residual);
    if (focks_.size() > capacity_) {
      focks_.erase(focks_.begin());
      residuals_.erase(residuals_.begin());
    }
    if (focks_.size() < 2) return fock;

    const std::size_t m = focks_.size();
    const std::size_t dim = m + 1;
    Matrix b(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    rhs[m] = -1.0;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        b[index(i, j, dim)] = dot(residuals_[i], residuals_[j]);
      }
      b[index(i, m, dim)] = -1.0;
      b[index(m, i, dim)] = -1.0;
    }
    std::vector<double> coefficients;
    if (!solve_linear(std::move(b), std::move(rhs), coefficients, dim)) return fock;

    Matrix extrapolated(fock.size(), 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t element = 0; element < fock.size(); ++element) {
        extrapolated[element] += coefficients[i] * focks_[i][element];
      }
    }
    return extrapolated;
  }

 private:
  std::size_t capacity_;
  std::vector<Matrix> focks_;
  std::vector<Matrix> residuals_;
};

double density_rms(const Matrix& a, const Matrix& b) {
  double square = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double delta = a[i] - b[i];
    square += delta * delta;
  }
  return std::sqrt(square / static_cast<double>(a.size()));
}

std::vector<double> analytic_forces(const integrals::IntegralData& ints,
                                    const Matrix& density,
                                    const Matrix& weighted_density) {
  const std::size_t n = ints.nbf;
  std::vector<double> forces(ints.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < ints.ncoord; ++coordinate) {
    const double* ds = ints.overlap_derivative.data() + coordinate * n * n;
    const double* dh = ints.hcore_derivative.data() + coordinate * n * n;
    const double* deri = ints.eri_derivative.data() + coordinate * n * n * n * n;
    double derivative = ints.nuclear_repulsion_derivative[coordinate];
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        derivative += density[index(i, j, n)] * dh[index(i, j, n)];
        derivative -= weighted_density[index(i, j, n)] * ds[index(i, j, n)];
        double coulomb_derivative = 0.0;
        double exchange_derivative = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          for (std::size_t l = 0; l < n; ++l) {
            const double pkl = density[index(k, l, n)];
            coulomb_derivative += pkl * deri[eri_index(i, j, k, l, n)];
            exchange_derivative += pkl * deri[eri_index(i, k, j, l, n)];
          }
        }
        derivative += 0.5 * density[index(i, j, n)] *
                      (coulomb_derivative - 0.5 * exchange_derivative);
      }
    }
    forces[coordinate] = -derivative;
  }
  return forces;
}

std::vector<double> analytic_uhf_forces(
    const integrals::IntegralData& ints,
    const Matrix& alpha_density,
    const Matrix& beta_density,
    const Matrix& alpha_weighted_density,
    const Matrix& beta_weighted_density) {
  const std::size_t n = ints.nbf;
  Matrix total_density(n * n);
  Matrix total_weighted(n * n);
  for (std::size_t element = 0; element < n * n; ++element) {
    total_density[element] = alpha_density[element] + beta_density[element];
    total_weighted[element] =
        alpha_weighted_density[element] + beta_weighted_density[element];
  }
  std::vector<double> forces(ints.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < ints.ncoord; ++coordinate) {
    const double* ds = ints.overlap_derivative.data() + coordinate * n * n;
    const double* dh = ints.hcore_derivative.data() + coordinate * n * n;
    const double* deri =
        ints.eri_derivative.data() + coordinate * n * n * n * n;
    double derivative = ints.nuclear_repulsion_derivative[coordinate];
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t ij = index(i, j, n);
        derivative += total_density[ij] * dh[ij];
        derivative -= total_weighted[ij] * ds[ij];
        double coulomb_derivative = 0.0;
        double alpha_exchange_derivative = 0.0;
        double beta_exchange_derivative = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          for (std::size_t l = 0; l < n; ++l) {
            const std::size_t kl = index(k, l, n);
            coulomb_derivative +=
                total_density[kl] * deri[eri_index(i, j, k, l, n)];
            alpha_exchange_derivative +=
                alpha_density[kl] * deri[eri_index(i, k, j, l, n)];
            beta_exchange_derivative +=
                beta_density[kl] * deri[eri_index(i, k, j, l, n)];
          }
        }
        derivative += 0.5 * total_density[ij] * coulomb_derivative;
        derivative -=
            0.5 * alpha_density[ij] * alpha_exchange_derivative;
        derivative -= 0.5 * beta_density[ij] * beta_exchange_derivative;
      }
    }
    forces[coordinate] = -derivative;
  }
  return forces;
}

void finalize_scf(const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer,
                  std::size_t occupied,
                  Matrix& density,
                  ScfResult& result) {
  const std::size_t n = ints.nbf;
  Matrix final_fock = build_fock(ints.hcore, ints.eri, density, n);
  EigenResult orbitals = generalized_eigen(final_fock, orthogonalizer, n);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final_fock = build_fock(ints.hcore, ints.eri, density, n);
  result.energy = electronic_energy(density, ints.hcore, final_fock) +
                  ints.nuclear_repulsion;
  const Matrix weighted = energy_weighted_density(
      orbitals.vectors, orbitals.values, n, occupied);
  result.forces = analytic_forces(ints, density, weighted);
  result.density = density;
}

void finalize_uhf(const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer,
                  std::size_t alpha_occupied,
                  std::size_t beta_occupied,
                  Matrix& alpha_density,
                  Matrix& beta_density,
                  ScfResult& result) {
  const std::size_t n = ints.nbf;
  auto [alpha_fock, beta_fock] = build_uhf_focks(
      ints.hcore, ints.eri, alpha_density, beta_density, n);
  EigenResult alpha_orbitals =
      generalized_eigen(alpha_fock, orthogonalizer, n);
  EigenResult beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
  alpha_density = density_from_orbitals(
      alpha_orbitals.vectors, n, alpha_occupied, 1.0);
  beta_density = density_from_orbitals(
      beta_orbitals.vectors, n, beta_occupied, 1.0);
  std::tie(alpha_fock, beta_fock) = build_uhf_focks(
      ints.hcore, ints.eri, alpha_density, beta_density, n);
  result.energy = uhf_electronic_energy(
      alpha_density, beta_density, ints.hcore, alpha_fock, beta_fock) +
      ints.nuclear_repulsion;
  const Matrix alpha_weighted = energy_weighted_density(
      alpha_orbitals.vectors, alpha_orbitals.values, n, alpha_occupied, 1.0);
  const Matrix beta_weighted = energy_weighted_density(
      beta_orbitals.vectors, beta_orbitals.values, n, beta_occupied, 1.0);
  result.forces = analytic_uhf_forces(
      ints, alpha_density, beta_density, alpha_weighted, beta_weighted);
  result.density = concatenate(alpha_density, beta_density);
}

/** Immutable integral state shared by all iterations of a DF SCF solve. */
struct DensityFittingScfData {
  integrals::IntegralData one_electron;
  integrals::DensityFittingIntegralData raw;
  DensityFittingThreeCenter three_center;
};

/** Assemble immutable DF state from already-evaluated one- and three-center data. */
DensityFittingScfData assemble_density_fitting_data(
    integrals::IntegralData one_electron,
    integrals::DensityFittingIntegralData raw,
    double relative_threshold) {
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0) ||
      !std::isfinite(relative_threshold)) {
    throw std::invalid_argument(
        "DF metric relative threshold must lie strictly between zero and one");
  }
  DensityFittingScfData data;
  data.one_electron = std::move(one_electron);
  data.raw = std::move(raw);
  const DensityFittingMetricFactor factor = factor_density_fitting_metric(
      data.raw.metric, data.raw.naux, relative_threshold);
  data.three_center = orthonormalize_density_fitting_three_center(
      data.raw.three_center, data.raw.nbf, factor);
  if (data.raw.nbf != data.one_electron.nbf) {
    throw std::runtime_error(
        "DF orbital and one-electron AO dimensions are inconsistent");
  }
  return data;
}

DensityFittingScfData prepare_density_fitting_data(
    const core::System& system,
    const core::System& auxiliary_system,
    double relative_threshold,
    int cuda_device_id = -1) {
  // A non-negative device selects the CUDA Cartesian evaluator for the raw
  // metric/three-center tensors.  The default keeps CPU-reference callers
  // entirely on the existing oracle path.
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0) ||
      !std::isfinite(relative_threshold)) {
    throw std::invalid_argument(
        "DF metric relative threshold must lie strictly between zero and one");
  }
  DensityFittingScfData data;
#if VIBEQC_HAS_CUDA
  if (cuda_device_id >= 0) {
    integrals::IntegralData cartesian_one_electron;
    std::string one_electron_detail;
    const vibeqc_status one_electron_status =
        build_cuda_one_electron_integrals(
            cuda_device_id, system, cartesian_one_electron,
            one_electron_detail);
    if (one_electron_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(
          one_electron_detail.empty()
              ? "CUDA one-electron integral generation failed"
              : one_electron_detail);
    }
    data.one_electron = integrals::transform_integrals(
        cartesian_one_electron, system);

    integrals::DensityFittingIntegralData cartesian;
    std::string detail;
    const vibeqc_status status = build_cuda_density_fitting_integrals(
        cuda_device_id, system, auxiliary_system, cartesian, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(
          detail.empty() ? "CUDA density-fitting integral generation failed"
                         : detail);
    }
    // Device recurrence operates on normalized Cartesian source AOs.  Apply
    // the same independently tested public spherical transform used by the
    // host oracle after the device values and derivatives are downloaded.
    data.raw = integrals::transform_density_fitting_integrals(
        cartesian, system, auxiliary_system);
  } else {
    data.one_electron = integrals::build_integrals(system);
    data.raw = integrals::build_density_fitting_integrals(
        system, auxiliary_system);
  }
#else
  (void)cuda_device_id;
  data.one_electron = integrals::build_integrals(system);
  data.raw = integrals::build_density_fitting_integrals(
      system, auxiliary_system);
#endif
  return assemble_density_fitting_data(
      std::move(data.one_electron), std::move(data.raw), relative_threshold);
}

Matrix build_density_fitting_rhf_fock(
    const Matrix& hcore,
    const DensityFittingThreeCenter& three_center,
    const Matrix& density) {
  const DensityFittingRhfJk jk = build_density_fitting_rhf_jk(
      three_center, density);
  Matrix fock = hcore;
  for (std::size_t element = 0; element < fock.size(); ++element) {
    fock[element] += jk.coulomb[element] - 0.5 * jk.exchange[element];
  }
  return fock;
}

std::pair<Matrix, Matrix> build_density_fitting_uhf_focks(
    const Matrix& hcore,
    const DensityFittingThreeCenter& three_center,
    const Matrix& alpha_density,
    const Matrix& beta_density) {
  const DensityFittingUhfJk jk = build_density_fitting_uhf_jk(
      three_center, alpha_density, beta_density);
  Matrix alpha_fock = hcore;
  Matrix beta_fock = hcore;
  for (std::size_t element = 0; element < hcore.size(); ++element) {
    alpha_fock[element] +=
        jk.coulomb[element] - jk.alpha_exchange[element];
    beta_fock[element] += jk.coulomb[element] - jk.beta_exchange[element];
  }
  return {std::move(alpha_fock), std::move(beta_fock)};
}

void finalize_density_fitting_rhf(
    const DensityFittingScfData& data,
    const Matrix& orthogonalizer,
    std::size_t occupied,
    Matrix& density,
    const ScfOptions& options,
    ScfResult& result,
    CudaDensityFittingJkPlan* cuda_plan = nullptr) {
#if !VIBEQC_HAS_CUDA
  (void)cuda_plan;
#endif
  const std::size_t n = data.one_electron.nbf;
  Matrix final_fock = build_density_fitting_rhf_fock(
      data.one_electron.hcore, data.three_center, density);
  EigenResult orbitals = generalized_eigen(final_fock, orthogonalizer, n);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final_fock = build_density_fitting_rhf_fock(
      data.one_electron.hcore, data.three_center, density);
  result.energy = electronic_energy(density, data.one_electron.hcore,
                                    final_fock) +
                  data.one_electron.nuclear_repulsion;
  const Matrix weighted = energy_weighted_density(
      orbitals.vectors, orbitals.values, n, occupied);
  bool device_force_response = false;
#if VIBEQC_HAS_CUDA
  if (cuda_plan != nullptr) {
    try {
      const std::vector<double> inverse = density_fitting_metric_pseudoinverse(
          data.raw, options.density_fitting_relative_threshold);
      const std::size_t metric_elements = data.raw.naux * data.raw.naux;
      const std::size_t derivative_elements = data.raw.ncoord * metric_elements;
      std::vector<double> inverse_derivative(derivative_elements, 0.0);
      for (std::size_t coordinate = 0; coordinate < data.raw.ncoord;
           ++coordinate) {
        const std::vector<double> response =
            density_fitting_metric_pseudoinverse_derivative(
                data.raw, inverse, coordinate);
        std::copy(response.begin(), response.end(),
                  inverse_derivative.begin() + coordinate * metric_elements);
      }
      std::vector<double> two_electron_derivative;
      std::string detail;
      const vibeqc_status status = execute_cuda_density_fitting_rhf_force_response(
          cuda_plan, data.raw.three_center, inverse,
          data.raw.three_center_derivative, inverse_derivative, data.raw.ncoord,
          density, two_electron_derivative, detail);
      if (status == VIBEQC_STATUS_SUCCESS &&
          two_electron_derivative.size() == data.raw.ncoord) {
        result.forces.assign(data.raw.ncoord, 0.0);
        const std::size_t matrix_elements = data.raw.nbf * data.raw.nbf;
        for (std::size_t coordinate = 0; coordinate < data.raw.ncoord;
             ++coordinate) {
          const double* overlap_derivative =
              data.one_electron.overlap_derivative.data() +
              coordinate * matrix_elements;
          const double* hcore_derivative =
              data.one_electron.hcore_derivative.data() +
              coordinate * matrix_elements;
          double derivative = two_electron_derivative[coordinate] +
                              data.one_electron
                                  .nuclear_repulsion_derivative[coordinate];
          for (std::size_t item = 0; item < matrix_elements; ++item) {
            derivative += density[item] * hcore_derivative[item] -
                          weighted[item] * overlap_derivative[item];
          }
          result.forces[coordinate] = -derivative;
        }
        device_force_response = true;
      }
    } catch (...) {
      // The host oracle below remains the correctness fallback when a device
      // force scratch allocation or metric preparation is unavailable.
    }
  }
#endif
  if (!device_force_response) {
    result.forces = build_density_fitting_rhf_forces(
        data.one_electron, data.raw, density, weighted,
        options.density_fitting_relative_threshold);
  }
  result.density = density;
}

void finalize_density_fitting_uhf(
    const DensityFittingScfData& data,
    const Matrix& orthogonalizer,
    std::size_t alpha_occupied,
    std::size_t beta_occupied,
    Matrix& alpha_density,
    Matrix& beta_density,
    const ScfOptions& options,
    ScfResult& result,
    CudaDensityFittingJkPlan* cuda_plan = nullptr) {
#if !VIBEQC_HAS_CUDA
  (void)cuda_plan;
#endif
  const std::size_t n = data.one_electron.nbf;
  auto [alpha_fock, beta_fock] = build_density_fitting_uhf_focks(
      data.one_electron.hcore, data.three_center, alpha_density,
      beta_density);
  EigenResult alpha_orbitals =
      generalized_eigen(alpha_fock, orthogonalizer, n);
  EigenResult beta_orbitals =
      generalized_eigen(beta_fock, orthogonalizer, n);
  alpha_density = density_from_orbitals(
      alpha_orbitals.vectors, n, alpha_occupied, 1.0);
  beta_density = density_from_orbitals(
      beta_orbitals.vectors, n, beta_occupied, 1.0);
  std::tie(alpha_fock, beta_fock) = build_density_fitting_uhf_focks(
      data.one_electron.hcore, data.three_center, alpha_density,
      beta_density);
  result.energy = uhf_electronic_energy(
      alpha_density, beta_density, data.one_electron.hcore, alpha_fock,
      beta_fock) + data.one_electron.nuclear_repulsion;
  const Matrix alpha_weighted = energy_weighted_density(
      alpha_orbitals.vectors, alpha_orbitals.values, n, alpha_occupied, 1.0);
  const Matrix beta_weighted = energy_weighted_density(
      beta_orbitals.vectors, beta_orbitals.values, n, beta_occupied, 1.0);
  bool device_force_response = false;
#if VIBEQC_HAS_CUDA
  if (cuda_plan != nullptr) {
    try {
      const std::vector<double> inverse = density_fitting_metric_pseudoinverse(
          data.raw, options.density_fitting_relative_threshold);
      const std::size_t metric_elements = data.raw.naux * data.raw.naux;
      std::vector<double> inverse_derivative(data.raw.ncoord * metric_elements,
                                              0.0);
      for (std::size_t coordinate = 0; coordinate < data.raw.ncoord;
           ++coordinate) {
        const std::vector<double> response =
            density_fitting_metric_pseudoinverse_derivative(
                data.raw, inverse, coordinate);
        std::copy(response.begin(), response.end(),
                  inverse_derivative.begin() + coordinate * metric_elements);
      }
      std::vector<double> two_electron_derivative;
      std::string detail;
      const vibeqc_status status = execute_cuda_density_fitting_uhf_force_response(
          cuda_plan, data.raw.three_center, inverse,
          data.raw.three_center_derivative, inverse_derivative, data.raw.ncoord,
          alpha_density, beta_density, two_electron_derivative, detail);
      if (status == VIBEQC_STATUS_SUCCESS &&
          two_electron_derivative.size() == data.raw.ncoord) {
        result.forces.assign(data.raw.ncoord, 0.0);
        const std::size_t matrix_elements = data.raw.nbf * data.raw.nbf;
        for (std::size_t coordinate = 0; coordinate < data.raw.ncoord;
             ++coordinate) {
          const double* overlap_derivative =
              data.one_electron.overlap_derivative.data() +
              coordinate * matrix_elements;
          const double* hcore_derivative =
              data.one_electron.hcore_derivative.data() +
              coordinate * matrix_elements;
          double derivative = two_electron_derivative[coordinate] +
                              data.one_electron
                                  .nuclear_repulsion_derivative[coordinate];
          for (std::size_t item = 0; item < matrix_elements; ++item) {
            const double total_density = alpha_density[item] + beta_density[item];
            const double total_weighted =
                alpha_weighted[item] + beta_weighted[item];
            derivative += total_density * hcore_derivative[item] -
                          total_weighted * overlap_derivative[item];
          }
          result.forces[coordinate] = -derivative;
        }
        device_force_response = true;
      }
    } catch (...) {
      // Preserve the validated host response if device force preparation fails.
    }
  }
#endif
  if (!device_force_response) {
    result.forces = build_density_fitting_uhf_forces(
        data.one_electron, data.raw, alpha_density, beta_density,
        alpha_weighted, beta_weighted,
        options.density_fitting_relative_threshold);
  }
  result.density = concatenate(alpha_density, beta_density);
}

}  // namespace

ScfResult run_rhf(const core::System& system,
                  const ScfOptions& options,
                        const std::vector<double>* initial_density) {
  const integrals::IntegralData ints =
      integrals::build_cartesian_integrals(system);
  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult orbitals;
  Matrix density = prepare_initial_density(system, ints, orthogonalizer, occupied,
                                           initial_density, orbitals);
  Diis diis(options.diis_history);

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const Matrix fock = build_fock(ints.hcore, ints.eri, density, n);
    const double energy = electronic_energy(density, ints.hcore, fock) +
                          ints.nuclear_repulsion;
    const Matrix residual = commutator_residual(fock, density, ints.overlap, n);
    const Matrix effective_fock = diis.update(fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    density = std::move(next_density);
  }

  if (!result.converged) return result;

  // Rebuild and diagonalize the un-extrapolated converged Fock matrix. The
  // resulting orbitals define the energy-weighted density in the Pulay term.
  finalize_scf(ints, orthogonalizer, occupied, density, result);
  return result;
}

ScfResult run_uhf(const core::System& system,
                  const ScfOptions& options,
                        const std::vector<double>* initial_density) {
  const integrals::IntegralData ints =
      integrals::build_cartesian_integrals(system);
  const std::size_t n = ints.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error(
        "basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] = prepare_initial_uhf_density(
      ints, orthogonalizer, alpha_occupied, beta_occupied, initial_density,
      alpha_orbitals, beta_orbitals);
  Diis diis(options.diis_history);

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    auto [alpha_fock, beta_fock] = build_uhf_focks(
        ints.hcore, ints.eri, alpha_density, beta_density, n);
    const double energy = uhf_electronic_energy(
                              alpha_density, beta_density, ints.hcore,
                              alpha_fock, beta_fock) +
                          ints.nuclear_repulsion;
    const Matrix alpha_residual = commutator_residual(
        alpha_fock, alpha_density, ints.overlap, n);
    const Matrix beta_residual = commutator_residual(
        beta_fock, beta_density, ints.overlap, n);
    const Matrix effective_joined = diis.update(
        concatenate(alpha_fock, beta_fock),
        concatenate(alpha_residual, beta_residual));
    std::tie(alpha_fock, beta_fock) =
        split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
    beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(
        alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(
        beta_orbitals.vectors, n, beta_occupied, 1.0);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(
        concatenate(next_alpha, next_beta),
        concatenate(alpha_density, beta_density));
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      alpha_density = std::move(next_alpha);
      beta_density = std::move(next_beta);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }

  if (!result.converged) return result;

  // As in RHF, rebuild from the un-extrapolated converged spin Fock matrices
  // before forming orbital-weighted Pulay densities and analytic forces.
  finalize_uhf(ints, orthogonalizer, alpha_occupied, beta_occupied,
               alpha_density, beta_density, result);
  return result;
}

ScfResult run_rhf_density_fitting(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    const std::vector<double>* initial_density) {
  const DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold);
  const std::size_t n = data.one_electron.nbf;
  const std::size_t occupied =
      static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error(
        "basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer =
      symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult orbitals;
  Matrix density = prepare_initial_density(
      system, data.one_electron, orthogonalizer, occupied, initial_density,
      orbitals);
  Diis diis(options.diis_history);

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations;
       ++iteration) {
    const Matrix fock = build_density_fitting_rhf_fock(
        data.one_electron.hcore, data.three_center, density);
    const double energy = electronic_energy(
                              density, data.one_electron.hcore, fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix residual = commutator_residual(
        fock, density, data.one_electron.overlap, n);
    const Matrix effective_fock = diis.update(fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(
        orbitals.vectors, n, occupied);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    density = std::move(next_density);
  }

  if (!result.converged) return result;
  finalize_density_fitting_rhf(
      data, orthogonalizer, occupied, density, options, result);
  return result;
}

ScfResult run_uhf_density_fitting(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    const std::vector<double>* initial_density) {
  const DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold);
  const std::size_t n = data.one_electron.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error(
        "basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer =
      symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] = prepare_initial_uhf_density(
      data.one_electron, orthogonalizer, alpha_occupied, beta_occupied,
      initial_density, alpha_orbitals, beta_orbitals);
  Diis diis(options.diis_history);

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations;
       ++iteration) {
    auto [alpha_fock, beta_fock] = build_density_fitting_uhf_focks(
        data.one_electron.hcore, data.three_center, alpha_density,
        beta_density);
    const double energy = uhf_electronic_energy(
                              alpha_density, beta_density,
                              data.one_electron.hcore, alpha_fock, beta_fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix alpha_residual = commutator_residual(
        alpha_fock, alpha_density, data.one_electron.overlap, n);
    const Matrix beta_residual = commutator_residual(
        beta_fock, beta_density, data.one_electron.overlap, n);
    const Matrix effective_joined = diis.update(
        concatenate(alpha_fock, beta_fock),
        concatenate(alpha_residual, beta_residual));
    std::tie(alpha_fock, beta_fock) =
        split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
    beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(
        alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(
        beta_orbitals.vectors, n, beta_occupied, 1.0);

    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(
        concatenate(next_alpha, next_beta),
        concatenate(alpha_density, beta_density));
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      alpha_density = std::move(next_alpha);
      beta_density = std::move(next_beta);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }

  if (!result.converged) return result;
  finalize_density_fitting_uhf(
      data, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
      beta_density, options, result);
  return result;
}

#if VIBEQC_HAS_CUDA

using CudaDensityFittingPlanPtr = std::unique_ptr<
    CudaDensityFittingJkPlan,
    decltype(&destroy_cuda_density_fitting_jk_plan)>;

CudaDensityFittingPlanPtr make_cuda_density_fitting_plan(
    const DensityFittingScfData& data,
    const ScfOptions& options,
    int device_id,
    std::size_t occupied,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics =
        nullptr) {
  CudaDensityFittingJkPlan* raw_plan = nullptr;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  std::size_t auxiliary_tile = 0;
  std::size_t ao_pair_tile = 0;
  if (options.density_fitting_memory_budget_bytes != 0) {
    const DensityFittingTilePlan tile_plan = plan_density_fitting_tiles(
        1, data.raw.nbf, data.raw.naux, std::max<std::size_t>(occupied, 1),
        options.density_fitting_memory_budget_bytes);
    auxiliary_tile = tile_plan.auxiliary_tile;
    ao_pair_tile = tile_plan.ao_pair_tile;
  }
  const vibeqc_status status = options.density_fitting_memory_budget_bytes != 0
      ? create_cuda_density_fitting_jk_plan_tiled(
            device_id, 1, data.raw.nbf, data.raw.naux, data.raw.metric,
            data.raw.three_center, options.density_fitting_relative_threshold,
            auxiliary_tile, ao_pair_tile, &raw_plan, diagnostics, detail)
      : create_cuda_density_fitting_jk_plan(
            device_id, 1, data.raw.nbf, data.raw.naux, data.raw.metric,
            data.raw.three_center, options.density_fitting_relative_threshold,
            0, &raw_plan, diagnostics, detail);
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw std::runtime_error(detail.empty()
                                 ? "CUDA density-fitting plan creation failed"
                                 : detail);
  }
  // Keep the raw plan owned while copying optional diagnostics; an allocation
  // failure in that copy must still release all CUDA resources.
  CudaDensityFittingPlanPtr owned_plan(
      raw_plan, &destroy_cuda_density_fitting_jk_plan);
  if (output_diagnostics != nullptr) {
    *output_diagnostics = diagnostics;
  }
  return owned_plan;
}

CudaDensityFittingPlanPtr make_cuda_density_fitting_batch_plan(
    const std::vector<DensityFittingScfData>& data,
    const ScfOptions& options,
    int device_id,
    std::size_t occupied,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics =
        nullptr) {
  if (data.empty()) {
    throw std::invalid_argument("CUDA density-fitting batch cannot be empty");
  }
  const std::size_t nbf = data.front().raw.nbf;
  const std::size_t naux = data.front().raw.naux;
  std::vector<double> metrics;
  std::vector<double> three_center;
  std::size_t metric_size = naux * naux;
  std::size_t tensor_size = nbf * nbf * naux;
  metrics.reserve(data.size() * metric_size);
  three_center.reserve(data.size() * tensor_size);
  for (const DensityFittingScfData& item : data) {
    if (item.raw.nbf != nbf || item.raw.naux != naux ||
        item.raw.metric.size() != metric_size ||
        item.raw.three_center.size() != tensor_size) {
      throw std::invalid_argument(
          "CUDA density-fitting bucket has incompatible auxiliary dimensions");
    }
    metrics.insert(metrics.end(), item.raw.metric.begin(), item.raw.metric.end());
    three_center.insert(three_center.end(), item.raw.three_center.begin(),
                        item.raw.three_center.end());
  }
  CudaDensityFittingJkPlan* raw_plan = nullptr;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  std::size_t auxiliary_tile = 0;
  std::size_t ao_pair_tile = 0;
  if (options.density_fitting_memory_budget_bytes != 0) {
    const DensityFittingTilePlan tile_plan = plan_density_fitting_tiles(
        data.size(), nbf, naux, std::max<std::size_t>(occupied, 1),
        options.density_fitting_memory_budget_bytes);
    auxiliary_tile = tile_plan.auxiliary_tile;
    ao_pair_tile = tile_plan.ao_pair_tile;
  }
  const vibeqc_status status = options.density_fitting_memory_budget_bytes != 0
      ? create_cuda_density_fitting_jk_plan_tiled(
            device_id, data.size(), nbf, naux, metrics, three_center,
            options.density_fitting_relative_threshold, auxiliary_tile,
            ao_pair_tile, &raw_plan, diagnostics, detail)
      : create_cuda_density_fitting_jk_plan(
            device_id, data.size(), nbf, naux, metrics, three_center,
            options.density_fitting_relative_threshold, 0, &raw_plan,
            diagnostics, detail);
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw std::runtime_error(detail.empty()
                                 ? "CUDA density-fitting batch plan creation failed"
                                 : detail);
  }
  // Keep the raw plan owned while copying optional diagnostics; an allocation
  // failure in that copy must still release all CUDA resources.
  CudaDensityFittingPlanPtr owned_plan(
      raw_plan, &destroy_cuda_density_fitting_jk_plan);
  if (output_diagnostics != nullptr) {
    *output_diagnostics = diagnostics;
  }
  return owned_plan;
}

core::System density_fitting_auxiliary_for_geometry(
    const std::optional<core::System>& auxiliary_template,
    const core::System& system) {
  if (!auxiliary_template.has_value()) return system;
  core::System auxiliary = *auxiliary_template;
  auxiliary.atoms = system.atoms;
  auxiliary.charge = system.charge;
  auxiliary.multiplicity = system.multiplicity;
  auxiliary.electron_count = system.electron_count;
  return auxiliary;
}

/**
 * Prepare CUDA DF data for a fleet while isolating failures to individual
 * systems. Raw metric/three-center tensors are generated in homogeneous
 * batches; one-electron tensors remain per-system because their public AO
 * representations may differ even when Cartesian dimensions match.
 */
std::vector<std::optional<DensityFittingScfData>>
prepare_cuda_density_fitting_batch(
    const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template,
    double relative_threshold, std::size_t output_budget_bytes, int device_id,
    std::vector<vibeqc_status>& statuses) {
  const std::size_t count = systems.size();
  statuses.assign(count, VIBEQC_STATUS_INTERNAL_ERROR);
  std::vector<std::optional<DensityFittingScfData>> prepared(count);
  std::vector<core::System> auxiliaries(count);
  std::vector<bool> auxiliary_valid(count, false);

  for (std::size_t source = 0; source < count; ++source) {
    try {
      auxiliaries[source] = density_fitting_auxiliary_for_geometry(
          auxiliary_template, systems[source]);
      auxiliary_valid[source] = true;
    } catch (const std::bad_alloc&) {
      statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }

  // Partition by Cartesian dimensions and atom count. This keeps the batch
  // kernel's packed strides valid while allowing ragged fleets to proceed.
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t source = 0; source < count; ++source) {
    if (!auxiliary_valid[source]) continue;
    const std::size_t orbital_count =
        molecule::cartesian_ao_count(systems[source]);
    const std::size_t auxiliary_count =
        molecule::cartesian_ao_count(auxiliaries[source]);
    const std::size_t atom_count = systems[source].atoms.size();
    bool placed = false;
    for (auto& group : groups) {
      const std::size_t representative = group.front();
      if (molecule::cartesian_ao_count(systems[representative]) ==
              orbital_count &&
          molecule::cartesian_ao_count(auxiliaries[representative]) ==
              auxiliary_count &&
          systems[representative].atoms.size() == atom_count) {
        group.push_back(source);
        placed = true;
        break;
      }
    }
    if (!placed) groups.push_back({source});
  }

  for (const auto& group : groups) {
    std::vector<core::System> orbital_batch;
    std::vector<core::System> auxiliary_batch;
    orbital_batch.reserve(group.size());
    auxiliary_batch.reserve(group.size());
    for (const std::size_t source : group) {
      orbital_batch.push_back(systems[source]);
      auxiliary_batch.push_back(auxiliaries[source]);
    }

    std::vector<integrals::DensityFittingIntegralData> raw_batch;
    std::string detail;
    const vibeqc_status batch_status =
        build_cuda_density_fitting_integrals_batch(
            device_id, orbital_batch, auxiliary_batch, raw_batch, detail,
            output_budget_bytes);
    if (batch_status == VIBEQC_STATUS_SUCCESS &&
        raw_batch.size() == group.size()) {
      std::vector<integrals::IntegralData> one_electron_batch;
      const vibeqc_status one_electron_batch_status =
          build_cuda_one_electron_integrals_batch(
              device_id, orbital_batch, one_electron_batch, detail);
      if (one_electron_batch_status == VIBEQC_STATUS_SUCCESS &&
          one_electron_batch.size() == group.size()) {
        for (std::size_t slot = 0; slot < group.size(); ++slot) {
          const std::size_t source = group[slot];
          try {
            integrals::IntegralData one_electron =
                integrals::transform_integrals(one_electron_batch[slot],
                                               systems[source]);
            integrals::DensityFittingIntegralData raw =
                integrals::transform_density_fitting_integrals(
                    raw_batch[slot], systems[source], auxiliaries[source]);
            prepared[source] = assemble_density_fitting_data(
                std::move(one_electron), std::move(raw), relative_threshold);
          } catch (const std::bad_alloc&) {
            statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
          } catch (const std::invalid_argument&) {
            statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
          } catch (...) {
            statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
          }
        }
        continue;
      }
    }

    // A batch-level launch can fail for resource or topology reasons. Retry
    // each item independently so one bad system never poisons its neighbors.
    for (const std::size_t source : group) {
      try {
        integrals::DensityFittingIntegralData cartesian;
        std::string item_detail;
        const vibeqc_status item_status = build_cuda_density_fitting_integrals(
            device_id, systems[source], auxiliaries[source], cartesian,
            item_detail);
        if (item_status != VIBEQC_STATUS_SUCCESS) {
          statuses[source] = item_status;
          continue;
        }
        integrals::IntegralData cartesian_one_electron;
        const vibeqc_status one_electron_status =
            build_cuda_one_electron_integrals(
                device_id, systems[source], cartesian_one_electron,
                item_detail);
        if (one_electron_status != VIBEQC_STATUS_SUCCESS) {
          statuses[source] = one_electron_status;
          continue;
        }
        integrals::IntegralData one_electron = integrals::transform_integrals(
            cartesian_one_electron, systems[source]);
        integrals::DensityFittingIntegralData raw =
            integrals::transform_density_fitting_integrals(
                cartesian, systems[source], auxiliaries[source]);
        prepared[source] = assemble_density_fitting_data(
            std::move(one_electron), std::move(raw), relative_threshold);
      } catch (const std::bad_alloc&) {
        statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
      } catch (const std::invalid_argument&) {
        statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
      } catch (...) {
        statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
      }
    }
  }
  return prepared;
}

ScfResult run_rhf_density_fitting_cuda_impl(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    int device_id,
    const std::vector<double>* initial_density) {
  const DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold,
      device_id);
  const std::size_t n = data.one_electron.nbf;
  const std::size_t occupied =
      static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error(
        "basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer =
      symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult orbitals;
  Matrix density = prepare_initial_density(
      system, data.one_electron, orthogonalizer, occupied, initial_density,
      orbitals);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  const CudaDensityFittingPlanPtr plan =
      make_cuda_density_fitting_plan(data, options, device_id, occupied);

  // Prefer the fully device-resident SCF loop.  It keeps the DF density,
  // Fock assembly, eigensolve, and convergence reductions on the plan stream;
  // the legacy host-orchestrated loop below remains a correctness-preserving
  // fallback for provider/workspace limitations or slow non-convergence.
  {
    std::vector<double> device_final_density;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string detail;
    const vibeqc_status device_status =
        run_cuda_density_fitting_rhf_device_scf(
            plan.get(), data.one_electron.hcore, orthogonalizer, density,
            {static_cast<std::int32_t>(occupied)},
            {data.one_electron.nuclear_repulsion}, options.max_iterations,
            options.energy_tolerance, options.density_tolerance,
            device_final_density, device_records, detail);
    if (device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == 1 &&
        device_records.front().converged) {
      density = std::move(device_final_density);
      result.iterations = device_records.front().iterations;
      result.energy = device_records.front().energy;
      result.energy_change = device_records.front().energy_change;
      result.density_rms = device_records.front().density_rms;
      result.converged = true;
      finalize_density_fitting_rhf(
          data, orthogonalizer, occupied, density, options, result,
          plan.get());
      return result;
    }
  }
  Diis diis(options.diis_history);
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations;
       ++iteration) {
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status status = execute_cuda_density_fitting_rhf_jk(
        plan.get(), density, coulomb, exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty()
                                   ? "CUDA density-fitting RHF J/K failed"
                                   : detail);
    }
    Matrix fock = data.one_electron.hcore;
    for (std::size_t element = 0; element < fock.size(); ++element) {
      fock[element] += coulomb[element] - 0.5 * exchange[element];
    }
    const double energy = electronic_energy(
                              density, data.one_electron.hcore, fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix residual = commutator_residual(
        fock, density, data.one_electron.overlap, n);
    const Matrix effective_fock = diis.update(fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(
        orbitals.vectors, n, occupied);
    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    density = std::move(next_density);
  }
  if (!result.converged) return result;
  // The final diagonalization is intentionally rebuilt from the same CPU
  // oracle tensor used for force response. The SCF iterations above exercise
  // the CUDA DF contractions, while this last step keeps the existing
  // variational weighted-density convention exact.
  finalize_density_fitting_rhf(
      data, orthogonalizer, occupied, density, options, result, plan.get());
  return result;
}

ScfResult run_uhf_density_fitting_cuda_impl(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    int device_id,
    const std::vector<double>* initial_density) {
  const DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold,
      device_id);
  const std::size_t n = data.one_electron.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error(
        "basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer =
      symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] = prepare_initial_uhf_density(
      data.one_electron, orthogonalizer, alpha_occupied, beta_occupied,
      initial_density, alpha_orbitals, beta_orbitals);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  const CudaDensityFittingPlanPtr plan =
      make_cuda_density_fitting_plan(
          data, options, device_id, std::max(alpha_occupied, beta_occupied));
  {
    std::vector<double> device_final_alpha;
    std::vector<double> device_final_beta;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string detail;
    const vibeqc_status device_status =
        run_cuda_density_fitting_uhf_device_scf(
            plan.get(), data.one_electron.hcore, orthogonalizer,
            alpha_density, beta_density,
            {static_cast<std::int32_t>(alpha_occupied)},
            {static_cast<std::int32_t>(beta_occupied)},
            {data.one_electron.nuclear_repulsion}, options.max_iterations,
            options.energy_tolerance, options.density_tolerance,
            device_final_alpha, device_final_beta, device_records, detail);
    if (device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == 1 &&
        device_records.front().converged) {
      alpha_density = std::move(device_final_alpha);
      beta_density = std::move(device_final_beta);
      result.iterations = device_records.front().iterations;
      result.energy = device_records.front().energy;
      result.energy_change = device_records.front().energy_change;
      result.density_rms = device_records.front().density_rms;
      result.converged = true;
      finalize_density_fitting_uhf(
          data, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
          beta_density, options, result, plan.get());
      return result;
    }
  }
  Diis diis(options.diis_history);
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations;
       ++iteration) {
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    const vibeqc_status status = execute_cuda_density_fitting_uhf_jk(
        plan.get(), alpha_density, beta_density, coulomb, alpha_exchange,
        beta_exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty()
                                   ? "CUDA density-fitting UHF J/K failed"
                                   : detail);
    }
    Matrix alpha_fock = data.one_electron.hcore;
    Matrix beta_fock = data.one_electron.hcore;
    for (std::size_t element = 0; element < alpha_fock.size(); ++element) {
      alpha_fock[element] += coulomb[element] - alpha_exchange[element];
      beta_fock[element] += coulomb[element] - beta_exchange[element];
    }
    const double energy = uhf_electronic_energy(
                              alpha_density, beta_density,
                              data.one_electron.hcore, alpha_fock, beta_fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix alpha_residual = commutator_residual(
        alpha_fock, alpha_density, data.one_electron.overlap, n);
    const Matrix beta_residual = commutator_residual(
        beta_fock, beta_density, data.one_electron.overlap, n);
    const Matrix effective_joined = diis.update(
        concatenate(alpha_fock, beta_fock),
        concatenate(alpha_residual, beta_residual));
    std::tie(alpha_fock, beta_fock) =
        split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
    beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(
        alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(
        beta_orbitals.vectors, n, beta_occupied, 1.0);
    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy)
                               ? std::abs(energy - previous_energy)
                               : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(
        concatenate(next_alpha, next_beta),
        concatenate(alpha_density, beta_density));
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      alpha_density = std::move(next_alpha);
      beta_density = std::move(next_beta);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }
  if (!result.converged) return result;
  finalize_density_fitting_uhf(
      data, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
      beta_density, options, result, plan.get());
  return result;
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_impl(
    const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics) {
  if (systems.size() != initial_densities.size()) {
    throw std::invalid_argument(
        "CUDA density-fitting RHF bucket density count mismatch");
  }
  std::vector<RhfBucketItem> outputs(systems.size());
  if (systems.empty()) return outputs;

  // Preparation failures are recorded per item.  The remaining compatible
  // systems still share one CUDA plan, preserving fleet failure isolation.
  std::vector<std::size_t> source_indices;
  std::vector<DensityFittingScfData> data;
  std::vector<Matrix> orthogonalizers;
  std::vector<Matrix> densities;
  std::vector<EigenResult> orbitals;
  std::vector<Diis> diis;
  std::vector<double> previous_energies;
  source_indices.reserve(systems.size());
  data.reserve(systems.size());
  orthogonalizers.reserve(systems.size());
  densities.reserve(systems.size());
  orbitals.reserve(systems.size());
  diis.reserve(systems.size());
  previous_energies.reserve(systems.size());

  std::vector<vibeqc_status> preparation_status;
  std::vector<std::optional<DensityFittingScfData>> batched_prepared;
  if (device_id >= 0) {
    batched_prepared = prepare_cuda_density_fitting_batch(
        systems, auxiliary_template,
        options.density_fitting_relative_threshold,
        options.density_fitting_memory_budget_bytes, device_id,
        preparation_status);
  }

  std::size_t nbf = 0;
  std::size_t naux = 0;
  for (std::size_t source = 0; source < systems.size(); ++source) {
    const std::size_t slot_before = data.size();
    try {
      DensityFittingScfData prepared;
      if (device_id >= 0) {
        if (!batched_prepared[source].has_value()) {
          outputs[source].status = preparation_status[source];
          continue;
        }
        prepared = std::move(*batched_prepared[source]);
      } else {
        const core::System auxiliary = density_fitting_auxiliary_for_geometry(
            auxiliary_template, systems[source]);
        prepared = prepare_density_fitting_data(
            systems[source], auxiliary,
            options.density_fitting_relative_threshold, device_id);
      }
      if (source_indices.empty()) {
        nbf = prepared.raw.nbf;
        naux = prepared.raw.naux;
      } else if (prepared.raw.nbf != nbf || prepared.raw.naux != naux) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const std::size_t occupied =
          static_cast<std::size_t>(systems[source].electron_count / 2);
      const Matrix orthogonalizer = symmetric_orthogonalizer(
          prepared.one_electron.overlap, prepared.one_electron.nbf);
      EigenResult initial_orbitals;
      Matrix density = prepare_initial_density(
          systems[source], prepared.one_electron, orthogonalizer, occupied,
          initial_densities[source], initial_orbitals);
      source_indices.push_back(source);
      data.push_back(std::move(prepared));
      orthogonalizers.push_back(orthogonalizer);
      densities.push_back(std::move(density));
      orbitals.push_back(std::move(initial_orbitals));
      diis.emplace_back(options.diis_history);
      previous_energies.push_back(std::numeric_limits<double>::infinity());
      outputs[source].scf.initial_density_used =
          initial_densities[source] != nullptr;
    } catch (const std::bad_alloc&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  if (data.empty()) return outputs;

  CudaDensityFittingPlanPtr plan(nullptr, &destroy_cuda_density_fitting_jk_plan);
  std::vector<CudaDensityFittingMetricDiagnostic> metric_diagnostics;
  try {
    const std::size_t occupied = static_cast<std::size_t>(
        systems[source_indices.front()].electron_count / 2);
    plan = make_cuda_density_fitting_batch_plan(
        data, options, device_id, occupied, &metric_diagnostics);
  } catch (const std::bad_alloc&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    return outputs;
  } catch (const std::invalid_argument&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return outputs;
  } catch (...) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_CUDA_ERROR;
    }
    return outputs;
  }
  if (output_diagnostics != nullptr) {
    for (std::size_t slot = 0; slot < metric_diagnostics.size(); ++slot) {
      metric_diagnostics[slot].system_index = source_indices[slot];
    }
    *output_diagnostics = metric_diagnostics;
  }

  // A compatible bucket can advance every density without host staging.  The
  // device driver returns only scalar convergence records and the final
  // densities needed by the existing analytic-force oracle.  If a provider
  // rejects the batched eigensolve or does not converge all items, retain the
  // independently isolated host-orchestrated path below.
  {
    const std::size_t matrix_size = nbf * nbf;
    std::vector<double> hcore(data.size() * matrix_size);
    std::vector<double> orthogonalizer(data.size() * matrix_size);
    std::vector<double> initial_density(data.size() * matrix_size);
    std::vector<double> nuclear(data.size());
    std::vector<std::int32_t> occupied(data.size());
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      std::copy(data[slot].one_electron.hcore.begin(),
                data[slot].one_electron.hcore.end(),
                hcore.begin() + slot * matrix_size);
      std::copy(orthogonalizers[slot].begin(), orthogonalizers[slot].end(),
                orthogonalizer.begin() + slot * matrix_size);
      std::copy(densities[slot].begin(), densities[slot].end(),
                initial_density.begin() + slot * matrix_size);
      nuclear[slot] = data[slot].one_electron.nuclear_repulsion;
      occupied[slot] = static_cast<std::int32_t>(
          systems[source_indices[slot]].electron_count / 2);
    }
    std::vector<double> device_final_density;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string device_detail;
    const vibeqc_status device_status =
        run_cuda_density_fitting_rhf_device_scf(
            plan.get(), hcore, orthogonalizer, initial_density, occupied,
            nuclear, options.max_iterations, options.energy_tolerance,
            options.density_tolerance, device_final_density, device_records,
            device_detail);
    const bool device_converged =
        device_status == VIBEQC_STATUS_SUCCESS &&
        device_records.size() == data.size() &&
        std::all_of(device_records.begin(), device_records.end(),
                    [](const CudaDensityFittingDeviceScfItem& item) {
                      return item.converged;
                    });
    if (device_converged) {
      for (std::size_t slot = 0; slot < data.size(); ++slot) {
        const std::size_t source = source_indices[slot];
        densities[slot].assign(
            device_final_density.begin() + slot * matrix_size,
            device_final_density.begin() + (slot + 1) * matrix_size);
        ScfResult& result = outputs[source].scf;
        result.iterations = device_records[slot].iterations;
        result.energy = device_records[slot].energy;
        result.energy_change = device_records[slot].energy_change;
        result.density_rms = device_records[slot].density_rms;
        result.converged = true;
        try {
          finalize_density_fitting_rhf(
              data[slot], orthogonalizers[slot],
              static_cast<std::size_t>(occupied[slot]), densities[slot],
              options, result, plan.get());
          outputs[source].status = VIBEQC_STATUS_SUCCESS;
        } catch (const std::bad_alloc&) {
          outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        } catch (const std::invalid_argument&) {
          outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        } catch (...) {
          outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
      }
      return outputs;
    }
  }

  const std::size_t matrix_size = nbf * nbf;
  std::vector<double> batch_density(data.size() * matrix_size);
  std::vector<bool> active(data.size(), true);
  std::size_t active_count = data.size();
  for (unsigned iteration = 1; iteration <= options.max_iterations &&
                                      active_count != 0;
       ++iteration) {
    for (std::size_t slot = 0; slot < densities.size(); ++slot) {
      std::copy(densities[slot].begin(), densities[slot].end(),
                batch_density.begin() + slot * matrix_size);
    }
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status jk_status = execute_cuda_density_fitting_rhf_jk(
        plan.get(), batch_density, coulomb, exchange, detail);
    if (jk_status != VIBEQC_STATUS_SUCCESS) {
      for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
        if (active[slot]) outputs[source_indices[slot]].status = jk_status;
        active[slot] = false;
      }
      break;
    }

    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      if (!active[slot]) continue;
      const std::size_t source = source_indices[slot];
      try {
        Matrix fock = data[slot].one_electron.hcore;
        const double* j = coulomb.data() + slot * matrix_size;
        const double* k = exchange.data() + slot * matrix_size;
        for (std::size_t element = 0; element < matrix_size; ++element) {
          fock[element] += j[element] - 0.5 * k[element];
        }
        const double energy = electronic_energy(
                                  densities[slot], data[slot].one_electron.hcore,
                                  fock) +
                              data[slot].one_electron.nuclear_repulsion;
        const Matrix residual = commutator_residual(
            fock, densities[slot], data[slot].one_electron.overlap, nbf);
        const Matrix effective_fock = diis[slot].update(fock, residual);
        orbitals[slot] = generalized_eigen(effective_fock,
                                           orthogonalizers[slot], nbf);
        Matrix next_density = density_from_orbitals(
            orbitals[slot].vectors, nbf,
            static_cast<std::size_t>(systems[source].electron_count / 2));
        ScfResult& result = outputs[source].scf;
        result.iterations = iteration;
        result.energy = energy;
        result.energy_change = std::isfinite(previous_energies[slot])
                                   ? std::abs(energy - previous_energies[slot])
                                   : std::numeric_limits<double>::infinity();
        result.density_rms = density_rms(next_density, densities[slot]);
        if (iteration > 1 &&
            result.energy_change < options.energy_tolerance &&
            result.density_rms < options.density_tolerance) {
          densities[slot] = std::move(next_density);
          result.converged = true;
          active[slot] = false;
          --active_count;
        } else {
          previous_energies[slot] = energy;
          densities[slot] = std::move(next_density);
        }
      } catch (const std::bad_alloc&) {
        outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        active[slot] = false;
        --active_count;
      } catch (...) {
        outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        active[slot] = false;
        --active_count;
      }
    }
  }

  for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
    const std::size_t source = source_indices[slot];
    ScfResult& result = outputs[source].scf;
    if (outputs[source].status != VIBEQC_STATUS_INTERNAL_ERROR) {
      continue;
    }
    if (!result.converged) {
      outputs[source].status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
      continue;
    }
    try {
      finalize_density_fitting_rhf(
          data[slot], orthogonalizers[slot],
          static_cast<std::size_t>(systems[source].electron_count / 2),
          densities[slot], options, result, plan.get());
      outputs[source].status = VIBEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_impl(
    const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics) {
  if (systems.size() != initial_densities.size()) {
    throw std::invalid_argument(
        "CUDA density-fitting UHF bucket density count mismatch");
  }
  std::vector<RhfBucketItem> outputs(systems.size());
  if (systems.empty()) return outputs;

  std::vector<std::size_t> source_indices;
  std::vector<DensityFittingScfData> data;
  std::vector<Matrix> orthogonalizers;
  std::vector<Matrix> alpha_densities;
  std::vector<Matrix> beta_densities;
  std::vector<EigenResult> alpha_orbitals;
  std::vector<EigenResult> beta_orbitals;
  std::vector<Diis> diis;
  std::vector<double> previous_energies;
  std::vector<vibeqc_status> preparation_status;
  std::vector<std::optional<DensityFittingScfData>> batched_prepared;
  if (device_id >= 0) {
    batched_prepared = prepare_cuda_density_fitting_batch(
        systems, auxiliary_template,
        options.density_fitting_relative_threshold,
        options.density_fitting_memory_budget_bytes, device_id,
        preparation_status);
  }
  std::size_t nbf = 0;
  std::size_t naux = 0;
  for (std::size_t source = 0; source < systems.size(); ++source) {
    const std::size_t slot_before = data.size();
    try {
      DensityFittingScfData prepared;
      if (device_id >= 0) {
        if (!batched_prepared[source].has_value()) {
          outputs[source].status = preparation_status[source];
          continue;
        }
        prepared = std::move(*batched_prepared[source]);
      } else {
        const core::System auxiliary = density_fitting_auxiliary_for_geometry(
            auxiliary_template, systems[source]);
        prepared = prepare_density_fitting_data(
            systems[source], auxiliary,
            options.density_fitting_relative_threshold, device_id);
      }
      if (source_indices.empty()) {
        nbf = prepared.raw.nbf;
        naux = prepared.raw.naux;
      } else if (prepared.raw.nbf != nbf || prepared.raw.naux != naux) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const auto [alpha_occupied, beta_occupied] =
          spin_occupations(systems[source]);
      if (alpha_occupied > nbf || beta_occupied > nbf) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const Matrix orthogonalizer = symmetric_orthogonalizer(
          prepared.one_electron.overlap, prepared.one_electron.nbf);
      EigenResult initial_alpha_orbitals;
      EigenResult initial_beta_orbitals;
      auto [alpha_density, beta_density] = prepare_initial_uhf_density(
          prepared.one_electron, orthogonalizer, alpha_occupied,
          beta_occupied, initial_densities[source], initial_alpha_orbitals,
          initial_beta_orbitals);
      source_indices.push_back(source);
      data.push_back(std::move(prepared));
      orthogonalizers.push_back(orthogonalizer);
      alpha_densities.push_back(std::move(alpha_density));
      beta_densities.push_back(std::move(beta_density));
      alpha_orbitals.push_back(std::move(initial_alpha_orbitals));
      beta_orbitals.push_back(std::move(initial_beta_orbitals));
      diis.emplace_back(options.diis_history);
      previous_energies.push_back(std::numeric_limits<double>::infinity());
      outputs[source].scf.initial_density_used =
          initial_densities[source] != nullptr;
    } catch (const std::bad_alloc&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  if (data.empty()) return outputs;

  CudaDensityFittingPlanPtr plan(nullptr, &destroy_cuda_density_fitting_jk_plan);
  std::vector<CudaDensityFittingMetricDiagnostic> metric_diagnostics;
  try {
    const auto [alpha_occupied, beta_occupied] =
        spin_occupations(systems[source_indices.front()]);
    plan = make_cuda_density_fitting_batch_plan(
        data, options, device_id, std::max(alpha_occupied, beta_occupied),
        &metric_diagnostics);
  } catch (const std::bad_alloc&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    return outputs;
  } catch (const std::invalid_argument&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return outputs;
  } catch (...) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_CUDA_ERROR;
    }
    return outputs;
  }
  if (output_diagnostics != nullptr) {
    for (std::size_t slot = 0; slot < metric_diagnostics.size(); ++slot) {
      metric_diagnostics[slot].system_index = source_indices[slot];
    }
    *output_diagnostics = metric_diagnostics;
  }

  {
    const std::size_t matrix_size = nbf * nbf;
    std::vector<double> hcore(data.size() * matrix_size);
    std::vector<double> orthogonalizer(data.size() * matrix_size);
    std::vector<double> initial_alpha(data.size() * matrix_size);
    std::vector<double> initial_beta(data.size() * matrix_size);
    std::vector<double> nuclear(data.size());
    std::vector<std::int32_t> alpha_occupied(data.size());
    std::vector<std::int32_t> beta_occupied(data.size());
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      std::copy(data[slot].one_electron.hcore.begin(),
                data[slot].one_electron.hcore.end(),
                hcore.begin() + slot * matrix_size);
      std::copy(orthogonalizers[slot].begin(), orthogonalizers[slot].end(),
                orthogonalizer.begin() + slot * matrix_size);
      std::copy(alpha_densities[slot].begin(), alpha_densities[slot].end(),
                initial_alpha.begin() + slot * matrix_size);
      std::copy(beta_densities[slot].begin(), beta_densities[slot].end(),
                initial_beta.begin() + slot * matrix_size);
      nuclear[slot] = data[slot].one_electron.nuclear_repulsion;
      const auto occupations = spin_occupations(systems[source_indices[slot]]);
      alpha_occupied[slot] = static_cast<std::int32_t>(occupations.first);
      beta_occupied[slot] = static_cast<std::int32_t>(occupations.second);
    }
    std::vector<double> device_final_alpha;
    std::vector<double> device_final_beta;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string device_detail;
    const vibeqc_status device_status =
        run_cuda_density_fitting_uhf_device_scf(
            plan.get(), hcore, orthogonalizer, initial_alpha, initial_beta,
            alpha_occupied, beta_occupied, nuclear, options.max_iterations,
            options.energy_tolerance, options.density_tolerance,
            device_final_alpha, device_final_beta, device_records,
            device_detail);
    const bool device_converged =
        device_status == VIBEQC_STATUS_SUCCESS &&
        device_records.size() == data.size() &&
        std::all_of(device_records.begin(), device_records.end(),
                    [](const CudaDensityFittingDeviceScfItem& item) {
                      return item.converged;
                    });
    if (device_converged) {
      for (std::size_t slot = 0; slot < data.size(); ++slot) {
        const std::size_t source = source_indices[slot];
        alpha_densities[slot].assign(
            device_final_alpha.begin() + slot * matrix_size,
            device_final_alpha.begin() + (slot + 1) * matrix_size);
        beta_densities[slot].assign(
            device_final_beta.begin() + slot * matrix_size,
            device_final_beta.begin() + (slot + 1) * matrix_size);
        ScfResult& result = outputs[source].scf;
        result.iterations = device_records[slot].iterations;
        result.energy = device_records[slot].energy;
        result.energy_change = device_records[slot].energy_change;
        result.density_rms = device_records[slot].density_rms;
        result.converged = true;
        try {
          const auto occupations = spin_occupations(systems[source]);
          finalize_density_fitting_uhf(
              data[slot], orthogonalizers[slot], occupations.first,
              occupations.second, alpha_densities[slot], beta_densities[slot],
              options, result, plan.get());
          outputs[source].status = VIBEQC_STATUS_SUCCESS;
        } catch (const std::bad_alloc&) {
          outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        } catch (const std::invalid_argument&) {
          outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        } catch (...) {
          outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
      }
      return outputs;
    }
  }

  const std::size_t matrix_size = nbf * nbf;
  std::vector<double> batch_alpha(data.size() * matrix_size);
  std::vector<double> batch_beta(data.size() * matrix_size);
  std::vector<bool> active(data.size(), true);
  std::size_t active_count = data.size();
  for (unsigned iteration = 1; iteration <= options.max_iterations &&
                                      active_count != 0;
       ++iteration) {
    for (std::size_t slot = 0; slot < alpha_densities.size(); ++slot) {
      std::copy(alpha_densities[slot].begin(), alpha_densities[slot].end(),
                batch_alpha.begin() + slot * matrix_size);
      std::copy(beta_densities[slot].begin(), beta_densities[slot].end(),
                batch_beta.begin() + slot * matrix_size);
    }
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    const vibeqc_status jk_status = execute_cuda_density_fitting_uhf_jk(
        plan.get(), batch_alpha, batch_beta, coulomb, alpha_exchange,
        beta_exchange, detail);
    if (jk_status != VIBEQC_STATUS_SUCCESS) {
      for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
        if (active[slot]) outputs[source_indices[slot]].status = jk_status;
        active[slot] = false;
      }
      break;
    }
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      if (!active[slot]) continue;
      const std::size_t source = source_indices[slot];
      try {
        const auto [alpha_occupied, beta_occupied] =
            spin_occupations(systems[source]);
        Matrix alpha_fock = data[slot].one_electron.hcore;
        Matrix beta_fock = data[slot].one_electron.hcore;
        const double* j = coulomb.data() + slot * matrix_size;
        const double* ak = alpha_exchange.data() + slot * matrix_size;
        const double* bk = beta_exchange.data() + slot * matrix_size;
        for (std::size_t element = 0; element < matrix_size; ++element) {
          alpha_fock[element] += j[element] - ak[element];
          beta_fock[element] += j[element] - bk[element];
        }
        const double energy = uhf_electronic_energy(
                                  alpha_densities[slot], beta_densities[slot],
                                  data[slot].one_electron.hcore, alpha_fock,
                                  beta_fock) +
                              data[slot].one_electron.nuclear_repulsion;
        const Matrix alpha_residual = commutator_residual(
            alpha_fock, alpha_densities[slot],
            data[slot].one_electron.overlap, nbf);
        const Matrix beta_residual = commutator_residual(
            beta_fock, beta_densities[slot],
            data[slot].one_electron.overlap, nbf);
        const Matrix effective_joined = diis[slot].update(
            concatenate(alpha_fock, beta_fock),
            concatenate(alpha_residual, beta_residual));
        std::tie(alpha_fock, beta_fock) =
            split_spin_matrices(effective_joined, matrix_size);
        alpha_orbitals[slot] = generalized_eigen(
            alpha_fock, orthogonalizers[slot], nbf);
        beta_orbitals[slot] = generalized_eigen(
            beta_fock, orthogonalizers[slot], nbf);
        Matrix next_alpha = density_from_orbitals(
            alpha_orbitals[slot].vectors, nbf, alpha_occupied, 1.0);
        Matrix next_beta = density_from_orbitals(
            beta_orbitals[slot].vectors, nbf, beta_occupied, 1.0);
        ScfResult& result = outputs[source].scf;
        result.iterations = iteration;
        result.energy = energy;
        result.energy_change = std::isfinite(previous_energies[slot])
                                   ? std::abs(energy - previous_energies[slot])
                                   : std::numeric_limits<double>::infinity();
        result.density_rms = density_rms(
            concatenate(next_alpha, next_beta),
            concatenate(alpha_densities[slot], beta_densities[slot]));
        if (iteration > 1 &&
            result.energy_change < options.energy_tolerance &&
            result.density_rms < options.density_tolerance) {
          alpha_densities[slot] = std::move(next_alpha);
          beta_densities[slot] = std::move(next_beta);
          result.converged = true;
          active[slot] = false;
          --active_count;
        } else {
          previous_energies[slot] = energy;
          alpha_densities[slot] = std::move(next_alpha);
          beta_densities[slot] = std::move(next_beta);
        }
      } catch (const std::bad_alloc&) {
        outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        active[slot] = false;
        --active_count;
      } catch (...) {
        outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        active[slot] = false;
        --active_count;
      }
    }
  }

  for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
    const std::size_t source = source_indices[slot];
    ScfResult& result = outputs[source].scf;
    if (outputs[source].status != VIBEQC_STATUS_INTERNAL_ERROR) {
      continue;
    }
    if (!result.converged) {
      outputs[source].status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
      continue;
    }
    try {
      const auto [alpha_occupied, beta_occupied] =
          spin_occupations(systems[source]);
      finalize_density_fitting_uhf(
          data[slot], orthogonalizers[slot], alpha_occupied, beta_occupied,
          alpha_densities[slot], beta_densities[slot], options, result,
          plan.get());
      outputs[source].status = VIBEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  return outputs;
}

#endif  // VIBEQC_HAS_CUDA

#if VIBEQC_HAS_CUDA

ScfResult run_rhf_density_fitting_cuda(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    int device_id,
    const std::vector<double>* initial_density) {
  return run_rhf_density_fitting_cuda_impl(
      system, auxiliary_system, options, device_id, initial_density);
}

ScfResult run_uhf_density_fitting_cuda(
    const core::System& system,
    const core::System& auxiliary_system,
    const ScfOptions& options,
    int device_id,
    const std::vector<double>* initial_density) {
  return run_uhf_density_fitting_cuda_impl(
      system, auxiliary_system, options, device_id, initial_density);
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  return run_rhf_density_fitting_cuda_bucket_impl(
      systems, auxiliary_template, options, initial_densities, device_id,
      diagnostics);
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  return run_uhf_density_fitting_cuda_bucket_impl(
      systems, auxiliary_template, options, initial_densities, device_id,
      diagnostics);
}

#endif  // VIBEQC_HAS_CUDA

#if !VIBEQC_HAS_CUDA
// Keep diagnostics identical to the CUDA backend's small persistent-ERI
// policy without exposing an implementation tuning threshold through the ABI.
constexpr std::size_t kDiagnosticPersistentEriAoLimit = 16;

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(
    const std::vector<core::System>& systems) {
  if (systems.empty()) {
    throw std::invalid_argument("a CUDA RHF basis layout requires systems");
  }
  const std::size_t nbf = molecule::ao_count(systems.front());
  const std::size_t direct_nbf =
      molecule::cartesian_ao_count(systems.front());
  std::size_t shell_count = 0;
  std::size_t shell_pair_count = 0;
  std::size_t shell_quartet_count = 0;
  std::size_t unique_primitives = 0;
  std::size_t expanded_primitives = 0;
  for (const core::System& system : systems) {
    if (molecule::ao_count(system) != nbf ||
        molecule::cartesian_ao_count(system) != direct_nbf) {
      throw std::invalid_argument("systems do not belong to one CUDA RHF bucket");
    }
    shell_count += system.shells.size();
    const std::size_t system_shell_pairs =
        system.shells.size() * (system.shells.size() + 1) / 2;
    shell_pair_count += system_shell_pairs;
    shell_quartet_count +=
        system_shell_pairs * (system_shell_pairs + 1) / 2;
    for (const core::Shell& shell : system.shells) {
      unique_primitives += shell.primitives.size();
      expanded_primitives += molecule::cartesian_count(shell.angular_momentum) *
                             shell.primitives.size();
    }
  }
  const std::size_t ao_count = systems.size() * nbf;
  const std::size_t direct_ao_count = systems.size() * direct_nbf;
  const std::size_t device_basis_bytes =
      (systems.size() + 1) * sizeof(std::int64_t) +
      shell_count * (sizeof(std::int32_t) + sizeof(std::uint8_t)) +
      3 * (shell_count + 1) * sizeof(std::int64_t) +
      2 * (systems.size() + 1) * sizeof(std::int64_t) +
      shell_pair_count * 3 * sizeof(std::int32_t) +
      ao_count *
          (sizeof(std::int32_t) + sizeof(std::uint8_t) +
           3 * molecule::kMaximumAoExpansionTerms * sizeof(std::uint8_t) +
           molecule::kMaximumAoExpansionTerms * sizeof(double)) +
      direct_ao_count *
          (sizeof(std::int32_t) + 3 * sizeof(std::uint8_t) + sizeof(double)) +
      (direct_nbf == nbf || nbf <= kDiagnosticPersistentEriAoLimit
           ? 0
           : systems.size() * nbf * direct_nbf * sizeof(double)) +
      unique_primitives * 2 * sizeof(double);
  return {systems.size(), shell_count, shell_pair_count, shell_quartet_count,
          ao_count, unique_primitives,
          expanded_primitives, device_basis_bytes,
          detail::direct_topology_requires_bounded_streaming(
              shell_quartet_count),
          detail::direct_topology_requires_bounded_streaming(
              shell_quartet_count)
              ? detail::kBoundedDirectQueueCapacity
              : 0};
}

ScfResult run_rhf_cuda(const core::System&,
                       const ScfOptions&,
                             int,
                             const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

ScfResult run_uhf_cuda(const core::System&,
                       const ScfOptions&,
                             int,
                             const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

ScfResult run_rhf_density_fitting_cuda(
    const core::System&,
    const core::System&,
    const ScfOptions&,
    int,
    const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

ScfResult run_uhf_density_fitting_cuda(
    const core::System&,
    const core::System&,
    const ScfOptions&,
    int,
    const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems,
    const std::optional<core::System>&,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems,
    const std::optional<core::System>&,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan**,
    const std::vector<core::System>& systems,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan**,
    const std::vector<core::System>& systems,
    const ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan*) noexcept {}

void set_rhf_cuda_bucket_warm_start_updates(
    CudaRhfBucketPlan*, bool) noexcept {}

void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan*) noexcept {}

bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan*, CudaRhfShellClassProfile&) noexcept {
  return false;
}

bool get_rhf_cuda_ppps_queue_profile(
    const CudaRhfBucketPlan*, CudaPppsQueueProfile&) noexcept {
  return false;
}

bool get_rhf_cuda_eigensolver_diagnostic(
    const CudaRhfBucketPlan*, CudaEigensolverDiagnostic&) noexcept {
  return false;
}

bool get_rhf_cuda_inactive_eigensolver_profile(
    const CudaRhfBucketPlan*, CudaInactiveEigensolverProfile&) noexcept {
  return false;
}
#endif

}  // namespace vibeqc::scf
