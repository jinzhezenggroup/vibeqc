#include "scf/rhf.hpp"

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace qce::scf {
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
                  core::ScfResult& result) {
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
                  core::ScfResult& result) {
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

}  // namespace

core::ScfResult run_rhf(const core::System& system,
                        const core::ScfOptions& options,
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

  core::ScfResult result;
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

core::ScfResult run_uhf(const core::System& system,
                        const core::ScfOptions& options,
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

  core::ScfResult result;
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

#if !QCE_HAS_CUDA
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
          expanded_primitives, device_basis_bytes};
}

core::ScfResult run_rhf_cuda(const core::System&,
                             const core::ScfOptions&,
                             int,
                             const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

core::ScfResult run_uhf_cuda(const core::System&,
                             const core::ScfOptions&,
                             int,
                             const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = QCE_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan**,
    const std::vector<core::System>& systems,
    const core::ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = QCE_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = QCE_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan**,
    const std::vector<core::System>& systems,
    const core::ScfOptions&,
    const std::vector<const std::vector<double>*>&,
    int,
    bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = QCE_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan*) noexcept {}

bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan*, CudaRhfShellClassProfile&) noexcept {
  return false;
}
#endif

}  // namespace qce::scf
