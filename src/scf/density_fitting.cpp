#include "scf/density_fitting.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vibeqc::scf {
namespace {

std::size_t index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

struct EigenResult {
  std::vector<double> values;
  std::vector<double> vectors;
};

// A cyclic Jacobi solve keeps the CPU oracle dependency-free while avoiding
// the O(n^4) search cost of choosing the largest pivot before every rotation.
// Production device factorization will use cuSOLVER instead of this routine.
EigenResult symmetric_eigen(std::vector<double> matrix, std::size_t n) {
  std::vector<double> vectors(matrix.size(), 0.0);
  for (std::size_t item = 0; item < n; ++item) {
    vectors[index(item, item, n)] = 1.0;
  }
  constexpr std::size_t maximum_sweeps = 100;
  bool converged = n == 1;
  for (std::size_t sweep = 0; sweep < maximum_sweeps && !converged;
       ++sweep) {
    double matrix_scale = 0.0;
    for (double value : matrix) {
      matrix_scale = std::max(matrix_scale, std::abs(value));
    }
    if (matrix_scale == 0.0) {
      converged = true;
      break;
    }
    const double tolerance = 1.0e-14 * matrix_scale;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = matrix[index(p, q, n)];
        if (std::abs(apq) <= tolerance) continue;

        const double app = matrix[index(p, p, n)];
        const double aqq = matrix[index(q, q, n)];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
          if (k == p || k == q) continue;
          const double mkp = matrix[index(k, p, n)];
          const double mkq = matrix[index(k, q, n)];
          matrix[index(k, p, n)] = matrix[index(p, k, n)] =
              cosine * mkp - sine * mkq;
          matrix[index(k, q, n)] = matrix[index(q, k, n)] =
              sine * mkp + cosine * mkq;
        }
        matrix[index(p, p, n)] = cosine * cosine * app -
            2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[index(q, q, n)] = sine * sine * app +
            2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[index(p, q, n)] = matrix[index(q, p, n)] = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
          const double vkp = vectors[index(row, p, n)];
          const double vkq = vectors[index(row, q, n)];
          vectors[index(row, p, n)] = cosine * vkp - sine * vkq;
          vectors[index(row, q, n)] = sine * vkp + cosine * vkq;
        }
      }
    }
    double largest_off_diagonal = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = row + 1; column < n; ++column) {
        largest_off_diagonal = std::max(
            largest_off_diagonal,
            std::abs(matrix[index(row, column, n)]));
      }
    }
    converged = largest_off_diagonal <= tolerance;
  }
  if (!converged) {
    throw std::runtime_error(
        "Coulomb metric eigensolver did not converge");
  }

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return matrix[index(a, a, n)] < matrix[index(b, b, n)];
  });
  EigenResult result;
  result.values.resize(n);
  result.vectors.resize(matrix.size());
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source = order[column];
    result.values[column] = matrix[index(source, source, n)];
    for (std::size_t row = 0; row < n; ++row) {
      result.vectors[index(row, column, n)] =
          vectors[index(row, source, n)];
    }
  }
  return result;
}

bool checked_multiply(
    std::size_t first, std::size_t second, std::size_t& product) {
  if (first != 0 &&
      second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

std::size_t checked_matrix_elements(std::size_t dimension,
                                    const char* description) {
  std::size_t elements = 0;
  if (dimension == 0 || !checked_multiply(dimension, dimension, elements)) {
    throw std::invalid_argument(description);
  }
  return elements;
}

std::size_t checked_three_center_elements(std::size_t nbf, std::size_t naux) {
  const std::size_t matrix_elements =
      checked_matrix_elements(nbf, "DF orbital dimension is invalid");
  std::size_t elements = 0;
  if (naux == 0 || !checked_multiply(matrix_elements, naux, elements)) {
    throw std::invalid_argument("DF three-center dimensions are invalid");
  }
  return elements;
}

void require_finite(const std::vector<double>& values,
                    const char* description) {
  if (!std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(description);
  }
}

std::size_t three_center_index(std::size_t mu, std::size_t nu,
                               std::size_t auxiliary, std::size_t nbf,
                               std::size_t naux) {
  return (mu * nbf + nu) * naux + auxiliary;
}

void validate_three_center(const DensityFittingThreeCenter& three_center) {
  const std::size_t expected = checked_three_center_elements(
      three_center.nbf, three_center.naux);
  if (three_center.values.size() != expected ||
      three_center.effective_rank == 0 ||
      three_center.effective_rank > three_center.naux) {
    throw std::invalid_argument(
        "orthonormalized DF three-center tensor is inconsistent");
  }
  require_finite(three_center.values,
                 "orthonormalized DF three-center entries must be finite");
}

void validate_density(const std::vector<double>& density,
                      std::size_t matrix_elements) {
  if (density.size() != matrix_elements) {
    throw std::invalid_argument("DF density dimensions are inconsistent");
  }
  require_finite(density, "DF density entries must be finite");
}

std::vector<double> build_coulomb(const DensityFittingThreeCenter& three_center,
                                  const std::vector<double>& density) {
  const std::size_t nbf = three_center.nbf;
  const std::size_t naux = three_center.naux;
  std::vector<double> auxiliary_density(naux, 0.0);
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      const double density_value = density[index(mu, nu, nbf)];
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        auxiliary_density[auxiliary] +=
            density_value *
            three_center
                .values[three_center_index(mu, nu, auxiliary, nbf, naux)];
      }
    }
  }

  std::vector<double> coulomb(nbf * nbf, 0.0);
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      double value = 0.0;
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        value += three_center
                     .values[three_center_index(mu, nu, auxiliary, nbf, naux)] *
                 auxiliary_density[auxiliary];
      }
      coulomb[index(mu, nu, nbf)] = value;
    }
  }
  return coulomb;
}

std::vector<double> build_exchange(
    const DensityFittingThreeCenter& three_center,
    const std::vector<double>& density) {
  const std::size_t nbf = three_center.nbf;
  const std::size_t naux = three_center.naux;
  std::vector<double> exchange(nbf * nbf, 0.0);
  std::vector<double> transformed_density(nbf * nbf, 0.0);
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    std::fill(transformed_density.begin(), transformed_density.end(), 0.0);
    // For each Q, form B_Q D and then (B_Q D) B_Q^T. This O(N^3 Naux)
    // ordering mirrors the two GEMMs used by the future blocked CUDA path and
    // avoids materializing any four-center ERIs in the CPU oracle.
    for (std::size_t mu = 0; mu < nbf; ++mu) {
      for (std::size_t lambda = 0; lambda < nbf; ++lambda) {
        double value = 0.0;
        for (std::size_t kappa = 0; kappa < nbf; ++kappa) {
          value +=
              three_center
                  .values[three_center_index(mu, kappa, auxiliary, nbf, naux)] *
              density[index(kappa, lambda, nbf)];
        }
        transformed_density[index(mu, lambda, nbf)] = value;
      }
    }
    for (std::size_t mu = 0; mu < nbf; ++mu) {
      for (std::size_t nu = 0; nu < nbf; ++nu) {
        double value = 0.0;
        for (std::size_t lambda = 0; lambda < nbf; ++lambda) {
          value +=
              transformed_density[index(mu, lambda, nbf)] *
              three_center
                  .values[three_center_index(nu, lambda, auxiliary, nbf, naux)];
        }
        exchange[index(mu, nu, nbf)] += value;
      }
    }
  }
  return exchange;
}

void validate_density_fitting_derivative_data(
    const integrals::DensityFittingIntegralData& data) {
  if (data.nbf == 0 || data.naux == 0) {
    throw std::invalid_argument("DF derivative dimensions must be positive");
  }
  std::size_t matrix_elements = 0;
  std::size_t metric_elements = 0;
  std::size_t three_center_elements = 0;
  std::size_t derivative_metric_elements = 0;
  std::size_t derivative_three_center_elements = 0;
  if (!checked_multiply(data.nbf, data.nbf, matrix_elements) ||
      !checked_multiply(data.naux, data.naux, metric_elements) ||
      !checked_multiply(matrix_elements, data.naux, three_center_elements) ||
      !checked_multiply(data.ncoord, metric_elements,
                        derivative_metric_elements) ||
      !checked_multiply(data.ncoord, three_center_elements,
                        derivative_three_center_elements) ||
      data.metric.size() != metric_elements ||
      data.three_center.size() != three_center_elements ||
      data.metric_derivative.size() != derivative_metric_elements ||
      data.three_center_derivative.size() !=
          derivative_three_center_elements) {
    throw std::invalid_argument(
        "DF derivative integral dimensions are inconsistent");
  }
  require_finite(data.metric, "DF metric entries must be finite");
  require_finite(data.three_center,
                 "DF three-center entries must be finite");
  require_finite(data.metric_derivative,
                 "DF metric derivative entries must be finite");
  require_finite(data.three_center_derivative,
                 "DF three-center derivative entries must be finite");
}

std::vector<double> metric_pseudoinverse(
    const integrals::DensityFittingIntegralData& data,
    double relative_threshold) {
  const DensityFittingMetricFactor factor = factor_density_fitting_metric(
      data.metric, data.naux, relative_threshold);
  std::vector<double> inverse(data.naux * data.naux, 0.0);
  // The symmetric inverse square root is also a convenient, stable way to
  // construct the metric pseudoinverse: M+ = M^{-1/2} M^{-1/2}.
  for (std::size_t row = 0; row < data.naux; ++row) {
    for (std::size_t column = 0; column < data.naux; ++column) {
      double value = 0.0;
      for (std::size_t item = 0; item < data.naux; ++item) {
        value += factor.inverse_square_root[index(row, item, data.naux)] *
                 factor.inverse_square_root[index(column, item, data.naux)];
      }
      inverse[index(row, column, data.naux)] = value;
    }
  }
  return inverse;
}

std::vector<double> metric_pseudoinverse_derivative(
    const integrals::DensityFittingIntegralData& data,
    const std::vector<double>& inverse, const double* metric_derivative) {
  const std::size_t naux = data.naux;
  std::vector<double> symmetric_metric(naux * naux, 0.0);
  std::vector<double> symmetric_derivative(naux * naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      symmetric_metric[index(row, column, naux)] = 0.5 *
          (data.metric[index(row, column, naux)] +
           data.metric[index(column, row, naux)]);
      symmetric_derivative[index(row, column, naux)] = 0.5 *
          (metric_derivative[index(row, column, naux)] +
           metric_derivative[index(column, row, naux)]);
    }
  }

  // For a fixed-rank symmetric positive-semidefinite metric, the derivative
  // of the Moore-Penrose inverse is
  //   dM+ = -M+ dM M+ + M+^2 dM (I-MM+) + (I-M+M) dM M+^2.
  // The projector terms are essential when the thresholded null space mixes
  // with retained auxiliary directions; the common -M+ dM M+ shortcut is
  // correct only for a strictly full-rank metric.
  std::vector<double> metric_inverse_squared(naux * naux, 0.0);
  std::vector<double> left_null_projector(naux * naux, 0.0);
  std::vector<double> right_null_projector(naux * naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      double inverse_squared = 0.0;
      double metric_times_inverse = 0.0;
      double inverse_times_metric = 0.0;
      for (std::size_t item = 0; item < naux; ++item) {
        inverse_squared += inverse[index(row, item, naux)] *
                           inverse[index(item, column, naux)];
        metric_times_inverse += symmetric_metric[index(row, item, naux)] *
                                inverse[index(item, column, naux)];
        inverse_times_metric += inverse[index(row, item, naux)] *
                                symmetric_metric[index(item, column, naux)];
      }
      metric_inverse_squared[index(row, column, naux)] = inverse_squared;
      left_null_projector[index(row, column, naux)] =
          (row == column ? 1.0 : 0.0) - metric_times_inverse;
      right_null_projector[index(row, column, naux)] =
          (row == column ? 1.0 : 0.0) - inverse_times_metric;
    }
  }

  // Evaluate the three matrix products as O(naux^3) contractions.  The
  // original four-index expansion is algebraically identical but becomes a
  // dominant cost for realistic auxiliary bases (and is unnecessary because
  // all factors are dense square matrices).
  const auto multiply_square = [&](const std::vector<double>& first,
                                   const std::vector<double>& second) {
    std::vector<double> product(naux * naux, 0.0);
    for (std::size_t row = 0; row < naux; ++row) {
      for (std::size_t item = 0; item < naux; ++item) {
        const double value = first[index(row, item, naux)];
        if (value == 0.0) continue;
        for (std::size_t column = 0; column < naux; ++column) {
          product[index(row, column, naux)] +=
              value * second[index(item, column, naux)];
        }
      }
    }
    return product;
  };
  const std::vector<double> inverse_derivative_left =
      multiply_square(inverse, symmetric_derivative);
  const std::vector<double> first_term =
      multiply_square(inverse_derivative_left, inverse);
  const std::vector<double> squared_derivative_left =
      multiply_square(metric_inverse_squared, symmetric_derivative);
  const std::vector<double> second_term =
      multiply_square(squared_derivative_left, left_null_projector);
  const std::vector<double> null_derivative_left =
      multiply_square(right_null_projector, symmetric_derivative);
  const std::vector<double> third_term =
      multiply_square(null_derivative_left, metric_inverse_squared);

  std::vector<double> derivative(naux * naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      const std::size_t item = index(row, column, naux);
      derivative[item] = -first_term[item] + second_term[item] +
                         third_term[item];
    }
  }
  // Symmetry is an invariant of the Coulomb metric and its Moore-Penrose
  // inverse. Enforce it explicitly so tiny eigensolver/BLAS roundoff cannot
  // leak a skew component into the subsequent quadratic contraction.
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = row + 1; column < naux; ++column) {
      const double symmetric = 0.5 *
          (derivative[index(row, column, naux)] +
           derivative[index(column, row, naux)]);
      derivative[index(row, column, naux)] = symmetric;
      derivative[index(column, row, naux)] = symmetric;
    }
  }
  require_finite(derivative,
                 "DF metric pseudoinverse derivative is non-finite");
  return derivative;
}

void validate_gradient_density(const std::vector<double>& density,
                               std::size_t nbf,
                               const char* description) {
  std::size_t matrix_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_elements) ||
      density.size() != matrix_elements) {
    throw std::invalid_argument(description);
  }
  require_finite(density, description);
}

double coulomb_quadratic_derivative(
    const integrals::DensityFittingIntegralData& data,
    const std::vector<double>& density, const std::vector<double>& inverse,
    const std::vector<double>& inverse_derivative,
    const double* three_center_derivative) {
  const std::size_t nbf = data.nbf;
  const std::size_t naux = data.naux;
  std::vector<double> charge(naux, 0.0);
  std::vector<double> derivative_charge(naux, 0.0);
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      const double density_value = density[index(mu, nu, nbf)];
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        const std::size_t item =
            three_center_index(mu, nu, auxiliary, nbf, naux);
        charge[auxiliary] += density_value * data.three_center[item];
        derivative_charge[auxiliary] +=
            density_value * three_center_derivative[item];
      }
    }
  }

  std::vector<double> metric_potential(naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      metric_potential[row] += inverse[index(row, column, naux)] *
                               charge[column];
    }
  }
  double derivative = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    derivative += derivative_charge[auxiliary] * metric_potential[auxiliary];
  }

  double metric_response = 0.0;
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      metric_response += charge[row] * inverse_derivative[
          index(row, column, naux)] * charge[column];
    }
  }
  // E_J = 1/2 rho^T M+ rho, hence the metric response enters as
  // +1/2 rho^T (dM+) rho.  (The sign is already carried by dM+.)
  return derivative + 0.5 * metric_response;
}

double exchange_quadratic_derivative(
    const integrals::DensityFittingIntegralData& data,
    const std::vector<double>& density, const std::vector<double>& inverse,
    const std::vector<double>& inverse_derivative,
    const double* three_center_derivative) {
  const std::size_t nbf = data.nbf;
  const std::size_t naux = data.naux;
  const std::size_t matrix_elements = nbf * nbf;
  // Rewrite the four-AO exchange contraction as two matrix products for each
  // auxiliary function.  Besides matching the CUDA RI-K schedule, this keeps
  // the independent force oracle practical for medium-sized test molecules:
  // the straightforward O(n^4 naux^2) loop is reduced to
  // O(n^3 naux + n^2 naux^2).
  std::vector<std::vector<double>> response(naux,
                                            std::vector<double>(matrix_elements));
  std::vector<std::vector<double>> derivative_response(
      naux, std::vector<double>(matrix_elements));
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    for (std::size_t i = 0; i < nbf; ++i) {
      for (std::size_t column = 0; column < nbf; ++column) {
        double transformed = 0.0;
        double derivative_transformed = 0.0;
        for (std::size_t k = 0; k < nbf; ++k) {
          const std::size_t tensor_item =
              three_center_index(i, k, auxiliary, nbf, naux);
          transformed += data.three_center[tensor_item] *
                         density[index(k, column, nbf)];
          derivative_transformed += three_center_derivative[tensor_item] *
                                    density[index(k, column, nbf)];
        }
        for (std::size_t row = 0; row < nbf; ++row) {
          response[auxiliary][index(row, column, nbf)] +=
              density[index(i, row, nbf)] * transformed;
          derivative_response[auxiliary][index(row, column, nbf)] +=
              density[index(i, row, nbf)] * derivative_transformed;
        }
      }
    }
  }

  double derivative = 0.0;
  for (std::size_t first_auxiliary = 0; first_auxiliary < naux;
       ++first_auxiliary) {
    for (std::size_t second_auxiliary = 0; second_auxiliary < naux;
         ++second_auxiliary) {
      double quadratic = 0.0;
      double derivative_quadratic = 0.0;
      for (std::size_t row = 0; row < nbf; ++row) {
        for (std::size_t column = 0; column < nbf; ++column) {
          const std::size_t pair = index(row, column, nbf);
          const std::size_t tensor_item =
              three_center_index(row, column, second_auxiliary, nbf, naux);
          quadratic += response[first_auxiliary][pair] *
                       data.three_center[tensor_item];
          derivative_quadratic +=
              derivative_response[first_auxiliary][pair] *
                  data.three_center[tensor_item] +
              response[first_auxiliary][pair] *
                  three_center_derivative[tensor_item];
        }
      }
      derivative +=
          derivative_quadratic *
              inverse[index(first_auxiliary, second_auxiliary, naux)] +
          quadratic *
              inverse_derivative[index(first_auxiliary, second_auxiliary,
                                       naux)];
    }
  }
  return derivative;
}

std::size_t workspace_bytes(
    std::size_t batch_tile,
    std::size_t ao_pair_tile,
    std::size_t auxiliary_tile,
    std::size_t occupied_tile,
    std::size_t batch_size,
    std::size_t nbf,
    std::size_t naux,
    std::size_t metric_bytes) {
  (void)batch_tile;
  (void)occupied_tile;
  // The CUDA plan keeps seven AO matrices and one auxiliary vector for the
  // complete batch.  Its streamed tile rounds the logical AO-pair budget up
  // to a whole row, so account for that physical capacity rather than the
  // planner's logical pair count. Setup/factorization storage is also charged
  // for every batch metric. This makes the planner's byte budget a conservative
  // bound on the actual device allocation, not just on one contraction tile.
  const long double matrix_elements = static_cast<long double>(batch_size) *
                                      static_cast<long double>(nbf) * nbf;
  const long double tensor_elements = matrix_elements * naux;
  const long double staged_rows = std::min<std::size_t>(
      nbf, std::max<std::size_t>(1, ao_pair_tile / std::max<std::size_t>(1, nbf)));
  const long double staged_pairs = staged_rows * nbf;
  const long double tile_elements = staged_pairs * auxiliary_tile;
  const long double setup_doubles =
      static_cast<long double>(batch_size) * (3.0L * naux * naux + 2.0L * naux);
  const long double contraction_doubles =
      7.0L * matrix_elements +
      static_cast<long double>(batch_size) * naux +
      3.0L * tile_elements +
      ((auxiliary_tile < naux || ao_pair_tile < nbf * nbf) ? tile_elements : 0.0L) +
      ((auxiliary_tile < naux || ao_pair_tile < nbf * nbf)
           ? 0.0L
           : tensor_elements);
  // Force-response scratch is allocated one system/coordinate at a time by
  // the finalizer and is not part of the persistent contraction-plan budget.
  // It is still reported in CUDA diagnostics as part of peak_device_bytes.
  const long double force_scratch_doubles = 0.0L;
  const long double bytes = static_cast<long double>(metric_bytes) * batch_size +
                            (setup_doubles + contraction_doubles +
                             force_scratch_doubles) * sizeof(double);
  if (bytes > static_cast<long double>(
                  std::numeric_limits<std::size_t>::max())) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(bytes);
}

}  // namespace

std::vector<double> density_fitting_metric_pseudoinverse(
    const integrals::DensityFittingIntegralData& integrals,
    double relative_threshold) {
  validate_density_fitting_derivative_data(integrals);
  return metric_pseudoinverse(integrals, relative_threshold);
}

std::vector<double> density_fitting_metric_pseudoinverse_derivative(
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& inverse, std::size_t coordinate) {
  validate_density_fitting_derivative_data(integrals);
  if (inverse.size() != integrals.naux * integrals.naux) {
    throw std::invalid_argument(
        "DF metric pseudoinverse dimensions are inconsistent");
  }
  if (coordinate >= integrals.ncoord) {
    throw std::invalid_argument("DF metric derivative coordinate is invalid");
  }
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  return metric_pseudoinverse_derivative(
      integrals, inverse,
      integrals.metric_derivative.data() + coordinate * metric_elements);
}

DensityFittingMetricFactor factor_density_fitting_metric(
    const std::vector<double>& metric,
    std::size_t dimension,
    double relative_threshold) {
  std::size_t metric_elements = 0;
  if (dimension == 0 ||
      !checked_multiply(dimension, dimension, metric_elements) ||
      metric.size() != metric_elements) {
    throw std::invalid_argument("metric dimensions are inconsistent");
  }
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0)) {
    throw std::invalid_argument(
        "metric relative threshold must lie strictly between zero and one");
  }
  std::vector<double> symmetric(metric.size());
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      const double value = 0.5 *
          (metric[index(row, column, dimension)] +
           metric[index(column, row, dimension)]);
      if (!std::isfinite(value)) {
        throw std::invalid_argument("metric entries must be finite");
      }
      symmetric[index(row, column, dimension)] = value;
    }
  }
  const EigenResult eigen = symmetric_eigen(std::move(symmetric), dimension);
  const double largest = eigen.values.back();
  if (!(largest > 0.0) || !std::isfinite(largest)) {
    throw std::runtime_error("Coulomb metric has no positive eigenspace");
  }
  DensityFittingMetricFactor result;
  result.dimension = dimension;
  result.absolute_threshold = relative_threshold * largest;
  result.inverse_square_root.assign(metric_elements, 0.0);
  double smallest_retained = largest;
  for (std::size_t item = 0; item < dimension; ++item) {
    const double value = eigen.values[item];
    if (value <= result.absolute_threshold) continue;
    ++result.effective_rank;
    smallest_retained = std::min(smallest_retained, value);
    const double scale = 1.0 / std::sqrt(value);
    for (std::size_t row = 0; row < dimension; ++row) {
      for (std::size_t column = 0; column < dimension; ++column) {
        result.inverse_square_root[index(row, column, dimension)] +=
            eigen.vectors[index(row, item, dimension)] * scale *
            eigen.vectors[index(column, item, dimension)];
      }
    }
  }
  if (result.effective_rank == 0) {
    throw std::runtime_error(
        "Coulomb metric threshold removed every auxiliary direction");
  }
  result.condition_number = largest / smallest_retained;
  return result;
}

DensityFittingThreeCenter orthonormalize_density_fitting_three_center(
    const std::vector<double>& three_center, std::size_t nbf,
    const DensityFittingMetricFactor& metric_factor) {
  const std::size_t naux = metric_factor.dimension;
  const std::size_t tensor_elements = checked_three_center_elements(nbf, naux);
  const std::size_t metric_elements =
      checked_matrix_elements(naux, "DF metric factor dimension is invalid");
  if (three_center.size() != tensor_elements ||
      metric_factor.inverse_square_root.size() != metric_elements ||
      metric_factor.effective_rank == 0 ||
      metric_factor.effective_rank > naux) {
    throw std::invalid_argument(
        "DF three-center tensor and metric factor are inconsistent");
  }
  require_finite(three_center, "DF three-center entries must be finite");
  require_finite(metric_factor.inverse_square_root,
                 "DF metric factor entries must be finite");

  DensityFittingThreeCenter result{
      nbf,
      naux,
      metric_factor.effective_rank,
      std::vector<double>(tensor_elements, 0.0),
  };
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      for (std::size_t target = 0; target < naux; ++target) {
        double value = 0.0;
        for (std::size_t source = 0; source < naux; ++source) {
          value +=
              three_center[three_center_index(mu, nu, source, nbf, naux)] *
              metric_factor.inverse_square_root[index(source, target, naux)];
        }
        result.values[three_center_index(mu, nu, target, nbf, naux)] = value;
      }
    }
  }
  return result;
}

DensityFittingRhfJk build_density_fitting_rhf_jk(
    const DensityFittingThreeCenter& three_center,
    const std::vector<double>& density) {
  validate_three_center(three_center);
  const std::size_t matrix_elements = checked_matrix_elements(
      three_center.nbf, "DF orbital dimension is invalid");
  validate_density(density, matrix_elements);
  return {
      three_center.nbf,
      build_coulomb(three_center, density),
      build_exchange(three_center, density),
  };
}

DensityFittingUhfJk build_density_fitting_uhf_jk(
    const DensityFittingThreeCenter& three_center,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density) {
  validate_three_center(three_center);
  const std::size_t matrix_elements = checked_matrix_elements(
      three_center.nbf, "DF orbital dimension is invalid");
  validate_density(alpha_density, matrix_elements);
  validate_density(beta_density, matrix_elements);
  std::vector<double> total_density(matrix_elements, 0.0);
  for (std::size_t element = 0; element < matrix_elements; ++element) {
    total_density[element] = alpha_density[element] + beta_density[element];
  }
  return {
      three_center.nbf,
      build_coulomb(three_center, total_density),
      build_exchange(three_center, alpha_density),
      build_exchange(three_center, beta_density),
  };
}

DensityFittingRhfGradient build_density_fitting_rhf_gradient(
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& density, double relative_threshold) {
  validate_density_fitting_derivative_data(integrals);
  validate_gradient_density(density, integrals.nbf,
                            "DF RHF gradient density is inconsistent");
  const std::vector<double> inverse =
      metric_pseudoinverse(integrals, relative_threshold);
  DensityFittingRhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative.assign(integrals.ncoord, 0.0);
  result.forces.assign(integrals.ncoord, 0.0);
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  const std::size_t three_center_elements =
      integrals.nbf * integrals.nbf * integrals.naux;
  for (std::size_t coordinate = 0; coordinate < integrals.ncoord;
       ++coordinate) {
    const double* metric_derivative =
        integrals.metric_derivative.data() + coordinate * metric_elements;
    const double* three_center_derivative =
        integrals.three_center_derivative.data() +
        coordinate * three_center_elements;
    const std::vector<double> inverse_derivative =
        metric_pseudoinverse_derivative(integrals, inverse,
                                        metric_derivative);
    const double coulomb = coulomb_quadratic_derivative(
        integrals, density, inverse, inverse_derivative,
        three_center_derivative);
    const double exchange = exchange_quadratic_derivative(
        integrals, density, inverse, inverse_derivative,
        three_center_derivative);
    // `coulomb` already differentiates 1/2 (P|P)DF, while `exchange`
    // differentiates the unweighted exchange quadratic. Apply the RHF
    // exchange coefficient here, matching the closed-shell convention used
    // throughout the existing SCF implementation.
    result.derivative[coordinate] = coulomb - 0.25 * exchange;
    result.forces[coordinate] = -result.derivative[coordinate];
  }
  return result;
}

DensityFittingUhfGradient build_density_fitting_uhf_gradient(
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, double relative_threshold) {
  validate_density_fitting_derivative_data(integrals);
  validate_gradient_density(alpha_density, integrals.nbf,
                            "DF UHF alpha gradient density is inconsistent");
  validate_gradient_density(beta_density, integrals.nbf,
                            "DF UHF beta gradient density is inconsistent");
  const std::vector<double> inverse =
      metric_pseudoinverse(integrals, relative_threshold);
  std::vector<double> total_density(alpha_density.size(), 0.0);
  for (std::size_t item = 0; item < total_density.size(); ++item) {
    total_density[item] = alpha_density[item] + beta_density[item];
  }

  DensityFittingUhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative.assign(integrals.ncoord, 0.0);
  result.forces.assign(integrals.ncoord, 0.0);
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  const std::size_t three_center_elements =
      integrals.nbf * integrals.nbf * integrals.naux;
  for (std::size_t coordinate = 0; coordinate < integrals.ncoord;
       ++coordinate) {
    const double* metric_derivative =
        integrals.metric_derivative.data() + coordinate * metric_elements;
    const double* three_center_derivative =
        integrals.three_center_derivative.data() +
        coordinate * three_center_elements;
    const std::vector<double> inverse_derivative =
        metric_pseudoinverse_derivative(integrals, inverse,
                                        metric_derivative);
    const double coulomb = coulomb_quadratic_derivative(
        integrals, total_density, inverse, inverse_derivative,
        three_center_derivative);
    const double alpha_exchange = exchange_quadratic_derivative(
        integrals, alpha_density, inverse, inverse_derivative,
        three_center_derivative);
    const double beta_exchange = exchange_quadratic_derivative(
        integrals, beta_density, inverse, inverse_derivative,
        three_center_derivative);
    // `coulomb` already differentiates 1/2 J(Pa+Pb), and each exchange
    // quadratic receives the UHF -1/2 coefficient.
    result.derivative[coordinate] =
        coulomb - 0.5 * alpha_exchange - 0.5 * beta_exchange;
    result.forces[coordinate] = -result.derivative[coordinate];
  }
  return result;
}

void validate_one_electron_force_data(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting) {
  if (one_electron.nbf != density_fitting.nbf ||
      one_electron.ncoord != density_fitting.ncoord) {
    throw std::invalid_argument(
        "one-electron and DF force dimensions are inconsistent");
  }
  std::size_t matrix_elements = 0;
  if (!checked_multiply(one_electron.nbf, one_electron.nbf,
                        matrix_elements)) {
    throw std::invalid_argument("one-electron force dimensions are invalid");
  }
  std::size_t derivative_elements = 0;
  if (!checked_multiply(one_electron.ncoord, matrix_elements,
                        derivative_elements) ||
      one_electron.overlap_derivative.size() != derivative_elements ||
      one_electron.hcore_derivative.size() != derivative_elements ||
      one_electron.nuclear_repulsion_derivative.size() !=
          one_electron.ncoord) {
    throw std::invalid_argument(
        "one-electron derivative data dimensions are inconsistent");
  }
  require_finite(one_electron.overlap_derivative,
                 "overlap derivative entries must be finite");
  require_finite(one_electron.hcore_derivative,
                 "core-Hamiltonian derivative entries must be finite");
  require_finite(one_electron.nuclear_repulsion_derivative,
                 "nuclear-repulsion derivative entries must be finite");
}

std::vector<double> build_density_fitting_rhf_forces(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& density,
    const std::vector<double>& weighted_density, double relative_threshold) {
  validate_one_electron_force_data(one_electron, density_fitting);
  validate_gradient_density(density, density_fitting.nbf,
                            "DF RHF force density is inconsistent");
  validate_gradient_density(
      weighted_density, density_fitting.nbf,
      "DF RHF weighted density is inconsistent");
  const DensityFittingRhfGradient two_electron =
      build_density_fitting_rhf_gradient(density_fitting, density,
                                         relative_threshold);
  const std::size_t matrix_elements = density_fitting.nbf * density_fitting.nbf;
  std::vector<double> forces(density_fitting.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < density_fitting.ncoord;
       ++coordinate) {
    const double* overlap_derivative =
        one_electron.overlap_derivative.data() + coordinate * matrix_elements;
    const double* hcore_derivative =
        one_electron.hcore_derivative.data() + coordinate * matrix_elements;
    double derivative = two_electron.derivative[coordinate] +
                         one_electron.nuclear_repulsion_derivative[coordinate];
    for (std::size_t item = 0; item < matrix_elements; ++item) {
      derivative += density[item] * hcore_derivative[item] -
                    weighted_density[item] * overlap_derivative[item];
    }
    forces[coordinate] = -derivative;
  }
  return forces;
}

std::vector<double> build_density_fitting_uhf_forces(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density,
    const std::vector<double>& alpha_weighted_density,
    const std::vector<double>& beta_weighted_density,
    double relative_threshold) {
  validate_one_electron_force_data(one_electron, density_fitting);
  validate_gradient_density(alpha_density, density_fitting.nbf,
                            "DF UHF alpha force density is inconsistent");
  validate_gradient_density(beta_density, density_fitting.nbf,
                            "DF UHF beta force density is inconsistent");
  validate_gradient_density(
      alpha_weighted_density, density_fitting.nbf,
      "DF UHF alpha weighted density is inconsistent");
  validate_gradient_density(
      beta_weighted_density, density_fitting.nbf,
      "DF UHF beta weighted density is inconsistent");
  const DensityFittingUhfGradient two_electron =
      build_density_fitting_uhf_gradient(
          density_fitting, alpha_density, beta_density, relative_threshold);
  const std::size_t matrix_elements = density_fitting.nbf * density_fitting.nbf;
  std::vector<double> forces(density_fitting.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < density_fitting.ncoord;
       ++coordinate) {
    const double* overlap_derivative =
        one_electron.overlap_derivative.data() + coordinate * matrix_elements;
    const double* hcore_derivative =
        one_electron.hcore_derivative.data() + coordinate * matrix_elements;
    double derivative = two_electron.derivative[coordinate] +
                         one_electron.nuclear_repulsion_derivative[coordinate];
    for (std::size_t item = 0; item < matrix_elements; ++item) {
      const double total_density = alpha_density[item] + beta_density[item];
      const double total_weighted = alpha_weighted_density[item] +
                                    beta_weighted_density[item];
      derivative += total_density * hcore_derivative[item] -
                    total_weighted * overlap_derivative[item];
    }
    forces[coordinate] = -derivative;
  }
  return forces;
}

DensityFittingTilePlan plan_density_fitting_tiles(
    std::size_t batch_size,
    std::size_t nbf,
    std::size_t naux,
    std::size_t occupied,
    std::size_t memory_budget_bytes) {
  if (batch_size == 0 || nbf == 0 || naux == 0 || occupied == 0) {
    throw std::invalid_argument(
        "DF planner dimensions must all be positive");
  }
  std::size_t metric_elements = 0;
  std::size_t metric_bytes = 0;
  if (!checked_multiply(naux, naux, metric_elements) ||
      !checked_multiply(metric_elements, sizeof(double), metric_bytes)) {
    throw std::overflow_error("DF metric storage overflows size_t");
  }
  if (nbf == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("DF AO-pair count overflows size_t");
  }
  std::size_t ao_pair_count = 0;
  if (!checked_multiply(nbf, nbf, ao_pair_count)) {
    throw std::overflow_error("DF AO-pair count overflows size_t");
  }
  DensityFittingTilePlan plan{
      // CUDA currently owns one persistent plan for the complete homogeneous
      // bucket, so batch tiling is deliberately disabled until execution can
      // submit independent sub-batches without changing result ordering.
      batch_size,
      std::min<std::size_t>(ao_pair_count, 8192),
      std::min<std::size_t>(naux, 128),
      std::min<std::size_t>(occupied, 32),
      0,
      false,
  };
  auto update_bytes = [&]() {
    plan.peak_workspace_bytes = workspace_bytes(
        plan.batch_tile, plan.ao_pair_tile, plan.auxiliary_tile,
        plan.occupied_tile, batch_size, nbf, naux, metric_bytes);
  };
  update_bytes();
  // A zero budget is the documented sentinel for the implementation's
  // default tile policy; only a positive budget requests shrinking.
  while (memory_budget_bytes != 0 &&
         plan.peak_workspace_bytes > memory_budget_bytes) {
    const long double pair_cost = static_cast<long double>(
        plan.ao_pair_tile) * plan.auxiliary_tile;
    const long double occupied_cost = static_cast<long double>(
        plan.occupied_tile) * nbf * plan.auxiliary_tile;
    if (plan.ao_pair_tile > 1 && pair_cost >= occupied_cost) {
      plan.ao_pair_tile = (plan.ao_pair_tile + 1) / 2;
    } else if (plan.occupied_tile > 1) {
      plan.occupied_tile = (plan.occupied_tile + 1) / 2;
    } else if (plan.auxiliary_tile > 1) {
      plan.auxiliary_tile = (plan.auxiliary_tile + 1) / 2;
    } else if (plan.ao_pair_tile > 1) {
      plan.ao_pair_tile = (plan.ao_pair_tile + 1) / 2;
    } else {
      throw std::invalid_argument(
          "DF memory budget cannot hold the metric and one contraction tile");
    }
    update_bytes();
  }
  plan.stores_full_three_center =
      plan.batch_tile == batch_size &&
      plan.ao_pair_tile == ao_pair_count &&
      plan.auxiliary_tile == naux;
  return plan;
}

}  // namespace vibeqc::scf
