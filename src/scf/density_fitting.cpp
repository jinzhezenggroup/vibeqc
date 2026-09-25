#include "scf/density_fitting.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/df_exchange_policy.hpp"
#include "scf/df_projected_exchange_schedule.hpp"
#include "scf/df_streamed_k_policy.hpp"
#include "tensor/cpu_linalg.hpp"
#include "tensor/symmetric_matrix_function.hpp"

namespace vibeqc::scf {
namespace {

std::size_t index(std::size_t row, std::size_t column, std::size_t n) { return row * n + column; }

using EigenResult = tensor::CpuSymmetricEigenResult;

EigenResult symmetric_eigen(std::vector<double> matrix, std::size_t n) {
  // Current endpoint evidence keeps the metric eigensolve on the deterministic
  // scalar schedule; dense response products below may still use the external provider.
  const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::scalar,
                                   tensor::CpuLinalgThreadOwnership::task_parallel, 1};
  return tensor::cpu_symmetric_eigen(std::move(matrix), n, plan);
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& product) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

std::size_t checked_matrix_elements(std::size_t dimension, const char* description) {
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

void require_finite(const std::vector<double>& values, const char* description) {
  if (!std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(description);
  }
}

std::size_t three_center_index(std::size_t mu, std::size_t nu, std::size_t auxiliary,
                               std::size_t nbf, std::size_t naux) {
  return (mu * nbf + nu) * naux + auxiliary;
}

void validate_three_center(const DensityFittingThreeCenter& three_center) {
  const std::size_t expected = checked_three_center_elements(three_center.nbf, three_center.naux);
  if (three_center.values.size() != expected || three_center.effective_rank == 0 ||
      three_center.effective_rank > three_center.naux ||
      (!three_center.auxiliary_major_values.empty() &&
       three_center.auxiliary_major_values.size() != expected)) {
    throw std::invalid_argument("orthonormalized DF three-center tensor is inconsistent");
  }
  require_finite(three_center.values, "orthonormalized DF three-center entries must be finite");
  if (!three_center.auxiliary_major_values.empty()) {
    // This cache is a layout copy, not a second scientific input. Both vectors
    // are publicly mutable; finite same-sized stale caches must not change J/K.
    const auto pairs = three_center.nbf * three_center.nbf;
    for (std::size_t auxiliary = 0; auxiliary < three_center.naux; ++auxiliary)
      for (std::size_t pair = 0; pair < pairs; ++pair)
        if (three_center.auxiliary_major_values[auxiliary * pairs + pair] !=
            three_center.values[pair * three_center.naux + auxiliary])
          throw std::invalid_argument("orthonormalized DF provider cache is stale");
  }
}

void validate_density(const std::vector<double>& density, std::size_t matrix_elements) {
  if (density.size() != matrix_elements) {
    throw std::invalid_argument("DF density dimensions are inconsistent");
  }
  require_finite(density, "DF density entries must be finite");
}

std::vector<double> build_coulomb(const DensityFittingThreeCenter& three_center,
                                  const std::vector<double>& density) {
  const std::size_t nbf = three_center.nbf;
  const std::size_t naux = three_center.naux;
  const std::size_t pairs = nbf * nbf;
  std::vector<double> auxiliary_density(naux, 0.0);
  std::vector<double> coulomb(pairs, 0.0);

  constexpr std::size_t kDenseCoulombMinimum = 16;
  if (nbf >= kDenseCoulombMinimum && tensor::cpu_openblas_built()) {
    std::vector<double> temporary_auxiliary_major;
    const double* b = three_center.auxiliary_major_values.data();
    if (three_center.auxiliary_major_values.empty()) {
      temporary_auxiliary_major.resize(naux * pairs);
      for (std::size_t pair = 0; pair < pairs; ++pair)
        for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary)
          temporary_auxiliary_major[auxiliary * pairs + pair] =
              three_center.values[pair * naux + auxiliary];
      b = temporary_auxiliary_major.data();
    }
    const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::automatic,
                                     tensor::CpuLinalgThreadOwnership::provider_parallel, 1};
    tensor::cpu_gemm('N', 'N', naux, 1, pairs, b, density.data(), auxiliary_density.data(), 1.0,
                     0.0, plan);
    tensor::cpu_gemm('T', 'N', pairs, 1, naux, b, auxiliary_density.data(), coulomb.data(), 1.0,
                     0.0, plan);
    return coulomb;
  }

  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      const double density_value = density[index(mu, nu, nbf)];
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        auxiliary_density[auxiliary] +=
            density_value * three_center.values[three_center_index(mu, nu, auxiliary, nbf, naux)];
      }
    }
  }
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      double value = 0.0;
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        value += three_center.values[three_center_index(mu, nu, auxiliary, nbf, naux)] *
                 auxiliary_density[auxiliary];
      }
      coulomb[index(mu, nu, nbf)] = value;
    }
  }
  return coulomb;
}

// Keep this hot scalar contraction aligned independently of surrounding
// dispatch changes. On GCC/Zen 2, a 16-byte function shift made the unchanged
// loops about 31% slower at nbf=16; 32-byte alignment restores the baseline
// without changing arithmetic or its reduction order. Other compilers may
// ignore this optional layout hint.
#if __has_cpp_attribute(gnu::aligned)
[[gnu::aligned(32)]]
#endif
std::vector<double> build_exchange(const DensityFittingThreeCenter& three_center,
                                   const std::vector<double>& density) {
  const std::size_t nbf = three_center.nbf;
  const std::size_t naux = three_center.naux;
  std::vector<double> exchange(nbf * nbf, 0.0);
  std::vector<double> transformed_density(nbf * nbf, 0.0);

  // B is stored AO-pair-major, so one B_Q matrix is strided. A prepared
  // provider retains the Q-major copy once; direct/oracle callers can still
  // create a bounded temporary without changing the public tensor contract.
  constexpr std::size_t kDenseExchangeMinimum = 16;
  const bool use_dense_provider = nbf >= kDenseExchangeMinimum && tensor::cpu_openblas_built();
  std::vector<double> temporary_auxiliary_major;
  const double* auxiliary_major = nullptr;
  tensor::CpuLinalgPlan dense_plan;
  if (use_dense_provider) {
    if (!three_center.auxiliary_major_values.empty()) {
      auxiliary_major = three_center.auxiliary_major_values.data();
    } else {
      temporary_auxiliary_major.resize(naux * nbf * nbf);
      for (std::size_t mu = 0; mu < nbf; ++mu)
        for (std::size_t nu = 0; nu < nbf; ++nu)
          for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary)
            temporary_auxiliary_major[auxiliary * nbf * nbf + index(mu, nu, nbf)] =
                three_center.values[three_center_index(mu, nu, auxiliary, nbf, naux)];
      auxiliary_major = temporary_auxiliary_major.data();
    }
    dense_plan = {tensor::CpuLinalgProvider::automatic,
                  tensor::CpuLinalgThreadOwnership::provider_parallel, 1};
  }

  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    if (use_dense_provider) {
      const double* bq = auxiliary_major + auxiliary * nbf * nbf;
      tensor::cpu_gemm('N', 'N', nbf, nbf, nbf, bq, density.data(), transformed_density.data(), 1.0,
                       0.0, dense_plan);
      tensor::cpu_gemm('N', 'T', nbf, nbf, nbf, transformed_density.data(), bq, exchange.data(),
                       1.0, 1.0, dense_plan);
      continue;
    }

    std::fill(transformed_density.begin(), transformed_density.end(), 0.0);
    for (std::size_t mu = 0; mu < nbf; ++mu) {
      for (std::size_t lambda = 0; lambda < nbf; ++lambda) {
        double value = 0.0;
        for (std::size_t kappa = 0; kappa < nbf; ++kappa) {
          value += three_center.values[three_center_index(mu, kappa, auxiliary, nbf, naux)] *
                   density[index(kappa, lambda, nbf)];
        }
        transformed_density[index(mu, lambda, nbf)] = value;
      }
    }
    for (std::size_t mu = 0; mu < nbf; ++mu) {
      for (std::size_t nu = 0; nu < nbf; ++nu) {
        double value = 0.0;
        for (std::size_t lambda = 0; lambda < nbf; ++lambda) {
          value += transformed_density[index(mu, lambda, nbf)] *
                   three_center.values[three_center_index(nu, lambda, auxiliary, nbf, naux)];
        }
        exchange[index(mu, nu, nbf)] += value;
      }
    }
  }
  return exchange;
}

void validate_density_fitting_derivative_data(const integrals::DensityFittingIntegralData& data) {
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
      !checked_multiply(data.ncoord, metric_elements, derivative_metric_elements) ||
      !checked_multiply(data.ncoord, three_center_elements, derivative_three_center_elements) ||
      data.metric.size() != metric_elements || data.three_center.size() != three_center_elements ||
      data.metric_derivative.size() != derivative_metric_elements ||
      data.three_center_derivative.size() != derivative_three_center_elements) {
    throw std::invalid_argument("DF derivative integral dimensions are inconsistent");
  }
  require_finite(data.metric, "DF metric entries must be finite");
  require_finite(data.three_center, "DF three-center entries must be finite");
  require_finite(data.metric_derivative, "DF metric derivative entries must be finite");
  require_finite(data.three_center_derivative, "DF three-center derivative entries must be finite");
}

std::vector<double> metric_pseudoinverse(const integrals::DensityFittingIntegralData& data,
                                         double relative_threshold) {
  const DensityFittingMetricFactor factor =
      factor_density_fitting_metric(data.metric, data.naux, relative_threshold);
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
    const integrals::DensityFittingIntegralData& data, const std::vector<double>& inverse,
    const double* metric_derivative, double relative_threshold = 0.0) {
  const std::size_t elements = data.naux * data.naux;
  return density_fitting_metric_inverse_response(
      data.metric, inverse, std::vector<double>(metric_derivative, metric_derivative + elements),
      data.naux, relative_threshold);
}

void validate_gradient_density(const std::vector<double>& density, std::size_t nbf,
                               const char* description) {
  std::size_t matrix_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_elements) || density.size() != matrix_elements) {
    throw std::invalid_argument(description);
  }
  require_finite(density, description);
}

double coulomb_quadratic_derivative(const integrals::DensityFittingIntegralData& data,
                                    const std::vector<double>& density,
                                    const std::vector<double>& inverse,
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
        const std::size_t item = three_center_index(mu, nu, auxiliary, nbf, naux);
        charge[auxiliary] += density_value * data.three_center[item];
        derivative_charge[auxiliary] += density_value * three_center_derivative[item];
      }
    }
  }

  std::vector<double> metric_potential(naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      metric_potential[row] += inverse[index(row, column, naux)] * charge[column];
    }
  }
  double derivative = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    derivative += derivative_charge[auxiliary] * metric_potential[auxiliary];
  }

  double metric_response = 0.0;
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      metric_response +=
          charge[row] * inverse_derivative[index(row, column, naux)] * charge[column];
    }
  }
  // E_J = 1/2 rho^T M+ rho, hence the metric response enters as
  // +1/2 rho^T (dM+) rho.  (The sign is already carried by dM+.)
  return derivative + 0.5 * metric_response;
}

double exchange_quadratic_derivative(const integrals::DensityFittingIntegralData& data,
                                     const std::vector<double>& density,
                                     const std::vector<double>& inverse,
                                     const std::vector<double>& inverse_derivative,
                                     const double* three_center_derivative) {
  const std::size_t nbf = data.nbf;
  const std::size_t naux = data.naux;
  const std::size_t matrix_elements = nbf * nbf;

  // The force contraction has an exact dense formulation:
  //   R_Q  = D^T B_Q D
  //   dR_Q = D^T dB_Q D
  // followed by auxiliary-space Gram contractions against B and dB.
  // Keep the historical scalar loop as the no-provider oracle, but use the
  // common dense-LA boundary when an external provider is available.
  constexpr std::size_t kDenseDerivativeMinimum = 16;
  if (nbf >= kDenseDerivativeMinimum && tensor::cpu_openblas_built()) {
    const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::automatic,
                                     tensor::CpuLinalgThreadOwnership::provider_parallel, 1};
    std::vector<double> b_aux(naux * matrix_elements);
    std::vector<double> db_aux(naux * matrix_elements);
    for (std::size_t row = 0; row < nbf; ++row) {
      for (std::size_t column = 0; column < nbf; ++column) {
        const std::size_t pair = index(row, column, nbf);
        for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
          const std::size_t source = three_center_index(row, column, auxiliary, nbf, naux);
          b_aux[auxiliary * matrix_elements + pair] = data.three_center[source];
          db_aux[auxiliary * matrix_elements + pair] = three_center_derivative[source];
        }
      }
    }

    std::vector<double> response(naux * matrix_elements);
    std::vector<double> derivative_response(naux * matrix_elements);
    std::vector<double> transformed(matrix_elements);
    std::vector<double> derivative_transformed(matrix_elements);
    for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
      const double* bq = b_aux.data() + auxiliary * matrix_elements;
      const double* dbq = db_aux.data() + auxiliary * matrix_elements;
      double* rq = response.data() + auxiliary * matrix_elements;
      double* drq = derivative_response.data() + auxiliary * matrix_elements;
      tensor::cpu_gemm('N', 'N', nbf, nbf, nbf, bq, density.data(), transformed.data(), 1.0, 0.0,
                       plan);
      tensor::cpu_gemm('T', 'N', nbf, nbf, nbf, density.data(), transformed.data(), rq, 1.0, 0.0,
                       plan);
      tensor::cpu_gemm('N', 'N', nbf, nbf, nbf, dbq, density.data(), derivative_transformed.data(),
                       1.0, 0.0, plan);
      tensor::cpu_gemm('T', 'N', nbf, nbf, nbf, density.data(), derivative_transformed.data(), drq,
                       1.0, 0.0, plan);
    }

    std::vector<double> quadratic(naux * naux);
    std::vector<double> derivative_quadratic(naux * naux);
    tensor::cpu_gemm('N', 'T', naux, naux, matrix_elements, response.data(), b_aux.data(),
                     quadratic.data(), 1.0, 0.0, plan);
    tensor::cpu_gemm('N', 'T', naux, naux, matrix_elements, derivative_response.data(),
                     b_aux.data(), derivative_quadratic.data(), 1.0, 0.0, plan);
    tensor::cpu_gemm('N', 'T', naux, naux, matrix_elements, response.data(), db_aux.data(),
                     derivative_quadratic.data(), 1.0, 1.0, plan);

    double derivative = 0.0;
    for (std::size_t item = 0; item < naux * naux; ++item)
      derivative +=
          derivative_quadratic[item] * inverse[item] + quadratic[item] * inverse_derivative[item];
    return derivative;
  }

  // Scalar/no-provider oracle: preserve the established reduction order.
  std::vector<std::vector<double>> response(naux, std::vector<double>(matrix_elements));
  std::vector<std::vector<double>> derivative_response(naux, std::vector<double>(matrix_elements));
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    for (std::size_t i = 0; i < nbf; ++i) {
      for (std::size_t column = 0; column < nbf; ++column) {
        double transformed = 0.0;
        double derivative_transformed = 0.0;
        for (std::size_t k = 0; k < nbf; ++k) {
          const std::size_t tensor_item = three_center_index(i, k, auxiliary, nbf, naux);
          transformed += data.three_center[tensor_item] * density[index(k, column, nbf)];
          derivative_transformed +=
              three_center_derivative[tensor_item] * density[index(k, column, nbf)];
        }
        for (std::size_t row = 0; row < nbf; ++row) {
          response[auxiliary][index(row, column, nbf)] += density[index(i, row, nbf)] * transformed;
          derivative_response[auxiliary][index(row, column, nbf)] +=
              density[index(i, row, nbf)] * derivative_transformed;
        }
      }
    }
  }

  double derivative = 0.0;
  for (std::size_t first_auxiliary = 0; first_auxiliary < naux; ++first_auxiliary) {
    for (std::size_t second_auxiliary = 0; second_auxiliary < naux; ++second_auxiliary) {
      double quadratic = 0.0;
      double derivative_quadratic = 0.0;
      for (std::size_t row = 0; row < nbf; ++row) {
        for (std::size_t column = 0; column < nbf; ++column) {
          const std::size_t pair = index(row, column, nbf);
          const std::size_t tensor_item =
              three_center_index(row, column, second_auxiliary, nbf, naux);
          quadratic += response[first_auxiliary][pair] * data.three_center[tensor_item];
          derivative_quadratic +=
              derivative_response[first_auxiliary][pair] * data.three_center[tensor_item] +
              response[first_auxiliary][pair] * three_center_derivative[tensor_item];
        }
      }
      derivative += derivative_quadratic * inverse[index(first_auxiliary, second_auxiliary, naux)] +
                    quadratic * inverse_derivative[index(first_auxiliary, second_auxiliary, naux)];
    }
  }
  return derivative;
}

struct DensityFittingReverseWeights {
  std::vector<double> metric;
  std::vector<double> three_center;
};

void validate_density_fitting_value_data(const integrals::DensityFittingIntegralData& data) {
  std::size_t matrix_elements = 0;
  std::size_t metric_elements = 0;
  std::size_t three_center_elements = 0;
  if (data.nbf == 0 || data.naux == 0 || !checked_multiply(data.nbf, data.nbf, matrix_elements) ||
      !checked_multiply(data.naux, data.naux, metric_elements) ||
      !checked_multiply(matrix_elements, data.naux, three_center_elements) ||
      data.metric.size() != metric_elements || data.three_center.size() != three_center_elements) {
    throw std::invalid_argument("DF value integral dimensions are inconsistent");
  }
  require_finite(data.metric, "DF metric entries must be finite");
  require_finite(data.three_center, "DF three-center entries must be finite");
}

void accumulate_coulomb_reverse_weights(const integrals::DensityFittingIntegralData& data,
                                        const std::vector<double>& density,
                                        const std::vector<double>& inverse, double scale,
                                        std::vector<double>& inverse_weights,
                                        std::vector<double>& three_center_weights) {
  if (scale == 0.0) return;
  const std::size_t nbf = data.nbf;
  const std::size_t naux = data.naux;
  std::vector<double> charge(naux, 0.0);
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      const double density_value = density[index(mu, nu, nbf)];
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        charge[auxiliary] +=
            density_value * data.three_center[three_center_index(mu, nu, auxiliary, nbf, naux)];
      }
    }
  }
  std::vector<double> potential(naux, 0.0);
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      potential[row] += inverse[index(row, column, naux)] * charge[column];
    }
  }
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      const double density_value = scale * density[index(mu, nu, nbf)];
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        three_center_weights[three_center_index(mu, nu, auxiliary, nbf, naux)] +=
            density_value * potential[auxiliary];
      }
    }
  }
  for (std::size_t row = 0; row < naux; ++row) {
    for (std::size_t column = 0; column < naux; ++column) {
      inverse_weights[index(row, column, naux)] += 0.5 * scale * charge[row] * charge[column];
    }
  }
}

void accumulate_exchange_reverse_weights(const integrals::DensityFittingIntegralData& data,
                                         const std::vector<double>& density,
                                         const std::vector<double>& inverse, double scale,
                                         std::vector<double>& inverse_weights,
                                         std::vector<double>& three_center_weights) {
  if (scale == 0.0) return;
  const std::size_t nbf = data.nbf;
  const std::size_t naux = data.naux;
  const std::size_t matrix_elements = nbf * nbf;
  const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::automatic,
                                   tensor::CpuLinalgThreadOwnership::provider_parallel, 1};

  std::vector<double> b_aux(naux * matrix_elements);
  for (std::size_t row = 0; row < nbf; ++row) {
    for (std::size_t column = 0; column < nbf; ++column) {
      const std::size_t pair = index(row, column, nbf);
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        b_aux[auxiliary * matrix_elements + pair] =
            data.three_center[three_center_index(row, column, auxiliary, nbf, naux)];
      }
    }
  }

  std::vector<double> response(naux * matrix_elements);
  std::vector<double> transformed(matrix_elements);
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    const double* bq = b_aux.data() + auxiliary * matrix_elements;
    double* rq = response.data() + auxiliary * matrix_elements;
    tensor::cpu_gemm('N', 'N', nbf, nbf, nbf, bq, density.data(), transformed.data(), 1.0, 0.0,
                     plan);
    tensor::cpu_gemm('T', 'N', nbf, nbf, nbf, density.data(), transformed.data(), rq, 1.0, 0.0,
                     plan);
  }

  std::vector<double> quadratic(naux * naux);
  tensor::cpu_gemm('N', 'T', naux, naux, matrix_elements, response.data(), b_aux.data(),
                   quadratic.data(), 1.0, 0.0, plan);
  for (std::size_t item = 0; item < quadratic.size(); ++item)
    inverse_weights[item] += scale * quadratic[item];

  std::vector<double> bar_b_aux(naux * matrix_elements);
  tensor::cpu_gemm('T', 'N', naux, matrix_elements, naux, inverse.data(), response.data(),
                   bar_b_aux.data(), scale, 0.0, plan);

  std::vector<double> mixed(naux * matrix_elements);
  tensor::cpu_gemm('N', 'N', naux, matrix_elements, naux, inverse.data(), b_aux.data(),
                   mixed.data(), 1.0, 0.0, plan);
  std::vector<double> left(matrix_elements);
  std::vector<double> indirect(matrix_elements);
  for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
    const double* xq = mixed.data() + auxiliary * matrix_elements;
    tensor::cpu_gemm('N', 'N', nbf, nbf, nbf, density.data(), xq, left.data(), 1.0, 0.0, plan);
    tensor::cpu_gemm('N', 'T', nbf, nbf, nbf, left.data(), density.data(), indirect.data(), 1.0,
                     0.0, plan);
    double* target = bar_b_aux.data() + auxiliary * matrix_elements;
    for (std::size_t item = 0; item < matrix_elements; ++item)
      target[item] += scale * indirect[item];
  }

  for (std::size_t row = 0; row < nbf; ++row) {
    for (std::size_t column = 0; column < nbf; ++column) {
      const std::size_t pair = index(row, column, nbf);
      for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
        three_center_weights[three_center_index(row, column, auxiliary, nbf, naux)] +=
            bar_b_aux[auxiliary * matrix_elements + pair];
      }
    }
  }
}

DensityFittingReverseWeights density_fitting_reverse_weights(
    const integrals::DensityFittingIntegralData& data,
    const std::vector<std::pair<const std::vector<double>*, JkCoefficients>>& terms,
    double relative_threshold) {
  validate_density_fitting_value_data(data);
  const std::vector<double> inverse = metric_pseudoinverse(data, relative_threshold);
  DensityFittingReverseWeights weights;
  std::vector<double> inverse_weights(data.naux * data.naux, 0.0);
  weights.three_center.assign(data.nbf * data.nbf * data.naux, 0.0);
  for (const auto& [density, coefficients] : terms) {
    if (!density || !std::isfinite(coefficients.coulomb) || !std::isfinite(coefficients.exchange))
      throw std::invalid_argument("DF weighted gradient coefficients are invalid");
    validate_gradient_density(*density, data.nbf, "DF weighted gradient density is inconsistent");
    accumulate_coulomb_reverse_weights(data, *density, inverse, coefficients.coulomb,
                                       inverse_weights, weights.three_center);
    accumulate_exchange_reverse_weights(data, *density, inverse, 0.5 * coefficients.exchange,
                                        inverse_weights, weights.three_center);
  }
  weights.metric = density_fitting_metric_inverse_response(data.metric, inverse, inverse_weights,
                                                           data.naux, relative_threshold);
  return weights;
}
std::size_t workspace_bytes(std::size_t ao_pair_tile, std::size_t auxiliary_tile,
                            std::size_t batch_size, std::size_t nbf, std::size_t naux,
                            std::size_t metric_bytes, std::size_t fixed_device_bytes,
                            bool generated_source, bool occupied_exchange,
                            bool retain_three_center = false) {
  // The CUDA plan keeps seven AO matrices and one auxiliary vector for the
  // complete batch.  Its streamed tile rounds the logical AO-pair budget up
  // to a whole row, so account for that physical capacity rather than the
  // planner's logical pair count. Setup/factorization storage is also charged
  // for every batch metric. This makes the planner's byte budget a conservative
  // bound on the actual device allocation, not just on one contraction tile.
  const long double matrix_elements =
      static_cast<long double>(batch_size) * static_cast<long double>(nbf) * nbf;
  const long double tensor_elements = matrix_elements * naux;
  const long double staged_rows = std::min<std::size_t>(
      nbf, std::max<std::size_t>(1, ao_pair_tile / std::max<std::size_t>(1, nbf)));
  const long double staged_pairs = staged_rows * nbf;
  const long double tile_elements = staged_pairs * auxiliary_tile;
  const bool resident =
      retain_three_center || (auxiliary_tile == naux && ao_pair_tile == nbf * nbf);
  const long double setup_doubles =
      static_cast<long double>(batch_size) * (3.0L * naux * naux + 2.0L * naux);
  // Metric setup and compact SCF use different solvers and dimensions. Both
  // have a fixed workspace floor; a quadratic-only metric estimate can admit
  // a tile that leaves no room for the actual small-system provider query.
  // Charge both owners conservatively, matching the native diagnostic, and
  // reject provider queries that exceed these shape-only allowances.
  const long double solver_workspace_bytes =
      static_cast<long double>(df_eigen_workspace_allowance(naux)) +
      df_scf_workspace_allowance(nbf, batch_size);
  const long double contraction_doubles =
      7.0L * matrix_elements + static_cast<long double>(batch_size) * naux + 3.0L * tile_elements +
      (resident ? 0.0L : tile_elements) +
      (resident ? tensor_elements * (generated_source ? 1.0L : 2.0L) : 0.0L);
  // Generated response staging has its own budget in the finalizer; this
  // planner covers value/SCF storage and reserves no retired coordinate scratch.
  // One-electron/Pulay assembly and the lazy device SCF driver retain up to
  // twenty AO matrices and one graph reservation per active batch item.
  // Only occupied mode adds two factor matrices and their generation controls,
  // preserving the dense planner's original minimum and residency thresholds.
  // These match the native plan's conservative RHF/UHF lazy-state allowance. Include the small
  // convergence/occupation vectors and metric status too; a positive budget cannot be spent
  // entirely before SCF state is allocated.
  const long double one_electron_doubles = (occupied_exchange ? 23.0L : 21.0L) * matrix_elements;
  const long double control_bytes =
      static_cast<long double>(batch_size) *
      (16 * sizeof(double) + 2 * sizeof(std::int32_t) + 2 * sizeof(std::uint8_t) +
       sizeof(std::uint32_t) + sizeof(int) +
       (occupied_exchange ? 2 * sizeof(std::uint32_t) + sizeof(int) : 0));
  const long double bytes =
      static_cast<long double>(fixed_device_bytes) + control_bytes +
      static_cast<long double>(df_eigen_device_reservation(nbf)) +
      static_cast<long double>(df_final_snapshot_device_reservation(nbf, batch_size) +
                               df_final_validation_device_reservation(nbf)) +
      static_cast<long double>(metric_bytes) * batch_size + solver_workspace_bytes +
      (setup_doubles + contraction_doubles + one_electron_doubles) * sizeof(double);
  if (bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(bytes);
}

}  // namespace

bool cpu_materialized_df_derivatives_requested() noexcept {
  const char* value = std::getenv("VIBEQC_CPU_DF_MATERIALIZED_DERIVATIVES");
  return value && value[0] == '1' && value[1] == '\0';
}

std::vector<double> density_fitting_metric_pseudoinverse(
    const integrals::DensityFittingIntegralData& integrals, double relative_threshold) {
  validate_density_fitting_derivative_data(integrals);
  return metric_pseudoinverse(integrals, relative_threshold);
}

std::vector<double> density_fitting_metric_pseudoinverse_derivative(
    const integrals::DensityFittingIntegralData& integrals, const std::vector<double>& inverse,
    std::size_t coordinate, double relative_threshold) {
  validate_density_fitting_derivative_data(integrals);
  if (inverse.size() != integrals.naux * integrals.naux) {
    throw std::invalid_argument("DF metric pseudoinverse dimensions are inconsistent");
  }
  if (coordinate >= integrals.ncoord) {
    throw std::invalid_argument("DF metric derivative coordinate is invalid");
  }
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  return metric_pseudoinverse_derivative(
      integrals, inverse, integrals.metric_derivative.data() + coordinate * metric_elements,
      relative_threshold);
}

std::vector<double> density_fitting_metric_inverse_response(const std::vector<double>& metric,
                                                            const std::vector<double>& inverse,
                                                            const std::vector<double>& response,
                                                            std::size_t n,
                                                            double relative_threshold) {
  return integrals::density_fitting_metric_response(metric, inverse, response, n,
                                                    relative_threshold,
                                                    tensor::SymmetricMatrixFunction::pseudoinverse);
}

std::vector<double> density_fitting_metric_inverse_square_root_response(
    const std::vector<double>& metric, const std::vector<double>& inverse_square_root,
    const std::vector<double>& response, std::size_t n, double relative_threshold) {
  return integrals::density_fitting_metric_response(metric, inverse_square_root, response, n,
                                                    relative_threshold,
                                                    tensor::SymmetricMatrixFunction::inverse_sqrt);
}

DensityFittingMetricFactor factor_density_fitting_metric(const std::vector<double>& metric,
                                                         std::size_t dimension,
                                                         double relative_threshold) {
  return integrals::factor_density_fitting_metric(metric, dimension, relative_threshold);
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
      metric_factor.effective_rank == 0 || metric_factor.effective_rank > naux) {
    throw std::invalid_argument("DF three-center tensor and metric factor are inconsistent");
  }
  require_finite(three_center, "DF three-center entries must be finite");
  require_finite(metric_factor.inverse_square_root, "DF metric factor entries must be finite");

  DensityFittingThreeCenter result{
      nbf,
      naux,
      metric_factor.effective_rank,
      std::vector<double>(tensor_elements, 0.0),
  };
  constexpr std::size_t kDenseTransformMinimum = 16;
  if (nbf >= kDenseTransformMinimum && tensor::cpu_openblas_built()) {
    const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::automatic,
                                     tensor::CpuLinalgThreadOwnership::provider_parallel, 1};
    tensor::cpu_gemm('N', 'N', nbf * nbf, naux, naux, three_center.data(),
                     metric_factor.inverse_square_root.data(), result.values.data(), 1.0, 0.0,
                     plan);
  } else {
    for (std::size_t mu = 0; mu < nbf; ++mu) {
      for (std::size_t nu = 0; nu < nbf; ++nu) {
        for (std::size_t target = 0; target < naux; ++target) {
          double value = 0.0;
          for (std::size_t source = 0; source < naux; ++source) {
            value += three_center[three_center_index(mu, nu, source, nbf, naux)] *
                     metric_factor.inverse_square_root[index(source, target, naux)];
          }
          result.values[three_center_index(mu, nu, target, nbf, naux)] = value;
        }
      }
    }
  }

  constexpr std::size_t kPersistentDenseExchangeMinimum = 16;
  if (nbf >= kPersistentDenseExchangeMinimum && tensor::cpu_openblas_built()) {
    result.auxiliary_major_values.resize(tensor_elements);
    for (std::size_t mu = 0; mu < nbf; ++mu)
      for (std::size_t nu = 0; nu < nbf; ++nu)
        for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary)
          result.auxiliary_major_values[auxiliary * nbf * nbf + index(mu, nu, nbf)] =
              result.values[three_center_index(mu, nu, auxiliary, nbf, naux)];
  }
  return result;
}

DensityFittingRhfJk build_density_fitting_rhf_jk(const DensityFittingThreeCenter& three_center,
                                                 const std::vector<double>& density,
                                                 JkTermSelection terms) {
  validate_three_center(three_center);
  const std::size_t matrix_elements =
      checked_matrix_elements(three_center.nbf, "DF orbital dimension is invalid");
  validate_density(density, matrix_elements);
  return {
      three_center.nbf,
      terms.coulomb ? build_coulomb(three_center, density) : std::vector<double>{},
      terms.exchange ? build_exchange(three_center, density) : std::vector<double>{},
  };
}

DensityFittingUhfJk build_density_fitting_uhf_jk(const DensityFittingThreeCenter& three_center,
                                                 const std::vector<double>& alpha_density,
                                                 const std::vector<double>& beta_density,
                                                 JkTermSelection terms) {
  validate_three_center(three_center);
  const std::size_t matrix_elements =
      checked_matrix_elements(three_center.nbf, "DF orbital dimension is invalid");
  validate_density(alpha_density, matrix_elements);
  validate_density(beta_density, matrix_elements);
  std::vector<double> total_density;
  if (terms.coulomb) {
    total_density.resize(matrix_elements);
    for (std::size_t element = 0; element < matrix_elements; ++element)
      total_density[element] = alpha_density[element] + beta_density[element];
  }
  return {
      three_center.nbf,
      terms.coulomb ? build_coulomb(three_center, total_density) : std::vector<double>{},
      terms.exchange ? build_exchange(three_center, alpha_density) : std::vector<double>{},
      terms.exchange ? build_exchange(three_center, beta_density) : std::vector<double>{},
  };
}

DensityFittingRhfGradient build_density_fitting_rhf_gradient(
    const integrals::DensityFittingIntegralData& integrals, const std::vector<double>& density,
    double relative_threshold, JkCoefficients coefficients) {
  if (!std::isfinite(coefficients.coulomb) || !std::isfinite(coefficients.exchange))
    throw std::invalid_argument("DF gradient coefficients must be finite");
  validate_density_fitting_derivative_data(integrals);
  validate_gradient_density(density, integrals.nbf, "DF RHF gradient density is inconsistent");
  const std::vector<double> inverse = metric_pseudoinverse(integrals, relative_threshold);
  DensityFittingRhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative.assign(integrals.ncoord, 0.0);
  result.forces.assign(integrals.ncoord, 0.0);
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  const std::size_t three_center_elements = integrals.nbf * integrals.nbf * integrals.naux;
  for (std::size_t coordinate = 0; coordinate < integrals.ncoord; ++coordinate) {
    const double* metric_derivative =
        integrals.metric_derivative.data() + coordinate * metric_elements;
    const double* three_center_derivative =
        integrals.three_center_derivative.data() + coordinate * three_center_elements;
    const std::vector<double> inverse_derivative =
        metric_pseudoinverse_derivative(integrals, inverse, metric_derivative, relative_threshold);
    const double coulomb =
        coefficients.coulomb == 0.0
            ? 0.0
            : coulomb_quadratic_derivative(integrals, density, inverse, inverse_derivative,
                                           three_center_derivative);
    const double exchange =
        coefficients.exchange == 0.0
            ? 0.0
            : exchange_quadratic_derivative(integrals, density, inverse, inverse_derivative,
                                            three_center_derivative);
    // `coulomb` already differentiates 1/2 (P|P)DF, while `exchange`
    // differentiates the unweighted exchange quadratic. The signed Fock
    // exchange coefficient therefore needs the extra energy factor 1/2.
    result.derivative[coordinate] =
        coefficients.coulomb * coulomb + 0.5 * coefficients.exchange * exchange;
    result.forces[coordinate] = -result.derivative[coordinate];
  }
  return result;
}

DensityFittingUhfGradient build_density_fitting_uhf_gradient(
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& alpha_density, const std::vector<double>& beta_density,
    double relative_threshold, JkCoefficients coefficients) {
  if (!std::isfinite(coefficients.coulomb) || !std::isfinite(coefficients.exchange))
    throw std::invalid_argument("DF gradient coefficients must be finite");
  validate_density_fitting_derivative_data(integrals);
  validate_gradient_density(alpha_density, integrals.nbf,
                            "DF UHF alpha gradient density is inconsistent");
  validate_gradient_density(beta_density, integrals.nbf,
                            "DF UHF beta gradient density is inconsistent");
  const std::vector<double> inverse = metric_pseudoinverse(integrals, relative_threshold);
  std::vector<double> total_density(alpha_density.size(), 0.0);
  for (std::size_t item = 0; item < total_density.size(); ++item) {
    total_density[item] = alpha_density[item] + beta_density[item];
  }

  DensityFittingUhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative.assign(integrals.ncoord, 0.0);
  result.forces.assign(integrals.ncoord, 0.0);
  const std::size_t metric_elements = integrals.naux * integrals.naux;
  const std::size_t three_center_elements = integrals.nbf * integrals.nbf * integrals.naux;
  for (std::size_t coordinate = 0; coordinate < integrals.ncoord; ++coordinate) {
    const double* metric_derivative =
        integrals.metric_derivative.data() + coordinate * metric_elements;
    const double* three_center_derivative =
        integrals.three_center_derivative.data() + coordinate * three_center_elements;
    const std::vector<double> inverse_derivative =
        metric_pseudoinverse_derivative(integrals, inverse, metric_derivative, relative_threshold);
    const double coulomb =
        coefficients.coulomb == 0.0
            ? 0.0
            : coulomb_quadratic_derivative(integrals, total_density, inverse, inverse_derivative,
                                           three_center_derivative);
    const double alpha_exchange =
        coefficients.exchange == 0.0
            ? 0.0
            : exchange_quadratic_derivative(integrals, alpha_density, inverse, inverse_derivative,
                                            three_center_derivative);
    const double beta_exchange =
        coefficients.exchange == 0.0
            ? 0.0
            : exchange_quadratic_derivative(integrals, beta_density, inverse, inverse_derivative,
                                            three_center_derivative);
    // Keep the established spin-by-spin summation order for standard UHF.
    result.derivative[coordinate] = coefficients.coulomb * coulomb +
                                    0.5 * coefficients.exchange * alpha_exchange +
                                    0.5 * coefficients.exchange * beta_exchange;
    result.forces[coordinate] = -result.derivative[coordinate];
  }
  return result;
}

DensityFittingRhfGradient build_density_fitting_rhf_weighted_gradient(
    const core::System& orbital_system, const core::System& auxiliary_system,
    const integrals::DensityFittingIntegralData& integrals, const std::vector<double>& density,
    double relative_threshold, JkCoefficients coefficients) {
  if (integrals.ncoord != orbital_system.atoms.size() * 3 ||
      integrals.nbf != molecule::ao_count(orbital_system) ||
      integrals.naux != molecule::ao_count(auxiliary_system)) {
    throw std::invalid_argument("DF weighted RHF gradient geometry dimensions are inconsistent");
  }
  const DensityFittingReverseWeights weights =
      density_fitting_reverse_weights(integrals, {{&density, coefficients}}, relative_threshold);
  DensityFittingRhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative = integrals::contract_weighted_density_fitting_derivative(
      orbital_system, auxiliary_system, weights.metric, weights.three_center);
  result.forces.resize(result.derivative.size());
  for (std::size_t coordinate = 0; coordinate < result.derivative.size(); ++coordinate)
    result.forces[coordinate] = -result.derivative[coordinate];
  return result;
}

DensityFittingUhfGradient build_density_fitting_uhf_weighted_gradient(
    const core::System& orbital_system, const core::System& auxiliary_system,
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& alpha_density, const std::vector<double>& beta_density,
    double relative_threshold, JkCoefficients coefficients) {
  if (integrals.ncoord != orbital_system.atoms.size() * 3 ||
      integrals.nbf != molecule::ao_count(orbital_system) ||
      integrals.naux != molecule::ao_count(auxiliary_system)) {
    throw std::invalid_argument("DF weighted UHF gradient geometry dimensions are inconsistent");
  }
  validate_gradient_density(alpha_density, integrals.nbf,
                            "DF UHF alpha weighted gradient density is inconsistent");
  validate_gradient_density(beta_density, integrals.nbf,
                            "DF UHF beta weighted gradient density is inconsistent");
  std::vector<double> total_density(alpha_density.size());
  for (std::size_t item = 0; item < total_density.size(); ++item)
    total_density[item] = alpha_density[item] + beta_density[item];
  const DensityFittingReverseWeights weights =
      density_fitting_reverse_weights(integrals,
                                      {{&total_density, {coefficients.coulomb, 0.0}},
                                       {&alpha_density, {0.0, coefficients.exchange}},
                                       {&beta_density, {0.0, coefficients.exchange}}},
                                      relative_threshold);
  DensityFittingUhfGradient result;
  result.ncoord = integrals.ncoord;
  result.derivative = integrals::contract_weighted_density_fitting_derivative(
      orbital_system, auxiliary_system, weights.metric, weights.three_center);
  result.forces.resize(result.derivative.size());
  for (std::size_t coordinate = 0; coordinate < result.derivative.size(); ++coordinate)
    result.forces[coordinate] = -result.derivative[coordinate];
  return result;
}
void validate_one_electron_force_data(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting) {
  if (one_electron.nbf != density_fitting.nbf || one_electron.ncoord != density_fitting.ncoord) {
    throw std::invalid_argument("one-electron and DF force dimensions are inconsistent");
  }
  std::size_t matrix_elements = 0;
  if (!checked_multiply(one_electron.nbf, one_electron.nbf, matrix_elements)) {
    throw std::invalid_argument("one-electron force dimensions are invalid");
  }
  std::size_t derivative_elements = 0;
  if (!checked_multiply(one_electron.ncoord, matrix_elements, derivative_elements) ||
      one_electron.overlap_derivative.size() != derivative_elements ||
      one_electron.hcore_derivative.size() != derivative_elements ||
      one_electron.nuclear_repulsion_derivative.size() != one_electron.ncoord) {
    throw std::invalid_argument("one-electron derivative data dimensions are inconsistent");
  }
  require_finite(one_electron.overlap_derivative, "overlap derivative entries must be finite");
  require_finite(one_electron.hcore_derivative,
                 "core-Hamiltonian derivative entries must be finite");
  require_finite(one_electron.nuclear_repulsion_derivative,
                 "nuclear-repulsion derivative entries must be finite");
}

std::vector<double> build_density_fitting_rhf_forces(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& density, const std::vector<double>& weighted_density,
    double relative_threshold) {
  validate_one_electron_force_data(one_electron, density_fitting);
  validate_gradient_density(density, density_fitting.nbf, "DF RHF force density is inconsistent");
  validate_gradient_density(weighted_density, density_fitting.nbf,
                            "DF RHF weighted density is inconsistent");
  const DensityFittingRhfGradient two_electron =
      build_density_fitting_rhf_gradient(density_fitting, density, relative_threshold);
  const std::size_t matrix_elements = density_fitting.nbf * density_fitting.nbf;
  std::vector<double> forces(density_fitting.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < density_fitting.ncoord; ++coordinate) {
    const double* overlap_derivative =
        one_electron.overlap_derivative.data() + coordinate * matrix_elements;
    const double* hcore_derivative =
        one_electron.hcore_derivative.data() + coordinate * matrix_elements;
    double derivative =
        two_electron.derivative[coordinate] + one_electron.nuclear_repulsion_derivative[coordinate];
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
    const std::vector<double>& alpha_density, const std::vector<double>& beta_density,
    const std::vector<double>& alpha_weighted_density,
    const std::vector<double>& beta_weighted_density, double relative_threshold) {
  validate_one_electron_force_data(one_electron, density_fitting);
  validate_gradient_density(alpha_density, density_fitting.nbf,
                            "DF UHF alpha force density is inconsistent");
  validate_gradient_density(beta_density, density_fitting.nbf,
                            "DF UHF beta force density is inconsistent");
  validate_gradient_density(alpha_weighted_density, density_fitting.nbf,
                            "DF UHF alpha weighted density is inconsistent");
  validate_gradient_density(beta_weighted_density, density_fitting.nbf,
                            "DF UHF beta weighted density is inconsistent");
  const DensityFittingUhfGradient two_electron = build_density_fitting_uhf_gradient(
      density_fitting, alpha_density, beta_density, relative_threshold);
  const std::size_t matrix_elements = density_fitting.nbf * density_fitting.nbf;
  std::vector<double> forces(density_fitting.ncoord, 0.0);
  for (std::size_t coordinate = 0; coordinate < density_fitting.ncoord; ++coordinate) {
    const double* overlap_derivative =
        one_electron.overlap_derivative.data() + coordinate * matrix_elements;
    const double* hcore_derivative =
        one_electron.hcore_derivative.data() + coordinate * matrix_elements;
    double derivative =
        two_electron.derivative[coordinate] + one_electron.nuclear_repulsion_derivative[coordinate];
    for (std::size_t item = 0; item < matrix_elements; ++item) {
      const double total_density = alpha_density[item] + beta_density[item];
      const double total_weighted = alpha_weighted_density[item] + beta_weighted_density[item];
      derivative +=
          total_density * hcore_derivative[item] - total_weighted * overlap_derivative[item];
    }
    forces[coordinate] = -derivative;
  }
  return forces;
}

static DensityFittingTilePlan plan_density_fitting_tiles_impl(
    std::size_t batch_size, std::size_t nbf, std::size_t naux, std::size_t occupied,
    std::size_t memory_budget_bytes, std::size_t fixed_device_bytes, bool generated_source,
    bool occupied_exchange) {
  if (batch_size == 0 || nbf == 0 || naux == 0 || occupied == 0) {
    throw std::invalid_argument("DF planner dimensions must all be positive");
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
        plan.ao_pair_tile, plan.auxiliary_tile, batch_size, nbf, naux, metric_bytes,
        fixed_device_bytes, generated_source, occupied_exchange, plan.stores_full_three_center);
  };
  // Full transformed storage is a latency policy only when its entire
  // contraction/setup/SCF allowance fits. Zero keeps the compatibility policy.
  if (memory_budget_bytes != 0) {
    plan.ao_pair_tile = ao_pair_count;
    // Generated B retention needs one full tensor plus three K panels. The
    // ordinary dense path caps interchangeable Q scratch at 128, but a
    // method-authorized occupied-RHF plan needs the complete Q extent so that
    // both SCF K and the exact raw-response owner can be reused. If that full
    // layout does not fit, the generated branch below finds the largest
    // bounded resident panel and the automatic wrapper drops the optional
    // factor reservation.
    plan.auxiliary_tile = generated_source && occupied_exchange ? naux
                          : generated_source                    ? std::min<std::size_t>(naux, 128)
                                                                : naux;
    plan.stores_full_three_center = generated_source;
    update_bytes();
    if (plan.peak_workspace_bytes <= memory_budget_bytes) {
      plan.stores_full_three_center = true;
      return plan;
    }
    if (generated_source) {
      const auto retained_minimum =
          workspace_bytes(ao_pair_count, 1, batch_size, nbf, naux, metric_bytes, fixed_device_bytes,
                          true, occupied_exchange, true);
      if (retained_minimum <= memory_budget_bytes) {
        // All persistent/setup/SCF/source reservations are already charged.
        // Find the largest bounded resident Q that fits this same live set.
        std::size_t lower = 1, upper = plan.auxiliary_tile;
        while (lower < upper) {
          const auto middle = lower + (upper - lower + 1) / 2;
          const auto bytes =
              workspace_bytes(ao_pair_count, middle, batch_size, nbf, naux, metric_bytes,
                              fixed_device_bytes, true, occupied_exchange, true);
          if (bytes <= memory_budget_bytes)
            lower = middle;
          else
            upper = middle - 1;
        }
        plan.auxiliary_tile = lower;
        update_bytes();
        return plan;
      }
      plan.stores_full_three_center = false;
      // A streamed source owns four equally sized tile buffers. Everything
      // else in workspace_bytes is independent of their shape. Spend only the
      // remaining allowance on whole row/Q units, then minimize actual raw
      // regeneration instead of falling back to the fixed 8192/128 plateau.
      const auto minimum = workspace_bytes(nbf, 1, batch_size, nbf, naux, metric_bytes,
                                           fixed_device_bytes, true, occupied_exchange);
      if (minimum > memory_budget_bytes || (nbf == 1 && naux == 1))
        throw DensityFittingBudgetError();
      std::size_t all_row_auxiliaries = 0;
      if (!checked_multiply(nbf, naux, all_row_auxiliaries))
        throw std::overflow_error("DF panel dimensions overflow size_t");
      const auto row_auxiliaries =
          std::min(all_row_auxiliaries - 1,
                   1 + (memory_budget_bytes - minimum) / (4 * sizeof(double) * nbf));
      const auto panel = df_streamed_k_panel(nbf, naux, row_auxiliaries * nbf);
      plan.ao_pair_tile = panel.rows * nbf;
      plan.auxiliary_tile = panel.output_auxiliaries;
      update_bytes();
      if (!panel.rows || plan.peak_workspace_bytes > memory_budget_bytes)
        throw DensityFittingBudgetError();
      return plan;
    }
    plan.ao_pair_tile = std::min<std::size_t>(ao_pair_count, 8192);
    plan.auxiliary_tile = std::min<std::size_t>(naux, 128);
  }
  update_bytes();
  // A zero budget is the documented sentinel for the implementation's
  // default tile policy; only a positive budget requests shrinking.
  while (memory_budget_bytes != 0 && plan.peak_workspace_bytes > memory_budget_bytes) {
    const long double pair_cost = static_cast<long double>(plan.ao_pair_tile) * plan.auxiliary_tile;
    const long double occupied_cost =
        static_cast<long double>(plan.occupied_tile) * nbf * plan.auxiliary_tile;
    if (plan.ao_pair_tile > 1 && pair_cost >= occupied_cost) {
      plan.ao_pair_tile = (plan.ao_pair_tile + 1) / 2;
    } else if (plan.occupied_tile > 1) {
      plan.occupied_tile = (plan.occupied_tile + 1) / 2;
    } else if (plan.auxiliary_tile > 1) {
      plan.auxiliary_tile = (plan.auxiliary_tile + 1) / 2;
    } else if (plan.ao_pair_tile > 1) {
      plan.ao_pair_tile = (plan.ao_pair_tile + 1) / 2;
    } else {
      throw DensityFittingBudgetError();
    }
    update_bytes();
  }
  plan.stores_full_three_center = plan.batch_tile == batch_size &&
                                  plan.ao_pair_tile == ao_pair_count && plan.auxiliary_tile == naux;
  return plan;
}

/** Automatic factors are optional: retain dense residency when it fits, and
 * reserve streamed factors only when the compiler predicts less raw work. */
DensityFittingTilePlan plan_density_fitting_tiles(std::size_t batch, std::size_t nbf,
                                                  std::size_t naux, std::size_t occupied,
                                                  std::size_t budget, std::size_t fixed,
                                                  bool generated_source,
                                                  std::size_t automatic_rhf_rank) {
  const bool automatic = df_occupied_exchange_auto_requested() &&
                         df_occupied_exchange_requested(nbf, naux, batch, automatic_rhf_rank);
  if (automatic) {
    try {
      auto plan = plan_density_fitting_tiles_impl(batch, nbf, naux, occupied, budget, fixed,
                                                  generated_source, true);
      const auto ao_pair_count = nbf * nbf;
      // A retained B tensor with a bounded Q panel cannot consume the
      // automatic occupied owner. Keep the ordinary dense/source plan in that
      // case instead of charging SCF factor state that execution cannot use.
      if (plan.stores_full_three_center && plan.ao_pair_tile == ao_pair_count &&
          plan.auxiliary_tile == naux) {
        plan.automatic_rhf_rank = automatic_rhf_rank;
        return plan;
      }
      if (generated_source && !plan.stores_full_three_center) {
        const auto dense = plan_density_fitting_tiles_impl(batch, nbf, naux, occupied, budget,
                                                           fixed, generated_source, false);
        // Optional factor storage must not turn a retained dense tensor into
        // regeneration or increase the fallback's source passes when a mixed
        // seed cannot be factored exactly.
        const auto capacity = (plan.ao_pair_tile / nbf) * nbf * plan.auxiliary_tile;
        const auto dense_capacity = (dense.ao_pair_tile / nbf) * nbf * dense.auxiliary_tile;
        const auto fallback = df_streamed_k_panel(nbf, naux, capacity);
        const auto original = df_streamed_k_panel(nbf, naux, dense_capacity);
        if (!dense.stores_full_three_center &&
            static_cast<long double>(fallback.row_tiles) * fallback.output_tiles <=
                static_cast<long double>(original.row_tiles) * original.output_tiles &&
            df_projected_exchange_schedule(nbf, naux, automatic_rhf_rank, capacity,
                                           df_triangular_exchange_requested())
                .rows) {
          plan.automatic_rhf_rank = automatic_rhf_rank;
          return plan;
        }
      }
    } catch (const DensityFittingBudgetError&) {
      // Retry the original dense budget before reporting an infeasible job.
    }
  }
  return plan_density_fitting_tiles_impl(batch, nbf, naux, occupied, budget, fixed,
                                         generated_source, df_occupied_exchange_requested());
}

static DensityFittingTilePlan plan_packed_density_fitting_tiles_impl(
    std::size_t batch, std::size_t n, std::size_t a, std::size_t rank, std::size_t budget,
    std::size_t fixed_device_bytes, bool occupied_exchange, bool retain_raw) {
  // Validate the actual number of simultaneous immutable owners before any
  // shape arithmetic; the native allocator must use the same representation.
  (void)df_packed_value_capacity(batch, n, a, rank, 1, retain_raw);
  std::size_t matrix{}, metric_elements{}, metric_bytes{};
  if (!checked_multiply(n, n, matrix) || !checked_multiply(a, a, metric_elements) ||
      !checked_multiply(metric_elements, sizeof(double), metric_bytes) ||
      matrix > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      a > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("packed DF dimensions exceed native indexing");
  const auto bytes = [&](std::size_t q) {
    const auto capacity = df_packed_value_capacity(batch, n, a, rank, q, retain_raw);
    const auto legacy = workspace_bytes(matrix, q, batch, n, a, metric_bytes, fixed_device_bytes,
                                        true, occupied_exchange, true);
    if (legacy == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("packed DF fixed reservation overflows size_t");
    // Keep the same metric, library, AO, lazy SCF and final-state reservations;
    // replace only the representation's actual simultaneous tensor allocations.
    const long double dense = static_cast<long double>(batch) * matrix * a * sizeof(double) +
                              3.0L * matrix * q * sizeof(double);
    const long double packed = (retain_raw ? 2.0L : 1.0L) * capacity.factor_bytes +
                               capacity.scratch_bytes;
    const long double total = static_cast<long double>(legacy) - dense + packed;
    if (total < 0 || total >= static_cast<long double>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("packed DF reservation overflows size_t");
    return static_cast<std::size_t>(total);
  };
  // Single-owner setup must fit one complete raw auxiliary pair before it
  // can whiten that pair into B; the retained raw owner has no such floor.
  const std::size_t minimum_q = retain_raw ? 1 : a / matrix + (a % matrix != 0);
  std::size_t q = std::max(minimum_q, std::min<std::size_t>(a, 128));
  if (budget && bytes(q) > budget) {
    if (bytes(minimum_q) > budget) throw DensityFittingBudgetError();
    std::size_t low = minimum_q, high = q;
    while (low < high) {
      const auto middle = low + (high - low + 1) / 2;
      if (bytes(middle) <= budget)
        low = middle;
      else
        high = middle - 1;
    }
    q = low;
  }
  return {batch,
          matrix,
          q,
          std::min<std::size_t>(rank, 32),
          bytes(q),
          true,
          {retain_raw ? DfPairStorage::SymmetricLower : DfPairStorage::SymmetricLowerSingle, rank}};
}

/** Packed U is independently requested storage; optional automatic SCF factors
 * may use it only if their additional charged capacity also fits. */
DensityFittingTilePlan plan_packed_density_fitting_tiles(std::size_t batch, std::size_t nbf,
                                                         std::size_t naux, std::size_t rank,
                                                         std::size_t budget, std::size_t fixed,
                                                         std::size_t automatic_rhf_rank,
                                                         bool retain_raw) {
  if (df_occupied_exchange_auto_requested() && automatic_rhf_rank <= rank &&
      df_occupied_exchange_requested(nbf, naux, batch, automatic_rhf_rank)) {
    try {
      auto plan =
          plan_packed_density_fitting_tiles_impl(batch, nbf, naux, rank, budget, fixed, true,
                                                 retain_raw);
      plan.automatic_rhf_rank = automatic_rhf_rank;
      return plan;
    } catch (const DensityFittingBudgetError&) {
      // Packing remains explicit; only the unused SCF reservation is dropped.
    }
  }
  try {
    return plan_packed_density_fitting_tiles_impl(batch, nbf, naux, rank, budget, fixed,
                                                  df_occupied_exchange_requested(), retain_raw);
  } catch (const DensityFittingBudgetError&) {
    if (retain_raw || !rank || !df_occupied_exchange_auto_requested()) throw;
    // A single fitted B may fit when a complete occupied U does not. Drop
    // only that optional SCF scratch: J/K remains exact via bounded panels.
    return plan_packed_density_fitting_tiles_impl(batch, nbf, naux, 0, budget, fixed, false,
                                                  false);
  }
}

std::size_t density_fitting_scf_diis_device_bytes(std::size_t batch, std::size_t nbf,
                                                  unsigned history) noexcept {
  if (history < 2) return 0;
  const long double dimension = static_cast<long double>(history) + 1;
  const long double matrices = (4.0L * history + 6) * static_cast<long double>(nbf) * nbf;
  const long double bytes =
      static_cast<long double>(batch) *
      ((matrices + dimension * dimension + dimension) * sizeof(double) + 2 * sizeof(std::uint32_t));
  return bytes >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(bytes);
}

std::size_t density_fitting_source_metadata_bytes(std::size_t batch, std::size_t atoms,
                                                  std::size_t shells, std::size_t cartesian_aos,
                                                  std::size_t primitives,
                                                  std::size_t transform_elements) {
  std::size_t bytes = 32;  // atom and three shell offset vectors' final entries
  const auto add = [&](std::size_t count, std::size_t width) {
    std::size_t product = 0;
    if (!checked_multiply(count, width, product) ||
        product > std::numeric_limits<std::size_t>::max() - bytes)
      throw std::overflow_error("DF source metadata overflows size_t");
    bytes += product;
  };
  add(batch, 8);
  add(atoms, 32);   // system/element indices and Cartesian positions
  add(shells, 29);  // center/angular values and three shell offset arrays
  add(cartesian_aos, 20 + 11 * molecule::kMaximumAoExpansionTerms);
  add(primitives, 16);
  // The v1 caller supplies the historical dense-transform element bound.
  // Keep that ABI conservative while also covering fixed-size sparse records
  // for very small Cartesian bases, where a record can exceed a dense row.
  std::size_t dense_bytes = 0, sparse_bytes = 0;
  if (!checked_multiply(transform_elements, 8, dense_bytes) ||
      !checked_multiply(
          cartesian_aos,
          sizeof(std::uint32_t) +
              molecule::kMaximumAoExpansionTerms * (sizeof(std::int32_t) + sizeof(double)) +
              alignof(double) - 1,
          sparse_bytes))
    throw std::overflow_error("DF transform metadata overflows size_t");
  add(std::max(dense_bytes, sparse_bytes), 1);
  return bytes;
}

}  // namespace vibeqc::scf
