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

std::size_t workspace_bytes(
    std::size_t batch_tile,
    std::size_t ao_pair_tile,
    std::size_t auxiliary_tile,
    std::size_t occupied_tile,
    std::size_t nbf,
    std::size_t metric_bytes) {
  // Three-center input, metric-transformed tile, and occupied-orbital
  // intermediates are simultaneously live in the conservative schedule.
  const long double doubles =
      2.0L * batch_tile * ao_pair_tile * auxiliary_tile +
      batch_tile * nbf * occupied_tile * auxiliary_tile;
  const long double bytes =
      static_cast<long double>(metric_bytes) + doubles * sizeof(double);
  if (bytes > static_cast<long double>(
                  std::numeric_limits<std::size_t>::max())) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(bytes);
}

}  // namespace

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
  // Divide one consecutive factor first so the triangular number can be
  // represented whenever its final value fits in size_t.
  std::size_t pair_factor = nbf;
  std::size_t consecutive_factor = nbf + 1;
  if ((pair_factor & 1U) == 0) {
    pair_factor /= 2;
  } else {
    consecutive_factor /= 2;
  }
  std::size_t ao_pair_count = 0;
  if (!checked_multiply(
          pair_factor, consecutive_factor, ao_pair_count)) {
    throw std::overflow_error("DF AO-pair count overflows size_t");
  }
  DensityFittingTilePlan plan{
      std::min<std::size_t>(batch_size, 4),
      std::min<std::size_t>(ao_pair_count, 8192),
      std::min<std::size_t>(naux, 128),
      std::min<std::size_t>(occupied, 32),
      0,
      false,
  };
  auto update_bytes = [&]() {
    plan.peak_workspace_bytes = workspace_bytes(
        plan.batch_tile, plan.ao_pair_tile, plan.auxiliary_tile,
        plan.occupied_tile, nbf, metric_bytes);
  };
  update_bytes();
  while (plan.peak_workspace_bytes > memory_budget_bytes) {
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
    } else if (plan.batch_tile > 1) {
      plan.batch_tile = (plan.batch_tile + 1) / 2;
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
