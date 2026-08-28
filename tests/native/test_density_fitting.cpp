#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/density_fitting.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": actual=" +
                             std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

std::size_t matrix_index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

std::size_t three_center_index(std::size_t i,
                               std::size_t j,
                               std::size_t auxiliary,
                               std::size_t n,
                               std::size_t naux) {
  return (i * n + j) * naux + auxiliary;
}

std::pair<std::vector<double>, std::vector<double>> reference_jk(
    const vibeqc::scf::DensityFittingThreeCenter& three_center,
    const std::vector<double>& density) {
  const std::size_t n = three_center.nbf;
  const std::size_t naux = three_center.naux;
  std::vector<double> coulomb(n * n, 0.0);
  std::vector<double> exchange(n * n, 0.0);
  // Deliberately reconstruct each required four-center RI integral. This
  // expensive formula is independent of the production-style matrix-product
  // ordering used by the implementation under test.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          double coulomb_integral = 0.0;
          double exchange_integral = 0.0;
          for (std::size_t auxiliary = 0; auxiliary < naux; ++auxiliary) {
            coulomb_integral +=
                three_center
                    .values[three_center_index(i, j, auxiliary, n, naux)] *
                three_center
                    .values[three_center_index(k, l, auxiliary, n, naux)];
            exchange_integral +=
                three_center
                    .values[three_center_index(i, k, auxiliary, n, naux)] *
                three_center
                    .values[three_center_index(j, l, auxiliary, n, naux)];
          }
          const double density_value = density[matrix_index(k, l, n)];
          coulomb[matrix_index(i, j, n)] += density_value * coulomb_integral;
          exchange[matrix_index(i, j, n)] += density_value * exchange_integral;
        }
      }
    }
  }
  return {std::move(coulomb), std::move(exchange)};
}

void require_matrix_close(const std::vector<double>& actual,
                          const std::vector<double>& expected, double tolerance,
                          const char* message) {
  require(actual.size() == expected.size(), message);
  for (std::size_t element = 0; element < actual.size(); ++element) {
    if (std::abs(actual[element] - expected[element]) > tolerance) {
      throw std::runtime_error(
          std::string(message) + " at element " + std::to_string(element) +
          ": actual=" + std::to_string(actual[element]) +
          " expected=" + std::to_string(expected[element]));
    }
  }
}

void append_values(std::vector<double>& destination,
                   const std::vector<double>& source) {
  destination.insert(destination.end(), source.begin(), source.end());
}

#if VIBEQC_HAS_CUDA
bool cuda_device_available() {
  // CUDA builds also run on login nodes. Probe through the public context so
  // real device tests execute only inside a scheduler-provided allocation.
  vibeqc_context_descriptor descriptor{
      sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
      VIBEQC_BACKEND_CUDA};
  vibeqc_context* context = nullptr;
  const vibeqc_status status = vibeqc_context_create(&descriptor, &context);
  if (context != nullptr) vibeqc_context_destroy(context);
  return status == VIBEQC_STATUS_SUCCESS;
}
#endif

vibeqc::core::System orbital_system() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.2, 1.0}}}, {0, 1, {{0.7, 1.0}}},
      {1, 0, {{1.2, 1.0}}}, {1, 1, {{0.7, 1.0}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "orbital basis normalization failed");
  return system;
}

vibeqc::core::System auxiliary_system() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{0.8, 1.0}}}, {0, 1, {{0.5, 1.0}}},
      {1, 0, {{0.8, 1.0}}}, {1, 1, {{0.5, 1.0}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "auxiliary basis normalization failed");
  return system;
}

vibeqc::core::System spherical_d_system(double exponent) {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 2, {{exponent, 1.0}}}};
  system.multiplicity = 2;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "spherical d basis normalization failed");
  return system;
}

}  // namespace

int main() {
  try {
    const vibeqc::core::System orbital = orbital_system();
    const vibeqc::core::System auxiliary = auxiliary_system();
    const vibeqc::integrals::DensityFittingIntegralData integrals =
        vibeqc::integrals::build_density_fitting_integrals(
            orbital, auxiliary);
    require(integrals.nbf == 8 && integrals.naux == 8 &&
                integrals.ncoord == 6,
            "DF integral dimensions are inconsistent");

    // Independent PySCF 2.14/libcint references use cart=True and the exact
    // one-primitive orbital/auxiliary bases constructed above.
    require_close(
        std::accumulate(
            integrals.three_center.begin(), integrals.three_center.end(), 0.0),
        83.84636438741627, 3.0e-11,
        "three-center checksum differs from libcint");
    require_close(
        std::accumulate(integrals.metric.begin(), integrals.metric.end(), 0.0),
        137.30042310245761, 3.0e-11,
        "Coulomb metric checksum differs from libcint");
    require_close(
        integrals.three_center[three_center_index(0, 0, 0, 8, 8)],
        4.100606497726132, 5.0e-13,
        "three-center sss value differs from libcint");
    require_close(
        integrals.three_center[three_center_index(3, 7, 4, 8, 8)],
        -0.794018075268033, 5.0e-13,
        "three-center Cartesian ordering differs from libcint");
    require_close(integrals.metric[matrix_index(0, 4, 8)],
                  12.412526419640978, 5.0e-13,
                  "two-center metric value differs from libcint");

    const auto spherical =
        vibeqc::integrals::build_density_fitting_integrals(
            spherical_d_system(0.9), spherical_d_system(0.55));
    require(spherical.nbf == 5 && spherical.naux == 5,
            "spherical d transforms produced incorrect dimensions");
    // These PySCF 2.14/libcint references exercise independent d-shell
    // transforms on both orbital indices and on the auxiliary index.
    require_close(
        std::accumulate(spherical.metric.begin(), spherical.metric.end(), 0.0),
        22.84794657156211, 2.0e-12,
        "spherical d metric checksum differs from libcint");
    require_close(
        std::accumulate(
            spherical.three_center.begin(), spherical.three_center.end(), 0.0),
        1.003026914171485, 2.0e-12,
        "spherical d three-center checksum differs from libcint");
    require_close(
        spherical.metric[matrix_index(0, 0, 5)],
        4.569589314312425, 8.0e-13,
        "spherical d metric ordering differs from libcint");
    require_close(
        spherical.three_center[three_center_index(0, 0, 2, 5, 5)],
        -0.3138232416719612, 8.0e-13,
        "spherical d three-center ordering differs from libcint");

    constexpr double displacement = 1.0e-5;
    vibeqc::core::System plus_orbital = orbital;
    vibeqc::core::System plus_auxiliary = auxiliary;
    vibeqc::core::System minus_orbital = orbital;
    vibeqc::core::System minus_auxiliary = auxiliary;
    plus_orbital.atoms[0].position[2] += displacement;
    plus_auxiliary.atoms[0].position[2] += displacement;
    minus_orbital.atoms[0].position[2] -= displacement;
    minus_auxiliary.atoms[0].position[2] -= displacement;
    const auto plus = vibeqc::integrals::build_density_fitting_integrals(
        plus_orbital, plus_auxiliary);
    const auto minus = vibeqc::integrals::build_density_fitting_integrals(
        minus_orbital, minus_auxiliary);
    const std::size_t three_item = three_center_index(0, 1, 4, 8, 8);
    const std::size_t metric_item = matrix_index(0, 4, 8);
    require_close(
        integrals.three_center_derivative[2 * integrals.three_center.size() +
                                          three_item],
        (plus.three_center[three_item] - minus.three_center[three_item]) /
            (2.0 * displacement),
        2.0e-9, "three-center analytic derivative failed finite differences");
    require_close(
        integrals.metric_derivative[2 * integrals.metric.size() + metric_item],
        (plus.metric[metric_item] - minus.metric[metric_item]) /
            (2.0 * displacement),
        2.0e-9, "metric analytic derivative failed finite differences");

    for (std::size_t axis = 0; axis < 3; ++axis) {
      require_close(
          integrals.metric_derivative[axis * integrals.metric.size() +
                                      metric_item] +
              integrals.metric_derivative[(axis + 3) *
                                               integrals.metric.size() +
                                           metric_item],
          0.0, 2.0e-12, "metric derivative violates translation invariance");
      require_close(
          integrals.three_center_derivative[
              axis * integrals.three_center.size() + three_item] +
              integrals.three_center_derivative[
                  (axis + 3) * integrals.three_center.size() + three_item],
          0.0, 2.0e-12,
          "three-center derivative violates translation invariance");
    }

    const vibeqc::scf::DensityFittingMetricFactor factor =
        vibeqc::scf::factor_density_fitting_metric(
            integrals.metric, integrals.naux, 1.0e-12);
    require(factor.effective_rank == 8 && factor.condition_number > 70.0 &&
                factor.condition_number < 80.0,
            "metric factorization reported incorrect conditioning");
    for (std::size_t row = 0; row < 8; ++row) {
      for (std::size_t column = 0; column < 8; ++column) {
        double value = 0.0;
        for (std::size_t p = 0; p < 8; ++p) {
          for (std::size_t q = 0; q < 8; ++q) {
            value += factor.inverse_square_root[matrix_index(row, p, 8)] *
                integrals.metric[matrix_index(p, q, 8)] *
                factor.inverse_square_root[matrix_index(q, column, 8)];
          }
        }
        require_close(value, row == column ? 1.0 : 0.0, 2.0e-11,
                      "metric inverse square root is inconsistent");
      }
    }
    const auto deficient = vibeqc::scf::factor_density_fitting_metric(
        {2.0, 2.0, 2.0, 2.0}, 2, 1.0e-10);
    require(deficient.effective_rank == 1 &&
                deficient.condition_number == 1.0,
            "metric threshold did not remove a dependent direction");

    const vibeqc::scf::DensityFittingThreeCenter orthonormal_three_center =
        vibeqc::scf::orthonormalize_density_fitting_three_center(
            integrals.three_center, integrals.nbf, factor);
    require(
        orthonormal_three_center.nbf == integrals.nbf &&
            orthonormal_three_center.naux == integrals.naux &&
            orthonormal_three_center.effective_rank == factor.effective_rank,
        "orthonormalized three-center dimensions are inconsistent");
    for (std::size_t i = 0; i < integrals.nbf; ++i) {
      for (std::size_t j = 0; j < integrals.nbf; ++j) {
        for (std::size_t auxiliary = 0; auxiliary < integrals.naux;
             ++auxiliary) {
          double expected = 0.0;
          for (std::size_t source = 0; source < integrals.naux; ++source) {
            expected += integrals.three_center[three_center_index(
                            i, j, source, integrals.nbf, integrals.naux)] *
                factor.inverse_square_root[
                    matrix_index(source, auxiliary, integrals.naux)];
          }
          require_close(orthonormal_three_center.values[three_center_index(
                            i, j, auxiliary, integrals.nbf, integrals.naux)],
                        expected, 2.0e-13,
                        "three-center metric transform is inconsistent");
          require_close(orthonormal_three_center.values[three_center_index(
                            i, j, auxiliary, integrals.nbf, integrals.naux)],
                        orthonormal_three_center.values[three_center_index(
                            j, i, auxiliary, integrals.nbf, integrals.naux)],
                        2.0e-13, "metric transform broke AO-pair symmetry");
        }
      }
    }

    std::vector<double> rhf_density(integrals.nbf * integrals.nbf, 0.0);
    std::vector<double> alpha_density(rhf_density.size(), 0.0);
    std::vector<double> beta_density(rhf_density.size(), 0.0);
    for (std::size_t i = 0; i < integrals.nbf; ++i) {
      const double first_i = 0.04 * static_cast<double>(i + 1);
      const double second_i =
          (i % 2 == 0 ? 0.03 : -0.02) * static_cast<double>(i + 1);
      const double beta_i =
          (i % 3 == 0 ? -0.025 : 0.015) * static_cast<double>(i + 2);
      for (std::size_t j = 0; j < integrals.nbf; ++j) {
        const double first_j = 0.04 * static_cast<double>(j + 1);
        const double second_j =
            (j % 2 == 0 ? 0.03 : -0.02) * static_cast<double>(j + 1);
        const double beta_j =
            (j % 3 == 0 ? -0.025 : 0.015) * static_cast<double>(j + 2);
        alpha_density[matrix_index(i, j, integrals.nbf)] =
            first_i * first_j + second_i * second_j;
        beta_density[matrix_index(i, j, integrals.nbf)] = beta_i * beta_j;
        rhf_density[matrix_index(i, j, integrals.nbf)] =
            2.0 * alpha_density[matrix_index(i, j, integrals.nbf)];
      }
    }
    // The CUDA exchange path deliberately composes row-major AO matrices
    // through column-major cuBLAS views. Exercise a non-symmetric input so a
    // missing layout transpose cannot hide behind the physical SCF symmetry.
    require(integrals.nbf >= 2,
            "DF layout regression requires at least two AO functions");
    rhf_density[matrix_index(0, 1, integrals.nbf)] += 0.017;
    rhf_density[matrix_index(1, 0, integrals.nbf)] -= 0.011;

    const vibeqc::scf::DensityFittingRhfJk rhf_jk =
        vibeqc::scf::build_density_fitting_rhf_jk(orthonormal_three_center,
                                                  rhf_density);
    const auto [reference_rhf_j, reference_rhf_k] =
        reference_jk(orthonormal_three_center, rhf_density);
    require(rhf_jk.nbf == integrals.nbf,
            "RHF RI-J/K reported the wrong AO dimension");
    require_matrix_close(rhf_jk.coulomb, reference_rhf_j, 2.0e-12,
                         "RHF RI-J contraction is inconsistent");
    require_matrix_close(rhf_jk.exchange, reference_rhf_k, 2.0e-12,
                         "RHF RI-K contraction is inconsistent");

    const vibeqc::scf::DensityFittingUhfJk uhf_jk =
        vibeqc::scf::build_density_fitting_uhf_jk(orthonormal_three_center,
                                                  alpha_density, beta_density);
    std::vector<double> total_density(alpha_density.size(), 0.0);
    for (std::size_t element = 0; element < total_density.size(); ++element) {
      total_density[element] = alpha_density[element] + beta_density[element];
    }
    const auto reference_total =
        reference_jk(orthonormal_three_center, total_density);
    const auto reference_alpha =
        reference_jk(orthonormal_three_center, alpha_density);
    const auto reference_beta =
        reference_jk(orthonormal_three_center, beta_density);
    require(uhf_jk.nbf == integrals.nbf,
            "UHF RI-J/K reported the wrong AO dimension");
    require_matrix_close(uhf_jk.coulomb, reference_total.first, 2.0e-12,
                         "UHF RI-J contraction is inconsistent");
    require_matrix_close(uhf_jk.alpha_exchange, reference_alpha.second,
                         2.0e-12,
                         "UHF alpha RI-K contraction is inconsistent");
    require_matrix_close(uhf_jk.beta_exchange, reference_beta.second, 2.0e-12,
                         "UHF beta RI-K contraction is inconsistent");

#if VIBEQC_HAS_CUDA
    {
      vibeqc::scf::CudaDensityFittingJkPlan* invalid_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          invalid_diagnostics;
      std::string invalid_detail;
      const std::size_t oversized_dimension =
          static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
      const vibeqc_status invalid_dimension_status =
          vibeqc::scf::create_cuda_density_fitting_jk_plan(
              0, 1, 1, oversized_dimension, {}, {}, 1.0e-12, 1,
              &invalid_plan, invalid_diagnostics, invalid_detail);
      require(invalid_dimension_status == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  invalid_plan == nullptr,
              "CUDA DF accepted an eigensolver dimension above its checked "
              "API range");
    }
    if (cuda_device_available()) {
      std::vector<double> second_metric = plus.metric;
      const std::size_t dependent = plus.naux - 1;
      for (std::size_t auxiliary = 0; auxiliary < plus.naux; ++auxiliary) {
        second_metric[matrix_index(auxiliary, dependent, plus.naux)] =
            plus.metric[matrix_index(auxiliary, 0, plus.naux)];
        second_metric[matrix_index(dependent, auxiliary, plus.naux)] =
            plus.metric[matrix_index(0, auxiliary, plus.naux)];
      }
      second_metric[matrix_index(dependent, dependent, plus.naux)] =
          plus.metric[matrix_index(0, 0, plus.naux)];
      const vibeqc::scf::DensityFittingMetricFactor second_factor =
          vibeqc::scf::factor_density_fitting_metric(
              second_metric, plus.naux, 1.0e-12);
      const vibeqc::scf::DensityFittingThreeCenter plus_three_center =
          vibeqc::scf::orthonormalize_density_fitting_three_center(
              plus.three_center, plus.nbf, second_factor);

      std::vector<double> batch_metrics;
      std::vector<double> batch_three_center;
      append_values(batch_metrics, integrals.metric);
      append_values(batch_metrics, second_metric);
      append_values(batch_three_center, integrals.three_center);
      append_values(batch_three_center, plus.three_center);

      std::vector<double> second_rhf_density = rhf_density;
      std::vector<double> second_alpha_density = alpha_density;
      std::vector<double> second_beta_density = beta_density;
      for (std::size_t element = 0; element < rhf_density.size(); ++element) {
        second_rhf_density[element] *= 0.73;
        second_alpha_density[element] =
            0.81 * alpha_density[element] + 0.09 * beta_density[element];
        second_beta_density[element] *= 0.64;
      }

      const auto second_rhf_jk = vibeqc::scf::build_density_fitting_rhf_jk(
          plus_three_center, second_rhf_density);
      const auto second_uhf_jk = vibeqc::scf::build_density_fitting_uhf_jk(
          plus_three_center, second_alpha_density, second_beta_density);
      std::vector<double> batch_rhf_density = rhf_density;
      std::vector<double> batch_alpha_density = alpha_density;
      std::vector<double> batch_beta_density = beta_density;
      append_values(batch_rhf_density, second_rhf_density);
      append_values(batch_alpha_density, second_alpha_density);
      append_values(batch_beta_density, second_beta_density);

      vibeqc::scf::CudaDensityFittingJkPlan* raw_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          diagnostics;
      std::string cuda_detail;
      const vibeqc_status create_status =
          vibeqc::scf::create_cuda_density_fitting_jk_plan(
              0, 2, integrals.nbf, integrals.naux, batch_metrics,
              batch_three_center, 1.0e-12, 3, &raw_plan, diagnostics,
              cuda_detail);
      require(create_status == VIBEQC_STATUS_SUCCESS, cuda_detail.c_str());
      using CudaPlan = std::unique_ptr<
          vibeqc::scf::CudaDensityFittingJkPlan,
          decltype(&vibeqc::scf::destroy_cuda_density_fitting_jk_plan)>;
      CudaPlan cuda_plan(
          raw_plan, &vibeqc::scf::destroy_cuda_density_fitting_jk_plan);
      require(diagnostics.size() == 2,
              "CUDA DF plan returned the wrong diagnostic count");
      require(diagnostics[0].solver_device_workspace_bytes > 0 &&
                  diagnostics[0].solver_device_workspace_bytes ==
                      diagnostics[1].solver_device_workspace_bytes &&
                  diagnostics[0].solver_host_workspace_bytes ==
                      diagnostics[1].solver_host_workspace_bytes,
              "CUDA DF generic eigensolver workspace diagnostics differ");
      require(diagnostics[0].effective_rank == factor.effective_rank &&
                  diagnostics[1].effective_rank == second_factor.effective_rank,
              "CUDA DF metric effective rank differs from the CPU oracle");
      require_close(diagnostics[0].condition_number, factor.condition_number,
                    2.0e-9,
                    "CUDA DF metric condition differs from the CPU oracle");
      require_close(
          diagnostics[1].condition_number, second_factor.condition_number,
          2.0e-9,
          "second CUDA DF metric condition differs from the CPU oracle");

      std::vector<double> cuda_j;
      std::vector<double> cuda_k;
      const vibeqc_status rhf_cuda_status =
          vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
              cuda_plan.get(), batch_rhf_density, cuda_j, cuda_k, cuda_detail);
      require(rhf_cuda_status == VIBEQC_STATUS_SUCCESS, cuda_detail.c_str());
      std::vector<double> expected_rhf_j = rhf_jk.coulomb;
      std::vector<double> expected_rhf_k = rhf_jk.exchange;
      append_values(expected_rhf_j, second_rhf_jk.coulomb);
      append_values(expected_rhf_k, second_rhf_jk.exchange);
      require_matrix_close(cuda_j, expected_rhf_j, 3.0e-11,
                           "CUDA RHF RI-J differs from the CPU oracle");
      require_matrix_close(cuda_k, expected_rhf_k, 3.0e-11,
                           "CUDA RHF RI-K differs from the CPU oracle");

      std::vector<double> cuda_alpha_k;
      std::vector<double> cuda_beta_k;
      const vibeqc_status uhf_cuda_status =
          vibeqc::scf::execute_cuda_density_fitting_uhf_jk(
              cuda_plan.get(), batch_alpha_density, batch_beta_density,
              cuda_j, cuda_alpha_k, cuda_beta_k, cuda_detail);
      require(uhf_cuda_status == VIBEQC_STATUS_SUCCESS, cuda_detail.c_str());
      std::vector<double> expected_uhf_j = uhf_jk.coulomb;
      std::vector<double> expected_alpha_k = uhf_jk.alpha_exchange;
      std::vector<double> expected_beta_k = uhf_jk.beta_exchange;
      append_values(expected_uhf_j, second_uhf_jk.coulomb);
      append_values(expected_alpha_k, second_uhf_jk.alpha_exchange);
      append_values(expected_beta_k, second_uhf_jk.beta_exchange);
      require_matrix_close(cuda_j, expected_uhf_j, 3.0e-11,
                           "CUDA UHF RI-J differs from the CPU oracle");
      require_matrix_close(
          cuda_alpha_k, expected_alpha_k, 3.0e-11,
          "CUDA UHF alpha RI-K differs from the CPU oracle");
      require_matrix_close(cuda_beta_k, expected_beta_k, 3.0e-11,
                           "CUDA UHF beta RI-K differs from the CPU oracle");

      const vibeqc_status invalid_cuda_status =
          vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
              cuda_plan.get(), {0.0}, cuda_j, cuda_k, cuda_detail);
      require(invalid_cuda_status == VIBEQC_STATUS_INVALID_ARGUMENT,
              "CUDA DF J/K accepted an invalid density shape");
    } else {
      std::cout << "CUDA DF J/K checks skipped: no allocated CUDA device\n";
    }
#endif

    const auto deficient_three_center =
        vibeqc::scf::orthonormalize_density_fitting_three_center({1.0, 3.0}, 1,
                                                                 deficient);
    const auto deficient_jk = vibeqc::scf::build_density_fitting_rhf_jk(
        deficient_three_center, {2.0});
    require(deficient_three_center.effective_rank == 1,
            "three-center tensor lost the metric effective rank");
    require_close(deficient_jk.coulomb[0], deficient_jk.exchange[0], 1.0e-14,
                  "one-AO rank-deficient RI-J and RI-K should coincide");

    bool rejected_bad_density = false;
    try {
      (void)vibeqc::scf::build_density_fitting_rhf_jk(
          orthonormal_three_center,
          std::vector<double>(rhf_density.size() - 1, 0.0));
    } catch (const std::invalid_argument&) {
      rejected_bad_density = true;
    }
    require(rejected_bad_density,
            "RI-J/K accepted a density with inconsistent dimensions");

    bool rejected_nonfinite_tensor = false;
    try {
      std::vector<double> nonfinite = integrals.three_center;
      nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
      (void)vibeqc::scf::orthonormalize_density_fitting_three_center(
          nonfinite, integrals.nbf, factor);
    } catch (const std::invalid_argument&) {
      rejected_nonfinite_tensor = true;
    }
    require(rejected_nonfinite_tensor,
            "metric transform accepted a non-finite three-center tensor");

    const auto small_plan = vibeqc::scf::plan_density_fitting_tiles(
        1, 8, 8, 2, 10 * 1024 * 1024);
    require(small_plan.stores_full_three_center &&
                small_plan.peak_workspace_bytes <= 10 * 1024 * 1024,
            "small DF plan should retain its complete tensor");
    const auto bounded_plan = vibeqc::scf::plan_density_fitting_tiles(
        4, 192, 600, 48, 1024ULL * 1024 * 1024);
    require(!bounded_plan.stores_full_three_center &&
                bounded_plan.peak_workspace_bytes <=
                    1024ULL * 1024 * 1024 &&
                bounded_plan.ao_pair_tile > 0 &&
                bounded_plan.auxiliary_tile > 0,
            "large DF plan exceeded its memory budget");
    bool rejected_tiny_budget = false;
    try {
      (void)vibeqc::scf::plan_density_fitting_tiles(1, 8, 8, 2, 512);
    } catch (const std::invalid_argument&) {
      rejected_tiny_budget = true;
    }
    require(rejected_tiny_budget,
            "DF planner accepted a budget smaller than its minimum tile");
    bool rejected_pair_overflow = false;
    try {
      (void)vibeqc::scf::plan_density_fitting_tiles(
          1, std::numeric_limits<std::size_t>::max(), 1, 1,
          std::numeric_limits<std::size_t>::max());
    } catch (const std::overflow_error&) {
      rejected_pair_overflow = true;
    }
    require(rejected_pair_overflow,
            "DF planner accepted an overflowing AO-pair count");

    std::cout << "validated density-fitting integrals, RI-J/K, and planning\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
