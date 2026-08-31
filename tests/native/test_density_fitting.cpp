#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/density_fitting.hpp"
#include "scf/mean_field.hpp"

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

double matrix_inner_product(const std::vector<double>& first,
                            const std::vector<double>& second) {
  require(first.size() == second.size(), "matrix inner-product dimensions differ");
  double result = 0.0;
  for (std::size_t item = 0; item < first.size(); ++item) {
    result += first[item] * second[item];
  }
  return result;
}

double rhf_df_two_electron_energy(
    const vibeqc::integrals::DensityFittingIntegralData& data,
    const std::vector<double>& density) {
  const auto factor = vibeqc::scf::factor_density_fitting_metric(
      data.metric, data.naux, 1.0e-12);
  const auto three_center =
      vibeqc::scf::orthonormalize_density_fitting_three_center(
          data.three_center, data.nbf, factor);
  const auto jk = vibeqc::scf::build_density_fitting_rhf_jk(
      three_center, density);
  return 0.5 * matrix_inner_product(density, jk.coulomb) -
         0.25 * matrix_inner_product(density, jk.exchange);
}

double uhf_df_two_electron_energy(
    const vibeqc::integrals::DensityFittingIntegralData& data,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density) {
  const auto factor = vibeqc::scf::factor_density_fitting_metric(
      data.metric, data.naux, 1.0e-12);
  const auto three_center =
      vibeqc::scf::orthonormalize_density_fitting_three_center(
          data.three_center, data.nbf, factor);
  const auto jk = vibeqc::scf::build_density_fitting_uhf_jk(
      three_center, alpha_density, beta_density);
  std::vector<double> total_density(alpha_density.size(), 0.0);
  for (std::size_t item = 0; item < total_density.size(); ++item) {
    total_density[item] = alpha_density[item] + beta_density[item];
  }
  return 0.5 * matrix_inner_product(total_density, jk.coulomb) -
         0.5 * matrix_inner_product(alpha_density, jk.alpha_exchange) -
         0.5 * matrix_inner_product(beta_density, jk.beta_exchange);
}

double rhf_df_total_energy(
    const vibeqc::integrals::IntegralData& one_electron,
    const vibeqc::integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& density) {
  return matrix_inner_product(density, one_electron.hcore) +
         rhf_df_two_electron_energy(density_fitting, density) +
         one_electron.nuclear_repulsion;
}

double uhf_df_total_energy(
    const vibeqc::integrals::IntegralData& one_electron,
    const vibeqc::integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density) {
  std::vector<double> total_density(alpha_density.size(), 0.0);
  for (std::size_t item = 0; item < total_density.size(); ++item) {
    total_density[item] = alpha_density[item] + beta_density[item];
  }
  return matrix_inner_product(total_density, one_electron.hcore) +
         uhf_df_two_electron_energy(density_fitting, alpha_density,
                                     beta_density) +
         one_electron.nuclear_repulsion;
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

vibeqc::core::System spherical_ds_system(bool reverse_shell_order) {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}};
  const vibeqc::core::Shell d_shell{0, 2, {{0.9, 1.0}}};
  const vibeqc::core::Shell s_shell{0, 0, {{0.6, 1.0}}};
  system.shells = reverse_shell_order
      ? std::vector<vibeqc::core::Shell>{s_shell, d_shell}
      : std::vector<vibeqc::core::Shell>{d_shell, s_shell};
  system.multiplicity = 1;
  system.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "spherical s/d basis normalization failed");
  return system;
}

vibeqc::core::System compact_auxiliary_system() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 0, {{0.5, 1.0}}}, {0, 0, {{0.3, 1.0}}}};
  system.multiplicity = 1;
  system.basis_representation = VIBEQC_BASIS_CARTESIAN;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "compact auxiliary basis normalization failed");
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
    // The CUDA evaluator returns Cartesian tensors. Verify the shared
    // transform helper reproduces the direct spherical reference exactly.
    vibeqc::core::System cartesian_spherical_orbital = spherical_d_system(0.9);
    vibeqc::core::System cartesian_spherical_auxiliary =
        spherical_d_system(0.55);
    cartesian_spherical_orbital.basis_representation =
        VIBEQC_BASIS_CARTESIAN;
    cartesian_spherical_auxiliary.basis_representation =
        VIBEQC_BASIS_CARTESIAN;
    const auto spherical_cartesian =
        vibeqc::integrals::build_density_fitting_integrals(
            cartesian_spherical_orbital, cartesian_spherical_auxiliary);
    const auto spherical_transformed =
        vibeqc::integrals::transform_density_fitting_integrals(
            spherical_cartesian, spherical_d_system(0.9),
            spherical_d_system(0.55));
    require_matrix_close(spherical_transformed.metric, spherical.metric,
                         2.0e-13,
                         "public spherical DF metric transform differs");
    require_matrix_close(spherical_transformed.three_center,
                         spherical.three_center, 2.0e-13,
                         "public spherical DF tensor transform differs");
    require_matrix_close(spherical_transformed.metric_derivative,
                         spherical.metric_derivative, 2.0e-13,
                         "public spherical DF metric derivative transform differs");
    require_matrix_close(spherical_transformed.three_center_derivative,
                         spherical.three_center_derivative, 2.0e-13,
                         "public spherical DF tensor derivative transform differs");

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

    const auto rhf_gradient =
        vibeqc::scf::build_density_fitting_rhf_gradient(
            integrals, rhf_density, 1.0e-12);
    require(rhf_gradient.ncoord == integrals.ncoord &&
                rhf_gradient.derivative.size() == integrals.ncoord &&
                rhf_gradient.forces.size() == integrals.ncoord,
            "RHF DF gradient dimensions are inconsistent");
    constexpr double gradient_displacement = 1.0e-3;
    vibeqc::core::System gradient_plus_orbital = orbital;
    vibeqc::core::System gradient_plus_auxiliary = auxiliary;
    vibeqc::core::System gradient_minus_orbital = orbital;
    vibeqc::core::System gradient_minus_auxiliary = auxiliary;
    gradient_plus_orbital.atoms[0].position[2] += gradient_displacement;
    gradient_plus_auxiliary.atoms[0].position[2] += gradient_displacement;
    gradient_minus_orbital.atoms[0].position[2] -= gradient_displacement;
    gradient_minus_auxiliary.atoms[0].position[2] -= gradient_displacement;
    const auto gradient_plus =
        vibeqc::integrals::build_density_fitting_integrals(
            gradient_plus_orbital, gradient_plus_auxiliary);
    const auto gradient_minus =
        vibeqc::integrals::build_density_fitting_integrals(
            gradient_minus_orbital, gradient_minus_auxiliary);
    const double rhf_plus_energy =
        rhf_df_two_electron_energy(gradient_plus, rhf_density);
    const double rhf_minus_energy =
        rhf_df_two_electron_energy(gradient_minus, rhf_density);
    require_close(
        rhf_gradient.derivative[2],
        (rhf_plus_energy - rhf_minus_energy) /
            (2.0 * gradient_displacement),
        2.0e-6,
        "RHF DF metric/three-center force response differs from finite differences");
    require_close(rhf_gradient.forces[2], -rhf_gradient.derivative[2],
                  1.0e-14, "RHF DF force sign is inconsistent");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      require_close(rhf_gradient.derivative[axis] +
                        rhf_gradient.derivative[axis + 3],
                    0.0, 3.0e-10,
                    "RHF DF gradient violates translation invariance");
    }

    const auto uhf_gradient =
        vibeqc::scf::build_density_fitting_uhf_gradient(
            integrals, alpha_density, beta_density, 1.0e-12);
    require(uhf_gradient.ncoord == integrals.ncoord &&
                uhf_gradient.derivative.size() == integrals.ncoord &&
                uhf_gradient.forces.size() == integrals.ncoord,
            "UHF DF gradient dimensions are inconsistent");
    const double uhf_plus_energy = uhf_df_two_electron_energy(
        gradient_plus, alpha_density, beta_density);
    const double uhf_minus_energy = uhf_df_two_electron_energy(
        gradient_minus, alpha_density, beta_density);
    require_close(
        uhf_gradient.derivative[2],
        (uhf_plus_energy - uhf_minus_energy) /
            (2.0 * gradient_displacement),
        2.0e-6,
        "UHF DF metric/three-center force response differs from finite differences");
    require_close(uhf_gradient.forces[2], -uhf_gradient.derivative[2],
                  1.0e-14, "UHF DF force sign is inconsistent");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      require_close(uhf_gradient.derivative[axis] +
                        uhf_gradient.derivative[axis + 3],
                    0.0, 3.0e-10,
                    "UHF DF gradient violates translation invariance");
    }

    // The complete force helpers add the ordinary one-electron, overlap
    // Pulay, and nuclear-repulsion pieces around the DF response. With a
    // zero weighted density the finite-difference check isolates their
    // assembly and still exercises the public full-force interface.
    const auto one_electron =
        vibeqc::integrals::build_integrals(orbital);
    const auto plus_one_electron =
        vibeqc::integrals::build_integrals(plus_orbital);
    const auto minus_one_electron =
        vibeqc::integrals::build_integrals(minus_orbital);
    const std::vector<double> zero_weighted_density(rhf_density.size(), 0.0);
    const auto rhf_forces = vibeqc::scf::build_density_fitting_rhf_forces(
        one_electron, integrals, rhf_density, zero_weighted_density, 1.0e-12);
    require(rhf_forces.size() == integrals.ncoord,
            "complete RHF DF force dimensions are inconsistent");
    require_close(
        -rhf_forces[2],
        (rhf_df_total_energy(plus_one_electron, plus, rhf_density) -
         rhf_df_total_energy(minus_one_electron, minus, rhf_density)) /
            (2.0 * displacement),
        2.0e-8,
        "complete RHF DF force differs from finite differences");

    const auto uhf_forces = vibeqc::scf::build_density_fitting_uhf_forces(
        one_electron, integrals, alpha_density, beta_density,
        zero_weighted_density, zero_weighted_density, 1.0e-12);
    require(uhf_forces.size() == integrals.ncoord,
            "complete UHF DF force dimensions are inconsistent");
    require_close(
        -uhf_forces[2],
        (uhf_df_total_energy(plus_one_electron, plus, alpha_density,
                             beta_density) -
         uhf_df_total_energy(minus_one_electron, minus, alpha_density,
                             beta_density)) /
            (2.0 * displacement),
        2.0e-8,
        "complete UHF DF force differs from finite differences");

    // A rank-deficient metric can rotate its retained and null auxiliary
    // spaces under displacement. Check the projector terms in d(M+) rather
    // than only the full-rank shortcut -M+ dM M+.
    vibeqc::integrals::DensityFittingIntegralData rotating_metric{
        1, 2, 1,
        {1.0, 0.0, 0.0, 0.0},
        {2.0, 3.0},
        {0.0, 1.0, 1.0, 0.0},
        {0.0, 0.0}};
    const auto rotating_gradient =
        vibeqc::scf::build_density_fitting_rhf_gradient(
            rotating_metric, {1.0}, 1.0e-12);
    // rho=(2,3), dM+ has unit off-diagonal entries, and the RHF exchange
    // equals Coulomb for one AO: dE2 = (1 - 1/4) * 2*rho0*rho1.
    require_close(rotating_gradient.derivative[0], 3.0, 1.0e-12,
                  "rank-deficient DF metric response omitted null-space mixing");

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
      // The CUDA integral evaluator should reproduce the independent host
      // oracle before any metric factorization or J/K staging occurs.  This
      // exercises both Cartesian recurrence output and first derivatives.
      vibeqc::integrals::DensityFittingIntegralData cuda_integrals;
      std::string cuda_integral_detail;
      require(vibeqc::scf::build_cuda_density_fitting_integrals(
                  0, orbital, auxiliary, cuda_integrals,
                  cuda_integral_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_integral_detail.c_str());
      require(cuda_integrals.nbf == integrals.nbf &&
                  cuda_integrals.naux == integrals.naux &&
                  cuda_integrals.ncoord == integrals.ncoord,
              "CUDA DF integral dimensions are inconsistent");
      require_matrix_close(cuda_integrals.metric, integrals.metric, 3.0e-11,
                           "CUDA DF metric differs from the CPU oracle");
      require_matrix_close(cuda_integrals.three_center,
                           integrals.three_center, 3.0e-11,
                           "CUDA DF three-center tensor differs from oracle");
      require_matrix_close(cuda_integrals.metric_derivative,
                           integrals.metric_derivative, 3.0e-10,
                           "CUDA DF metric derivative differs from oracle");
      require_matrix_close(cuda_integrals.three_center_derivative,
                           integrals.three_center_derivative, 3.0e-10,
                           "CUDA DF three-center derivative differs from oracle");

      vibeqc::integrals::IntegralData cuda_one_electron;
      std::string cuda_one_electron_detail;
      require(vibeqc::scf::build_cuda_one_electron_integrals(
                  0, orbital, cuda_one_electron,
                  cuda_one_electron_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_one_electron_detail.c_str());
      const auto host_one_electron =
          vibeqc::integrals::build_integrals(orbital);
      require_matrix_close(cuda_one_electron.overlap,
                           host_one_electron.overlap, 3.0e-11,
                           "CUDA overlap differs from the CPU oracle");
      require_matrix_close(cuda_one_electron.hcore,
                           host_one_electron.hcore, 3.0e-11,
                           "CUDA Hcore differs from the CPU oracle");
      require_matrix_close(cuda_one_electron.overlap_derivative,
                           host_one_electron.overlap_derivative, 3.0e-10,
                           "CUDA overlap derivative differs from oracle");
      require_matrix_close(cuda_one_electron.hcore_derivative,
                           host_one_electron.hcore_derivative, 3.0e-10,
                           "CUDA Hcore derivative differs from oracle");
      require_close(cuda_one_electron.nuclear_repulsion,
                    host_one_electron.nuclear_repulsion, 3.0e-12,
                    "CUDA nuclear repulsion differs from oracle");
      require_matrix_close(cuda_one_electron.nuclear_repulsion_derivative,
                           host_one_electron.nuclear_repulsion_derivative,
                           3.0e-11,
                           "CUDA nuclear derivative differs from oracle");

      // Exercise the production bucket bridge itself (not only the lower-level
      // J/K API): both systems must share one batched plan while retaining
      // input order and independent SCF results.
      vibeqc::core::System second_orbital = orbital;
      second_orbital.atoms[1].position[2] += 0.08;
      vibeqc::core::System second_auxiliary = auxiliary;
      second_auxiliary.atoms[1].position[2] += 0.08;
      std::vector<vibeqc::integrals::DensityFittingIntegralData>
          batch_integrals;
      std::string batch_integral_detail;
      require(vibeqc::scf::build_cuda_density_fitting_integrals_batch(
                  0, {orbital, second_orbital},
                  {auxiliary, second_auxiliary}, batch_integrals,
                  batch_integral_detail) == VIBEQC_STATUS_SUCCESS,
              batch_integral_detail.c_str());
      require(batch_integrals.size() == 2,
              "CUDA DF integral batch returned the wrong result count");
      const auto second_host_integrals =
          vibeqc::integrals::build_density_fitting_integrals(
              second_orbital, second_auxiliary);
      require_matrix_close(batch_integrals[0].metric, integrals.metric,
                           3.0e-11,
                           "batched CUDA DF metric differs for item zero");
      require_matrix_close(batch_integrals[0].three_center,
                           integrals.three_center, 3.0e-11,
                           "batched CUDA DF tensor differs for item zero");
      require_matrix_close(batch_integrals[1].metric,
                           second_host_integrals.metric, 3.0e-11,
                           "batched CUDA DF metric differs for item one");
      require_matrix_close(batch_integrals[1].three_center,
                           second_host_integrals.three_center, 3.0e-11,
                           "batched CUDA DF tensor differs for item one");
      require_matrix_close(batch_integrals[1].metric_derivative,
                           second_host_integrals.metric_derivative, 3.0e-10,
                           "batched CUDA DF metric derivative differs");
      require_matrix_close(batch_integrals[1].three_center_derivative,
                           second_host_integrals.three_center_derivative,
                           3.0e-10,
                           "batched CUDA DF tensor derivative differs");
      // A positive output budget exercises the bounded chunk-selection path
      // while preserving the same public results.
      std::vector<vibeqc::integrals::DensityFittingIntegralData>
          chunked_batch_integrals;
      std::string chunked_batch_detail;
      require(vibeqc::scf::build_cuda_density_fitting_integrals_batch(
                  0, {orbital, second_orbital},
                  {auxiliary, second_auxiliary}, chunked_batch_integrals,
                  chunked_batch_detail, 32768U) == VIBEQC_STATUS_SUCCESS,
              chunked_batch_detail.c_str());
      require_matrix_close(chunked_batch_integrals[0].metric,
                           batch_integrals[0].metric, 3.0e-11,
                           "chunked CUDA DF metric differs for item zero");
      require_matrix_close(chunked_batch_integrals[1].three_center,
                           batch_integrals[1].three_center, 3.0e-11,
                           "chunked CUDA DF tensor differs for item one");
      std::vector<vibeqc::integrals::IntegralData> batch_one_electron;
      std::string batch_one_electron_detail;
      require(vibeqc::scf::build_cuda_one_electron_integrals_batch(
                  0, {orbital, second_orbital}, batch_one_electron,
                  batch_one_electron_detail) == VIBEQC_STATUS_SUCCESS,
              batch_one_electron_detail.c_str());
      require(batch_one_electron.size() == 2,
              "CUDA one-electron batch returned the wrong result count");
      require_matrix_close(batch_one_electron[0].overlap,
                           host_one_electron.overlap, 3.0e-11,
                           "batched CUDA overlap differs for item zero");
      require_matrix_close(batch_one_electron[1].overlap,
                           vibeqc::integrals::build_integrals(second_orbital)
                               .overlap,
                           3.0e-11,
                           "batched CUDA overlap differs for item one");
      vibeqc::scf::ScfOptions bucket_options;
      bucket_options.density_fitting_mode = VIBEQC_DENSITY_FITTING_CUDA;
      const std::vector<vibeqc::core::System> bucket_systems{
          orbital, second_orbital};
      const std::vector<const std::vector<double>*> bucket_initial{
          nullptr, nullptr};
      const auto bucket_results =
          vibeqc::scf::run_rhf_density_fitting_cuda_bucket(
              bucket_systems, auxiliary, bucket_options, bucket_initial, 0);
      require(bucket_results.size() == 2,
              "CUDA DF bucket returned the wrong result count");
      require(bucket_results[0].status == VIBEQC_STATUS_SUCCESS &&
                  bucket_results[1].status == VIBEQC_STATUS_SUCCESS,
              "CUDA DF bucket SCF item failed");
      require(bucket_results[0].scf.converged &&
                  bucket_results[1].scf.converged,
              "CUDA DF bucket did not converge both systems");

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
      require(diagnostics[0].device_resident_bytes > 0 &&
                  diagnostics[0].peak_device_bytes >=
                      diagnostics[0].device_resident_bytes &&
                  diagnostics[0].host_resident_bytes > 0 &&
                  diagnostics[0].peak_host_bytes >=
                      diagnostics[0].host_resident_bytes &&
                  diagnostics[0].system_index == 0 &&
                  diagnostics[1].system_index == 1 &&
                  diagnostics[0].auxiliary_tile == 3 &&
                  diagnostics[0].streamed && diagnostics[1].streamed,
              "CUDA DF streamed allocation diagnostics are inconsistent");
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

      // The default planner keeps small auxiliary bases resident on device.
      // Exercise that path with the intentionally non-symmetric RHF density
      // above so the resident cuBLAS layout conversion is covered too.
      vibeqc::scf::CudaDensityFittingJkPlan* resident_raw_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          resident_diagnostics;
      std::string resident_detail;
      const vibeqc_status resident_create_status =
          vibeqc::scf::create_cuda_density_fitting_jk_plan(
              0, 1, integrals.nbf, integrals.naux, integrals.metric,
              integrals.three_center, 1.0e-12, 0, &resident_raw_plan,
              resident_diagnostics, resident_detail);
      require(resident_create_status == VIBEQC_STATUS_SUCCESS,
              resident_detail.c_str());
      CudaPlan resident_plan(
          resident_raw_plan,
          &vibeqc::scf::destroy_cuda_density_fitting_jk_plan);
      require(resident_diagnostics.size() == 1 &&
                  !resident_diagnostics[0].streamed,
              "CUDA DF resident diagnostics are inconsistent");
      std::vector<double> resident_j;
      std::vector<double> resident_k;
      const vibeqc_status resident_status =
          vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
              resident_plan.get(), rhf_density, resident_j, resident_k,
              resident_detail);
      require(resident_status == VIBEQC_STATUS_SUCCESS,
              resident_detail.c_str());
      require_matrix_close(resident_j, rhf_jk.coulomb, 3.0e-11,
                           "resident CUDA RHF RI-J differs from the CPU oracle");
      require_matrix_close(resident_k, rhf_jk.exchange, 3.0e-11,
                           "resident CUDA RHF RI-K differs from the CPU oracle");

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

      // Exercise the planner-aware AO-row streaming entry point with a tile
      // that is deliberately not divisible by nbf. This catches partial-row
      // staging and the column-major/row-major exchange scatter while
      // retaining the same packed batch layout as the compatibility plan.
      vibeqc::scf::CudaDensityFittingJkPlan* tiled_raw_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          tiled_diagnostics;
      std::string tiled_detail;
      const vibeqc_status tiled_create_status =
          vibeqc::scf::create_cuda_density_fitting_jk_plan_tiled(
              0, 2, integrals.nbf, integrals.naux, batch_metrics,
              batch_three_center, 1.0e-12, 3, 3, &tiled_raw_plan,
              tiled_diagnostics, tiled_detail);
      require(tiled_create_status == VIBEQC_STATUS_SUCCESS,
              tiled_detail.c_str());
      CudaPlan tiled_plan(
          tiled_raw_plan, &vibeqc::scf::destroy_cuda_density_fitting_jk_plan);
      require(tiled_diagnostics.size() == 2 &&
                  tiled_diagnostics[0].streamed &&
                  tiled_diagnostics[0].auxiliary_tile == 3,
              "CUDA DF AO-pair tiled diagnostics are inconsistent");
      std::vector<double> tiled_j;
      std::vector<double> tiled_k;
      const vibeqc_status tiled_rhf_status =
          vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
              tiled_plan.get(), batch_rhf_density, tiled_j, tiled_k,
              tiled_detail);
      require(tiled_rhf_status == VIBEQC_STATUS_SUCCESS,
              tiled_detail.c_str());
      require_matrix_close(
          tiled_j, expected_rhf_j, 3.0e-11,
          "AO-pair tiled CUDA RHF RI-J differs from the CPU oracle");
      require_matrix_close(
          tiled_k, expected_rhf_k, 3.0e-11,
          "AO-pair tiled CUDA RHF RI-K differs from the CPU oracle");
      std::vector<double> tiled_alpha_k;
      std::vector<double> tiled_beta_k;
      const vibeqc_status tiled_uhf_status =
          vibeqc::scf::execute_cuda_density_fitting_uhf_jk(
              tiled_plan.get(), batch_alpha_density, batch_beta_density,
              tiled_j, tiled_alpha_k, tiled_beta_k, tiled_detail);
      require(tiled_uhf_status == VIBEQC_STATUS_SUCCESS,
              tiled_detail.c_str());
      require_matrix_close(
          tiled_j, expected_uhf_j, 3.0e-11,
          "AO-pair tiled CUDA UHF RI-J differs from the CPU oracle");
      require_matrix_close(
          tiled_alpha_k, expected_alpha_k, 3.0e-11,
          "AO-pair tiled CUDA alpha RI-K differs from the CPU oracle");
      require_matrix_close(
          tiled_beta_k, expected_beta_k, 3.0e-11,
          "AO-pair tiled CUDA beta RI-K differs from the CPU oracle");

      // Source-backed budget replay regenerates transformed tiles directly on
      // the device instead of retaining the raw three-center tensor on host.
      vibeqc::scf::CudaDensityFittingIntegralSource* source = nullptr;
      std::vector<double> source_metrics;
      std::size_t source_nbf = 0;
      std::size_t source_naux = 0;
      std::string source_detail;
      require(vibeqc::scf::create_cuda_density_fitting_integral_source(
                  0, {orbital}, {auxiliary}, &source, source_metrics,
                  source_nbf, source_naux, source_detail) ==
                  VIBEQC_STATUS_SUCCESS,
              source_detail.c_str());
      vibeqc::scf::CudaDensityFittingJkPlan* source_raw_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          source_diagnostics;
      require(vibeqc::scf::create_cuda_density_fitting_jk_plan_from_source(
                  0, &source, 1, source_nbf, source_naux, source_metrics,
                  1.0e-12, 3, 3, &source_raw_plan, source_diagnostics,
                  source_detail) == VIBEQC_STATUS_SUCCESS,
              source_detail.c_str());
      vibeqc::scf::destroy_cuda_density_fitting_integral_source(source);
      CudaPlan source_plan(
          source_raw_plan, &vibeqc::scf::destroy_cuda_density_fitting_jk_plan);
      require(source_diagnostics.size() == 1 && source_diagnostics[0].streamed,
              "source-backed CUDA DF diagnostics are inconsistent");
      std::vector<double> source_j;
      std::vector<double> source_k;
      require(vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
                  source_plan.get(), rhf_density, source_j, source_k,
                  source_detail) == VIBEQC_STATUS_SUCCESS,
              source_detail.c_str());
      require_matrix_close(source_j, rhf_jk.coulomb, 3.0e-11,
                           "source-backed CUDA RHF RI-J differs from oracle");
      require_matrix_close(source_k, rhf_jk.exchange, 3.0e-11,
                           "source-backed CUDA RHF RI-K differs from oracle");

      // Exercise a genuinely heterogeneous source batch.  The two orbital
      // systems have identical public dimensions but reverse their spherical
      // s/d shell order, while the compact auxiliary basis has a different
      // Cartesian count.  This catches both per-system transform reuse and
      // the auxiliary-metric Cartesian offset in source replay.
      const vibeqc::core::System source_orbital_a =
          spherical_ds_system(false);
      const vibeqc::core::System source_orbital_b =
          spherical_ds_system(true);
      const vibeqc::core::System source_auxiliary =
          compact_auxiliary_system();
      const auto source_host_a =
          vibeqc::integrals::build_density_fitting_integrals(
              source_orbital_a, source_auxiliary);
      const auto source_host_b =
          vibeqc::integrals::build_density_fitting_integrals(
              source_orbital_b, source_auxiliary);
      require(source_host_a.nbf == source_host_b.nbf &&
                  source_host_a.naux == source_host_b.naux &&
                  source_host_a.nbf !=
                      vibeqc::molecule::cartesian_ao_count(source_auxiliary),
              "heterogeneous source fixture dimensions are not diagnostic");
      vibeqc::scf::CudaDensityFittingIntegralSource* batch_source = nullptr;
      std::vector<double> batch_source_metrics;
      std::size_t batch_source_nbf = 0;
      std::size_t batch_source_naux = 0;
      std::string batch_source_detail;
      require(vibeqc::scf::create_cuda_density_fitting_integral_source(
                  0, {source_orbital_a, source_orbital_b},
                  {source_auxiliary, source_auxiliary}, &batch_source,
                  batch_source_metrics, batch_source_nbf, batch_source_naux,
                  batch_source_detail) == VIBEQC_STATUS_SUCCESS,
              batch_source_detail.c_str());
      vibeqc::scf::CudaDensityFittingJkPlan* batch_source_raw_plan = nullptr;
      std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic>
          batch_source_diagnostics;
      require(vibeqc::scf::create_cuda_density_fitting_jk_plan_from_source(
                  0, &batch_source, 2, batch_source_nbf, batch_source_naux,
                  batch_source_metrics, 1.0e-12, 2, 8,
                  &batch_source_raw_plan, batch_source_diagnostics,
                  batch_source_detail) == VIBEQC_STATUS_SUCCESS,
              batch_source_detail.c_str());
      vibeqc::scf::destroy_cuda_density_fitting_integral_source(batch_source);
      CudaPlan batch_source_plan(
          batch_source_raw_plan,
          &vibeqc::scf::destroy_cuda_density_fitting_jk_plan);
      const std::vector<double> source_density_a(source_host_a.nbf *
                                                     source_host_a.nbf,
                                                 0.0);
      std::vector<double> source_density_b = source_density_a;
      for (std::size_t diagonal = 0; diagonal < source_host_a.nbf;
           ++diagonal) {
        source_density_b[matrix_index(diagonal, diagonal,
                                      source_host_a.nbf)] =
            0.05 * static_cast<double>(diagonal + 1);
      }
      std::vector<double> batch_source_density = source_density_a;
      append_values(batch_source_density, source_density_b);
      std::vector<double> batch_source_j;
      std::vector<double> batch_source_k;
      require(vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
                  batch_source_plan.get(), batch_source_density,
                  batch_source_j, batch_source_k, batch_source_detail) ==
                  VIBEQC_STATUS_SUCCESS,
              batch_source_detail.c_str());
      const auto source_factor_a = vibeqc::scf::factor_density_fitting_metric(
          source_host_a.metric, source_host_a.naux, 1.0e-12);
      const auto source_factor_b = vibeqc::scf::factor_density_fitting_metric(
          source_host_b.metric, source_host_b.naux, 1.0e-12);
      const auto source_three_a =
          vibeqc::scf::orthonormalize_density_fitting_three_center(
              source_host_a.three_center, source_host_a.nbf, source_factor_a);
      const auto source_three_b =
          vibeqc::scf::orthonormalize_density_fitting_three_center(
              source_host_b.three_center, source_host_b.nbf, source_factor_b);
      const auto source_jk_a = vibeqc::scf::build_density_fitting_rhf_jk(
          source_three_a, source_density_a);
      const auto source_jk_b = vibeqc::scf::build_density_fitting_rhf_jk(
          source_three_b, source_density_b);
      std::vector<double> expected_source_j = source_jk_a.coulomb;
      append_values(expected_source_j, source_jk_b.coulomb);
      std::vector<double> expected_source_k = source_jk_a.exchange;
      append_values(expected_source_k, source_jk_b.exchange);
      require_matrix_close(batch_source_j, expected_source_j, 4.0e-10,
                           "heterogeneous source RHF RI-J differs from oracle");
      require_matrix_close(batch_source_k, expected_source_k, 4.0e-10,
                           "heterogeneous source RHF RI-K differs from oracle");

      const vibeqc_status invalid_cuda_status =
          vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
              cuda_plan.get(), {0.0}, cuda_j, cuda_k, cuda_detail);
      require(invalid_cuda_status == VIBEQC_STATUS_INVALID_ARGUMENT,
              "CUDA DF J/K accepted an invalid density shape");

      // The final CUDA SCF bridge uses the same plan for raw two-electron
      // force response.  Exercise both spin conventions against the
      // independent metric/derivative oracle, including a single-item slice
      // submitted to a multi-system plan.
      const std::vector<double> inverse =
          vibeqc::scf::density_fitting_metric_pseudoinverse(
              integrals, 1.0e-12);
      std::vector<double> inverse_derivative(
          integrals.ncoord * integrals.naux * integrals.naux, 0.0);
      for (std::size_t coordinate = 0; coordinate < integrals.ncoord;
           ++coordinate) {
        const auto response =
            vibeqc::scf::density_fitting_metric_pseudoinverse_derivative(
                integrals, inverse, coordinate);
        std::copy(response.begin(), response.end(),
                  inverse_derivative.begin() +
                      coordinate * integrals.naux * integrals.naux);
      }
      std::vector<double> cuda_rhf_derivative;
      require(vibeqc::scf::execute_cuda_density_fitting_rhf_force_response(
                  cuda_plan.get(), integrals.three_center, inverse,
                  integrals.three_center_derivative, inverse_derivative,
                  integrals.ncoord, rhf_density, cuda_rhf_derivative,
                  cuda_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_detail.c_str());
      const auto host_rhf_gradient =
          vibeqc::scf::build_density_fitting_rhf_gradient(
              integrals, rhf_density, 1.0e-12);
      require_matrix_close(cuda_rhf_derivative, host_rhf_gradient.derivative,
                           5.0e-10,
                           "CUDA RHF force response differs from oracle");

      std::vector<double> cuda_uhf_derivative;
      require(vibeqc::scf::execute_cuda_density_fitting_uhf_force_response(
                  cuda_plan.get(), integrals.three_center, inverse,
                  integrals.three_center_derivative, inverse_derivative,
                  integrals.ncoord, alpha_density, beta_density,
                  cuda_uhf_derivative, cuda_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_detail.c_str());
      const auto host_uhf_gradient =
          vibeqc::scf::build_density_fitting_uhf_gradient(
              integrals, alpha_density, beta_density, 1.0e-12);
      require_matrix_close(cuda_uhf_derivative, host_uhf_gradient.derivative,
                           5.0e-10,
                           "CUDA UHF force response differs from oracle");

      // Also exercise the documented packed system-then-coordinate layout.
      const vibeqc::integrals::DensityFittingIntegralData second_integrals{
          integrals.nbf,
          integrals.naux,
          integrals.ncoord,
          second_metric,
          plus.three_center,
          integrals.metric_derivative,
          integrals.three_center_derivative,
      };
      const std::vector<double> second_inverse =
          vibeqc::scf::density_fitting_metric_pseudoinverse(
              second_integrals, 1.0e-12);
      std::vector<double> packed_inverse = inverse;
      append_values(packed_inverse, second_inverse);
      std::vector<double> packed_inverse_derivative = inverse_derivative;
      std::vector<double> second_inverse_derivative(
          integrals.ncoord * integrals.naux * integrals.naux, 0.0);
      for (std::size_t coordinate = 0; coordinate < integrals.ncoord;
           ++coordinate) {
        const auto response =
            vibeqc::scf::density_fitting_metric_pseudoinverse_derivative(
                second_integrals, second_inverse, coordinate);
        std::copy(response.begin(), response.end(),
                  second_inverse_derivative.begin() +
                      coordinate * integrals.naux * integrals.naux);
      }
      append_values(packed_inverse_derivative, second_inverse_derivative);
      std::vector<double> packed_derivative_raw =
          integrals.three_center_derivative;
      append_values(packed_derivative_raw, integrals.three_center_derivative);
      std::vector<double> packed_cuda_derivative;
      require(vibeqc::scf::execute_cuda_density_fitting_rhf_force_response(
                  cuda_plan.get(), batch_three_center, packed_inverse,
                  packed_derivative_raw, packed_inverse_derivative,
                  integrals.ncoord, batch_rhf_density, packed_cuda_derivative,
                  cuda_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_detail.c_str());
      const auto second_rhf_gradient =
          vibeqc::scf::build_density_fitting_rhf_gradient(
              second_integrals, second_rhf_density, 1.0e-12);
      std::vector<double> expected_packed_derivative =
          host_rhf_gradient.derivative;
      append_values(expected_packed_derivative,
                    second_rhf_gradient.derivative);
      require_matrix_close(packed_cuda_derivative,
                           expected_packed_derivative, 5.0e-10,
                           "packed CUDA RHF force response differs from oracle");
      std::vector<double> packed_alpha_density = alpha_density;
      std::vector<double> packed_beta_density = beta_density;
      append_values(packed_alpha_density, second_alpha_density);
      append_values(packed_beta_density, second_beta_density);
      std::vector<double> packed_cuda_uhf_derivative;
      require(vibeqc::scf::execute_cuda_density_fitting_uhf_force_response(
                  cuda_plan.get(), batch_three_center, packed_inverse,
                  packed_derivative_raw, packed_inverse_derivative,
                  integrals.ncoord, packed_alpha_density,
                  packed_beta_density, packed_cuda_uhf_derivative,
                  cuda_detail) == VIBEQC_STATUS_SUCCESS,
              cuda_detail.c_str());
      const auto second_uhf_gradient =
          vibeqc::scf::build_density_fitting_uhf_gradient(
              second_integrals, second_alpha_density, second_beta_density,
              1.0e-12);
      std::vector<double> expected_packed_uhf_derivative =
          host_uhf_gradient.derivative;
      append_values(expected_packed_uhf_derivative,
                    second_uhf_gradient.derivative);
      require_matrix_close(
          packed_cuda_uhf_derivative, expected_packed_uhf_derivative,
          5.0e-10, "packed CUDA UHF force response differs from oracle");
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
    const auto default_plan = vibeqc::scf::plan_density_fitting_tiles(
        1, 8, 8, 2, 0);
    require(default_plan.batch_tile > 0 && default_plan.auxiliary_tile > 0,
            "zero DF planner budget should select the default policy");
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
