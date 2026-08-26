#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/density_fitting.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

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

    std::cout << "validated density-fitting integral and planning foundation\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
