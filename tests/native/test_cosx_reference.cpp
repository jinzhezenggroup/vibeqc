#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cosx_reference.hpp"
#include "dft/grid.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4}}};
  system.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "COSX H2 normalization failed");
  return system;
}

double max_abs_diff(const std::vector<double>& first, const std::vector<double>& second) {
  require(first.size() == second.size(), "matrix comparison size mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    error = std::max(error, std::abs(first[i] - second[i]));
  }
  return error;
}

std::vector<double> direct_exchange(const vibeqc::core::System& system,
                                    const std::vector<double>& density) {
  const auto integrals = vibeqc::integrals::build_integrals(system, false, true);
  const std::size_t n = integrals.nbf;
  require(density.size() == n * n, "direct exchange density size mismatch");
  std::vector<double> exchange(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          exchange[i * n + j] += density[k * n + l] * integrals.eri[((i * n + k) * n + j) * n + l];
        }
      }
    }
  }
  return exchange;
}

std::vector<double> numerical_esp(const vibeqc::core::System& system,
                                  const std::array<double, 3>& probe,
                                  const vibeqc::dft::GridSpec& spec) {
  const vibeqc::dft::MolecularGrid grid(system, spec);
  const vibeqc::dft::AoBasis basis(system);
  const std::size_t n = basis.nao;
  std::vector<double> ao(grid.point_count() * n);
  basis.evaluate(grid.points().data(), grid.point_count(), 0, 0, n, ao.data(), ao.size());
  std::vector<double> value(n * n, 0.0);
  for (std::size_t point = 0; point < grid.point_count(); ++point) {
    double r2 = 0.0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double delta = grid.points()[3 * point + axis] - probe[axis];
      r2 += delta * delta;
    }
    require(r2 > 1.0e-20, "numerical ESP probe coincides with a quadrature point");
    const double factor = grid.weights()[point] / std::sqrt(r2);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        value[i * n + j] += factor * ao[point * n + i] * ao[point * n + j];
      }
    }
  }
  return value;
}

std::vector<double> brute_discrete_exchange(const vibeqc::core::System& system,
                                            const vibeqc::dft::MolecularGrid& grid,
                                            const std::vector<double>& density) {
  const vibeqc::dft::AoBasis basis(system);
  const std::size_t n = basis.nao;
  std::vector<double> ao(grid.point_count() * n);
  basis.evaluate(grid.points().data(), grid.point_count(), 0, 0, n, ao.data(), ao.size());
  const auto esp = vibeqc::integrals::build_esp_integrals(system, grid.points());
  std::vector<double> raw(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          double one_sided_eri = 0.0;
          for (std::size_t point = 0; point < grid.point_count(); ++point) {
            one_sided_eri += grid.weights()[point] * ao[point * n + i] * ao[point * n + k] *
                             esp.values[(point * n + j) * n + l];
          }
          raw[i * n + j] += density[k * n + l] * one_sided_eri;
        }
      }
    }
  }
  return raw;
}

}  // namespace

int main() {
  try {
    const auto system = h2();
    const std::vector<double> density{0.8, 0.2, 0.2, 0.6};

    const vibeqc::dft::GridSpec tiny{1, 2, 2, 4, 3, 1.0e-12};
    const vibeqc::dft::MolecularGrid tiny_grid(system, tiny);
    const auto rhf =
        vibeqc::dft::build_cosx_reference(system, tiny_grid.points(), tiny_grid.weights(), density,
                                          vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
    const auto brute = brute_discrete_exchange(system, tiny_grid, density);
    require(rhf.nbf == 2 && rhf.npoint == tiny_grid.point_count(),
            "COSX reference dimensions are wrong");
    require(max_abs_diff(rhf.raw_exchange, brute) < 3.0e-15,
            "COSX compact contraction differs from the explicit discrete four-index oracle");
    for (std::size_t i = 0; i < rhf.nbf; ++i) {
      for (std::size_t j = 0; j < rhf.nbf; ++j) {
        const double expected = 0.5 * (brute[i * rhf.nbf + j] + brute[j * rhf.nbf + i]);
        require(std::abs(rhf.exchange[i * rhf.nbf + j] - expected) < 3.0e-15,
                "COSX reference symmetrization is inconsistent");
      }
    }
    double expected_rhf_energy = 0.0;
    for (std::size_t i = 0; i < density.size(); ++i) {
      expected_rhf_energy -= 0.25 * density[i] * rhf.exchange[i];
    }
    require(std::abs(rhf.exchange_energy - expected_rhf_energy) < 2.0e-15,
            "COSX RHF exchange-energy spin factor is wrong");

    const auto point_derivative = vibeqc::dft::build_cosx_point_derivative_reference(
        system, tiny_grid.points(), tiny_grid.weights(), density,
        vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
    require(std::abs(point_derivative.value.exchange_energy - rhf.exchange_energy) < 2.0e-15 &&
                point_derivative.point_gradient.size() == 3 * tiny_grid.point_count(),
            "COSX point derivative changed the underlying discrete energy");
    constexpr double point_step = 1.0e-5;
    const std::size_t checked_points = std::min<std::size_t>(3, tiny_grid.point_count());
    for (std::size_t point = 0; point < checked_points; ++point) {
      for (unsigned axis = 0; axis < 3; ++axis) {
        auto plus = tiny_grid.points();
        auto minus = tiny_grid.points();
        plus[3 * point + axis] += point_step;
        minus[3 * point + axis] -= point_step;
        const auto plus_value =
            vibeqc::dft::build_cosx_reference(system, plus, tiny_grid.weights(), density,
                                              vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
        const auto minus_value =
            vibeqc::dft::build_cosx_reference(system, minus, tiny_grid.weights(), density,
                                              vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
        const double finite_difference =
            (plus_value.exchange_energy - minus_value.exchange_energy) / (2.0 * point_step);
        require(std::abs(finite_difference - point_derivative.point_gradient[3 * point + axis]) <
                    2.0e-8,
                "COSX analytic explicit-point derivative disagrees with finite differences");
      }
    }

    std::vector<double> half_density = density;
    for (double& value : half_density) value *= 0.5;
    const auto spin = vibeqc::dft::build_cosx_reference(
        system, tiny_grid.points(), tiny_grid.weights(), half_density,
        vibeqc::dft::CosxDensityConvention::spin_resolved);
    for (std::size_t i = 0; i < rhf.exchange.size(); ++i) {
      require(std::abs(spin.exchange[i] - 0.5 * rhf.exchange[i]) < 2.0e-15,
              "COSX exchange is not linear in one-spin density");
    }
    require(std::abs(2.0 * spin.exchange_energy - rhf.exchange_energy) < 2.0e-15,
            "equal-spin UHF and RHF COSX exchange energies disagree");

    const std::array<double, 3> probe{0.4, -0.3, 3.1};
    const std::vector<double> probe_xyz{probe[0], probe[1], probe[2]};
    const auto analytic_esp = vibeqc::integrals::build_esp_integrals(system, probe_xyz);
    require(analytic_esp.nbf == 2 && analytic_esp.npoint == 1 && analytic_esp.values.size() == 4,
            "analytic ESP reference dimensions are wrong");
    const auto analytic_esp_derivative =
        vibeqc::integrals::build_esp_integrals_with_probe_derivatives(system, probe_xyz);
    require(max_abs_diff(analytic_esp.values, analytic_esp_derivative.values) < 1.0e-15 &&
                analytic_esp_derivative.probe_derivative.size() == 12,
            "analytic ESP derivative changed the value path");
    constexpr double esp_step = 1.0e-5;
    for (unsigned axis = 0; axis < 3; ++axis) {
      auto plus = probe_xyz;
      auto minus = probe_xyz;
      plus[axis] += esp_step;
      minus[axis] -= esp_step;
      const auto plus_value = vibeqc::integrals::build_esp_integrals(system, plus);
      const auto minus_value = vibeqc::integrals::build_esp_integrals(system, minus);
      for (std::size_t element = 0; element < analytic_esp.values.size(); ++element) {
        const double finite_difference =
            (plus_value.values[element] - minus_value.values[element]) / (2.0 * esp_step);
        require(std::abs(finite_difference -
                         analytic_esp_derivative.probe_derivative[axis * 4 + element]) < 2.0e-8,
                "analytic ESP probe derivative disagrees with finite differences");
      }
    }
    const auto coarse_esp =
        numerical_esp(system, probe, vibeqc::dft::GridSpec{1, 20, 10, 20, 3, 1.0e-12});
    const auto fine_esp =
        numerical_esp(system, probe, vibeqc::dft::GridSpec{1, 72, 24, 48, 3, 1.0e-12});
    const double coarse_esp_error = max_abs_diff(coarse_esp, analytic_esp.values);
    const double fine_esp_error = max_abs_diff(fine_esp, analytic_esp.values);
    require(fine_esp_error < coarse_esp_error && fine_esp_error < 2.0e-4,
            "independent numerical ESP quadrature does not converge to the analytic ESP matrix");

    const auto direct = direct_exchange(system, density);
    const vibeqc::dft::MolecularGrid coarse_grid(system,
                                                 vibeqc::dft::GridSpec{1, 12, 8, 16, 3, 1.0e-12});
    const vibeqc::dft::MolecularGrid fine_grid(system,
                                               vibeqc::dft::GridSpec{1, 36, 14, 28, 3, 1.0e-12});
    const auto coarse_cosx = vibeqc::dft::build_cosx_reference(
        system, coarse_grid.points(), coarse_grid.weights(), density,
        vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
    const auto fine_cosx =
        vibeqc::dft::build_cosx_reference(system, fine_grid.points(), fine_grid.weights(), density,
                                          vibeqc::dft::CosxDensityConvention::rhf_spin_summed);
    const double coarse_cosx_error = max_abs_diff(coarse_cosx.exchange, direct);
    const double fine_cosx_error = max_abs_diff(fine_cosx.exchange, direct);
    require(fine_cosx_error < coarse_cosx_error && fine_cosx_error < 2.0e-6,
            "COSX grid refinement does not approach analytic direct exchange");

    bool malformed_probe_rejected = false;
    try {
      (void)vibeqc::integrals::build_esp_integrals(system, std::vector<double>{0.0, 1.0});
    } catch (const std::invalid_argument&) {
      malformed_probe_rejected = true;
    }
    require(malformed_probe_rejected, "ESP reference accepted incomplete xyz coordinates");

    bool fitted_reference_rejected = false;
    try {
      auto spec = vibeqc::dft::CosxReferenceSpec{};
      spec.overlap_fitting = true;
      (void)vibeqc::dft::build_cosx_reference(
          system, tiny_grid.points(), tiny_grid.weights(), density,
          vibeqc::dft::CosxDensityConvention::rhf_spin_summed, spec);
    } catch (const std::invalid_argument&) {
      fitted_reference_rejected = true;
    }
    require(fitted_reference_rejected,
            "COSX reference v1 silently accepted an unversioned fitting approximation");

    std::cout
        << "COSX v1 discrete reference, ESP oracle, spin factors and grid convergence passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
