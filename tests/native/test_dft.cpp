#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "molecule/basis.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.4}}};
  system.shells = {
      {0, 0, {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423},
              {0.168855404, 0.4446345422}}},
      {1, 0, {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423},
              {0.168855404, 0.4446345422}}},
  };
  std::string detail;
  if (vibeqc::molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}
}  // namespace

int main() {
  try {
    const auto system = h2();
    const vibeqc::dft::GridSpec small{1, 2, 2, 4, 3, 1.0e-12};
    const vibeqc::dft::MolecularGrid grid(system, small);
    require(grid.point_count() == 32, "small DFT grid point count is wrong");
    const std::array<double, 3> first{0.21877959948235706, 0.0, -0.15470053837925155};
    for (unsigned axis = 0; axis < 3; ++axis)
      require(std::abs(grid.points()[axis] - first[axis]) < 3.0e-15,
              "small DFT grid ordering/value differs from GridSpec v1");
    require(std::abs(grid.weights()[0] - 0.09065640301154727) < 3.0e-14,
            "small DFT grid Becke weight differs from GridSpec v1");

    const vibeqc::dft::AoBasis basis(system);
    const std::vector<double> density{0.8, 0.2, 0.2, 0.6};
    const auto reference = vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, density, 7);
    require(std::isfinite(reference.energy) && reference.energy < 0.0,
            "LDA fixed-density energy is invalid");
    require(reference.potential.size() == 4 && reference.points == 32,
            "LDA fixed-density result dimensions are wrong");
    const std::vector<double> direction{0.3, -0.2, -0.2, 0.1};
    for (double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
      std::vector<double> plus = density, minus = density;
      for (std::size_t i = 0; i < density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, plus, 11).energy -
           vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, minus, 11).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i)
        trace += reference.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 2.0e-7,
              "LDA potential violates delta E = Tr(V delta D)");
    }
    std::vector<double> zero(4);
    const auto vacuum = vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, zero);
    require(vacuum.energy == 0.0 && vacuum.electrons == 0.0 &&
                std::count(vacuum.potential.begin(), vacuum.potential.end(), 0.0) == 4,
            "LDA tail-v1 zero-density limit is wrong");
    const std::vector<double> tiny{std::numeric_limits<double>::denorm_min(), 0.0, 0.0,
                                   0.0};
    const auto tail = vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, tiny);
    require(std::isfinite(tail.energy) &&
                std::all_of(tail.potential.begin(), tail.potential.end(),
                            [](double value) { return std::isfinite(value); }),
            "LDA sixth-root tail algebra is nonfinite at the smallest positive density");
    bool negative_rejected = false;
    try {
      const std::vector<double> negative{-0.8, 0.0, 0.0, -0.6};
      (void)vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, negative);
    } catch (const std::domain_error&) {
      negative_rejected = true;
    }
    require(negative_rejected, "LDA tail-v1 accepted negative real-space density");
    bool nonfinite_rejected = false;
    try {
      auto nonfinite = density;
      nonfinite[0] = std::numeric_limits<double>::infinity();
      (void)vibeqc::dft::integrate_lda_xc_pw_rks(basis, grid, nonfinite);
    } catch (const std::invalid_argument&) {
      nonfinite_rejected = true;
    }
    require(nonfinite_rejected, "LDA tail-v1 accepted a nonfinite AO density matrix");

    const auto pbe = vibeqc::dft::integrate_pbe_rks(basis, grid, density, 5);
    require(std::isfinite(pbe.energy) && pbe.energy < 0.0 &&
                pbe.potential.size() == density.size() && pbe.points == 32,
            "PBE fixed-density integral is invalid");
    for (double value : pbe.potential)
      require(std::isfinite(value), "PBE fixed-density potential is nonfinite");
    const auto pbe_interior_tail =
        vibeqc::dft::integrate_pbe_rks_with_tail(basis, grid, density, 5);
    require(std::abs(pbe_interior_tail.energy - pbe.energy) < 2.0e-14 &&
                pbe_interior_tail.potential == pbe.potential,
            "PBE production tail-v2 changed an interior point result");

    // Values from tests/reference_data/xc_integration/h2.npz, generated by
    // PySCF 2.14.0 / Libxc 7.0.0 on this exact 32-point GridSpec v1 grid.
    const std::vector<double> reference_density{1.2007575959127958, 0.011302590336886256,
                                                0.011302590336886256, 0.462144452714005};
    const vibeqc::dft::MolecularGrid default_grid(system);
    bool default_grid_rejected = false;
    try {
      (void)vibeqc::dft::integrate_pbe_rks(basis, default_grid, reference_density);
    } catch (const std::domain_error&) {
      default_grid_rejected = true;
    }
    require(default_grid_rejected, "PBE tail-v1 accepted an unsupported default-grid tail");
    const auto pbe_tail =
        vibeqc::dft::integrate_pbe_rks_with_tail(basis, default_grid, reference_density, 113);
    require(std::isfinite(pbe_tail.energy) && std::isfinite(pbe_tail.electrons) &&
                std::all_of(pbe_tail.potential.begin(), pbe_tail.potential.end(),
                            [](double value) { return std::isfinite(value); }),
            "PBE production tail-v2 produced a nonfinite default-grid result");
    for (double step : {1.0e-5, 3.0e-6}) {
      std::vector<double> plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < reference_density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_pbe_rks_with_tail(basis, default_grid, plus, 113).energy -
           vibeqc::dft::integrate_pbe_rks_with_tail(basis, default_grid, minus, 113).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i)
        trace += pbe_tail.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 2.0e-6,
              "PBE production tail-v2 violates delta E = Tr(V delta D)");
    }
    const auto reference_pbe = vibeqc::dft::integrate_pbe_rks(basis, grid, reference_density, 7);
    require(std::abs(reference_pbe.energy - (-0.23010116952210713)) < 2.0e-10,
            "native PBE energy differs from the independent fixture");
    require(std::abs(reference_pbe.electrons - 0.5309124083146364) < 2.0e-10,
            "native PBE electron integral differs from the independent fixture");
    const std::array<double, 4> reference_potential{
        -0.1903934858683413, -0.07901244667607567, -0.07901244667607567,
        -0.15198583764323192};
    for (std::size_t i = 0; i < reference_potential.size(); ++i)
      require(std::abs(reference_pbe.potential[i] - reference_potential[i]) < 2.0e-10,
              "native PBE potential differs from the independent fixture");

    std::vector<double> alpha_density(density.size()), beta_density(density.size());
    for (std::size_t i = 0; i < density.size(); ++i) {
      alpha_density[i] = 0.5 * density[i];
      beta_density[i] = 0.5 * density[i];
    }
    const auto lda_uks =
        vibeqc::dft::integrate_lda_xc_pw_uks(basis, grid, alpha_density, beta_density, 7);
    require(std::abs(lda_uks.energy - reference.energy) < 2.0e-12 &&
                std::abs(lda_uks.electrons[0] - 0.5 * reference.electrons) < 2.0e-12 &&
                std::abs(lda_uks.electrons[1] - 0.5 * reference.electrons) < 2.0e-12,
            "polarized LDA fixed-density energy or electron split differs from RKS");
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < reference.potential.size(); ++i)
        require(std::abs(lda_uks.potential[spin][i] - reference.potential[i]) < 2.0e-12,
                "polarized LDA equal-spin potential differs from RKS");
    for (double step : {1.0e-4, 3.0e-5}) {
      std::vector<double> plus = density, minus = density;
      for (std::size_t i = 0; i < density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_pbe_rks(basis, grid, plus, 9).energy -
           vibeqc::dft::integrate_pbe_rks(basis, grid, minus, 9).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) trace += pbe.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 2.0e-6,
              "PBE potential violates delta E = Tr(V delta D)");
    }
    const auto pbe_vacuum = vibeqc::dft::integrate_pbe_rks(basis, grid, zero);
    require(pbe_vacuum.energy == 0.0 && pbe_vacuum.electrons == 0.0 &&
                std::count(pbe_vacuum.potential.begin(), pbe_vacuum.potential.end(), 0.0) == 4,
            "PBE tail-v1 zero-density limit is wrong");
    bool pbe_tail_rejected = false;
    try {
      const std::vector<double> tiny{1.0e-30, 0.0, 0.0, 1.0e-30};
      (void)vibeqc::dft::integrate_pbe_rks(basis, grid, tiny);
    } catch (const std::domain_error&) {
      pbe_tail_rejected = true;
    }
    require(pbe_tail_rejected, "PBE tail-v1 accepted an out-of-domain density");
    std::cout << "DFT GridSpec v1 and generated LDA fixed-density gates passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
