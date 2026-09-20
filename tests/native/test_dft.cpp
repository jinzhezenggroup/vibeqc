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
#include "xc_cpu_generated.hpp"

#if VIBEQC_HAS_CUDA
extern "C" void grid_cuda_fail_next_allocation_for_test_v1();
extern "C" void grid_cuda_fail_next_host_allocation_for_test_v1();
extern "C" int grid_cuda_create_v1(int, int, int, const std::size_t*, const double*, std::size_t,
                                   unsigned, std::size_t, void**, char*, std::size_t);
#endif

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
  if (vibeqc::molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}
}  // namespace

int main() {
  try {
    const auto b3_point = vibeqc::dft::generated::b3lyp_polarized(0.3, 0.2, 0.015, 0.003, 0.01);
    // Pinned independently with PySCF 2.14.0 / Libxc 7.0.0 B3LYP.
    const std::array<double, 6> b3_oracle{-0.26232290280116083,  -0.7122471845474007,
                                          -0.647404446702975,    -0.014654598299102754,
                                          0.0016566689256405436, -0.022748740576278376};
    require(std::abs(b3_point.energy_density - b3_oracle[0]) < 2e-13,
            "B3LYP semilocal scalar differs from pinned Libxc oracle");
    for (std::size_t i = 0; i < 5; ++i)
      require(std::abs(b3_point.feature_derivative[i] - b3_oracle[i + 1]) < 2e-13,
              "B3LYP semilocal derivative differs from pinned Libxc oracle");
    require(std::abs(vibeqc::dft::generated::kB3lypExactExchange - 0.2) < 1e-16,
            "B3LYP generated exact-exchange fraction disagrees with MethodIR");

    const auto cam_point =
        vibeqc::dft::generated::cam_b3lyp_polarized(0.3, 0.2, 0.015, 0.003, 0.01);
    const std::array<double, 6> cam_oracle{-0.22534883092171914, -0.6376091098611569,
                                           -0.5721238021867927,  -0.012862968002481867,
                                           0.001656668925640544, -0.01898573870842516};
    require(std::abs(cam_point.energy_density - cam_oracle[0]) < 2e-13,
            "CAM semilocal scalar differs from pinned Libxc oracle");
    for (std::size_t i = 0; i < 5; ++i)
      require(std::abs(cam_point.feature_derivative[i] - cam_oracle[i + 1]) < 2e-13,
              "CAM semilocal derivative differs from pinned Libxc oracle");

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
#if VIBEQC_HAS_CUDA
    // The compiler grid boundary survives the resident-KS replacement. Its
    // host/device failures must preserve OOM and clear unpublished ownership.
    // These injected failures precede device setup; KS arena OOM behavior is
    // separately exercised through the public resource-budget tests.
    const std::size_t dimensions[]{basis.natom, basis.nprimitive, basis.nao};
    for (auto fail : {grid_cuda_fail_next_allocation_for_test_v1,
                      grid_cuda_fail_next_host_allocation_for_test_v1}) {
      char error[256]{};
      void* failed_owner = error;
      fail();
      require(grid_cuda_create_v1(0, 8, 0, dimensions, basis.packed.data(), 7, 0, 0, &failed_owner,
                                  error, sizeof(error)) == VIBEQC_STATUS_OUT_OF_MEMORY &&
                  failed_owner == nullptr,
              "compiler CUDA grid allocation failure lost its status or output invariant");
    }
#endif
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
    const std::vector<double> tiny{std::numeric_limits<double>::denorm_min(), 0.0, 0.0, 0.0};
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
            "PBE scaled-v1 domain changed an interior point result");

    // Values from tests/reference_data/xc_integration/h2.npz, generated by
    // PySCF 2.14.0 / Libxc 7.0.0 on this exact 32-point GridSpec v1 grid.
    const std::vector<double> reference_density{1.2007575959127958, 0.011302590336886256,
                                                0.011302590336886256, 0.462144452714005};
    const vibeqc::dft::GridSpec cam_interior_spec{1, 1, 2, 4, 3, 1.0e-12};
    const vibeqc::dft::MolecularGrid cam_interior_grid(system, cam_interior_spec);
    const auto b3 =
        vibeqc::dft::integrate_b3lyp_rks(basis, cam_interior_grid, reference_density, 7);
    require(std::isfinite(b3.energy) && b3.potential.size() == reference_density.size(),
            "B3LYP MethodIR semilocal integration is invalid on the audited grid");
    for (double step : {1.0e-5, 3.0e-6}) {
      std::vector<double> plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < reference_density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_b3lyp_rks(basis, cam_interior_grid, plus, 7).energy -
           vibeqc::dft::integrate_b3lyp_rks(basis, cam_interior_grid, minus, 7).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) trace += b3.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 3.0e-6,
              "B3LYP semilocal potential violates delta E = Tr(V delta D)");
    }

    const auto cam =
        vibeqc::dft::integrate_cam_b3lyp_rks(basis, cam_interior_grid, reference_density, 7);
    require(std::isfinite(cam.energy) && cam.potential.size() == reference_density.size(),
            "CAM-B3LYP MethodIR semilocal integration is invalid on the audited grid");
    for (double step : {1.0e-5, 3.0e-6}) {
      std::vector<double> plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < reference_density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_cam_b3lyp_rks(basis, cam_interior_grid, plus, 7).energy -
           vibeqc::dft::integrate_cam_b3lyp_rks(basis, cam_interior_grid, minus, 7).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) trace += cam.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 3.0e-6,
              "CAM-B3LYP semilocal potential violates delta E = Tr(V delta D)");
    }
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
            "PBE scaled-v1 domain produced a nonfinite default-grid result");
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
              "PBE scaled-v1 domain violates delta E = Tr(V delta D)");
    }
    const auto reference_pbe = vibeqc::dft::integrate_pbe_rks(basis, grid, reference_density, 7);
    require(std::abs(reference_pbe.energy - (-0.23010116952210713)) < 2.0e-10,
            "native PBE energy differs from the independent fixture");
    require(std::abs(reference_pbe.electrons - 0.5309124083146364) < 2.0e-10,
            "native PBE electron integral differs from the independent fixture");
    const std::array<double, 4> reference_potential{-0.1903934858683413, -0.07901244667607567,
                                                    -0.07901244667607567, -0.15198583764323192};
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
    const auto pbe_uks =
        vibeqc::dft::integrate_pbe_uks_with_tail(basis, grid, alpha_density, beta_density, 7);
    require(std::abs(pbe_uks.energy - pbe.energy) < 2.0e-12 &&
                std::abs(pbe_uks.electrons[0] - 0.5 * pbe.electrons) < 2.0e-12 &&
                std::abs(pbe_uks.electrons[1] - 0.5 * pbe.electrons) < 2.0e-12,
            "polarized PBE equal-spin energy or electron split differs from RKS");
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < pbe.potential.size(); ++i)
        require(std::abs(pbe_uks.potential[spin][i] - pbe.potential[i]) < 2.0e-11,
                "polarized PBE equal-spin potential differs from RKS");
    for (std::size_t spin = 0; spin < 2; ++spin) {
      for (double step : {1.0e-5, 3.0e-6}) {
        auto plus_alpha = alpha_density, minus_alpha = alpha_density;
        auto plus_beta = beta_density, minus_beta = beta_density;
        auto& plus = spin == 0 ? plus_alpha : plus_beta;
        auto& minus = spin == 0 ? minus_alpha : minus_beta;
        for (std::size_t i = 0; i < direction.size(); ++i) {
          plus[i] += step * direction[i];
          minus[i] -= step * direction[i];
        }
        const double finite_difference =
            (vibeqc::dft::integrate_pbe_uks_with_tail(basis, grid, plus_alpha, plus_beta, 9)
                 .energy -
             vibeqc::dft::integrate_pbe_uks_with_tail(basis, grid, minus_alpha, minus_beta, 9)
                 .energy) /
            (2.0 * step);
        double trace = 0.0;
        for (std::size_t i = 0; i < direction.size(); ++i)
          trace += pbe_uks.potential[spin][i] * direction[i];
        require(std::abs(finite_difference - trace) < 2.0e-6,
                "polarized PBE potential violates delta E = Tr(V_s delta D_s)");
      }
    }
    const auto fully_polarized_lda =
        vibeqc::dft::integrate_lda_xc_pw_uks(basis, default_grid, reference_density, zero, 113);
    const auto fully_polarized_pbe =
        vibeqc::dft::integrate_pbe_uks_with_tail(basis, default_grid, reference_density, zero, 113);
    // A vanishing minority density must approach the same functional as an
    // exactly empty spin. The former PBE-to-LDA dispatch caused a finite jump
    // here, despite both spin densities and gradients varying continuously.
    const auto polarized_small =
        vibeqc::dft::integrate_pbe_uks_with_tail(basis, grid, density, zero, 7);
    for (double fraction : {1e-15, 1e-12, 0.999e-10, 1.001e-10}) {
      auto minority = density;
      for (double& value : minority) value *= fraction;
      const auto nearby =
          vibeqc::dft::integrate_pbe_uks_with_tail(basis, grid, density, minority, 7);
      require(std::abs(nearby.energy - polarized_small.energy) < 1e-8,
              "PBE energy jumps between empty and nearly empty spin densities");
      for (std::size_t i = 0; i < density.size(); ++i)
        require(std::abs(nearby.potential[0][i] - polarized_small.potential[0][i]) < 1e-8,
                "PBE majority-spin potential jumps across the minority-spin boundary");
    }
    for (const auto* integral : {&fully_polarized_lda, &fully_polarized_pbe}) {
      require(std::isfinite(integral->energy) &&
                  std::all_of(integral->potential[0].begin(), integral->potential[0].end(),
                              [](double value) { return std::isfinite(value); }) &&
                  std::all_of(integral->potential[1].begin(), integral->potential[1].end(),
                              [](double value) { return std::isfinite(value); }),
              "polarized XC tail is nonfinite at complete spin polarization");
    }
    for (double step : {1.0e-5, 3.0e-6}) {
      auto plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < direction.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_pbe_uks_with_tail(basis, default_grid, plus, zero, 113).energy -
           vibeqc::dft::integrate_pbe_uks_with_tail(basis, default_grid, minus, zero, 113).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i)
        trace += fully_polarized_pbe.potential[0][i] * direction[i];
      require(std::abs(finite_difference - trace) < 2.0e-6,
              "fully polarized PBE active-spin potential violates delta E = Tr(Va delta Da)");
    }
    for (const auto integrate :
         {vibeqc::dft::integrate_lda_xc_pw_uks, vibeqc::dft::integrate_pbe_uks}) {
      // Unequal, nondiagonal spin states detect wrong total-density Hartree
      // conventions and spin/symmetric-matrix factors that singlets conceal.
      const std::vector<double> a{0.8, 0.12, 0.12, 0.4}, b{0.3, -0.05, -0.05, 0.2};
      const auto value = integrate(basis, grid, a, b, 7);
      const auto swapped = integrate(basis, grid, b, a, 11);
      require(std::abs(value.energy - swapped.energy) < 2e-14,
              "XC energy is not invariant under exchanging spins");
      for (unsigned spin = 0; spin < 2; ++spin) {
        for (std::size_t i = 0; i < 4; ++i)
          require(std::abs(value.potential[spin][i] - swapped.potential[1 - spin][i]) < 2e-14,
                  "XC spin potentials did not exchange independently");
        for (const auto delta : {direction, std::vector<double>{0, 0.2, 0.2, 0}}) {
          auto plus_a = a, minus_a = a, plus_b = b, minus_b = b;
          constexpr double step = 1e-5;
          for (std::size_t i = 0; i < 4; ++i) {
            (spin == 0 ? plus_a : plus_b)[i] += step * delta[i];
            (spin == 0 ? minus_a : minus_b)[i] -= step * delta[i];
          }
          const double fd = (integrate(basis, grid, plus_a, plus_b, 9).energy -
                             integrate(basis, grid, minus_a, minus_b, 13).energy) /
                            (2 * step);
          double trace = 0;
          for (std::size_t i = 0; i < 4; ++i) trace += value.potential[spin][i] * delta[i];
          require(std::abs(fd - trace) < 2e-9, "UKS discrete variational identity failed");
          require(std::abs(fd - 0.5 * trace) > 1e-4 && std::abs(fd - 2.0 * trace) > 1e-4,
                  "UKS derivative test cannot detect wrong symmetric/spin factors");
        }
      }
      const auto empty = integrate(basis, grid, a, zero, 7);
      require(empty.electrons[1] == 0.0 && std::isfinite(empty.energy),
              "empty spin channel was rejected or assigned electrons");
    }
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
