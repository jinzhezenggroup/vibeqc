#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/xc.hpp"
#include "dft/xc_point.hpp"
#include "molecule/basis.hpp"
#include "xc_cpu_generated.hpp"

#if VIBEQC_HAS_CUDA
extern "C" void grid_cuda_fail_next_allocation_for_test_v1();
extern "C" void grid_cuda_fail_next_host_allocation_for_test_v1();
extern "C" int grid_cuda_create_v1(int, int, int, const std::size_t*, const double*, std::size_t,
                                   unsigned, std::size_t, void**, char*, std::size_t);
#endif

namespace {
// Canonical graph features include tau even for GGA. Lowering must preserve
// the five-derivative GGA ABI while MGGA exposes all seven active derivatives.
using namespace vibeqc::dft::generated;
static_assert(std::extent_v<decltype(B3lypPolarizedValue::feature_derivative)> == 5);
static_assert(std::extent_v<decltype(CamB3lypPolarizedValue::feature_derivative)> == 5);
static_assert(std::extent_v<decltype(Pw91PolarizedValue::feature_derivative)> == 5);
static_assert(std::extent_v<decltype(ScanPolarizedValue::feature_derivative)> == 7);
static_assert(std::extent_v<decltype(R2scanPolarizedValue::feature_derivative)> == 7);
static_assert(std::extent_v<decltype(Wb97mvPolarizedValue::feature_derivative)> == 7);

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

    const auto wb_point =
        vibeqc::dft::generated::wb97mv_polarized(0.3, 0.2, 0.015, 0.003, 0.01, 0.08, 0.05);
    const std::array<double, 8> wb_oracle{
        -0.20814586702136345, -0.5948515352856814,  -0.5697576268765359, -0.01082457361721382, 0.0,
        -0.01425699937015483, -0.05540206273456351, -0.05804981432341172};
    require(std::abs(wb_point.energy_density - wb_oracle[0]) < 3e-12,
            "omegaB97M-V semilocal scalar differs from pinned Libxc oracle");
    for (std::size_t i = 0; i < 7; ++i)
      require(std::abs(wb_point.feature_derivative[i] - wb_oracle[i + 1]) < 3e-11,
              "omegaB97M-V semilocal derivative differs from pinned Libxc oracle");

    const auto wb_tail = vibeqc::dft::generated::wb97mv_polarized(5e-11, 5e-11, 2.5e-31, 2.5e-31,
                                                                  2.5e-31, 5e-13, 5e-13);
    require(std::isfinite(wb_tail.energy_density),
            "omegaB97M-V large-a production energy is nonfinite");
    for (double derivative : wb_tail.feature_derivative)
      require(std::isfinite(derivative), "omegaB97M-V large-a production derivative is nonfinite");

    require(std::abs(vibeqc::dft::generated::kWb97mvDensityThreshold - 1.0e-13) < 1e-30 &&
                std::abs(vibeqc::dft::generated::kWb97mvTauThreshold - 1.0e-20) < 1e-37 &&
                std::abs(vibeqc::dft::generated::kWb97mvSmoothLrCutoff - 1.35) < 1e-15 &&
                vibeqc::dft::generated::kWb97mvSmoothLrOrder == 16,
            "omegaB97M-V generated work_mgga/smooth-LR policy changed");

    const double wb_screened_rho[2]{4.0e-14, 5.0e-14};
    const double wb_screened_gradient[2][3]{{1.0e-20, 0.0, 0.0}, {0.0, 1.0e-20, 0.0}};
    const double wb_screened_tau[2]{1.0e-21, 2.0e-21};
    const auto wb_screened =
        vibeqc::dft::evaluate_wb97mv_point(wb_screened_rho, wb_screened_gradient, wb_screened_tau);
    require(wb_screened.energy == 0.0 && wb_screened.rho[0] == 0.0 && wb_screened.rho[1] == 0.0,
            "omegaB97M-V work_mgga total-density screen is not exact zero");

    const double wb_minority_rho[2]{0.0, 1.0e-4};
    const double wb_minority_gradient[2][3]{};
    const double wb_minority_tau[2]{0.0, 1.0e-5};
    const auto wb_minority =
        vibeqc::dft::evaluate_wb97mv_point(wb_minority_rho, wb_minority_gradient, wb_minority_tau);
    require(std::isfinite(wb_minority.energy) && std::isfinite(wb_minority.rho[0]) &&
                std::isfinite(wb_minority.rho[1]),
            "omegaB97M-V work_mgga spin-feature floors are nonfinite");

    const double wb_vacuum_rho[2]{};
    const double wb_vacuum_gradient[2][3]{};
    const double wb_vacuum_tau[2]{};
    const auto wb_vacuum =
        vibeqc::dft::evaluate_wb97mv_point(wb_vacuum_rho, wb_vacuum_gradient, wb_vacuum_tau);
    require(wb_vacuum.energy == 0.0 && wb_vacuum.rho[0] == 0.0 && wb_vacuum.rho[1] == 0.0 &&
                wb_vacuum.kinetic[0] == 0.0 && wb_vacuum.kinetic[1] == 0.0,
            "omegaB97M-V exact vacuum is not canonical zero");
    const auto pw91_point = vibeqc::dft::generated::pw91_polarized(0.3, 0.2, 0.015, 0.003, 0.01);
    // Pinned independently with PySCF 2.14.0 / Libxc 7.0.0 PW91.
    const std::array<double, 6> pw91_oracle{-0.3282121838488419, -0.8954942661475697,
                                            -0.8067117696168564, -0.007339659490082176,
                                            0.02173376863067051, -0.02517931481430781};
    require(std::abs(pw91_point.energy_density - pw91_oracle[0]) < 2e-13,
            "PW91 generic-GGA scalar differs from pinned Libxc oracle");
    for (std::size_t i = 0; i < 5; ++i)
      require(std::abs(pw91_point.feature_derivative[i] - pw91_oracle[i + 1]) < 2e-13,
              "PW91 generic-GGA derivative differs from pinned Libxc oracle");

    const auto scan_point =
        vibeqc::dft::generated::scan_polarized(0.55, 0.25, 0.025, 0.006, 0.018, 0.3, 0.16);
    // Sum of pinned independent Libxc 7.0.0 MGGA_X_SCAN + MGGA_C_SCAN bulk fixtures.
    const std::array<double, 8> scan_oracle{
        -0.6716431723099776,   -1.2121359952027473,   -0.975507137482233,  -0.00557843033514503,
        0.0063013418044912395, -0.029931891669623255, 0.02202134320410421, 0.04542853626023215};
    require(std::abs(scan_point.energy_density - scan_oracle[0]) < 2e-13,
            "SCAN generic-MGGA scalar differs from pinned Libxc oracle");
    for (std::size_t i = 0; i < 7; ++i)
      require(std::abs(scan_point.feature_derivative[i] - scan_oracle[i + 1]) < 2e-13,
              "SCAN generic-MGGA derivative differs from pinned Libxc oracle");

    std::ifstream xc_fixture(VIBEQC_SOURCE_DIR "/tests/data/xc/scf_domain.tsv");
    require(static_cast<bool>(xc_fixture), "missing independent XC SCF-domain fixture");
    std::string xc_line;
    std::size_t lda_rows = 0, pbe_rows = 0;
    while (std::getline(xc_fixture, xc_line)) {
      if (xc_line.empty() || xc_line[0] == '#') continue;
      std::istringstream row(xc_line);
      int pbe = 0, oracle = 0;
      double rho[2]{}, gradient[2][3]{}, expected[9]{};
      row >> pbe >> oracle >> rho[0] >> rho[1];
      for (auto& spin : gradient)
        for (double& component : spin) row >> component;
      for (double& component : expected) row >> component;
      require(static_cast<bool>(row), "malformed XC SCF-domain fixture");
      if (pbe == 0) {
        const auto value = vibeqc::dft::generated::lda_xc_pw_polarized_production(rho[0], rho[1]);
        const double actual[]{value.energy_density, value.feature_derivative[0],
                              value.feature_derivative[1]};
        for (unsigned i = 0; i < 3; ++i) {
          const double tolerance = 5.0e-10 * std::abs(expected[i]) + 1.0e-322;
          require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= tolerance,
                  "compiler-owned polarized LDA differs from independent SCF-domain reference");
        }
        for (unsigned i = 3; i < 9; ++i)
          require(expected[i] == 0.0, "independent LDA fixture has a gradient coefficient");
        ++lda_rows;
      } else {
        const auto value =
            vibeqc::dft::generated::pbe_polarized_production(rho[0], rho[1], gradient);
        const double actual[]{
            value.energy_density, value.rho[0],         value.rho[1],
            value.gradient[0][0], value.gradient[0][1], value.gradient[0][2],
            value.gradient[1][0], value.gradient[1][1], value.gradient[1][2],
        };
        for (unsigned i = 0; i < 9; ++i) {
          const double tolerance = 5.0e-10 * std::abs(expected[i]) + 1.0e-322;
          require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= tolerance,
                  "compiler-owned polarized PBE differs from independent SCF-domain reference");
        }
        ++pbe_rows;
      }
    }
    require(lda_rows == 36, "incomplete independent polarized LDA reference coverage");
    require(pbe_rows == 61, "incomplete independent polarized PBE reference coverage");
    const auto lda_vacuum = vibeqc::dft::generated::lda_xc_pw_polarized_production(0.0, 0.0);
    require(lda_vacuum.energy_density == 0.0 && lda_vacuum.feature_derivative[0] == 0.0 &&
                lda_vacuum.feature_derivative[1] == 0.0,
            "compiler-owned polarized LDA vacuum limit is wrong");
    const double zero_gradient[2][3]{};
    const auto generated_pbe_vacuum =
        vibeqc::dft::generated::pbe_polarized_production(0.0, 0.0, zero_gradient);
    require(generated_pbe_vacuum.energy_density == 0.0 && generated_pbe_vacuum.rho[0] == 0.0 &&
                generated_pbe_vacuum.rho[1] == 0.0,
            "compiler-owned polarized PBE vacuum limit is wrong");

    {
      const double rho[2]{0.3, 0.2};
      const double gradient[2][3]{{0.11, -0.07, 0.03}, {-0.02, 0.09, -0.04}};
      for (const auto scales :
           {std::array<double, 2>{0.75, 1.0}, std::array<double, 2>{0.37, 0.61}}) {
        const auto expected =
            vibeqc::dft::point::evaluate(true, rho, gradient, scales[0], scales[1]);
        const auto actual = vibeqc::dft::generated::pbe_polarized_production(
            rho[0], rho[1], gradient, scales[0], scales[1]);
        const double generated[]{
            actual.energy_density, actual.rho[0],         actual.rho[1],
            actual.gradient[0][0], actual.gradient[0][1], actual.gradient[0][2],
            actual.gradient[1][0], actual.gradient[1][1], actual.gradient[1][2],
        };
        const double reference[]{
            expected.energy,         expected.rho[0],         expected.rho[1],
            expected.gradient[0][0], expected.gradient[0][1], expected.gradient[0][2],
            expected.gradient[1][0], expected.gradient[1][1], expected.gradient[1][2],
        };
        require(expected.valid, "independent scaled PBE oracle rejected an interior point");
        for (unsigned i = 0; i < 9; ++i)
          require(std::abs(generated[i] - reference[i]) <
                      3.0e-13 * std::max(1.0, std::abs(reference[i])),
                  "generated PBE X/C scaling differs from the independent point oracle");
      }
    }

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
    // The optimized triangular contraction must retain the established
    // tolerance for slightly asymmetric caller storage by consuming D_uv+D_vu,
    // rather than silently trusting only one triangle.
    const std::vector<double> near_symmetric_density{0.8, 0.2 + 5.0e-12,
                                                     0.2 - 5.0e-12, 0.6};
    const auto near_symmetric =
        vibeqc::dft::integrate_pbe_rks(basis, grid, near_symmetric_density, 5);
    require(std::abs(near_symmetric.energy - pbe.energy) < 2.0e-14,
            "triangular PBE contraction changed accepted near-symmetric density semantics");
    for (std::size_t i = 0; i < pbe.potential.size(); ++i)
      require(std::abs(near_symmetric.potential[i] - pbe.potential[i]) < 2.0e-14,
              "triangular PBE potential changed accepted near-symmetric density semantics");
    // #237 Slice A: delta-D may be signed/indefinite. Contract its
    // *linear* rho/grad-rho features separately, then recompute nonlinear PBE
    // from the reconstructed total features. The exact incremental result must
    // match a full target-density build, including exact Vxc differences.
    const std::vector<double> incremental_delta{3.0e-3, -4.0e-3, -4.0e-3, -2.0e-3};
    std::vector<double> incremental_target = density;
    for (std::size_t i = 0; i < density.size(); ++i) incremental_target[i] += incremental_delta[i];
    const auto incremental = vibeqc::dft::integrate_pbe_rks_incremental_exact(basis, grid, density,
                                                                              incremental_delta, 5);
    const auto incremental_full =
        vibeqc::dft::integrate_pbe_rks_with_tail(basis, grid, incremental_target, 5);
    const auto incremental_anchor =
        vibeqc::dft::integrate_pbe_rks_with_tail(basis, grid, density, 5);
    require(std::abs(incremental.total.energy - incremental_full.energy) < 2.0e-14 &&
                std::abs(incremental.total.electrons - incremental_full.electrons) < 2.0e-14,
            "exact incremental PBE total differs from a full target build");
    require(std::abs(incremental.energy_difference -
                     (incremental_full.energy - incremental_anchor.energy)) < 2.0e-14,
            "exact incremental PBE energy difference is not anchor-relative");
    for (std::size_t i = 0; i < incremental_full.potential.size(); ++i) {
      require(std::abs(incremental.total.potential[i] - incremental_full.potential[i]) < 2.0e-14,
              "exact incremental PBE potential differs from a full target build");
      require(std::abs(incremental.potential_difference[i] -
                       (incremental_full.potential[i] - incremental_anchor.potential[i])) < 2.0e-14,
              "exact incremental PBE potential difference is not anchor-relative");
    }
    const std::vector<double> zero_delta(density.size(), 0.0);
    const auto unchanged =
        vibeqc::dft::integrate_pbe_rks_incremental_exact(basis, grid, density, zero_delta, 3);
    require(std::abs(unchanged.energy_difference) < 2.0e-14 &&
                std::all_of(unchanged.potential_difference.begin(),
                            unchanged.potential_difference.end(),
                            [](double value) { return std::abs(value) < 2.0e-14; }),
            "zero delta-D did not cancel exactly in incremental PBE");

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

    const auto pw91 =
        vibeqc::dft::integrate_pw91_rks(basis, cam_interior_grid, reference_density, 7);
    require(std::isfinite(pw91.energy) && pw91.potential.size() == reference_density.size(),
            "PW91 generic-GGA native integration is invalid on the audited interior grid");
    for (double step : {1.0e-5, 3.0e-6}) {
      std::vector<double> plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < reference_density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_pw91_rks(basis, cam_interior_grid, plus, 7).energy -
           vibeqc::dft::integrate_pw91_rks(basis, cam_interior_grid, minus, 7).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) trace += pw91.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 3.0e-6,
              "PW91 generic-GGA potential violates delta E = Tr(V delta D)");
    }
    std::vector<double> pw91_alpha(reference_density.size()), pw91_beta(reference_density.size());
    for (std::size_t i = 0; i < reference_density.size(); ++i)
      pw91_alpha[i] = pw91_beta[i] = 0.5 * reference_density[i];
    const auto pw91_uks =
        vibeqc::dft::integrate_pw91_uks(basis, cam_interior_grid, pw91_alpha, pw91_beta, 7);
    require(std::abs(pw91_uks.energy - pw91.energy) < 2.0e-12,
            "PW91 generic-GGA equal-spin UKS energy differs from RKS");
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < pw91.potential.size(); ++i)
        require(std::abs(pw91_uks.potential[spin][i] - pw91.potential[i]) < 2.0e-11,
                "PW91 generic-GGA equal-spin UKS potential differs from RKS");

    const auto r2scan =
        vibeqc::dft::integrate_r2scan_rks(basis, cam_interior_grid, reference_density, 7);
    require(std::isfinite(r2scan.energy) && r2scan.potential.size() == reference_density.size(),
            "r2SCAN generic-semilo native integration is invalid on the audited grid");
    for (double step : {1.0e-5, 3.0e-6}) {
      std::vector<double> plus = reference_density, minus = reference_density;
      for (std::size_t i = 0; i < reference_density.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_r2scan_rks(basis, cam_interior_grid, plus, 7).energy -
           vibeqc::dft::integrate_r2scan_rks(basis, cam_interior_grid, minus, 7).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i)
        trace += r2scan.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 3.0e-6,
              "r2SCAN generic-semilo potential violates delta E = Tr(V delta D)");
    }
    std::vector<double> r2scan_alpha(reference_density.size()),
        r2scan_beta(reference_density.size());
    for (std::size_t i = 0; i < reference_density.size(); ++i)
      r2scan_alpha[i] = r2scan_beta[i] = 0.5 * reference_density[i];
    const auto r2scan_uks =
        vibeqc::dft::integrate_r2scan_uks(basis, cam_interior_grid, r2scan_alpha, r2scan_beta, 7);
    require(std::abs(r2scan_uks.energy - r2scan.energy) < 2.0e-12,
            "r2SCAN generic-semilo equal-spin UKS energy differs from RKS");
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < r2scan.potential.size(); ++i)
        require(std::abs(r2scan_uks.potential[spin][i] - r2scan.potential[i]) < 2.0e-11,
                "r2SCAN generic-semilo equal-spin UKS potential differs from RKS");
    // Unequal spins exercise the independent vtau/2 channels and ragged AO tiles.
    for (std::size_t i = 0; i < reference_density.size(); ++i) {
      r2scan_alpha[i] = 0.7 * reference_density[i];
      r2scan_beta[i] = 0.3 * reference_density[i];
    }
    const auto open_r2scan =
        vibeqc::dft::integrate_r2scan_uks(basis, cam_interior_grid, r2scan_alpha, r2scan_beta, 3);
    const auto whole_r2scan = vibeqc::dft::integrate_r2scan_uks(
        basis, cam_interior_grid, r2scan_alpha, r2scan_beta, cam_interior_grid.point_count());
    require(std::abs(open_r2scan.energy - whole_r2scan.energy) < 2.0e-13,
            "r2SCAN semilocal energy depends on AO tile partition");
    for (unsigned spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < reference_density.size(); ++i)
        require(
            std::abs(open_r2scan.potential[spin][i] - whole_r2scan.potential[spin][i]) < 2.0e-12,
            "r2SCAN unequal-spin potential depends on AO tile partition");
    for (double step : {1.0e-5, 3.0e-6}) {
      auto ap = r2scan_alpha, am = r2scan_alpha, bp = r2scan_beta, bm = r2scan_beta;
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) {
        ap[i] += step * direction[i];
        am[i] -= step * direction[i];
        bp[i] -= 0.4 * step * direction[i];
        bm[i] += 0.4 * step * direction[i];
        trace += (open_r2scan.potential[0][i] - 0.4 * open_r2scan.potential[1][i]) * direction[i];
      }
      const double difference =
          (vibeqc::dft::integrate_r2scan_uks(basis, cam_interior_grid, ap, bp, 3).energy -
           vibeqc::dft::integrate_r2scan_uks(basis, cam_interior_grid, am, bm, 3).energy) /
          (2.0 * step);
      require(std::isfinite(difference) && std::abs(difference - trace) < 3.0e-6,
              "r2SCAN unequal-spin potential violates delta E = Tr(Va dDa + Vb dDb)");
    }

    // A genuine unequal-spin perturbation must use both independent Vxc blocks.
    for (std::size_t i = 0; i < reference_density.size(); ++i) {
      pw91_alpha[i] = 0.7 * reference_density[i];
      pw91_beta[i] = 0.3 * reference_density[i];
    }
    const auto open_pw91 =
        vibeqc::dft::integrate_pw91_uks(basis, cam_interior_grid, pw91_alpha, pw91_beta, 3);
    for (double step : {1.0e-5, 3.0e-6}) {
      auto ap = pw91_alpha, am = pw91_alpha, bp = pw91_beta, bm = pw91_beta;
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i) {
        ap[i] += step * direction[i];
        am[i] -= step * direction[i];
        bp[i] -= 0.4 * step * direction[i];
        bm[i] += 0.4 * step * direction[i];
        trace += (open_pw91.potential[0][i] - 0.4 * open_pw91.potential[1][i]) * direction[i];
      }
      const double difference =
          (vibeqc::dft::integrate_pw91_uks(basis, cam_interior_grid, ap, bp, 3).energy -
           vibeqc::dft::integrate_pw91_uks(basis, cam_interior_grid, am, bm, 3).energy) /
          (2.0 * step);
      require(std::isfinite(difference) && std::abs(difference - trace) < 3.0e-6,
              "PW91 unequal-spin potential violates delta E = Tr(Va dDa + Vb dDb)");
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
    const auto wb_rks = vibeqc::dft::integrate_wb97mv_rks(basis, grid, density, 7);
    const auto wb_uks =
        vibeqc::dft::integrate_wb97mv_uks(basis, grid, alpha_density, beta_density, 7);
    require(std::isfinite(wb_rks.energy) && std::isfinite(wb_uks.energy) &&
                std::abs(wb_uks.energy - wb_rks.energy) < 2.0e-11,
            "omegaB97M-V equal-spin RKS/UKS semilocal energies disagree");
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < wb_rks.potential.size(); ++i)
        require(std::abs(wb_uks.potential[spin][i] - wb_rks.potential[i]) < 3.0e-10,
                "omegaB97M-V equal-spin RKS/UKS semilocal potentials disagree");
    {
      constexpr double step = 3.0e-6;
      auto plus = density, minus = density;
      for (std::size_t i = 0; i < direction.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const double finite_difference =
          (vibeqc::dft::integrate_wb97mv_rks(basis, grid, plus, 7).energy -
           vibeqc::dft::integrate_wb97mv_rks(basis, grid, minus, 7).energy) /
          (2.0 * step);
      double trace = 0.0;
      for (std::size_t i = 0; i < direction.size(); ++i)
        trace += wb_rks.potential[i] * direction[i];
      require(std::abs(finite_difference - trace) < 4.0e-6,
              "omegaB97M-V semilocal potential violates delta E = Tr(V delta D)");
    }

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
