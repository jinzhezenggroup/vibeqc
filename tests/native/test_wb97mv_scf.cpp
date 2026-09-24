#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace vibeqc;

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

core::System input_system(unsigned kind) {
  core::System system;
  if (kind == 2) {
    system.atoms = {{2, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.7}}};
    system.shells = {{0, 0, {{1.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
    system.charge = 1;
  } else {
    const unsigned count = kind == 1 ? 3 : 2;
    system.multiplicity = kind == 1 ? 2 : 1;
    for (unsigned i = 0; i < count; ++i) {
      system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, 1.5 * i}});
      system.shells.push_back({i,
                               0,
                               {{3.425250914, 0.1543289673},
                                {0.6239137298, 0.5353281423},
                                {0.168855404, 0.4446345422}}});
    }
  }
  return system;
}

vibeqc_ks_options options(bool unrestricted) {
  static const std::array<vibeqc_ks_semilocal_component, 2> components{
      {{"MGGA_X_WB97M_V", 1.0}, {"MGGA_C_WB97M_V", 1.0}}};
  static const std::array<vibeqc_ks_exchange_term, 2> restricted_exchange{{
      {VIBEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.075},
      {VIBEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -0.5},
  }};
  static const std::array<vibeqc_ks_exchange_term, 2> unrestricted_exchange{{
      {VIBEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.15},
      {VIBEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -1.0},
  }};
  const auto& exchange = unrestricted ? unrestricted_exchange : restricted_exchange;
  vibeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = VIBEQC_ABI_VERSION;
  ks.scf_domain = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";
  ks.grid_version = 1;
  ks.radial_points = 12;
  ks.angular_polar = 4;
  ks.angular_azimuth = 8;
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.tile_points = 64;
  ks.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.spin_channels = unrestricted ? 2 : 1;
  ks.semilocal_components = components.data();
  ks.semilocal_component_count = components.size();
  ks.semilocal_range_omega = 0.3;
  ks.exchange_terms = exchange.data();
  ks.exchange_term_count = exchange.size();
  ks.has_nonlocal_correlation = 1;
  ks.nonlocal_variant = VIBEQC_NONLOCAL_VV10;
  ks.nonlocal_b = 6.0;
  ks.nonlocal_c = 0.01;
  ks.nonlocal_coefficient = 1.0;
  ks.nonlocal_maximum_bytes = 1 << 24;
  return ks;
}

vibeqc_method_descriptor descriptor(const vibeqc_ks_options& ks) {
  vibeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = VIBEQC_ABI_VERSION;
  method.method = ks.spin_channels == 2 ? VIBEQC_METHOD_WB97M_V_UKS : VIBEQC_METHOD_WB97M_V;
  method.max_iterations = 180;
  method.diis_history = 8;
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-10;
  method.screening_tolerance = 1e-14;
  method.ks_options = &ks;
  return method;
}

const methods::Capabilities capabilities{VIBEQC_METHOD_WB97M_V,
                                         VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                                         VIBEQC_PROPERTY_ENERGY, false, true};

template <class Values>
void array(std::ostream& out, const Values& values) {
  out << '[';
  bool comma = false;
  for (const auto& value : values) {
    if (comma) out << ',';
    out << value;
    comma = true;
  }
  out << ']';
}

void dump(std::ostream& out, const std::string& name, const core::System& input,
          const dft::MolecularGrid& grid, bool unrestricted, const methods::Result& result,
          const dft::VerifiedKsFinalState& state) {
  out << std::setprecision(17) << "{\"name\":\"" << name << "\",\"atoms\":[";
  for (std::size_t i = 0; i < input.atoms.size(); ++i) {
    if (i) out << ',';
    const auto& a = input.atoms[i];
    out << '[' << a.atomic_number << ',';
    array(out, a.position);
    out << ']';
  }
  out << "],\"shells\":[";
  for (std::size_t i = 0; i < input.shells.size(); ++i) {
    if (i) out << ',';
    const auto& shell = input.shells[i];
    out << '[' << shell.atom_index << ',' << shell.angular_momentum << ",[";
    for (std::size_t j = 0; j < shell.primitives.size(); ++j) {
      if (j) out << ',';
      out << '[' << shell.primitives[j].exponent << ',' << shell.primitives[j].coefficient << ']';
    }
    out << "]]";
  }
  out << "],\"charge\":" << input.charge << ",\"spin\":" << input.multiplicity - 1
      << ",\"unrestricted\":" << (unrestricted ? "true" : "false")
      << ",\"energy\":" << result.energy << ",\"residual\":" << *result.physical_residual_rms
      << ",\"density\":[";
  for (std::size_t s = 0; s < state.density.size(); ++s) {
    if (s) out << ',';
    array(out, state.density[s]);
  }
  out << "],\"fock\":[";
  for (std::size_t s = 0; s < state.fock.size(); ++s) {
    if (s) out << ',';
    array(out, state.fock[s]);
  }
  out << "],\"points\":";
  array(out, grid.points());
  out << ",\"weights\":";
  array(out, grid.weights());
  out << "}\n";
}

void nonlocal_density_domain() {
  using namespace dft::nlc;
  auto system = input_system(2);
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS, detail);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 12, 4, 8, 3, 1e-12});
  const std::size_t n = basis.nao, points = grid.point_count();
  vibeqc_status status;
  const Vv10Parameters parameters{Vv10Variant::vv10, 6.0, 0.01, 1.0};
  auto plan =
      Vv10Plan::prepare(VIBEQC_BACKEND_CPU_REFERENCE, -1, static_cast<std::uint32_t>(points), 64,
                        parameters, 1 << 24, detail, status);
  require(plan && status == VIBEQC_STATUS_SUCCESS, detail);
  const std::vector<double> density{1.2, 0.05, 0.05, 0.5};
  const auto screened =
      integrate_vv10_rks(basis, grid, density, *plan, 64, {}, Vv10DensityDomain::MolecularV1);
  // Independent active-set compaction verifies that zero-weight padding removes
  // both pair domains, not just the outer energy quadrature.
  std::vector<double> ao(4 * points * n);
  basis.evaluate(grid.points().data(), points, 1, 0, n, ao.data(), ao.size());
  std::vector<double> xyz, weights, rho, gradients;
  std::vector<std::size_t> active;
  for (std::size_t p = 0; p < points; ++p) {
    const auto* phi = ao.data() + p * n;
    double r = 0.0, g[3]{};
    for (std::size_t mu = 0; mu < n; ++mu) {
      double weighted = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu) weighted += density[mu * n + nu] * phi[nu];
      r += phi[mu] * weighted;
      for (unsigned k = 0; k < 3; ++k) g[k] += 2.0 * ao[((k + 1) * points + p) * n + mu] * weighted;
    }
    if (r >= 1e-8) {
      active.push_back(p);
      xyz.insert(xyz.end(), grid.points().begin() + 3 * p, grid.points().begin() + 3 * p + 3);
      weights.push_back(grid.weights()[p]);
      rho.push_back(r);
      gradients.insert(gradients.end(), g, g + 3);
    }
  }
  require(!active.empty() && active.size() < points,
          "molecular VV10 fixture lacks an active/tail split");
  auto compact =
      Vv10Plan::prepare(VIBEQC_BACKEND_CPU_REFERENCE, -1, static_cast<std::uint32_t>(active.size()),
                        64, parameters, 1 << 24, detail, status);
  require(compact && status == VIBEQC_STATUS_SUCCESS, detail);
  std::vector<double> vrho(active.size()), vsigma(active.size()), potential(n * n);
  double energy = 0.0;
  require(compact->execute(xyz, weights, rho, gradients, energy, vrho, vsigma, {}, {}, detail) ==
              VIBEQC_STATUS_SUCCESS,
          detail);
  for (std::size_t a = 0; a < active.size(); ++a) {
    const auto p = active[a];
    const auto* phi = ao.data() + p * n;
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t nu = 0; nu < n; ++nu) {
        double value = vrho[a] * phi[mu] * phi[nu];
        for (unsigned k = 0; k < 3; ++k) {
          const auto* jet = ao.data() + ((k + 1) * points + p) * n;
          value += 2.0 * vsigma[a] * gradients[3 * a + k] * (jet[mu] * phi[nu] + phi[mu] * jet[nu]);
        }
        potential[mu * n + nu] += weights[a] * value;
      }
  }
  require(std::abs(energy - screened.energy) < 1e-13,
          "VV10 padding differs from compact active-set energy");
  for (std::size_t i = 0; i < potential.size(); ++i)
    require(std::abs(potential[i] - screened.potential[i]) < 1e-13,
            "VV10 padding leaked inactive points into the AO potential");
  const auto vacuum = integrate_vv10_rks(basis, grid, std::vector<double>(n * n), *plan, 64, {},
                                         Vv10DensityDomain::MolecularV1);
  require(vacuum.energy == 0.0 && std::all_of(vacuum.potential.begin(), vacuum.potential.end(),
                                              [](double x) { return x == 0.0; }),
          "screened VV10 vacuum is not zero");
  for (unsigned invalid = 0; invalid < 3; ++invalid) {
    auto bad = density;
    if (invalid == 0)
      for (auto& x : bad) x = -x;
    if (invalid == 1) bad[0] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try {
      (void)integrate_vv10_rks(
          basis, grid, bad, *plan, 64, {},
          invalid == 2 ? static_cast<Vv10DensityDomain>(99) : Vv10DensityDomain::MolecularV1);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "VV10 molecular screening concealed invalid inputs/domain");
  }
  const std::vector<double> zero_rho(points), zero_gradient(3 * points);
  require(plan->execute(grid.points(), grid.weights(), zero_rho, zero_gradient, energy, {}, {}, {},
                        {}, detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "molecular screening weakened the strict raw-pair density contract");
}

void run_case(unsigned kind, bool unrestricted, std::ostream* output) {
  const auto input = input_system(kind);
  auto system = input;
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS, detail);
  auto ks = options(unrestricted);
  auto method = descriptor(ks);
  core::ContextState context;
  context.device_id = -1;
  auto plan = methods::detail::prepare_dft_calculation(capabilities, context, system, method);
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*plan, token, detail) != VIBEQC_STATUS_SUCCESS,
          "unexecuted WB97M-V owner published a state");
  auto result = plan->execute(false);
  require(result.convergence.converged && result.ks_diagnostic && result.physical_residual_rms &&
              *result.physical_residual_rms < 1e-9,
          "WB97M-V native endpoint did not converge");
  // Independently reconverged PySCF 2.14.0 / Libxc 7.0.0 on the exact
  // basis/grid above; reproduce with tools/verify_wb97mv_scf.py and this test's dump.
  const double oracle = kind == 0   ? -1.1419425514308112
                        : kind == 1 ? -1.5764935673604195
                                    : -2.3282082665978300;
  require(std::abs(result.energy - oracle) < 2e-10,
          "complete WB97M-V SCF energy differs from its independent matched-grid oracle");
  require(result.ks_diagnostic->scf_domain_version == 3 && result.ks_diagnostic->ao_order == 1,
          "WB97M-V lost its production point-domain or AO-jet identity");
  require(std::abs(result.ks_diagnostic->components.total() - result.energy) < 1e-12,
          "WB97M-V energy components do not describe its final density");
  require(methods::detail::dft_final_state_token(*plan, token, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  dft::VerifiedKsFinalState state;
  auto status = methods::detail::read_dft_final_state(*plan, token, false, state, detail);
  require(status == VIBEQC_STATUS_SUCCESS, detail);
  require(state.identity.model.range_correction && state.identity.model.nonlocal_correlation,
          "WB97M-V final-state identity omitted the LR-K/VV10 owners");
  require(state.identity.model.nonlocal_density_domain == dft::nlc::Vv10DensityDomain::MolecularV1,
          "WB97M-V final state lost its explicit nonlocal density domain");
  auto forged = token;
  forged.identity.model.nonlocal_correlation->b += 0.1;
  dft::VerifiedKsFinalState rejected;
  require(methods::detail::read_dft_final_state(*plan, forged, false, rejected, detail) !=
              VIBEQC_STATUS_SUCCESS,
          "changed VV10 identity authorized an old state");

  forged = token;
  forged.identity.model.nonlocal_density_domain = dft::nlc::Vv10DensityDomain::StrictPositive;
  require(methods::detail::read_dft_final_state(*plan, forged, false, rejected, detail) !=
              VIBEQC_STATUS_SUCCESS,
          "changed VV10 density policy authorized an old state");

  const std::string name = std::string(kind == 0   ? "h2"
                                       : kind == 1 ? "h3"
                                                   : "heh") +
                           (unrestricted ? "-uks" : "-rks");
  if (output) {
    dft::MolecularGrid grid(system, {1, ks.radial_points, ks.angular_polar, ks.angular_azimuth,
                                     ks.partition_iterations, ks.coincident_tolerance});
    dump(*output, name, input, grid, unrestricted, result, state);
  }
  // Pointees have been snapshotted; replay must not observe caller mutation.
  ks.exchange_terms = nullptr;
  ks.exchange_term_count = 0;
  ks.nonlocal_b = 9.0;
  const auto warm = plan->execute(false);
  require(warm.convergence.converged && warm.ks_diagnostic->initial_density_used &&
              std::abs(warm.energy - result.energy) < 1e-9,
          "WB97M-V warm replay changed the immutable model/energy");
  require(methods::detail::read_dft_final_state(*plan, token, false, rejected, detail) !=
              VIBEQC_STATUS_SUCCESS,
          "WB97M-V replay did not revoke the previous token");
  bool force_rejected = false;
  try {
    (void)plan->execute(true);
  } catch (const methods::MethodError& error) {
    force_rejected = error.status() == VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(force_rejected, "energy-only WB97M-V implementation accepted analytic forces");
  std::cout << name << " E=" << std::setprecision(16) << result.energy
            << " residual=" << *result.physical_residual_rms << '\n';

  // Invalid or incomplete compositions must never execute a partial method.
  for (unsigned failure = 0; failure < 9; ++failure) {
    ks = options(unrestricted);
    std::array<vibeqc_ks_exchange_term, 2> exchange{ks.exchange_terms[0], ks.exchange_terms[1]};
    std::array<vibeqc_ks_semilocal_component, 2> components{ks.semilocal_components[0],
                                                            ks.semilocal_components[1]};
    ks.exchange_terms = exchange.data();
    ks.semilocal_components = components.data();
    switch (failure) {
      case 0:
        ks.exchange_terms = nullptr;
        ks.exchange_term_count = 0;
        break;
      case 1:
        ks.has_nonlocal_correlation = 0;
        break;
      case 2:
        exchange[0].omega = 0.4;
        break;
      case 3:
        ks.nonlocal_b = 5.9;
        break;
      case 4:
        ks.nonlocal_variant = VIBEQC_NONLOCAL_RVV10;
        break;
      case 5:
        exchange[1].coefficient = 0.8;
        break;
      case 6:
        components[0].coefficient = 0.9;
        break;
      case 7:
        ks.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
        break;
      case 8:
        ks.nonlocal_maximum_bytes = 1;
        break;
    }
    bool refused = false;
    try {
      (void)methods::detail::prepare_dft_calculation(capabilities, context, system, method);
    } catch (const methods::MethodError&) {
      refused = true;
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "incomplete/mutated WB97M-V composition was accepted");
  }
  if (kind == 2 && !unrestricted) {
    ks = options(false);
    method.max_iterations = 1;
    auto failed = methods::detail::prepare_dft_calculation(capabilities, context, system, method);
    require(!failed->execute(false).convergence.converged,
            "single-iteration WB97M-V falsely converged");
    require(methods::detail::dft_final_state_token(*failed, token, detail) != VIBEQC_STATUS_SUCCESS,
            "unconverged WB97M-V solve published a successful state");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::ofstream output;
    if (argc == 2) {
      output.open(argv[1]);
      require(static_cast<bool>(output), "unable to write WB97M-V independent-oracle input");
    } else
      require(argc == 1, "usage: vibeqc_wb97mv_scf_tests [oracle-input.jsonl]");
    int32_t available = 1;
    require(vibeqc_method_available(VIBEQC_METHOD_WB97M_V, &available) == VIBEQC_STATUS_SUCCESS &&
                available,
            "WB97M-V composition is missing its public energy admission");
    nonlocal_density_domain();
    run_case(0, false, output.is_open() ? &output : nullptr);
    run_case(0, true, output.is_open() ? &output : nullptr);
    run_case(1, true, output.is_open() ? &output : nullptr);
    run_case(2, false, output.is_open() ? &output : nullptr);
    run_case(2, true, output.is_open() ? &output : nullptr);
    std::cout << "WB97M-V self-consistent composition and owner gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
