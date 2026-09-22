#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace vibeqc;
using dft::XcDensityFallback;
using dft::XcDensityRoute;
using dft::XcDensitySource;
using scf::DensityFactorIdentity;
using scf::DensityFactorSpin;
using scf::OccupiedDensityFactor;
using Matrix = std::vector<double>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

core::System hydrogens(std::size_t count, double displacement = 0.0) {
  core::System system;
  for (std::size_t atom = 0; atom < count; ++atom) {
    system.atoms.push_back({1, {0.1 * atom, 0.2 * atom, 1.4 * atom + displacement * atom * atom}});
    system.shells.push_back(
        {static_cast<std::uint32_t>(atom),
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "invalid native D/C fixture");
  return system;
}

core::System water() {
  auto system = hydrogens(2);
  system.atoms = {
      {8, {0, 0, 0}}, {1, {0, -1.43233673, 1.10715266}}, {1, {0, 1.43233673, 1.10715266}}};
  auto hydrogen = system.shells.front();
  system.shells = {
      {0, 0, {{130.70932, 0.15432897}, {23.808861, 0.53532814}, {6.4436083, 0.44463454}}},
      {0, 0, {{5.0331513, -0.09996723}, {1.1695961, 0.39951283}, {0.3803890, 0.70011547}}},
      {0, 1, {{5.0331513, 0.15591627}, {1.1695961, 0.60768372}, {0.3803890, 0.39195739}}},
  };
  for (unsigned atom : {1U, 2U}) {
    hydrogen.atom_index = atom;
    system.shells.push_back(hydrogen);
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "invalid water D/C fixture");
  return system;
}

void same_xc(const dft::XcIntegral& actual, const dft::XcIntegral& expected) {
  require(std::abs(actual.energy - expected.energy) < 3e-11 &&
              std::abs(actual.electrons - expected.electrons) < 3e-11,
          "native D/C XC energy/electron mismatch");
  require(actual.potential.size() == expected.potential.size(), "XC potential size mismatch");
  for (std::size_t i = 0; i < actual.potential.size(); ++i)
    require(std::abs(actual.potential[i] - expected.potential[i]) < 3e-11,
            "native D/C XC potential mismatch");
}

void fixed_density() {
  const auto system = hydrogens(2);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 2, 2, 4, 3, 1e-12});
  const DensityFactorIdentity identity{11, 17, 3, 3};
  // Off-diagonal cross terms and two occupied columns are essential here;
  // diagonal-only or one-orbital tests cannot detect accidental truncation.
  const Matrix coefficients{0.6, -0.2, 0.3, 0.7}, occupations{2, 2};
  const OccupiedDensityFactor factor(identity, DensityFactorSpin::Restricted, 2, coefficients,
                                     occupations);
  const Matrix density(factor.density().begin(), factor.density().end());
  const XcDensitySource source{XcDensityRoute::OccupiedOrbitals, &factor, identity};
  const OccupiedDensityFactor alpha(identity, DensityFactorSpin::Alpha, 2, coefficients,
                                    Matrix{1, 1});
  const OccupiedDensityFactor wrong_basis(identity, DensityFactorSpin::Restricted, 1, Matrix{0.6},
                                          Matrix{2});
  const OccupiedDensityFactor empty(identity, DensityFactorSpin::Restricted, 2, {}, {});
  for (const auto integrate :
       {dft::integrate_lda_xc_pw_rks, dft::integrate_pbe_rks, dft::integrate_pbe_rks_with_tail}) {
    const auto expected = integrate(basis, grid, density, 7, {});
    for (std::size_t tile : {1U, 7U, 113U}) {
      const auto actual = integrate(basis, grid, density, tile, source);
      same_xc(actual, expected);
      const auto& record = actual.density_diagnostic;
      require(record.executed == XcDensityRoute::OccupiedOrbitals &&
                  record.fallback == XcDensityFallback::None && record.nocc == 2 &&
                  record.npoint == grid.point_count() && record.active_ao == 2 &&
                  record.max_tile_points == std::min(tile, grid.point_count()),
              "native XC executed-route/shape diagnostic mismatch");
      const std::size_t jets = integrate == dft::integrate_lda_xc_pw_rks ? 1 : 4;
      require(record.ingredient_mask == (jets == 1 ? 1U : 3U) &&
                  record.owned_numeric_bytes ==
                      (jets * record.max_tile_points * 2 + 4) * sizeof(double) &&
                  record.borrowed_factor_bytes == factor.numeric_capacity_bytes() &&
                  record.borrowed_density_bytes == runtime::vector_bytes(density),
              "native XC output pruning or capacity accounting mismatch");
    }
    const auto fallback = [&](const Matrix& d, XcDensitySource candidate,
                              XcDensityFallback reason) {
      const auto actual = integrate(basis, grid, d, 7, candidate);
      const auto original = integrate(basis, grid, d, 7, {});
      require(actual.energy == original.energy && actual.potential == original.potential &&
                  actual.electrons == original.electrons &&
                  actual.density_diagnostic.executed == XcDensityRoute::DensityMatrix &&
                  actual.density_diagnostic.fallback == reason,
              "fallback changed original D or reported the wrong reason");
    };
    fallback(density, {XcDensityRoute::OccupiedOrbitals}, XcDensityFallback::MissingFactor);
    fallback(density,
             {XcDensityRoute::OccupiedOrbitals, &factor, identity, dft::XcDensityRole::Response},
             XcDensityFallback::Response);
    fallback(density, {XcDensityRoute::OccupiedOrbitals, &alpha, identity},
             XcDensityFallback::Spin);
    fallback(density, {XcDensityRoute::OccupiedOrbitals, &wrong_basis, identity},
             XcDensityFallback::Basis);
    for (unsigned field = 0; field < 4; ++field) {
      auto stale = identity;
      if (field == 0) ++stale.basis;
      if (field == 1) ++stale.reference;
      if (field == 2) ++stale.orbital_generation;
      if (field == 3) ++stale.density_generation;
      fallback(density, {XcDensityRoute::OccupiedOrbitals, &factor, stale},
               XcDensityFallback::Identity);
    }
    auto mixed = density;
    mixed[0] += 0.03;
    fallback(mixed, source, XcDensityFallback::Density);
    const auto vacuum =
        integrate(basis, grid, Matrix(4), 7, {XcDensityRoute::OccupiedOrbitals, &empty, identity});
    require(vacuum.energy == 0 && vacuum.electrons == 0 &&
                vacuum.density_diagnostic.executed == XcDensityRoute::OccupiedOrbitals &&
                std::all_of(vacuum.potential.begin(), vacuum.potential.end(),
                            [](double x) { return x == 0; }),
            "empty occupied channel did not retain the analytic vacuum");

    // Orbital-direction finite differences exercise the C route itself on
    // both sides. Compare with Tr(V dD), not merely with an electron count.
    const Matrix direction{0.1, 0.3, -0.2, 0.15};
    Matrix delta_density(4);
    for (std::size_t mu = 0; mu < 2; ++mu)
      for (std::size_t nu = 0; nu < 2; ++nu)
        for (std::size_t o = 0; o < 2; ++o)
          delta_density[mu * 2 + nu] += 2 * (direction[mu * 2 + o] * coefficients[nu * 2 + o] +
                                             coefficients[mu * 2 + o] * direction[nu * 2 + o]);
    for (double step : {1e-4, 3e-5}) {
      auto plus = coefficients, minus = coefficients;
      for (std::size_t i = 0; i < plus.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      const OccupiedDensityFactor fp(identity, DensityFactorSpin::Restricted, 2, plus, occupations);
      const OccupiedDensityFactor fm(identity, DensityFactorSpin::Restricted, 2, minus,
                                     occupations);
      const auto ep = integrate(basis, grid, Matrix(fp.density().begin(), fp.density().end()), 7,
                                {XcDensityRoute::OccupiedOrbitals, &fp, identity})
                          .energy;
      const auto em = integrate(basis, grid, Matrix(fm.density().begin(), fm.density().end()), 7,
                                {XcDensityRoute::OccupiedOrbitals, &fm, identity})
                          .energy;
      require(std::abs((ep - em) / (2 * step) -
                       scf::reference::dot(expected.potential, delta_density)) < 2e-8,
              "native C potential violates the orbital directional derivative");
    }
  }
}

void native_scf() {
  bool saw_periodic_rebuild = false;
  bool saw_drift_rebuild = false;
  bool saw_noise_rebuild = false;
  std::uint64_t previous_h2_incremental_identity = 0;
  for (const auto system : {hydrogens(2), hydrogens(2, 0.17), water()}) {
    const dft::AoBasis basis(system);
    const dft::MolecularGrid grid(system, {1, 16, 8, 16, 3, 1e-12});
    scf::FockBuildSpec spec;
    spec.derivative_order = 0;
    spec.exchange.present = false;
    const scf::PreparedFockPlan plan(system, nullptr,
                                     scf::resolve_fock_build(spec, scf::FockBackend::Cpu));
    scf::ScfOptions options;
    options.compute_forces = false;
    options.max_iterations = 150;
    for (const auto run : {scf::run_lda_rks, scf::run_pbe_rks}) {
      options.xc_density_route = XcDensityRoute::DensityMatrix;
      const auto d = run(plan, basis, grid, options, nullptr);
      require(d.converged && !d.xc_density_factor &&
                  d.xc_density_diagnostic.density_calls == d.fock_builds &&
                  d.xc_density_diagnostic.orbital_calls == 0 &&
                  d.xc_density_diagnostic.packed_coefficient_elements == 0,
              "default native D candidate changed or failed");
      if (run == scf::run_pbe_rks) {
        options.experimental_incremental_xc = true;
        options.incremental_xc_max_updates = 1;
        options.incremental_xc_max_density_rms = 1.0e6;
        options.incremental_xc_noise_density_rms = 0.0;
        const auto incremental = run(plan, basis, grid, options, nullptr);
        const auto& inc = incremental.dft_diagnostic.incremental_xc;
        require(
            incremental.converged && inc.enabled && inc.model_identity != 0 &&
                inc.full_builds >= 3 && inc.incremental_updates >= 1 &&
                inc.strict_final_builds >= inc.strict_refinement_iterations + inc.final_audits &&
                inc.strict_refinement_iterations >= 2 && inc.final_audits >= 1 &&
                inc.audit_failures == 0 && inc.anchor_generation >= 1 &&
                inc.retained_anchor_bytes == 0 && inc.peak_update_buffer_bytes != 0 &&
                inc.peak_replacement_overlap_bytes != 0 &&
                std::abs(incremental.energy - d.energy) < 2e-10 &&
                scf::reference::density_rms(incremental.density, d.density) < 2e-8 &&
                incremental.physical_residual_rms < options.density_tolerance,
            "incremental PBE SCF/rebuild/final-verification parity failed");
        if (basis.nao == 2) {
          if (previous_h2_incremental_identity != 0)
            require(previous_h2_incremental_identity != inc.model_identity,
                    "same-shaped changed geometry reused incremental XC model identity");
          previous_h2_incremental_identity = inc.model_identity;
        }
        saw_periodic_rebuild = saw_periodic_rebuild || inc.periodic_rebuilds != 0;

        options.incremental_xc_max_updates = 1000;
        options.incremental_xc_max_density_rms = 1.0e-20;
        const auto drift_rebuild = run(plan, basis, grid, options, nullptr);
        require(drift_rebuild.converged &&
                    drift_rebuild.dft_diagnostic.incremental_xc.final_audits >= 1 &&
                    std::abs(drift_rebuild.energy - d.energy) < 2e-10,
                "incremental PBE drift-policy run changed the strict endpoint");
        saw_drift_rebuild =
            saw_drift_rebuild || drift_rebuild.dft_diagnostic.incremental_xc.drift_rebuilds != 0;

        options.incremental_xc_max_density_rms = 1.0e6;
        options.incremental_xc_noise_density_rms = 1.0e6;
        const auto noise_rebuild = run(plan, basis, grid, options, nullptr);
        require(noise_rebuild.converged &&
                    noise_rebuild.dft_diagnostic.incremental_xc.final_audits >= 1 &&
                    std::abs(noise_rebuild.energy - d.energy) < 2e-10,
                "incremental PBE noise-policy run changed the strict endpoint");
        saw_noise_rebuild =
            saw_noise_rebuild || noise_rebuild.dft_diagnostic.incremental_xc.noise_rebuilds != 0;

        options.incremental_xc_noise_density_rms = 0.0;
        options.incremental_xc_max_updates = 4;
        options.incremental_xc_max_density_rms = 5.0e-2;
        options.strict_initial_density = true;
        const auto warm_incremental = run(plan, basis, grid, options, &d.density);
        options.strict_initial_density = false;
        require(warm_incremental.converged && warm_incremental.initial_density_used &&
                    warm_incremental.dft_diagnostic.incremental_xc.model_identity !=
                        inc.model_identity &&
                    warm_incremental.dft_diagnostic.incremental_xc.final_audits >= 1 &&
                    std::abs(warm_incremental.energy - d.energy) < 2e-10,
                "incremental PBE warm replay reused stale anchor state");

        options.max_iterations = 1;
        const auto failed_incremental = run(plan, basis, grid, options, nullptr);
        options.max_iterations = 150;
        require(!failed_incremental.converged &&
                    failed_incremental.dft_diagnostic.incremental_xc.model_identity !=
                        warm_incremental.dft_diagnostic.incremental_xc.model_identity &&
                    failed_incremental.dft_diagnostic.incremental_xc.retained_anchor_bytes == 0 &&
                    failed_incremental.dft_diagnostic.incremental_xc.final_audits == 0,
                "failed incremental PBE solve published or retained stale anchor state");
        options.experimental_incremental_xc = false;
      }
      options.xc_density_route = XcDensityRoute::OccupiedOrbitals;
      runtime::cpu_resource_observation = {.active = true};
      const auto c = run(plan, basis, grid, options, nullptr);
      const auto observed = runtime::cpu_resource_observation.peak_bytes;
      runtime::cpu_resource_observation = {};
      require(c.converged && std::abs(c.energy - d.energy) < 2e-10 &&
                  scf::reference::density_rms(c.density, d.density) < 2e-8 &&
                  c.xc_density_diagnostic.physical_residual < options.density_tolerance &&
                  d.xc_density_diagnostic.physical_residual < options.density_tolerance,
              "same-loop native D/C SCF endpoint gate failed");
      const auto check_current = [&](const scf::ScfResult& result) {
        require(result.xc_density_factor && result.xc_density_factor->matches(
                                                result.xc_density_diagnostic.final_identity,
                                                DensityFactorSpin::Restricted, result.density),
                "native RKS exported stale final orbitals");
      };
      check_current(c);
      require(c.xc_density_diagnostic.orbital_calls == c.fock_builds &&
                  c.xc_density_diagnostic.fallback_calls == 0 &&
                  c.xc_density_diagnostic.final_identity.orbital_generation == c.iterations + 2 &&
                  c.xc_density_diagnostic.packed_coefficient_elements ==
                      (c.iterations + 2) * basis.nao * (system.electron_count / 2) &&
                  observed >= c.xc_density_diagnostic.xc_peak_bytes +
                                  c.xc_density_factor->numeric_capacity_bytes(),
              "native RKS producer generations or resources mismatch");
      options.strict_initial_density = true;
      const auto warm = run(plan, basis, grid, options, &d.density);
      options.strict_initial_density = false;
      check_current(warm);
      require(warm.converged && warm.initial_density_used &&
                  warm.xc_density_diagnostic.density_calls == 1 &&
                  warm.xc_density_diagnostic.fallback_calls == 1 &&
                  warm.xc_density_diagnostic.orbital_calls + 1 == warm.fock_builds &&
                  warm.xc_density_diagnostic.final_identity.reference !=
                      c.xc_density_diagnostic.final_identity.reference &&
                  std::abs(warm.energy - d.energy) < 2e-10,
              "external warm seed used stale orbitals or changed the endpoint");
      options.max_iterations = 1;
      const auto unfinished = run(plan, basis, grid, options, nullptr);
      options.max_iterations = 150;
      require(!unfinished.converged && unfinished.fock_builds == 1,
              "nonconvergence test unexpectedly converged");
      check_current(unfinished);
      check_current(c);  // Later replays must not mutate an earlier snapshot.
      std::cout << "native RKS " << (run == scf::run_lda_rks ? "LDA" : "PBE-scaled-v1")
                << " nao=" << basis.nao << " iterations D/C=" << d.iterations << '/' << c.iterations
                << " E error=" << std::abs(c.energy - d.energy)
                << " C residual=" << c.xc_density_diagnostic.physical_residual << '\n';
    }
  }
  require(saw_periodic_rebuild, "incremental PBE fixtures never exercised periodic rebuild");
  require(saw_drift_rebuild, "incremental PBE fixtures never exercised drift rebuild");
  require(saw_noise_rebuild, "incremental PBE fixtures never exercised noise rebuild");
}
}  // namespace

int main() {
  try {
    fixed_density();
    native_scf();
    std::cout << "native D/C source, fallback, directional and SCF gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
