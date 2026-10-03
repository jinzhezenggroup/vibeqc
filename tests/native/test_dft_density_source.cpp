#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace generativeqc;
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
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
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
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
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

  const auto ao_cache = dft::prepare_rks_ao_cache(basis, grid, 1U);
  const auto retained = dft::integrate_pbe_rks_with_tail_scaled_retaining_features(
      basis, grid, density, 7, {}, 1.0, 1.0, &ao_cache);
  const Matrix signed_delta{1.0e-3, -4.0e-4, -4.0e-4, 7.0e-4};
  const auto reference =
      dft::integrate_pbe_rks_incremental_exact(basis, grid, density, signed_delta, 7, 1.0, 1.0);
  const auto cached = dft::integrate_pbe_rks_incremental_exact(
      basis, grid, density, signed_delta, 7, 1.0, 1.0, &retained.features, &ao_cache);
  same_xc(cached.total, reference.total);
  require(
      retained.features.points == grid.point_count() && retained.features.nao == basis.nao &&
          retained.features.numeric_capacity_bytes() == 4 * grid.point_count() * sizeof(double) &&
          std::abs(cached.anchor_energy - reference.anchor_energy) < 3e-11 &&
          std::abs(cached.energy_difference - reference.energy_difference) < 3e-11 &&
          cached.potential_difference.size() == reference.potential_difference.size(),
      "retained incremental PBE feature cache shape or scalar parity failed");
  for (std::size_t i = 0; i < cached.potential_difference.size(); ++i)
    require(std::abs(cached.potential_difference[i] - reference.potential_difference[i]) < 3e-11,
            "retained incremental PBE feature cache changed the potential difference");
}

template <unsigned Mask>
dft::SemilocalPointValue mask_test_point(const double rho[2], const double (&gradient)[2][3],
                                         const double tau[2]) {
  // Synthetic linear point program checks ingredient plumbing and derivative
  // factors only. Existing independent physics point/integration tests remain
  // the scientific oracles; this callback does not replace them.
  dft::SemilocalPointValue out;
  for (unsigned spin = 0; spin < 2; ++spin) {
    out.rho[spin] = 0.3 + 0.2 * spin;
    out.energy += out.rho[spin] * rho[spin];
    if constexpr ((Mask & 6U) != 0)
      for (unsigned axis = 0; axis < 3; ++axis) {
        out.gradient[spin][axis] = 0.1 * (axis + 1) * (spin + 1);
        out.energy += out.gradient[spin][axis] * gradient[spin][axis];
      }
    if constexpr ((Mask & 8U) != 0) {
      const double tau_coefficient = 0.2 + 0.1 * spin;
      out.kinetic[spin] = 0.5 * tau_coefficient;
      out.energy += tau_coefficient * tau[spin];
    }
  }
  return out;
}

void generic_feature_masks() {
  const auto system = hydrogens(2);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 2, 2, 4, 3, 1e-12});
  const DensityFactorIdentity identity{23, 29, 5, 7};
  const Matrix coefficients{0.6, -0.2, 0.3, 0.7}, occupations{2, 2};
  const OccupiedDensityFactor factor(identity, DensityFactorSpin::Restricted, 2, coefficients,
                                     occupations);
  const Matrix density(factor.density().begin(), factor.density().end());
  const XcDensitySource source{XcDensityRoute::OccupiedOrbitals, &factor, identity};
  const std::array<dft::SemilocalPointProgram, 3> programs{{
      {"mask-1 traversal test", "test://mask-1", 1U, 1U, mask_test_point<1>},
      {"mask-7 traversal test", "test://mask-7", 7U, 1U, mask_test_point<7>},
      {"mask-15 traversal test", "test://mask-15", 15U, 1U, mask_test_point<15>},
  }};
  for (const auto& program : programs) {
    const auto expected = dft::integrate_semilocal_rks(basis, grid, density, program, 7);
    for (std::size_t tile : {1U, 7U, 113U}) {
      const auto actual = dft::integrate_semilocal_rks(basis, grid, density, program, tile, source);
      same_xc(actual, expected);
      require(actual.density_diagnostic.executed == XcDensityRoute::OccupiedOrbitals &&
                  actual.density_diagnostic.ingredient_mask == program.ingredient_mask,
              "generic mask lost its occupied-factor route or ingredient diagnostic");
    }

    auto near_symmetric = density;
    near_symmetric[1] += 5.0e-13;
    near_symmetric[2] -= 5.0e-13;
    same_xc(dft::integrate_semilocal_rks(basis, grid, near_symmetric, program, 3), expected);
    auto near_alpha = near_symmetric, alpha = density, beta = density;
    for (std::size_t i = 0; i < density.size(); ++i) {
      near_alpha[i] *= 0.7;
      alpha[i] *= 0.7;
      beta[i] *= 0.3;
    }
    const auto spin_reference = dft::integrate_semilocal_uks(basis, grid, alpha, beta, program, 7);
    const auto spin_near = dft::integrate_semilocal_uks(basis, grid, near_alpha, beta, program, 3);
    require(std::abs(spin_near.energy - spin_reference.energy) < 3e-11 &&
                std::abs(spin_near.electrons[0] - spin_reference.electrons[0]) < 3e-11 &&
                std::abs(spin_near.electrons[1] - spin_reference.electrons[1]) < 3e-11,
            "generic UKS mask changed near-symmetric spin features");
    for (unsigned spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < density.size(); ++i)
        require(std::abs(spin_near.potential[spin][i] - spin_reference.potential[spin][i]) < 3e-11,
                "generic UKS mask changed the spin potential");

    const Matrix response_density{0.2, -0.375 + 5.0e-13, -0.375 - 5.0e-13, -0.1};
    const auto dense_response =
        dft::integrate_semilocal_rks(basis, grid, response_density, program, 7);
    const auto response = dft::integrate_semilocal_rks(
        basis, grid, response_density, program, 7,
        {XcDensityRoute::OccupiedOrbitals, &factor, identity, dft::XcDensityRole::Response});
    require(response.energy == dense_response.energy &&
                response.electrons == dense_response.electrons &&
                response.potential == dense_response.potential &&
                response.density_diagnostic.executed == XcDensityRoute::DensityMatrix &&
                response.density_diagnostic.fallback == XcDensityFallback::Response,
            "generic mask replaced signed response features with occupied-state features");
  }
  for (unsigned mask :
       {0U, 2U, 3U, 4U, 6U, 8U, 9U, 14U, 16U, std::numeric_limits<unsigned>::max()}) {
    auto invalid = programs[0];
    invalid.ingredient_mask = mask;
    bool rejected = false;
    try {
      dft::validate_semilocal_point_program(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "generic semilocal program accepted an unsupported ingredient mask");
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
      const auto check_physical = [&](const scf::ScfResult& result) {
        const auto& ints = plan.one_electron();
        const auto xc = run == scf::run_pbe_rks
                            ? dft::integrate_pbe_rks_with_tail(basis, grid, result.density)
                            : dft::integrate_lda_xc_pw_rks(basis, grid, result.density);
        const auto jk = plan.build(result.density);
        auto fock = scf::assemble_fock(plan.strategy(), ints.hcore, jk).alpha;
        for (std::size_t i = 0; i < fock.size(); ++i) fock[i] += xc.potential[i];
        const double energy = ints.nuclear_repulsion +
                              scf::reference::dot(result.density, ints.hcore) +
                              0.5 * scf::reference::dot(result.density, jk.coulomb) + xc.energy;
        const auto residual =
            scf::reference::commutator_residual(fock, result.density, ints.overlap, basis.nao);
        double maximum = 0.0;
        for (double value : residual) maximum = std::max(maximum, std::abs(value));
        require(std::abs(energy - result.energy) < 1e-12 &&
                    std::abs(scf::reference::residual_rms(residual) -
                             result.physical_residual_rms) < 1e-13,
                "RKS returned D/KS-energy/RMS from different physical states");
        if (result.converged)
          require(maximum <= std::min(1e-8, options.density_tolerance),
                  "RKS accepted an RMS-small maximum-large physical commutator");
        require(result.dft_diagnostic.history.size() == result.iterations &&
                    result.iterations <=
                        (options.experimental_incremental_xc ? 2U : 1U) * options.max_iterations,
                "RKS closure exceeded the existing iteration/history budget");
      };
      options.xc_density_route = XcDensityRoute::DensityMatrix;
      const auto d = run(plan, basis, grid, options, nullptr);
      check_physical(d);
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
        check_physical(incremental);
        const auto& inc = incremental.dft_diagnostic.incremental_xc;
        require(
            incremental.converged && inc.enabled && inc.model_identity != 0 &&
                inc.full_builds >= 3 && inc.incremental_updates >= 1 &&
                inc.strict_final_builds >= inc.strict_refinement_iterations + inc.final_audits &&
                inc.strict_refinement_iterations >= 2 && inc.final_audits >= 1 &&
                inc.audit_failures == 0 && inc.anchor_generation >= 1 &&
                inc.anchor_feature_builds >= 1 &&
                inc.anchor_feature_reuses == inc.incremental_updates &&
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
        check_physical(drift_rebuild);
        require(drift_rebuild.converged &&
                    drift_rebuild.dft_diagnostic.incremental_xc.final_audits >= 1 &&
                    std::abs(drift_rebuild.energy - d.energy) < 2e-10,
                "incremental PBE drift-policy run changed the strict endpoint");
        saw_drift_rebuild =
            saw_drift_rebuild || drift_rebuild.dft_diagnostic.incremental_xc.drift_rebuilds != 0;

        options.incremental_xc_max_density_rms = 1.0e6;
        options.incremental_xc_noise_density_rms = 1.0e6;
        const auto noise_rebuild = run(plan, basis, grid, options, nullptr);
        check_physical(noise_rebuild);
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
        check_physical(warm_incremental);
        options.strict_initial_density = false;
        require(warm_incremental.converged && warm_incremental.initial_density_used &&
                    warm_incremental.dft_diagnostic.incremental_xc.model_identity !=
                        inc.model_identity &&
                    warm_incremental.dft_diagnostic.incremental_xc.final_audits >= 1 &&
                    std::abs(warm_incremental.energy - d.energy) < 2e-10,
                "incremental PBE warm replay reused stale anchor state");

        options.max_iterations = 1;
        const auto failed_incremental = run(plan, basis, grid, options, nullptr);
        check_physical(failed_incremental);
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
      check_physical(c);
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
      check_physical(warm);
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
      check_physical(unfinished);
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
    generic_feature_masks();
    native_scf();
    std::cout << "native D/C source, fallback, directional and SCF gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
