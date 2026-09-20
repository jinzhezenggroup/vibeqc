#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "dft/ks_final_state.hpp"

namespace {
using namespace vibeqc;
using namespace vibeqc::dft;
using namespace vibeqc::scf;
using reference::Matrix;

void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
void near(double actual, double expected, const char* detail) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 1e-12, detail);
}

ResolvedFockBuild ks_fock(FockSpin spin) {
  FockBuildSpec spec;
  spec.spin = spin;
  spec.derivative_order = 0;
  spec.exchange.present = false;
  return resolve_fock_build(spec, FockBackend::Cuda);
}

struct Fixture {
  KsFinalStateIdentity id{{{11, 23, 7, 9}, 5, ks_fock(FockSpin::Restricted), {1}},
                          {1, 1, {}, 256, false, 1, 0, 31}};
  Matrix s{2, 0, 0, 4}, h{-4, 0, 0, 8};
  KsPhysicalState physical{id, true, {{1, 0, 0, 0}}, {{-2, 0, 0, 12}}, {.3, -2, .4, -.1}, -1.4, 0};
  KsFinalStateCandidate candidate{id, 9, true, {{{-1, 3}, {1 / std::sqrt(2.0), 0, 0, .5}}}};
  solver::FinalStateLimits limits;

  bool validate(bool weighted = false, VerifiedKsFinalState* retained = nullptr) const {
    VerifiedKsFinalState state;
    std::string detail;
    const bool valid =
        validate_ks_final_state(id, s, h, physical, candidate, limits, weighted, state, detail);
    if (valid && retained) *retained = std::move(state);
    return valid;
  }
  void sync() { physical.identity = candidate.identity = id; }
};

void analytic_rks_and_uks() {
  Fixture rks;
  VerifiedKsFinalState energy, gradient;
  require(rks.validate(false, &energy) && rks.validate(true, &gradient),
          "analytic RKS state rejected");
  require(energy.weighted_density.empty() && gradient.weighted_density.size() == 1,
          "energy-only validation constructed W or gradient validation omitted it");
  near(gradient.weighted_density[0][0], -1, "wrong restricted W");
  near(gradient.diagnostic.component_energy, -1.4, "wrong KS component energy");
  near(gradient.diagnostic.reported_energy_error, 0, "identical KS energies disagreed");

  Fixture cpu;
  // Resolve the complete CPU strategy, including its schedule; relabeling a
  // CUDA backend alone must remain invalid under the shared provider contract.
  cpu.id.determinant.model = resolve_fock_build(cpu.id.determinant.model.spec, FockBackend::Cpu);
  cpu.id.model.device = -1;
  cpu.sync();
  require(cpu.validate(true, &gradient), "CPU RKS state with explicit host device rejected");
  near(gradient.weighted_density[0][0], -1, "wrong CPU RKS W");
  cpu.id.model.device = 0;
  cpu.sync();
  require(!cpu.validate(), "CPU accepted a CUDA device ordinal");

  for (const std::size_t beta : {0U, 1U}) {
    Fixture uks;
    uks.id.determinant.model = ks_fock(FockSpin::Unrestricted);
    uks.id.determinant.occupied = {1, beta};
    uks.id.model.spins = 2;
    uks.physical.density = {{.5, 0, 0, 0}, {.5 * beta, 0, 0, 0}};
    uks.physical.fock.push_back(uks.physical.fock[0]);
    uks.candidate.spins.push_back(uks.candidate.spins[0]);
    uks.sync();
    VerifiedKsFinalState state;
    require(uks.validate(true, &state), "analytic UKS state rejected");
    require(state.weighted_density.size() == 2, "UKS lost a spin W block");
    near(state.weighted_density[0][0], -.5, "wrong alpha W");
    near(state.weighted_density[1][0], -.5 * beta, "wrong beta W");

    Fixture cpu_uks = uks;
    cpu_uks.id.determinant.model =
        resolve_fock_build(cpu_uks.id.determinant.model.spec, FockBackend::Cpu);
    cpu_uks.id.model.device = -1;
    cpu_uks.sync();
    require(cpu_uks.validate(true, &state), "CPU UKS final-state proof rejected");
    require(state.weighted_density.size() == 2, "CPU UKS lost a spin W block");
  }
}

void identity_rejection() {
  const std::vector<std::function<void(KsFinalStateIdentity&)>> changes{
      [](auto& id) { ++id.determinant.factor.basis; },
      [](auto& id) { ++id.determinant.factor.reference; },
      [](auto& id) { ++id.determinant.factor.orbital_generation; },
      [](auto& id) { ++id.determinant.factor.density_generation; },
      [](auto& id) { ++id.determinant.solve_epoch; },
      [](auto& id) { ++id.model.owner; },
      [](auto& id) { ++id.model.grid.radial_points; },
      [](auto& id) { id.model.functional = (id.model.functional + 1U) % 3U; },
      [](auto& id) { ++id.model.tile_points; },
      [](auto& id) { ++id.model.device; },
      [](auto& id) { ++id.model.scf_domain_version; },
      [](auto& id) { id.determinant.model.screening_tolerance *= 2; }};
  for (const auto& change : changes) {
    Fixture a;
    change(a.physical.identity);
    require(!a.validate(), "stale physical KS identity passed");
    a = Fixture{};
    change(a.candidate.identity);
    require(!a.validate(), "stale orbital KS identity passed");
  }
}

void model_and_state_rejection() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::function<void(Fixture&)>> changes{
      [](auto& a) { a.id.model.owner = 0; },
      [](auto& a) { a.id.model.tile_points = 0; },
      [](auto& a) { a.id.model.version = 2; },
      [](auto& a) { a.id.model.scf_domain_version = 2; },
      [](auto& a) { a.id.model.functional = 3; },
      [](auto& a) { a.id.model.spins = 2; },
      [](auto& a) { a.id.model.device = -1; },
      [](auto& a) { a.id.model.grid.version = 99; },
      [](auto& a) { a.id.determinant.model.backend = FockBackend::Cpu; },
      [](auto& a) { a.id.determinant.model.spec.derivative_order = 1; },
      [](auto& a) { a.id.determinant.model.spec.exchange.present = true; },
      [](auto& a) { a.id.determinant.model.spec.coulomb.coefficient = .5; },
      [](auto& a) { a.physical.physical = false; },
      [](auto& a) { a.candidate.physical_origin = false; },
      [](auto& a) { a.candidate.fock_density_generation = 0; },
      [](auto& a) { --a.candidate.fock_density_generation; },
      [](auto& a) { ++a.candidate.fock_density_generation; },
      [](auto& a) { a.physical.density[0][0] += 1e-5; },
      [](auto& a) { a.physical.fock[0][0] += 1e-5; },
      [](auto& a) { a.candidate.spins[0].values[0] += 1e-5; },
      [](auto& a) { a.physical.physical_residual = 2e-9; },
      [](auto& a) { a.physical.physical_residual = -1e-12; },
      [](auto& a) { a.physical.reported_energy += 1e-5; },
      [nan](auto& a) { a.physical.components.xc = nan; },
      [nan](auto& a) { a.physical.components.exact_exchange = nan; },
      [nan](auto& a) { a.physical.physical_residual = nan; }};
  for (const auto& change : changes) {
    Fixture a;
    change(a);
    a.sync();
    require(!a.validate(), "invalid KS model or final state passed");
  }
}

}  // namespace

int main() {
  try {
    analytic_rks_and_uks();
    identity_rejection();
    model_and_state_rejection();
    std::cout << "KS final-state contract tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
