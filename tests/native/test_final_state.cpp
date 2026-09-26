#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "scf/solver/final_state.hpp"
#ifdef VIBEQC_TEST_DEVICE_FINAL_STATE
#include <cuda_runtime_api.h>

#include <memory>

#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#endif

namespace {
using namespace vibeqc::scf;
using namespace vibeqc::scf::solver;
using reference::Matrix;
const FinalStateOperations* backend = nullptr;
void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
void near(double actual, double expected, const char* detail) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 1e-12, detail);
}

// All expected frames, densities, energies and W below are analytic. These
// tests never call either the production or reference eigensolver.
struct Fixture {
  FinalStateIdentity id{
      {11, 23, 7, 9},
      5,
      resolve_fock_build(make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu),
      {1}};
  Matrix s{2, 0, 0, 4}, h{-4, 0, 0, 8}, x{1 / std::sqrt(2.0), 0, 0, .5};
  std::vector<Matrix> d{{1, 0, 0, 0}};
  PhysicalFockFrame f{id, true, {{-2, 0, 0, 12}}};
  FinalFrameCandidate c{id, 8, true, {{{-1, 3}, x}}};
  FinalStateLimits limits;
  double nuclear{.3};

  bool valid() const {
    FinalStateDiagnostic diagnostic;
    std::string detail;
    return validate_final_state(id, s, h, nuclear, d, f, c, limits, diagnostic, detail, backend);
  }
  PhysicalFockOperation physical() const {
    return [this](const FinalStateIdentity& current, const std::vector<Matrix>& density) {
      require(density.size() == d.size(), "provider lost a spin density");
      return PhysicalFockFrame{current, true, f.spins};
    };
  }
  initial_guess::EigenOperation eigen() const {
    return [this](const Matrix& input, const Matrix* overlap, const Matrix* orthogonalizer,
                  std::size_t n) {
      require(input == f.spins[0] && overlap && *overlap == s && orthogonalizer &&
                  *orthogonalizer == x && n == 2,
              "correction lost its actual F/S/X inputs");
      return c.spins[0];
    };
  }
  FinalStateSelection select(bool weighted = false, bool force = false) const {
    return select_final_state(id, s, h, x, nuclear, d, &c, physical(), eigen(), limits, weighted,
                              force, backend);
  }
};

void analytic_spin_states() {
  for (const int beta : {-1, 0, 1}) {
    Fixture a;
    if (beta >= 0) {
      a.id.model = resolve_fock_build(make_hf_fock_spec(FockSpin::Unrestricted), FockBackend::Cpu);
      a.id.occupied = {1, static_cast<std::size_t>(beta)};
      a.d = {{.5, 0, 0, 0}, {.5 * beta, 0, 0, 0}};
      a.f.identity = a.c.identity = a.id;
      a.f.spins.push_back(a.f.spins[0]);
      a.c.spins.push_back(a.c.spins[0]);
    }
    require(a.valid(), "analytic physical spin state rejected");
    const auto energy = a.select();
    const auto force = a.select(true);
    require(energy.state && force.state && energy.reused && force.reused,
            "valid candidate was not reused");
    require(energy.fock_evaluations == 1 && energy.eigen_solves == 0 &&
                energy.density_updates == 0 && energy.state->weighted_density.empty(),
            "energy-only reuse performed unnecessary work");
    near(energy.state->diagnostic.energy, beta == 0 ? -1.2 : -2.7, "wrong spin energy");
    near(force.state->weighted_density[0][0], beta < 0 ? -1 : -.5, "wrong Pulay weight");
    if (beta >= 0) near(force.state->weighted_density[1][0], -.5 * beta, "wrong beta W");
    const auto rebuilt = a.select(true, true);
    require(rebuilt.state && !rebuilt.reused && rebuilt.fock_evaluations == 2 &&
                rebuilt.eigen_solves == 2 * a.d.size() && rebuilt.density_updates == 1 &&
                rebuilt.fixed_point_checks == 1 && rebuilt.fixed_point_eigen_solves == a.d.size(),
            "forced rebuild did not solve, project and evaluate the new D");
    near(rebuilt.state->diagnostic.energy, energy.state->diagnostic.energy,
         "rebuild energy changed");
    require(
        rebuilt.state->identity.factor.density_generation == a.id.factor.density_generation + 1 &&
            rebuilt.state->identity.factor.orbital_generation == a.id.factor.orbital_generation + 1,
        "correction did not advance determinant identity");
    for (std::size_t spin = 0; spin < a.d.size(); ++spin)
      for (std::size_t k = 0; k < 4; ++k)
        near(rebuilt.state->weighted_density[spin][k], force.state->weighted_density[spin][k],
             "forced rebuild W disagrees with analytic reuse");
  }
}

void identity_and_physical_origin() {
  const std::vector<std::function<void(FinalStateIdentity&)>> changes{
      [](auto& id) { ++id.factor.basis; },
      [](auto& id) { ++id.factor.reference; },
      [](auto& id) { ++id.factor.orbital_generation; },
      [](auto& id) { ++id.factor.density_generation; },
      [](auto& id) { ++id.solve_epoch; },
      [](auto& id) { id.occupied[0] = 0; },
      [](auto& id) { id.model.screening_tolerance *= 2; },
      [](auto& id) {
        id.model = resolve_fock_build(
            make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted),
            FockBackend::Cuda);
      }};
  for (const auto& change : changes) {
    Fixture a;
    change(a.c.identity);
    require(!a.valid(), "stale candidate identity accepted numerically identical data");
    a = Fixture{};
    change(a.f.identity);
    require(!a.valid(), "stale Fock identity accepted numerically identical data");
  }
  Fixture a;
  a.c.physical_origin = false;
  require(!a.valid(), "effective/DIIS-origin frame passed");
  auto corrected = a.select();
  require(corrected.state && corrected.eigen_solves == 1 && corrected.candidate_rejections == 1,
          "effective-origin frame did not require a physical correction");
  a = Fixture{};
  a.f.physical = false;
  require(!a.valid(), "nonphysical target Fock passed");
  for (const std::uint64_t origin : {0ULL, 10ULL}) {
    a = Fixture{};
    a.c.fock_density_generation = origin;
    require(!a.valid(), "unknown/future Fock origin generation passed");
  }
  // Old input-D provenance alone neither rejects a converged frame nor
  // authorizes it: the current Fock residual must still pass.
  a = Fixture{};
  a.f.spins[0][0] += 1e-5;
  require(!a.valid(), "F[D_old] frame passed mismatching current physical F");
}

void malformed_states_and_strict_gates() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::function<void(Fixture&)>> changes{
      [](auto& a) { a.id.factor.basis = 0; },
      [](auto& a) { a.id.factor.reference = 0; },
      [](auto& a) { a.id.solve_epoch = 0; },
      [](auto& a) { a.id.factor.orbital_generation = 0; },
      [](auto& a) { a.id.factor.density_generation = 0; },
      [](auto& a) { a.id.model.spec.spin = static_cast<FockSpin>(99); },
      [](auto& a) { a.id.model.precision = static_cast<FockPrecision>(99); },
      [](auto& a) { a.id.occupied = {3}; },
      [](auto& a) { a.d.clear(); },
      [](auto& a) { a.s.pop_back(); },
      [](auto& a) { a.h[1] = .1; },
      [](auto& a) { a.f.spins[0][1] = .1; },
      [](auto& a) { a.d[0] = {0, 0, 0, .5}; },
      [](auto& a) { a.d[0] = {.5, 0, 0, .25}; },
      [](auto& a) { a.d[0][0] += 1e-6; },
      [](auto& a) { a.c.spins[0].values[0] += 1e-6; },
      [](auto& a) { a.c.spins[0].vectors[0] *= 1.01; },
      [](auto& a) { a.c.spins.clear(); },
      [](auto& a) { a.c.spins[0].values.pop_back(); },
      [](auto& a) { a.c.spins[0].vectors.assign(4, std::numeric_limits<double>::max()); },
      [nan](auto& a) { a.d[0][0] = nan; },
      [nan](auto& a) { a.nuclear = nan; },
      [nan](auto& a) { a.limits.energy_tolerance = nan; },
      [](auto& a) { a.limits.density_tolerance = 0; },
      [](auto& a) { a.limits.maximum_corrections = std::numeric_limits<unsigned>::max(); }};
  for (const auto& change : changes) {
    Fixture a;
    change(a);
    require(!a.valid(), "malformed/inconsistent state passed");
  }
  Fixture a;
  a.d[0][0] += 1e-10;
  require(a.valid(), "default gate rejected a sub-tolerance density perturbation");
  a.limits.density_tolerance = 1e-12;
  require(!a.valid(), "tighter requested density tolerance was ignored");
  a = Fixture{};
  a.limits.density_tolerance = 1;
  a.d[0][0] += 1e-6;
  require(!a.valid(), "caller loosened the physical-reference absolute gates");
}

void degenerate_gauge() {
  Fixture a;
  a.id.occupied = {2};
  a.d = {{1, 0, 0, .5}};
  a.f = {a.id, true, {{-2, 0, 0, -4}}};
  const double t = std::sqrt(.5);
  a.c = {a.id, 8, true, {{{-1, -1}, {.5, -.5, .5 * t, .5 * t}}}};
  require(a.valid(), "occupied degenerate rotation failed gauge-invariant validation");
  const auto result = a.select(true);
  require(result.state.has_value(), "rotated degenerate state could not be selected");
  near(result.state->weighted_density[0][0], -1, "degenerate W changed with gauge");
  near(result.state->weighted_density[0][3], -.5, "degenerate W changed with gauge");
}

void malformed_candidates_and_provider_dimensions() {
  const Fixture a;
  for (const bool values : {false, true}) {
    auto candidate = a.c;
    (values ? candidate.spins[0].values : candidate.spins[0].vectors).pop_back();
    // A detached candidate can be replaced by a fresh, independently checked
    // solve. The same malformed dimensions from that solve are a hard failure.
    const auto corrected =
        select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, &candidate, a.physical(), a.eigen(),
                           a.limits, true, false, backend);
    require(corrected.state && !corrected.reused && corrected.candidate_rejections == 1 &&
                corrected.eigen_solves == 2 && corrected.density_updates == 1 &&
                corrected.fixed_point_eigen_solves == 1,
            "malformed detached candidate bypassed bounded correction");
    const initial_guess::EigenOperation malformed = [&](const auto&, const auto*, const auto*,
                                                        auto) { return candidate.spins[0]; };
    const auto failed = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, nullptr,
                                           a.physical(), malformed, a.limits, true, false, backend);
    require(!failed.state && failed.status == FinalStateStatus::ProviderFailure &&
                failed.eigen_solves == 1 && failed.density_updates == 0,
            "malformed eigen provider projected a density or published a state");
  }
}

void physical_reference_caps() {
  Fixture a;
  a.s = {1e-8, 0, 0, 1};
  a.x = {1e4, 0, 0, 1};
  a.c.spins[0] = {{-1, 3}, a.x};
  a.f.spins[0] = {-1e-8 + 1e-15, 0, 0, 3};
  a.d = {{2e8, 0, 0, 0}};
  require(a.valid(), "analytic scaled-residual fixture failed the generic contract");
  const auto ill_conditioned_force = a.select(true);
  require(ill_conditioned_force.state.has_value(), "ill-conditioned accepted state lost W");
  require(std::abs(ill_conditioned_force.state->weighted_density[0][0] - (-2e8 + 20)) < 1e-7,
          "ill-conditioned W used lagged eigenvalues instead of the physical Fock");
  a.limits.require_canonicality = true;
  require(!a.valid(), "physical-reference CFC amplification escaped absolute canonicality");
  a = Fixture{};
  a.s = a.x = {1, 0, 0, 1};
  a.c.spins[0] = {{-1, -.9}, a.x};
  a.f.spins[0] = {-1, 0, 0, -.9};
  // The occupied/virtual perturbation cancels to first order in idempotency
  // and has sub-gate RMS/commutator, but exceeds export's maximum D drift.
  a.d = {{2, 1.1e-8, 1.1e-8, 0}};
  require(a.valid(), "analytic maximum-density fixture failed the generic contract");
  a.limits.require_canonicality = true;
  require(!a.valid(), "physical-reference maximum density drift was replaced by RMS");
  a = Fixture{};
  a.limits.require_canonicality = true;
  require(a.select(true).state.has_value(), "canonical analytic state could not supply W/export");
}

void corrections_and_factor_invalidation() {
  Fixture a;
  // The first projection reaches the stationary D, but its energy differs
  // from the supplied D. A second evaluation confirms energy convergence.
  a.d[0] = {0, 0, 0, .5};
  a.limits.maximum_corrections = 1;
  auto result = a.select(true);
  require(!result.state && result.status == FinalStateStatus::NumericalFailure &&
              result.fock_evaluations == 2 && result.density_updates == 1,
          "one physical rebuild bypassed the energy-change gate");
  a.limits.maximum_corrections = 2;
  result = a.select(true);
  require(result.state && result.fock_evaluations == 3 && result.eigen_solves == 3 &&
              result.fixed_point_eigen_solves == 1,
          "bounded correction did not reach a consistent analytic state");
  near(result.state->diagnostic.energy, -2.7, "energy was not evaluated at returned D");
  require(result.state->identity.factor.density_generation == 11,
          "repeated corrections lost density generation");
  Fixture original;
  const Matrix occupied{original.x[0], 0}, occupations{2};
  OccupiedDensityFactor factor(original.id.factor, DensityFactorSpin::Restricted, 2, occupied,
                               occupations);
  require(factor.matches(original.id.factor, DensityFactorSpin::Restricted, factor.density()),
          "factor fixture is invalid");
  require(!factor.matches(result.state->identity.factor, DensityFactorSpin::Restricted,
                          factor.density()),
          "old factor survived corrected generations");
  original.limits.maximum_corrections = 0;
  require(original.select().state.has_value(), "validation-only mode rejected reuse");
  require(!original.select(false, true).state, "zero correction budget allowed forced rebuild");
  for (const bool orbital : {false, true}) {
    Fixture exhausted;
    auto& generation =
        orbital ? exhausted.id.factor.orbital_generation : exhausted.id.factor.density_generation;
    generation = std::numeric_limits<std::uint64_t>::max();
    exhausted.f.identity = exhausted.c.identity = exhausted.id;
    const auto failure = exhausted.select(false, true);
    require(!failure.state && failure.eigen_solves == 0, "generation overflow allowed correction");
  }
  require(Fixture{}.select().state.has_value(), "failed correction poisoned independent reuse");
}

void physical_fixed_point_and_consistent_weight() {
  // A small physical gap makes the old commutator/eigenframe gates pass even
  // though the occupied projector changes by more than the requested tolerance.
  // All matrices and the physical eigenvectors are analytic, including UHF's
  // empty beta channel. The probe must be reused when it triggers correction.
  for (const int beta : {-1, 0, 1}) {
    Fixture a;
    const double weight = beta < 0 ? 2 : 1;
    const double theta = 1.2e-10 / weight, c = std::cos(theta), s = std::sin(theta);
    a.s = a.x = {1, 0, 0, 1};
    a.h = {-1, 0, 0, -.99};
    a.f.spins = {a.h};
    a.c.spins = {{{-1, -.99}, {c, -s, s, c}}};
    a.d = {{weight * c * c, weight * c * s, weight * c * s, weight * s * s}};
    a.limits.density_tolerance = 1e-10;
    a.limits.energy_tolerance = 1e-12;
    if (beta >= 0) {
      a.id.model = resolve_fock_build(make_hf_fock_spec(FockSpin::Unrestricted), FockBackend::Cpu);
      a.id.occupied = {1, static_cast<std::size_t>(beta)};
      a.d.push_back({static_cast<double>(beta), 0, 0, 0});
      a.f.spins.push_back(a.h);
      a.c.spins.push_back({{-1, -.99}, a.x});
    }
    a.f.identity = a.c.identity = a.id;
    require(a.valid(), "small-gap fixture does not isolate the missing physical projector gate");
    const initial_guess::EigenOperation exact = [&](const Matrix& f, const Matrix*, const Matrix*,
                                                    std::size_t) {
      require(f == a.h, "fixed-point probe did not solve the actual physical Fock");
      return reference::EigenResult{{-1, -.99}, a.x};
    };
    const auto selected = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, &a.c,
                                             a.physical(), exact, a.limits, true, false, backend);
    require(selected.state && !selected.reused && selected.fock_evaluations == 2 &&
                selected.density_updates == 1 && selected.fixed_point_checks == 2 &&
                selected.fixed_point_rejections == 1 && selected.eigen_solves == 2 * a.d.size() &&
                selected.fixed_point_eigen_solves == selected.eigen_solves,
            "physical fixed-point correction was skipped, duplicated or not counted");
    near(selected.state->density[0][1], 0, "force retained the nonstationary density");
    near(selected.state->weighted_density[0][0], -weight, "physical Pulay weight is incorrect");
    require(
        selected.state->identity.factor.density_generation == a.id.factor.density_generation + 1,
        "fixed-point correction retained an old occupied-factor generation");
    a.limits.maximum_corrections = 0;
    const auto exhausted = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, &a.c,
                                              a.physical(), exact, a.limits, true, false, backend);
    require(
        !exhausted.state && exhausted.fixed_point_rejections == 1 && exhausted.density_updates == 0,
        "zero correction budget published a nonstationary force state");
  }
  Fixture a;
  a.s = a.x = {1, 0, 0, 1};
  a.d = {{2, 0, 0, 0}};
  a.f.spins[0] = {-1 + 1e-12, 0, 0, 3};
  a.c.spins[0] = {{-1, 3}, a.x};
  // Density is already a fixed point. A sub-gate diagonal eigenvalue lag must
  // not leak into the derivative weight when the current physical F is known.
  const auto selected = a.select(true);
  require(selected.state && selected.reused, "stationary density was needlessly replaced");
  near(selected.state->weighted_density[0][0], -2 + 2e-12,
       "force weight was built from lagged orbital eigenvalues");
}

void provider_failure_and_nonlinear_exhaustion() {
  Fixture a;
  for (const int fault : {0, 1, 2, 3, 4}) {
    const PhysicalFockOperation faulty = [&](const auto& id, const auto&) -> PhysicalFockFrame {
      if (fault == 0) throw std::runtime_error("physical evaluation failed");
      if (fault == 1) throw std::bad_alloc();
      if (fault == 2) throw 7;
      auto f = a.f;
      f.identity = id;
      if (fault == 3) --f.identity.factor.density_generation;
      if (fault == 4) f.physical = false;
      return f;
    };
    auto result = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, &a.c, faulty, a.eigen(),
                                     a.limits, true, false, backend);
    require(!result.state && result.fock_evaluations == 1 && result.eigen_solves == 0 &&
                result.status == (fault == 1 ? FinalStateStatus::OutOfMemory
                                             : FinalStateStatus::ProviderFailure),
            "invalid physical provider published a state or silently fell back");
  }
  for (const bool throws : {false, true}) {
    const initial_guess::EigenOperation bad = [&](const auto&, const auto*, const auto*,
                                                  auto) -> reference::EigenResult {
      if (throws) throw std::runtime_error("eigen provider failed");
      auto c = a.c.spins[0];
      c.values[0] += .1;
      return c;
    };
    const auto result = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, nullptr,
                                           a.physical(), bad, a.limits, true, false, backend);
    require(!result.state && result.status == FinalStateStatus::ProviderFailure &&
                result.eigen_solves == 1 && result.density_updates == 0,
            "failed eigen provider projected D or published a state");
  }
  // F[D] raises the occupied level, so each exact diagonalization swaps the
  // determinant. Every returned frame then fails against the newly evaluated
  // physical F: a small energy change cannot authorize this two-cycle.
  a.s = a.x = {1, 0, 0, 1};
  a.h = {0, 0, 0, 0};
  a.d = {{2, 0, 0, 0}};
  const PhysicalFockOperation oscillating = [](const auto& id, const auto& d) {
    return PhysicalFockFrame{id, true, {d[0]}};
  };
  const initial_guess::EigenOperation diagonal = [](const Matrix& f, const auto*, const auto*,
                                                    auto) {
    return f[0] > f[3] ? reference::EigenResult{{f[3], f[0]}, {0, 1, 1, 0}}
                       : reference::EigenResult{{f[0], f[3]}, {1, 0, 0, 1}};
  };
  const auto result = select_final_state(a.id, a.s, a.h, a.x, a.nuclear, a.d, nullptr, oscillating,
                                         diagonal, a.limits, true, false, backend);
  require(!result.state && result.status == FinalStateStatus::NumericalFailure &&
              result.fock_evaluations == 5 && result.eigen_solves == 4 &&
              result.density_updates == 4 && result.candidate_rejections == 4,
          "nonlinear correction cycle bypassed the strict physical residual");
}
}  // namespace

int main() {
  try {
#ifdef VIBEQC_TEST_DEVICE_FINAL_STATE
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    CudaDensityFittingJkPlan* raw{};
    std::vector<CudaDensityFittingMetricDiagnostic> metric;
    std::string detail;
    const auto status = create_cuda_density_fitting_jk_plan_tiled(
        0, 1, 2, 1, {1}, Matrix(4, 0), 1e-10, 1, 4, &raw, metric, detail);
    require(status == VIBEQC_STATUS_SUCCESS, detail.c_str());
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
        raw, destroy_cuda_density_fitting_jk_plan);
    const auto operations = cuda_density_fitting_final_state_operations(raw);
    backend = &operations;
#endif
    analytic_spin_states();
    identity_and_physical_origin();
    malformed_states_and_strict_gates();
    degenerate_gauge();
    malformed_candidates_and_provider_dimensions();
    physical_reference_caps();
    corrections_and_factor_invalidation();
    physical_fixed_point_and_consistent_weight();
    provider_failure_and_nonlinear_exhaustion();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
