#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/xc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/proposal_control.hpp"
#include "solver/self_consistent.hpp"
#include "xc_cpu_generated.hpp"

namespace vibeqc::scf {

FockBuildSpec make_global_hybrid_fock_spec(FockSpin spin, double exact_exchange) {
  if (!std::isfinite(exact_exchange))
    throw std::invalid_argument("global exact-exchange fraction must be finite");
  auto spec = make_hf_fock_spec(spin);
  spec.derivative_order = 0;
  const double spin_factor = spin == FockSpin::Restricted ? -0.5 : -1.0;
  spec.exchange.coefficient = spin_factor * exact_exchange;
  return spec;
}

FockBuildSpec make_rsh_primary_fock_spec(FockSpin spin, double short_range_exchange) {
  return make_global_hybrid_fock_spec(spin, short_range_exchange);
}

FockBuildSpec make_rsh_correction_fock_spec(FockSpin spin, double short_range_exchange,
                                            double long_range_exchange, double omega) {
  if (!std::isfinite(short_range_exchange) || !std::isfinite(long_range_exchange) ||
      !std::isfinite(omega) || omega < 0.0)
    throw std::invalid_argument(
        "RSH exchange fractions/omega must be finite and omega nonnegative");
  auto spec = make_hf_fock_spec(spin);
  spec.derivative_order = 0;
  spec.coulomb.present = false;
  const double spin_factor = spin == FockSpin::Restricted ? -0.5 : -1.0;
  spec.exchange = {true, spin_factor * (long_range_exchange - short_range_exchange),
                   FockOperator::LongRange, omega, FockApproximation::Exact};
  return spec;
}

void require_wb97mv_composition(const ResolvedFockBuild& primary,
                                const ResolvedFockBuild& correction,
                                const dft::nlc::Vv10Parameters& nonlocal) {
  validate_resolved_fock_build(primary);
  validate_resolved_fock_build(correction);
  const auto spin = primary.spec.spin;
  const auto expected_primary =
      resolve_fock_build(make_rsh_primary_fock_spec(spin, dft::generated::kWb97mvShortExchange),
                         FockBackend::Cpu, primary.screening_tolerance);
  const auto expected_correction =
      resolve_fock_build(make_rsh_correction_fock_spec(spin, dft::generated::kWb97mvShortExchange,
                                                       dft::generated::kWb97mvLongExchange,
                                                       dft::generated::kWb97mvOmega),
                         FockBackend::Cpu, primary.screening_tolerance);
  if (primary.backend != FockBackend::Cpu || correction.backend != FockBackend::Cpu ||
      primary != expected_primary || correction != expected_correction ||
      nonlocal.variant != dft::nlc::Vv10Variant::vv10 ||
      nonlocal.b != dft::generated::kWb97mvNonlocalB ||
      nonlocal.c != dft::generated::kWb97mvNonlocalC ||
      nonlocal.coefficient != dft::generated::kWb97mvNonlocalCoefficient)
    throw std::invalid_argument(
        "WB97M-V composition disagrees with the generated B97M/RSH/VV10 manifest");
}

namespace {

using initial_guess::prepare_initial_density;
using reference::commutator_residual;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::dot;
using reference::EigenResult;
using reference::generalized_eigen;
using reference::Matrix;
using reference::residual_rms;
using reference::symmetric_orthogonalizer;
using solver::Diis;
using solver::validate_seed;

// A run owns its immutable basis/grid binding and reference identity. IDs
// never alias across prepared replays or concurrent callers; exhaustion fails
// before wraparound rather than authorizing a previously exported factor.
std::uint64_t next_rks_identity() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("RKS density identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

struct RksEvaluation {
  Matrix fock;
  double energy{};
  dft::EnergyComponents components;
  dft::XcDensityDiagnostic density_diagnostic;
};

struct RksXcEvaluator {
  using Direct = dft::XcIntegral (*)(const dft::AoBasis&, const dft::MolecularGrid&, const Matrix&,
                                     dft::XcDensitySource, std::size_t, double, double);
  using CachedDirect = dft::XcIntegral (*)(
      const dft::AoBasis&, const dft::MolecularGrid&, const Matrix&, dft::XcDensitySource,
      std::size_t, double, double, const dft::RksAoCache&);
  Direct direct{};
  CachedDirect cached_direct{};
  const dft::SemilocalPointProgram* program{};

  RksXcEvaluator(Direct value, CachedDirect cached = nullptr)
      : direct(value), cached_direct(cached) {}
  RksXcEvaluator(const dft::SemilocalPointProgram& value) : program(&value) {}

  dft::XcIntegral operator()(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                             const Matrix& density, dft::XcDensitySource source, std::size_t tile,
                             double exchange_scale, double correlation_scale,
                             const dft::RksAoCache* cache = nullptr) const {
    if (program) {
      if (exchange_scale != 1.0 || correlation_scale != 1.0)
        throw std::invalid_argument("generic semilocal RKS does not accept legacy XC scaling");
      return dft::integrate_semilocal_rks(basis, grid, density, *program, tile, source);
    }
    if (cache && cached_direct)
      return cached_direct(basis, grid, density, source, tile, exchange_scale, correlation_scale,
                           *cache);
    if (!direct) throw std::logic_error("RKS XC evaluator is empty");
    return direct(basis, grid, density, source, tile, exchange_scale, correlation_scale);
  }
};

dft::XcIntegral evaluate_lda_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source,
                                    std::size_t tile, double exchange_scale,
                                    double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled LDA RKS is not qualified");
  return dft::integrate_lda_xc_pw_rks(basis, grid, density, tile, source);
}

dft::XcIntegral evaluate_pbe_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                    const Matrix& density, dft::XcDensitySource source,
                                    std::size_t tile, double exchange_scale,
                                    double correlation_scale) {
  return dft::integrate_pbe_rks_with_tail_scaled(basis, grid, density, tile, source, exchange_scale,
                                                 correlation_scale);
}

dft::XcIntegral evaluate_pbe_xc_rks_cached(
    const dft::AoBasis& basis, const dft::MolecularGrid& grid, const Matrix& density,
    dft::XcDensitySource source, std::size_t tile, double exchange_scale,
    double correlation_scale, const dft::RksAoCache& cache) {
  return dft::integrate_pbe_rks_with_tail_scaled_cached(
      basis, grid, density, tile, source, exchange_scale, correlation_scale, cache);
}

std::uint64_t fingerprint_mix(std::uint64_t hash, std::uint64_t value) noexcept {
  hash ^= value;
  return hash * 1099511628211ULL;
}

std::uint64_t fingerprint_double(double value) noexcept {
  std::uint64_t bits{};
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t geometry_fingerprint(const core::System& system) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(system.atoms.size()));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(system.charge));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(system.multiplicity));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(system.electron_count));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(system.basis_representation));
  for (const auto& atom : system.atoms) {
    hash = fingerprint_mix(hash, static_cast<std::uint64_t>(atom.atomic_number));
    hash = fingerprint_mix(hash, static_cast<std::uint64_t>(atom.ecp_core));
    for (const double coordinate : atom.position)
      hash = fingerprint_mix(hash, fingerprint_double(coordinate));
  }
  return hash;
}

std::uint64_t basis_fingerprint(const dft::AoBasis& basis) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(basis.natom));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(basis.nprimitive));
  hash = fingerprint_mix(hash, static_cast<std::uint64_t>(basis.nao));
  for (const double value : basis.packed) hash = fingerprint_mix(hash, fingerprint_double(value));
  return hash;
}

/** Immutable scientific/execution identity for one exact incremental-XC solve.
 * The unique owner prevents same-shaped models from aliasing; the fingerprints
 * and explicit policy fields make the binding auditable instead of relying on
 * dimensions alone. A new prepared replay always mints a new owner. */
struct IncrementalPbeRksModelIdentity {
  std::uint64_t owner{};
  std::uint64_t geometry{};
  std::uint64_t basis{};
  dft::GridSpec grid;
  std::size_t grid_points{};
  std::uint32_t functional_family{1};  // PBE generated semilocal family.
  std::uint32_t regularization_version{1};
  double exchange_scale{1.0};
  double correlation_scale{1.0};
  double screening_tolerance{};
  int32_t requested_precision{VIBEQC_PRECISION_FP64};
  std::uint32_t effective_precision_bits{64};
  ScfOptions::XcExecutionSchedule math_mode{ScfOptions::XcExecutionSchedule::DeviceFused};
  dft::XcDensityRoute source{dft::XcDensityRoute::DensityMatrix};

  bool operator==(const IncrementalPbeRksModelIdentity&) const = default;
};

struct IncrementalPbeRksAnchorIdentity {
  IncrementalPbeRksModelIdentity model;
  std::uint64_t source_generation{};
};

/** Exact #237 Slice-B controller. The anchor is deliberately run-local: the
 * enclosing RKS solve owns immutable basis/grid references, and a prepared
 * replay constructs a fresh controller, so geometry/basis/grid changes cannot
 * inherit stale XC state. Anchor replacement is transactional: a failed full
 * build leaves the preceding anchor untouched. */
struct IncrementalPbeRksState {
  const dft::AoBasis& basis;
  const dft::MolecularGrid& grid;
  IncrementalPbeRksModelIdentity model;
  std::size_t tile{};
  double exchange_scale{1.0}, correlation_scale{1.0};
  std::size_t max_updates{};
  double max_density_rms{};
  double noise_density_rms{};
  std::size_t stagnation_iterations{};
  dft::IncrementalXcDiagnostic& diagnostic;
  Matrix anchor_density;
  std::optional<IncrementalPbeRksAnchorIdentity> anchor_identity;
  std::size_t updates_since_rebuild{};
  std::size_t stagnation_count{};
  double best_residual{std::numeric_limits<double>::infinity()};
  bool rebuild_for_stagnation{};
  bool strict_only{};

  std::size_t numeric_capacity() const noexcept { return runtime::vector_bytes(anchor_density); }

  void validate_identity(const IncrementalPbeRksModelIdentity& current) const {
    if (!(current == model))
      throw std::invalid_argument("incremental XC model identity changed within one SCF solve");
    if (anchor_identity && !(anchor_identity->model == current))
      throw std::logic_error("incremental XC anchor identity is stale");
  }

  double anchor_delta_rms(const Matrix& density) const {
    if (density.size() != anchor_density.size())
      throw std::invalid_argument("incremental XC anchor density dimensions changed");
    double square_sum = 0.0;
    for (std::size_t i = 0; i < density.size(); ++i) {
      const double delta = density[i] - anchor_density[i];
      square_sum += delta * delta;
    }
    return std::sqrt(square_sum / static_cast<double>(density.size()));
  }

  dft::XcIntegral full_build(const Matrix& density, std::size_t external_retained_bytes,
                             bool periodic, bool drift, bool noise, bool stagnation, bool fallback,
                             bool strict_final, bool replace_anchor) {
    const auto old_anchor_bytes = numeric_capacity();
    auto full = dft::integrate_pbe_rks_with_tail_scaled(basis, grid, density, tile, {},
                                                        exchange_scale, correlation_scale);
    ++diagnostic.full_builds;
    diagnostic.periodic_rebuilds += periodic ? 1U : 0U;
    diagnostic.drift_rebuilds += drift ? 1U : 0U;
    diagnostic.noise_rebuilds += noise ? 1U : 0U;
    diagnostic.stagnation_rebuilds += stagnation ? 1U : 0U;
    diagnostic.fallback_rebuilds += fallback ? 1U : 0U;
    diagnostic.strict_final_builds += strict_final ? 1U : 0U;
    if (replace_anchor) {
      // Construct both replacement pieces before mutating the accepted anchor.
      // If allocation/full evaluation fails, the preceding anchor stays valid.
      Matrix replacement_density = density;
      if (diagnostic.anchor_generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("incremental XC anchor generation exhausted");
      const IncrementalPbeRksAnchorIdentity replacement_identity{model,
                                                                 diagnostic.anchor_generation + 1};
      const auto replacement_bytes = runtime::vector_bytes(replacement_density);
      diagnostic.peak_replacement_overlap_bytes =
          std::max(diagnostic.peak_replacement_overlap_bytes,
                   runtime::add_capacity(old_anchor_bytes, replacement_bytes));
      runtime::sample_cpu_capacity(runtime::add_capacity(
          external_retained_bytes,
          runtime::add_capacity(runtime::add_capacity(old_anchor_bytes, replacement_bytes),
                                full.density_diagnostic.owned_numeric_bytes)));
      anchor_density.swap(replacement_density);
      anchor_identity = replacement_identity;
      diagnostic.anchor_generation = replacement_identity.source_generation;
      diagnostic.retained_anchor_bytes = numeric_capacity();
      updates_since_rebuild = 0;
      rebuild_for_stagnation = false;
    } else {
      runtime::sample_cpu_capacity(runtime::add_capacity(
          external_retained_bytes, full.density_diagnostic.owned_numeric_bytes));
    }
    return full;
  }

  dft::XcIntegral evaluate(const Matrix& density, const IncrementalPbeRksModelIdentity& current,
                           std::size_t external_retained_bytes, bool strict_final = false) {
    validate_identity(current);
    if (strict_only || strict_final)
      return full_build(density, external_retained_bytes, false, false, false, false, false, true,
                        false);
    if (anchor_density.empty())
      return full_build(density, external_retained_bytes, false, false, false, false, false, false,
                        true);
    if (updates_since_rebuild >= max_updates)
      return full_build(density, external_retained_bytes, true, false, false, false, false, false,
                        true);
    const double drift = anchor_delta_rms(density);
    diagnostic.max_anchor_delta_rms = std::max(diagnostic.max_anchor_delta_rms, drift);
    if (!std::isfinite(drift) || drift > max_density_rms)
      return full_build(density, external_retained_bytes, false, true, false, false, false, false,
                        true);
    if (noise_density_rms > 0.0 && drift > 0.0 && drift < noise_density_rms)
      return full_build(density, external_retained_bytes, false, false, true, false, false, false,
                        true);
    if (rebuild_for_stagnation)
      return full_build(density, external_retained_bytes, false, false, false, true, false, false,
                        true);

    Matrix delta(density.size());
    for (std::size_t i = 0; i < density.size(); ++i) delta[i] = density[i] - anchor_density[i];
    const auto update_bytes = runtime::vector_bytes(delta);
    diagnostic.peak_update_buffer_bytes =
        std::max(diagnostic.peak_update_buffer_bytes, update_bytes);
    try {
      auto incremental = dft::integrate_pbe_rks_incremental_exact(
          basis, grid, anchor_density, delta, tile, exchange_scale, correlation_scale);
      runtime::sample_cpu_capacity(runtime::add_capacity(
          external_retained_bytes,
          runtime::add_capacity(runtime::add_capacity(numeric_capacity(), update_bytes),
                                incremental.total.density_diagnostic.owned_numeric_bytes)));
      ++diagnostic.incremental_updates;
      ++updates_since_rebuild;
      return std::move(incremental.total);
    } catch (const std::domain_error&) {
      return full_build(density, runtime::add_capacity(external_retained_bytes, update_bytes),
                        false, false, false, false, true, false, true);
    } catch (const std::runtime_error&) {
      return full_build(density, runtime::add_capacity(external_retained_bytes, update_bytes),
                        false, false, false, false, true, false, true);
    }
  }

  void observe_progress(double physical_residual) noexcept {
    if (strict_only || stagnation_iterations == 0 || !std::isfinite(physical_residual)) return;
    if (physical_residual < best_residual * (1.0 - 1.0e-3)) {
      best_residual = physical_residual;
      stagnation_count = 0;
      return;
    }
    if (++stagnation_count >= stagnation_iterations) {
      rebuild_for_stagnation = true;
      stagnation_count = 0;
    }
  }

  void enter_strict_refinement() noexcept {
    strict_only = true;
    Matrix empty;
    anchor_density.swap(empty);
    anchor_identity.reset();
    updates_since_rebuild = 0;
    diagnostic.retained_anchor_bytes = 0;
  }
};

dft::XcIntegral evaluate_r2scan_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                       const Matrix& density, dft::XcDensitySource source,
                                       std::size_t tile, double exchange_scale,
                                       double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled r2SCAN RKS is not qualified");
  return dft::integrate_r2scan_rks(basis, grid, density, tile, source);
}

dft::XcIntegral evaluate_b3lyp_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                      const Matrix& density, dft::XcDensitySource source,
                                      std::size_t tile, double exchange_scale,
                                      double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled B3LYP RKS is not qualified");
  return dft::integrate_b3lyp_rks(basis, grid, density, tile, source);
}

dft::XcIntegral evaluate_wb97mv_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                       const Matrix& density, dft::XcDensitySource source,
                                       std::size_t tile, double exchange_scale,
                                       double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled WB97M-V semilocal execution is not qualified");
  return dft::integrate_wb97mv_rks(basis, grid, density, tile, source);
}

dft::XcIntegral evaluate_cam_b3lyp_xc_rks(const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                                          const Matrix& density, dft::XcDensitySource source,
                                          std::size_t tile, double exchange_scale,
                                          double correlation_scale) {
  if (exchange_scale != 1.0 || correlation_scale != 1.0)
    throw std::invalid_argument("scaled CAM-B3LYP RKS is not qualified");
  return dft::integrate_cam_b3lyp_rks(basis, grid, density, tile, source);
}

RksEvaluation evaluate_rks(const PreparedFockPlan& plan,
                           const PreparedFockPlan* long_range_correction, const dft::AoBasis& basis,
                           const dft::MolecularGrid& grid, const Matrix& density,
                           RksXcEvaluator evaluate_xc, const char* method_name,
                           dft::XcDensitySource source, std::size_t retained_capacity,
                           std::size_t tile, double exchange_scale, double correlation_scale,
                           dft::nlc::Vv10Plan* nonlocal_correlation,
                           dft::nlc::Vv10DensityDomain nonlocal_domain,
                           const dft::RksAoCache* ao_cache,
                           std::optional<dft::XcIntegral> xc_override = std::nullopt) {
  const auto& strategy = plan.strategy();
  const auto& ints = plan.one_electron();
  const auto jk = plan.build(density);
  RksEvaluation result;
  result.fock = assemble_fock(strategy, ints.hcore, jk).alpha;
  const auto primary_energy = contract_fock_energy_components(strategy, jk, density);
  double exact_exchange = primary_energy.exchange;
  DirectJkMatrices correction_jk;
  if (long_range_correction) {
    correction_jk = long_range_correction->build(density);
    const auto& correction_strategy = long_range_correction->strategy();
    if (correction_jk.exchange_alpha.size() != result.fock.size())
      throw std::runtime_error("RSH correction exchange dimensions do not match the Fock matrix");
    for (std::size_t i = 0; i < result.fock.size(); ++i)
      result.fock[i] +=
          correction_strategy.spec.exchange.coefficient * correction_jk.exchange_alpha[i];
    exact_exchange +=
        contract_fock_energy_components(correction_strategy, correction_jk, density).exchange;
  }
  auto xc = xc_override.has_value()
                ? std::move(*xc_override)
                : evaluate_xc(basis, grid, density, source, tile, exchange_scale,
                              correlation_scale, ao_cache);
  result.density_diagnostic = xc.density_diagnostic;
  dft::nlc::Vv10Integral nonlocal;
  if (nonlocal_correlation)
    nonlocal = dft::nlc::integrate_vv10_rks(basis, grid, density, *nonlocal_correlation, tile, {},
                                            nonlocal_domain);
  // The semilocal/nonlocal AO tiles and potentials are live together with
  // these J/Fock buffers. Their reported peaks exclude borrowed D/factors.
  const auto scientific_peak =
      runtime::add_capacity(xc.density_diagnostic.owned_numeric_bytes,
                            nonlocal_correlation ? nonlocal.owned_numeric_bytes : 0);
  runtime::sample_cpu_capacity(runtime::add_capacity(
      retained_capacity,
      runtime::add_capacity(
          scientific_peak,
          runtime::vector_capacities(result.fock, jk.coulomb, jk.exchange_alpha, jk.exchange_beta,
                                     correction_jk.coulomb, correction_jk.exchange_alpha,
                                     correction_jk.exchange_beta))));
  if (xc.potential.size() != result.fock.size() ||
      (nonlocal_correlation && nonlocal.potential.size() != result.fock.size()))
    throw std::runtime_error(std::string(method_name) +
                             " XC potential dimensions do not match the Fock matrix");
  for (std::size_t i = 0; i < result.fock.size(); ++i) {
    result.fock[i] += xc.potential[i];
    if (nonlocal_correlation) result.fock[i] += nonlocal.potential[i];
  }
  result.components = {ints.nuclear_repulsion, dot(density, ints.hcore), primary_energy.coulomb,
                       xc.energy + (nonlocal_correlation ? nonlocal.energy : 0.0), exact_exchange};
  result.energy = result.components.total();
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method_name + " RKS energy");
  return result;
}

ScfResult run_rks(
    const PreparedFockPlan& plan, const PreparedFockPlan* long_range_correction,
    const dft::AoBasis& basis, const dft::MolecularGrid& grid, const ScfOptions& options,
    const std::vector<double>* initial_density, RksXcEvaluator evaluate_xc, const char* method_name,
    dft::nlc::Vv10Plan* nonlocal_correlation,
    dft::nlc::Vv10DensityDomain nonlocal_domain = dft::nlc::Vv10DensityDomain::StrictPositive) {
  if (options.xc_density_route != dft::XcDensityRoute::DensityMatrix &&
      options.xc_density_route != dft::XcDensityRoute::OccupiedOrbitals)
    throw std::invalid_argument("unsupported RKS XC density route");
  const auto& strategy = plan.strategy();
  validate_resolved_fock_build(strategy);
  const auto& system = plan.system();
  const auto& ints = plan.one_electron();
  if (options.compute_forces)
    throw std::invalid_argument(std::string(method_name) + " RKS forces are not implemented");
  if (strategy.backend != FockBackend::Cpu || strategy.spec.spin != FockSpin::Restricted ||
      strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
      strategy.spec.coulomb.coefficient != 1.0 ||
      (strategy.spec.exchange.present &&
       (strategy.spec.exchange.op != FockOperator::FullRange ||
        (strategy.spec.exchange.approximation != FockApproximation::Exact &&
         strategy.spec.exchange.approximation != FockApproximation::DensityFitted))))
    throw std::invalid_argument(std::string(method_name) +
                                " RKS requires a CPU full-range exact or fitted J/K Fock strategy");
  if (long_range_correction) {
    const auto& correction = long_range_correction->strategy();
    validate_resolved_fock_build(correction);
    const bool primary_exchange =
        strategy.spec.exchange.present &&
        strategy.spec.exchange.approximation == FockApproximation::Exact &&
        strategy.spec.exchange.op == FockOperator::FullRange && strategy.spec.exchange.omega == 0.0;
    const bool correction_exchange =
        correction.backend == FockBackend::Cpu && correction.spec.spin == FockSpin::Restricted &&
        correction.spec.derivative_order == 0 && !correction.spec.coulomb.present &&
        correction.spec.exchange.present &&
        correction.spec.exchange.approximation == FockApproximation::Exact &&
        correction.spec.exchange.op == FockOperator::LongRange;
    if (!primary_exchange || !correction_exchange)
      throw std::invalid_argument("RSH RKS requires full-range primary K plus direct long-range K");
  }
  if (system.electron_count <= 0 || system.electron_count % 2 || system.multiplicity != 1)
    throw std::invalid_argument(std::string(method_name) +
                                " RKS requires a closed-shell electron count");
  if (basis.nao != ints.nbf || basis.natom != system.atoms.size() || grid.point_count() == 0 ||
      grid.system().atoms.size() != system.atoms.size())
    throw std::invalid_argument(std::string(method_name) +
                                " RKS prepared grid/basis state is inconsistent");

  // The source already owns its auxiliary basis. Comparing against a null
  // auxiliary request would incorrectly replace it with the orbital basis.
  if (!plan.matches_system(grid.system()) || basis.packed != dft::AoBasis(system).packed)
    throw std::invalid_argument("RKS refuses a stale geometry, basis, charge or spin binding");
  if (long_range_correction &&
      (!long_range_correction->matches(system, nullptr, long_range_correction->strategy(), -1, 0) ||
       long_range_correction->one_electron().nbf != ints.nbf))
    throw std::invalid_argument("RSH correction refuses a stale or incompatible source binding");

  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) throw std::runtime_error("basis has fewer orbitals than occupied pairs");
  const Matrix orthogonalizer = symmetric_orthogonalizer(ints.overlap, n);
  std::optional<EigenResult> initial_orbitals;
  Matrix density = prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density,
                                           initial_orbitals);
  // Only a cold seed carries a core frame. Warm consumers solve their first
  // target Fock before reading orbitals; RKS packs its initial factor cold-only.
  EigenResult orbitals = std::move(initial_orbitals).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0);
    density = *initial_density;
  }
  Diis diis(options.diis_history, true);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  auto& ks = result.dft_diagnostic;
  ks.occupations = {occupied, occupied};
  ks.grid_points = grid.point_count();
  ks.tile_points = std::min(options.xc_tile_points, grid.point_count());
  ks.ao_order = evaluate_xc.program ? (evaluate_xc.program->ingredient_mask == 1U ? 0U : 1U)
                                    : (std::string_view(method_name) == "LDA" ? 0U : 1U);
  ks.scf_domain_version = evaluate_xc.program
                              ? evaluate_xc.program->domain_version
                              : (std::string_view(method_name) == "WB97M-V"
                                     ? 3U
                                     : (std::string_view(method_name) == "B3LYP" ? 2U : 1U));
  auto& diagnostic = result.xc_density_diagnostic;
  diagnostic.physical_residual = std::numeric_limits<double>::infinity();
  const bool incremental_xc = options.experimental_incremental_xc;
  if (incremental_xc && (evaluate_xc.program || std::string_view(method_name) != "PBE" ||
                         options.xc_density_route != dft::XcDensityRoute::DensityMatrix ||
                         nonlocal_correlation || options.incremental_xc_max_updates == 0 ||
                         !std::isfinite(options.incremental_xc_max_density_rms) ||
                         options.incremental_xc_max_density_rms <= 0.0 ||
                         !std::isfinite(options.incremental_xc_noise_density_rms) ||
                         options.incremental_xc_noise_density_rms < 0.0))
    throw std::invalid_argument(
        "incremental XC requires CPU semilocal PBE RKS, density-matrix XC, positive bounded "
        "policy");
  ks.incremental_xc.enabled = incremental_xc;
  std::optional<IncrementalPbeRksModelIdentity> incremental_identity;
  std::optional<IncrementalPbeRksState> incremental_state;
  if (incremental_xc) {
    const auto owner = next_rks_identity();
    incremental_identity.emplace(IncrementalPbeRksModelIdentity{
        owner, geometry_fingerprint(system), basis_fingerprint(basis), grid.spec(),
        grid.point_count(), 1U, ks.scf_domain_version, options.semilocal_exchange_scale,
        options.semilocal_correlation_scale, options.screening_tolerance,
        static_cast<int32_t>(options.precision_mode.value_or(VIBEQC_PRECISION_FP64)), 64U,
        options.xc_execution_schedule, dft::XcDensityRoute::DensityMatrix});
    ks.incremental_xc.model_identity = owner;
    incremental_state.emplace(IncrementalPbeRksState{
        basis, grid, *incremental_identity, options.xc_tile_points,
        options.semilocal_exchange_scale, options.semilocal_correlation_scale,
        options.incremental_xc_max_updates, options.incremental_xc_max_density_rms,
        options.incremental_xc_noise_density_rms, options.incremental_xc_stagnation_iterations,
        ks.incremental_xc});
  }
  // Repeated CPU SCF builds see an immutable geometry/grid. Retain the complete
  // order-1 AO grid only when its exact FP64 footprint is bounded; larger
  // workloads keep the existing tile-streaming recomputation path.
  constexpr std::size_t kCpuRksAoCacheMaximumBytes = 64ULL * 1024ULL * 1024ULL;
  std::optional<dft::RksAoCache> ao_cache;
  if (!incremental_xc && evaluate_xc.cached_direct) {
    const auto cache_bytes = dft::rks_ao_cache_bytes(basis, grid, ks.ao_order);
    if (cache_bytes <= kCpuRksAoCacheMaximumBytes)
      ao_cache.emplace(dft::prepare_rks_ao_cache(basis, grid, ks.ao_order));
  }
  std::shared_ptr<const OccupiedDensityFactor> factor;
  DensityFactorIdentity identity{};
  const bool use_orbitals = options.xc_density_route == dft::XcDensityRoute::OccupiedOrbitals;
  if (use_orbitals) {
    const auto owner = next_rks_identity();
    identity = {owner, owner, 0, 0};
  }
  const auto retained_without_incremental = [&](const Matrix& current_density) {
    const auto provider_capacity = runtime::add_capacity(
        plan.cpu_observation_capacity(),
        long_range_correction ? long_range_correction->cpu_observation_capacity() : 0);
    return runtime::add_capacity(
        runtime::add_capacity(provider_capacity, diis.numeric_capacity()),
        runtime::add_capacity(
            runtime::add_capacity(factor ? factor->numeric_capacity_bytes() : 0,
                                  ao_cache ? ao_cache->numeric_capacity_bytes() : 0),
            runtime::vector_capacities(orthogonalizer, current_density, orbitals.values,
                                       orbitals.vectors, basis.packed, grid.points(),
                                       grid.weights(), grid.owners(), ks.history)));
  };
  const auto retained_capacity = [&](const Matrix& current_density) {
    return runtime::add_capacity(retained_without_incremental(current_density),
                                 incremental_state ? incremental_state->numeric_capacity() : 0);
  };
  const auto make_current_factor = [&](const Matrix& current_density,
                                       std::size_t extra_live_bytes = 0) {
    // Only an actual unmixed eigensolver state advances both generations.
    // Release the previous snapshot before packing the new one. Current D
    // remains owned for convergence tests and the Coulomb provider.
    factor.reset();
    ++identity.orbital_generation;
    ++identity.density_generation;
    Matrix packed(n * occupied), occupations(occupied, 2.0);
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t o = 0; o < occupied; ++o)
        packed[mu * occupied + o] = orbitals.vectors[mu * n + o];
    factor = std::make_shared<const OccupiedDensityFactor>(identity, DensityFactorSpin::Restricted,
                                                           n, packed, occupations);
    const auto packing_bytes = runtime::vector_capacities(packed, occupations);
    diagnostic.packed_coefficient_elements += packed.size();
    diagnostic.factor_peak_bytes =
        std::max(diagnostic.factor_peak_bytes,
                 runtime::add_capacity(factor->numeric_capacity_bytes(), packing_bytes));
    runtime::sample_cpu_capacity(
        runtime::add_capacity(retained_capacity(current_density),
                              runtime::add_capacity(packing_bytes, extra_live_bytes)));
  };
  if (use_orbitals && !initial_density) make_current_factor(density);
  const auto evaluate_current = [&](const Matrix& current_density, std::size_t extra_live_bytes = 0,
                                    bool require_full_xc = false) {
    std::optional<dft::XcIntegral> xc_override;
    if (incremental_state) {
      if (!incremental_identity) throw std::logic_error("incremental XC identity was not prepared");
      xc_override.emplace(incremental_state->evaluate(
          current_density, *incremental_identity,
          runtime::add_capacity(retained_without_incremental(current_density), extra_live_bytes),
          require_full_xc));
    }
    auto physical =
        evaluate_rks(plan, long_range_correction, basis, grid, current_density, evaluate_xc,
                     method_name, {options.xc_density_route, factor.get(), identity},
                     runtime::add_capacity(retained_capacity(current_density), extra_live_bytes),
                     options.xc_tile_points, options.semilocal_exchange_scale,
                     options.semilocal_correlation_scale, nonlocal_correlation, nonlocal_domain,
                     ao_cache ? &*ao_cache : nullptr, std::move(xc_override));
    const auto& record = physical.density_diagnostic;
    if (record.executed == dft::XcDensityRoute::OccupiedOrbitals)
      ++diagnostic.orbital_calls;
    else
      ++diagnostic.density_calls;
    if (record.fallback != dft::XcDensityFallback::None) ++diagnostic.fallback_calls;
    diagnostic.xc_peak_bytes = std::max(diagnostic.xc_peak_bytes, record.owned_numeric_bytes);
    return physical;
  };
  const auto next_density_from_orbitals = [&](const Matrix& current_density,
                                              std::size_t extra_live_bytes = 0) {
    if (use_orbitals) make_current_factor(current_density, extra_live_bytes);
    // Reuse the producer's exact witness rather than reconstructing D twice.
    Matrix next = use_orbitals ? Matrix(factor->density().begin(), factor->density().end())
                               : density_from_orbitals(orbitals.vectors, n, occupied);
    runtime::sample_cpu_capacity(runtime::add_capacity(
        retained_capacity(current_density),
        runtime::add_capacity(runtime::vector_bytes(next), extra_live_bytes)));
    return next;
  };
  const auto retain_factor = [&] {
    result.xc_density_factor = factor;
    diagnostic.final_identity = identity;
  };
  struct RksLoopEvaluation {
    std::shared_ptr<const OccupiedDensityFactor> current_factor;
    DensityFactorIdentity current_identity{};
    dft::EnergyComponents components;
    Matrix next_density;
    double energy{};
    double state_rms{};
    double residual_rms{};
    double spin_electrons{};
  };

  if (incremental_xc) {
    if (!incremental_state) throw std::logic_error("incremental XC controller was not prepared");
    const double residual_tolerance = std::min(1.0e-9, options.density_tolerance);
    const auto run_stage = [&](Matrix stage_density, bool strict_full, unsigned iteration_offset,
                               unsigned iteration_budget) {
      const ::vibeqc::solver::SelfConsistentPolicy stage_policy{
          iteration_budget, options.energy_tolerance, options.density_tolerance, residual_tolerance,
          true};
      return ::vibeqc::solver::run_self_consistent(
          std::move(stage_density), stage_policy,
          [&](const Matrix& current_density, unsigned) {
            const auto current_factor = factor;
            const auto current_identity = identity;
            ++result.fock_builds;
            const auto physical = evaluate_current(current_density, 0, strict_full);
            const Matrix residual =
                commutator_residual(physical.fock, current_density, ints.overlap, n);
            const Matrix effective_fock = diis.update(physical.fock, residual);
            orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
            const auto iteration_bytes =
                runtime::vector_capacities(physical.fock, residual, effective_fock);
            Matrix next_density = next_density_from_orbitals(current_density, iteration_bytes);
            runtime::sample_cpu_capacity(runtime::add_capacity(
                retained_capacity(current_density),
                runtime::add_capacity(iteration_bytes, runtime::vector_bytes(next_density))));
            const double state_rms = density_rms(next_density, current_density);
            const double physical_residual = residual_rms(residual);
            const double spin_electrons = dot(current_density, ints.overlap) / 2.0;
            return RksLoopEvaluation{current_factor,          current_identity, physical.components,
                                     std::move(next_density), physical.energy,  state_rms,
                                     physical_residual,       spin_electrons};
          },
          [&](Matrix& current_density, RksLoopEvaluation evaluation,
              const ::vibeqc::solver::SelfConsistentProgress& progress) {
            if (strict_full && progress.converged) {
              // The independent final audit must rebuild the exact density that
              // actually passed the strict physical criteria, not an unchecked
              // orbital proposal generated after that evaluation.
              factor = std::move(evaluation.current_factor);
              identity = evaluation.current_identity;
              return std::move(current_density);
            }
            if (!progress.converged && progress.iteration == iteration_budget) {
              factor = std::move(evaluation.current_factor);
              identity = evaluation.current_identity;
              return std::move(current_density);
            }
            return std::move(evaluation.next_density);
          },
          [&](const ::vibeqc::solver::SelfConsistentProgress& progress,
              const RksLoopEvaluation& evaluation) {
            const unsigned reported_iteration = iteration_offset + progress.iteration;
            result.iterations = reported_iteration;
            result.energy = progress.energy;
            result.energy_change = progress.energy_change;
            result.density_rms = progress.state_rms;
            ks.physical_residual = progress.residual_rms;
            result.physical_residual_rms = ks.physical_residual;
            ks.components = evaluation.components;
            ks.electrons = {evaluation.spin_electrons, evaluation.spin_electrons};
            ks.density_change = progress.state_rms;
            ks.history.push_back({reported_iteration,
                                  evaluation.components,
                                  progress.energy_change,
                                  progress.state_rms,
                                  progress.residual_rms,
                                  {evaluation.spin_electrons, evaluation.spin_electrons}});
            if (!strict_full) incremental_state->observe_progress(progress.residual_rms);
          });
    };

    // Stage one may converge through exact anchor-relative updates or exhaust
    // its normal budget. Either way, it is only an accelerator: strict target
    // work gets a fresh DIIS history and its own full budget, matching the
    // existing mixed-precision refinement semantics.
    auto stage = run_stage(std::move(density), false, 0, options.max_iterations);
    density = std::move(stage.state);
    const unsigned accelerated_iterations = stage.progress.iteration;
    result.converged = false;
    diis.clear();
    incremental_state->enter_strict_refinement();

    unsigned strict_iterations = 0;
    while (strict_iterations < options.max_iterations) {
      const unsigned remaining = options.max_iterations - strict_iterations;
      if (remaining < 2) break;
      const unsigned offset = accelerated_iterations + strict_iterations;
      auto strict = run_stage(std::move(density), true, offset, remaining);
      density = std::move(strict.state);
      strict_iterations += strict.progress.iteration;
      ks.incremental_xc.strict_refinement_iterations += strict.progress.iteration;
      if (!strict.converged) break;

      // Independent physical audit: rebuild the requested target at the exact
      // candidate density after SCF declared convergence. A failed audit does
      // not publish success; it clears nonlinear history and spends the
      // remaining strict budget on further full iterations.
      ++result.fock_builds;
      ++ks.incremental_xc.final_audits;
      auto audit = evaluate_current(density, 0, true);
      const auto audit_residual = commutator_residual(audit.fock, density, ints.overlap, n);
      const double physical_residual = residual_rms(audit_residual);
      const double energy_change = std::abs(audit.energy - result.energy);
      const bool audit_passed = energy_change < options.energy_tolerance &&
                                result.density_rms < options.density_tolerance &&
                                physical_residual < residual_tolerance;
      diagnostic.physical_residual = physical_residual;
      ks.physical_residual = physical_residual;
      result.physical_residual_rms = physical_residual;
      result.energy_change = energy_change;
      result.energy = audit.energy;
      ks.components = audit.components;
      const double final_spin_electrons = dot(density, ints.overlap) / 2.0;
      ks.electrons = {final_spin_electrons, final_spin_electrons};
      ks.density_change = result.density_rms;
      runtime::sample_cpu_capacity(runtime::add_capacity(
          retained_capacity(density), runtime::vector_capacities(audit.fock, audit_residual)));
      if (audit_passed) {
        result.converged = true;
        if (options.retain_ks_state) result.ks_physical_fock = std::move(audit.fock);
        break;
      }
      ++ks.incremental_xc.audit_failures;
      diis.clear();
    }

    diagnostic.physical_residual = ks.physical_residual;
    retain_factor();
    result.density = std::move(density);
    return result;
  }

  const ::vibeqc::solver::SelfConsistentPolicy policy{
      options.max_iterations, options.energy_tolerance, options.density_tolerance,
      std::min(1.0e-9, options.density_tolerance), true};
  auto outcome = ::vibeqc::solver::run_self_consistent(
      std::move(density), policy,
      [&](const Matrix& current_density, unsigned) {
        const auto current_factor = factor;
        const auto current_identity = identity;
        ++result.fock_builds;
        const auto physical = evaluate_current(current_density);
        const Matrix residual =
            commutator_residual(physical.fock, current_density, ints.overlap, n);
        const Matrix effective_fock = diis.update(physical.fock, residual);
        orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
        const auto iteration_bytes =
            runtime::vector_capacities(physical.fock, residual, effective_fock);
        Matrix next_density = next_density_from_orbitals(current_density, iteration_bytes);

        runtime::sample_cpu_capacity(runtime::add_capacity(
            retained_capacity(current_density),
            runtime::add_capacity(iteration_bytes, runtime::vector_bytes(next_density))));
        const double state_rms = density_rms(next_density, current_density);
        const double physical_residual = residual_rms(residual);
        const double spin_electrons = dot(current_density, ints.overlap) / 2.0;
        return RksLoopEvaluation{current_factor,          current_identity, physical.components,
                                 std::move(next_density), physical.energy,  state_rms,
                                 physical_residual,       spin_electrons};
      },
      [&](Matrix& current_density, RksLoopEvaluation evaluation,
          const ::vibeqc::solver::SelfConsistentProgress& progress) {
        if (!progress.converged && progress.iteration == options.max_iterations) {
          // A failed return must keep E/residual/D/factor on the same physical
          // generation rather than publishing the last unchecked proposal.
          factor = std::move(evaluation.current_factor);
          identity = evaluation.current_identity;
          return std::move(current_density);
        }
        return std::move(evaluation.next_density);
      },
      [&](const ::vibeqc::solver::SelfConsistentProgress& progress,
          const RksLoopEvaluation& evaluation) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = progress.state_rms;
        ks.physical_residual = progress.residual_rms;
        result.physical_residual_rms = ks.physical_residual;
        ks.components = evaluation.components;
        ks.electrons = {evaluation.spin_electrons, evaluation.spin_electrons};
        ks.density_change = progress.state_rms;
        ks.history.push_back({progress.iteration,
                              evaluation.components,
                              progress.energy_change,
                              progress.state_rms,
                              progress.residual_rms,
                              {evaluation.spin_electrons, evaluation.spin_electrons}});
      });
  density = std::move(outcome.state);
  result.converged = outcome.converged;

  if (!result.converged) {
    diagnostic.physical_residual = ks.physical_residual;
    retain_factor();
    result.density = std::move(density);
    return result;
  }

  result.fock_builds += 2;
  auto final = evaluate_current(density, 0, true);
  orbitals = generalized_eigen(final.fock, orthogonalizer, n);
  Matrix projected = next_density_from_orbitals(density, runtime::vector_bytes(final.fock));
  result.density_rms = density_rms(projected, density);
  density = std::move(projected);
  final = evaluate_current(density, runtime::vector_bytes(final.fock), true);
  const auto final_residual = commutator_residual(final.fock, density, ints.overlap, n);
  diagnostic.physical_residual = residual_rms(final_residual);
  ks.physical_residual = diagnostic.physical_residual;
  result.physical_residual_rms = ks.physical_residual;
  ks.components = final.components;
  const double final_spin_electrons = dot(density, ints.overlap) / 2.0;
  ks.electrons = {final_spin_electrons, final_spin_electrons};
  ks.density_change = result.density_rms;
  result.energy_change = std::abs(final.energy - result.energy);
  result.converged = result.energy_change < options.energy_tolerance &&
                     result.density_rms < options.density_tolerance &&
                     ks.physical_residual < std::min(1.0e-9, options.density_tolerance);
  runtime::sample_cpu_capacity(runtime::add_capacity(
      retained_capacity(density), runtime::vector_capacities(final.fock, final_residual)));
  retain_factor();
  result.energy = final.energy;
  if (result.converged && options.retain_ks_state) result.ks_physical_fock = std::move(final.fock);
  result.density = std::move(density);
  return result;
}

}  // namespace

ScfResult run_lda_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_rks(plan, nullptr, basis, grid, options, initial_density, evaluate_lda_xc_rks, "LDA",
                 nullptr);
}

ScfResult run_pbe_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density) {
  return run_rks(plan, nullptr, basis, grid, options, initial_density,
                 RksXcEvaluator(evaluate_pbe_xc_rks, evaluate_pbe_xc_rks_cached), "PBE", nullptr);
}

ScfResult run_pbe_rks_nonlocal(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                               const dft::MolecularGrid& grid, const ScfOptions& options,
                               const std::vector<double>* initial_density,
                               dft::nlc::Vv10Plan& nonlocal_correlation) {
  return run_rks(plan, nullptr, basis, grid, options, initial_density,
                 RksXcEvaluator(evaluate_pbe_xc_rks, evaluate_pbe_xc_rks_cached), "PBE",
                 &nonlocal_correlation);
}

ScfResult run_pbe_rsh_rks(const PreparedFockPlan& primary,
                          const PreparedFockPlan& long_range_correction, const dft::AoBasis& basis,
                          const dft::MolecularGrid& grid, const ScfOptions& options,
                          const std::vector<double>* initial_density,
                          dft::nlc::Vv10Plan* nonlocal_correlation) {
  return run_rks(primary, &long_range_correction, basis, grid, options, initial_density,
                 RksXcEvaluator(evaluate_pbe_xc_rks, evaluate_pbe_xc_rks_cached), "PBE-RSH",
                 nonlocal_correlation);
}

ScfResult run_r2scan_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                         const dft::MolecularGrid& grid, const ScfOptions& options,
                         const std::vector<double>* initial_density) {
  return run_rks(plan, nullptr, basis, grid, options, initial_density, evaluate_r2scan_xc_rks,
                 "R2SCAN", nullptr);
}

ScfResult run_semilocal_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const dft::SemilocalPointProgram& program,
                            const std::vector<double>* initial_density) {
  dft::validate_semilocal_point_program(program);
  return run_rks(plan, nullptr, basis, grid, options, initial_density, RksXcEvaluator(program),
                 program.identifier, nullptr);
}

ScfResult run_b3lyp_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                        const dft::MolecularGrid& grid, const ScfOptions& options,
                        const std::vector<double>* initial_density) {
  auto spec =
      make_global_hybrid_fock_spec(FockSpin::Restricted, dft::generated::kB3lypExactExchange);
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE) {
    spec.coulomb.approximation = FockApproximation::DensityFitted;
    spec.exchange.approximation = FockApproximation::DensityFitted;
  }
  const auto expected = resolve_fock_build(spec, FockBackend::Cpu, options.screening_tolerance,
                                           options.density_fitting_relative_threshold);
  if (plan.strategy() != expected)
    throw std::invalid_argument("B3LYP plan does not match the generated MethodIR composition");
  return run_rks(plan, nullptr, basis, grid, options, initial_density, evaluate_b3lyp_xc_rks,
                 "B3LYP", nullptr);
}

ScfResult run_wb97mv_rks(const PreparedFockPlan& primary, const PreparedFockPlan& correction,
                         const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                         const ScfOptions& options, dft::nlc::Vv10Plan& nonlocal,
                         const std::vector<double>* initial_density) {
  require_wb97mv_composition(primary.strategy(), correction.strategy(), nonlocal.parameters());
  if (nonlocal.backend() != VIBEQC_BACKEND_CPU_REFERENCE ||
      nonlocal.resources().point_count != grid.point_count())
    throw std::invalid_argument("WB97M-V nonlocal owner is incompatible with the KS grid/backend");
  return run_rks(primary, &correction, basis, grid, options, initial_density,
                 evaluate_wb97mv_xc_rks, "WB97M-V", &nonlocal,
                 dft::nlc::Vv10DensityDomain::MolecularV1);
}

ScfResult run_cam_b3lyp_rks(const PreparedFockPlan& primary,
                            const PreparedFockPlan& long_range_correction,
                            const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                            const ScfOptions& options, const std::vector<double>* initial_density) {
  const auto expected_primary = resolve_fock_build(
      make_rsh_primary_fock_spec(FockSpin::Restricted, dft::generated::kCamB3lypShortExchange),
      FockBackend::Cpu);
  const auto expected_correction =
      resolve_fock_build(make_rsh_correction_fock_spec(
                             FockSpin::Restricted, dft::generated::kCamB3lypShortExchange,
                             dft::generated::kCamB3lypLongExchange, dft::generated::kCamB3lypOmega),
                         FockBackend::Cpu);
  if (primary.strategy() != expected_primary ||
      long_range_correction.strategy() != expected_correction)
    throw std::invalid_argument("CAM-B3LYP plans do not match the generated MethodIR composition");
  return run_rks(primary, &long_range_correction, basis, grid, options, initial_density,
                 evaluate_cam_b3lyp_xc_rks, "CAM-B3LYP", nullptr);
}

}  // namespace vibeqc::scf
