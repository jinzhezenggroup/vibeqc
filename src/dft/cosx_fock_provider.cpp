#include "dft/cosx_fock_provider.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc::dft {
namespace {

constexpr std::size_t kDefaultDeviceBudget = 256U * 1024U * 1024U;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::size_t add_size(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first) throw std::bad_alloc();
  return first + second;
}

GridSpec cosx_grid_spec(const scf::ResolvedFockBuild& strategy) {
  scf::validate_resolved_fock_build(strategy);
  require(strategy.backend == scf::FockBackend::Cuda &&
              strategy.schedule == scf::FockSchedule::CudaIndependent &&
              strategy.spec.derivative_order == 0 && strategy.spec.exchange.present &&
              strategy.spec.exchange.approximation == scf::FockApproximation::SeminumericalCosx,
          "prepared COSX Fock requires an energy-only CUDA COSX exchange strategy");
  scf::require_fock_provider_executable(scf::FockApproximation::SeminumericalCosx,
                                        scf::FockBackend::Cuda);
  const auto& source = strategy.spec.exchange.cosx;
  GridSpec grid;
  grid.version = source.grid_version;
  grid.radial_points = source.radial_points;
  grid.angular_polar = source.angular_polar;
  grid.angular_azimuth = source.angular_azimuth;
  grid.partition_iterations = source.partition_iterations;
  grid.coincident_tolerance = source.coincident_tolerance;
  grid.element_radii = source.element_radii;
  validate_grid_spec(grid);
  return grid;
}

scf::ResolvedFockBuild coulomb_strategy(const scf::ResolvedFockBuild& strategy) {
  auto spec = strategy.spec;
  spec.derivative_order = 0;
  spec.exchange.present = false;
  return scf::resolve_fock_build(spec, scf::FockBackend::Cuda, strategy.screening_tolerance,
                                 strategy.metric_relative_threshold);
}

void validate_density(const scf::ResolvedFockBuild& strategy, std::size_t nbf,
                      const std::vector<double>& density, const std::vector<double>& beta) {
  if (!nbf || nbf > std::numeric_limits<std::size_t>::max() / nbf)
    throw std::invalid_argument("invalid COSX Fock AO dimension");
  const std::size_t matrix = nbf * nbf;
  require(
      density.size() == matrix &&
          (strategy.spec.spin == scf::FockSpin::Restricted ? beta.empty() : beta.size() == matrix),
      "COSX Fock density/spin layout mismatch");
  for (const auto* values : {&density, &beta})
    for (double value : *values) require(std::isfinite(value), "nonfinite COSX Fock density");
}

}  // namespace

struct PreparedCosxFockPlan::Impl {
  core::System orbital;
  scf::ResolvedFockBuild strategy;
  MolecularGrid cosx_grid;
  std::unique_ptr<scf::PreparedFockPlan> coulomb;
  std::unique_ptr<CudaCosxStagingPlan> exchange;
  CosxFockPreparationDiagnostic diagnostic;

  Impl(const core::System& system, const core::System* auxiliary, scf::ResolvedFockBuild resolved,
       std::size_t tile_points, int device, std::size_t requested_budget)
      : orbital(system),
        strategy(std::move(resolved)),
        cosx_grid(orbital, cosx_grid_spec(strategy)) {
    require(device >= 0 && tile_points > 0, "invalid prepared COSX device or tile size");
    const auto available = requested_budget ? requested_budget : kDefaultDeviceBudget;
    const auto cosx_resources =
        cuda_cosx_staging_diagnostic(orbital, cosx_grid.point_count(), tile_points);
    if (cosx_resources.device_bytes >= available) throw std::bad_alloc();
    const auto j_budget = available - cosx_resources.device_bytes;

    const auto j_strategy = coulomb_strategy(strategy);
    coulomb =
        std::make_unique<scf::PreparedFockPlan>(orbital, auxiliary, j_strategy, device, j_budget);
    exchange =
        std::make_unique<CudaCosxStagingPlan>(orbital, cosx_grid.points(), cosx_grid.weights(),
                                              tile_points, device, cosx_resources.device_bytes);

    diagnostic.strategy = strategy;
    diagnostic.coulomb = coulomb->diagnostic();
    diagnostic.exchange = exchange->diagnostic();
    diagnostic.device_bytes =
        add_size(diagnostic.coulomb.device_bytes, diagnostic.exchange.device_bytes);
    diagnostic.device_budget_bytes = available;
    diagnostic.tile_points = diagnostic.exchange.tile_points;
    if (diagnostic.device_bytes > available)
      throw std::runtime_error("prepared COSX providers exceed admitted device budget");
  }

  scf::DirectJkMatrices build(const std::vector<double>& density, const std::vector<double>& beta) {
    const auto n = coulomb->one_electron().nbf;
    validate_density(strategy, n, density, beta);

    scf::DirectJkMatrices result;
    result.nbf = n;
    if (strategy.spec.coulomb.present) {
      auto j = coulomb->build(density, beta);
      if (j.nbf != n || j.coulomb.size() != n * n)
        throw std::runtime_error("prepared Coulomb provider returned an invalid matrix");
      result.coulomb = std::move(j.coulomb);
    }

    if (strategy.spec.spin == scf::FockSpin::Restricted) {
      auto k = exchange->build(density, CosxDensityConvention::rhf_spin_summed);
      if (k.nbf != n || k.exchange.size() != n * n)
        throw std::runtime_error("prepared COSX provider returned an invalid RHF matrix");
      result.exchange_alpha = std::move(k.exchange);
    } else {
      auto alpha = exchange->build(density, CosxDensityConvention::spin_resolved);
      auto beta_exchange = exchange->build(beta, CosxDensityConvention::spin_resolved);
      if (alpha.nbf != n || beta_exchange.nbf != n || alpha.exchange.size() != n * n ||
          beta_exchange.exchange.size() != n * n)
        throw std::runtime_error("prepared COSX provider returned an invalid UHF matrix");
      result.exchange_alpha = std::move(alpha.exchange);
      result.exchange_beta = std::move(beta_exchange.exchange);
    }
    return result;
  }
};

PreparedCosxFockPlan::PreparedCosxFockPlan(const core::System& orbital,
                                           const core::System* auxiliary,
                                           scf::ResolvedFockBuild strategy, std::size_t tile_points,
                                           int device_id, std::size_t device_budget_bytes)
    : impl_(std::make_unique<Impl>(orbital, auxiliary, std::move(strategy), tile_points, device_id,
                                   device_budget_bytes)) {}
PreparedCosxFockPlan::~PreparedCosxFockPlan() = default;

const scf::ResolvedFockBuild& PreparedCosxFockPlan::strategy() const noexcept {
  return impl_->strategy;
}
const core::System& PreparedCosxFockPlan::system() const noexcept { return impl_->orbital; }
const integrals::IntegralData& PreparedCosxFockPlan::one_electron() const noexcept {
  return impl_->coulomb->one_electron();
}
const MolecularGrid& PreparedCosxFockPlan::grid() const noexcept { return impl_->cosx_grid; }
const CosxFockPreparationDiagnostic& PreparedCosxFockPlan::diagnostic() const noexcept {
  return impl_->diagnostic;
}
scf::DirectJkMatrices PreparedCosxFockPlan::build(const std::vector<double>& density,
                                                  const std::vector<double>& beta) {
  return impl_->build(density, beta);
}

}  // namespace vibeqc::dft
