#include "scf/fock_prepared.hpp"

#include <limits>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

namespace vibeqc::scf {
namespace {
constexpr std::size_t kDefaultDeviceBudget = 256U * 1024U * 1024U;
void checked(vibeqc_status status, const std::string& detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
bool needs(const FockBuildSpec& spec, FockApproximation approximation) {
  return (spec.coulomb.present && spec.coulomb.approximation == approximation) ||
         (spec.exchange.present && spec.exchange.approximation == approximation);
}
std::size_t add_size(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::bad_alloc();
  return a + b;
}
std::size_t multiply_size(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::bad_alloc();
  return a * b;
}
/** Size the same combined orbital/auxiliary/dummy source as the shared
 * packer before it allocates GPU metadata. Keep the byte formula centralized
 * in the existing DF source resource model. */
std::size_t df_source_bytes(const core::System& orbital, const core::System& auxiliary) {
  const auto orbital_cartesian = molecule::cartesian_ao_count(orbital);
  const auto auxiliary_cartesian = molecule::cartesian_ao_count(auxiliary);
  std::size_t primitives = 1;  // The DF source's constant dummy function.
  for (const auto* system : {&orbital, &auxiliary})
    for (const auto& shell : system->shells)
      primitives = add_size(primitives, shell.primitives.size());
  return density_fitting_source_metadata_bytes(
      1, orbital.atoms.size(),
      add_size(add_size(orbital.shells.size(), auxiliary.shells.size()), 1),
      add_size(add_size(orbital_cartesian, auxiliary_cartesian), 1), primitives,
      add_size(multiply_size(orbital_cartesian, molecule::ao_count(orbital)),
               multiply_size(auxiliary_cartesian, molecule::ao_count(auxiliary))));
}
FockExecutionVariant execution_variant(const ResolvedFockBuild& strategy) noexcept {
#if VIBEQC_HAS_CUDA
  if (strategy.backend == FockBackend::Cpu) return {};
  FockExecutionVariant result;
  result.one_electron_value_mapping = cuda_policy::one_electron_value_mapping_requested();
  if (needs(strategy.spec, FockApproximation::DensityFitted)) {
    result.df_value_mapping = cuda_policy::df_value_mapping_requested();
    if (strategy.spec.derivative_order)
      result.df_derivative_mapping = cuda_policy::df_derivative_mapping_requested();
  }
  return result;
#else
  (void)strategy;
  return {};
#endif
}
/** Compare normalized scientific inputs directly. Hash collisions or pointer
 * reuse must never validate a stale geometry/basis. Unused auxiliary inputs
 * are excluded by the caller, just like absent terms in FockBuildSpec. */
bool same_system(const core::System& a, const core::System& b) noexcept {
  if (a.charge != b.charge || a.multiplicity != b.multiplicity ||
      a.electron_count != b.electron_count || a.basis_representation != b.basis_representation ||
      a.atoms.size() != b.atoms.size() || a.shells.size() != b.shells.size())
    return false;
  for (std::size_t i = 0; i < a.atoms.size(); ++i)
    if (a.atoms[i].atomic_number != b.atoms[i].atomic_number ||
        a.atoms[i].position != b.atoms[i].position)
      return false;
  for (std::size_t i = 0; i < a.shells.size(); ++i) {
    const auto& x = a.shells[i];
    const auto& y = b.shells[i];
    if (x.atom_index != y.atom_index || x.angular_momentum != y.angular_momentum ||
        x.primitives.size() != y.primitives.size())
      return false;
    for (std::size_t j = 0; j < x.primitives.size(); ++j)
      if (x.primitives[j].exponent != y.primitives[j].exponent ||
          x.primitives[j].coefficient != y.primitives[j].coefficient)
        return false;
  }
  return true;
}
}  // namespace

struct PreparedFockPlan::Impl {
  core::System orbital;
  std::optional<core::System> auxiliary;
  int device_id;
  std::size_t requested_budget;
  FockPreparationDiagnostic diagnostic;
  integrals::IntegralData exact;
  std::optional<DensityFittingScfData> fitted;
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> cuda_exact{
      nullptr, &destroy_cuda_direct_jk_plan};
  std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>
      cuda_df{nullptr, &destroy_cuda_density_fitting_jk_plan};
  std::optional<CpuFockPlanView> cpu_view;
  std::optional<CudaFockPlanView> cuda_view;

  const integrals::IntegralData& one_electron() const {
    return fitted ? fitted->one_electron : exact;
  }

  Impl(const core::System& system, const core::System* aux, ResolvedFockBuild strategy, int device,
       std::size_t budget)
      : orbital(system),
        device_id(strategy.backend == FockBackend::Cuda ? device : -1),
        requested_budget(strategy.backend == FockBackend::Cuda ? budget : 0) {
    validate_resolved_fock_build(strategy);
    diagnostic.strategy = strategy;
    diagnostic.variant = execution_variant(strategy);
    const bool has_df = needs(strategy.spec, FockApproximation::DensityFitted);
    const bool has_exact = needs(strategy.spec, FockApproximation::Exact);
    const bool derivatives = strategy.spec.derivative_order != 0;
    if (has_df) {
      auxiliary = aux ? *aux : system;
      fitted.emplace();
    }
    if (strategy.backend == FockBackend::Cpu) {
      auto ints = integrals::build_integrals(system, derivatives);
      if (has_df) {
        fitted->one_electron = std::move(ints);
        fitted->raw = integrals::build_density_fitting_integrals(system, *auxiliary, derivatives);
        fitted->metric_relative_threshold = strategy.metric_relative_threshold;
        fitted->three_center = orthonormalize_density_fitting_three_center(
            fitted->raw.three_center, fitted->raw.nbf,
            factor_density_fitting_metric(fitted->raw.metric, fitted->raw.naux,
                                          strategy.metric_relative_threshold));
      } else
        exact = std::move(ints);
      const auto& data = one_electron();
      auto provider = [&](const FockTermSpec& term) -> std::optional<CpuFockProviderView> {
        if (!term.present) return {};
        return term.approximation == FockApproximation::Exact ? CpuFockProviderView(data)
                                                              : CpuFockProviderView(*fitted);
      };
      cpu_view.emplace(strategy, data.nbf, data.ncoord, provider(strategy.spec.coulomb),
                       provider(strategy.spec.exchange));
      diagnostic.nbf = data.nbf;
      diagnostic.ncoord = data.ncoord;
      return;
    }

    if (device < 0) throw std::invalid_argument("CUDA prepared Fock requires a device");
    std::string detail;
    // The Cartesian temporary dies before persistent J/K source allocation.
    {
      integrals::IntegralData cartesian;
      checked(build_cuda_one_electron_integrals(device, system, cartesian, detail, derivatives,
                                                derivatives),
              detail);
      auto ints = integrals::transform_integrals(cartesian, system);
      if (has_df)
        fitted->one_electron = std::move(ints);
      else
        exact = std::move(ints);
    }
    diagnostic.nbf = one_electron().nbf;
    diagnostic.ncoord = system.atoms.size() * 3;
    const auto available = budget ? budget : kDefaultDeviceBudget;
    diagnostic.device_budget_bytes = available;
    if (has_exact) {
      const auto direct_budget = has_df ? available / 2 : available;
      if (!direct_budget) throw std::bad_alloc();
      CudaDirectJkPlan* raw{};
      checked(create_cuda_direct_jk_plan(device, {system}, strategy.spec.derivative_order,
                                         strategy.screening_tolerance, direct_budget, &raw,
                                         diagnostic.direct, detail),
              detail);
      cuda_exact.reset(raw);
      diagnostic.device_bytes = diagnostic.direct.device_bytes;
    }
    if (has_df) {
      const auto remainder = available - diagnostic.device_bytes;
      const auto plan_budget = remainder / 2;
      if (!plan_budget) throw std::bad_alloc();
      auto& data = *fitted;
      data.raw.nbf = diagnostic.nbf;
      data.raw.naux = molecule::ao_count(*auxiliary);
      data.raw.ncoord = diagnostic.ncoord;
      data.metric_relative_threshold = strategy.metric_relative_threshold;
      data.df_gradient_orbital = system;
      data.df_gradient_auxiliary = *auxiliary;
      data.df_gradient_mapping = diagnostic.variant.df_derivative_mapping;
      data.df_gradient_budget = remainder - plan_budget;
      // Reuse the existing tile planner before and after source metadata is
      // known. Half the available allowance is reserved for response staging.
      (void)plan_density_fitting_tiles(1, data.raw.nbf, data.raw.naux, data.raw.nbf, plan_budget,
                                       df_source_bytes(system, *auxiliary));
      CudaDensityFittingIntegralSource* raw_source{};
      std::vector<double> metrics;
      std::size_t nbf{}, naux{};
      checked(create_cuda_density_fitting_integral_source(device, {system}, {*auxiliary},
                                                          &raw_source, metrics, nbf, naux, detail),
              detail);
      std::unique_ptr<CudaDensityFittingIntegralSource,
                      decltype(&destroy_cuda_density_fitting_integral_source)>
          source(raw_source, &destroy_cuda_density_fitting_integral_source);
      diagnostic.fitted_source = cuda_density_fitting_integral_source_diagnostic(source.get());
      const auto tiles = plan_density_fitting_tiles(
          1, nbf, naux, nbf, plan_budget,
          cuda_density_fitting_integral_source_device_bytes(source.get()));
      CudaDensityFittingJkPlan* raw_plan{};
      // from_source owns the transferred handle on both success and failure.
      raw_source = source.release();
      checked(create_cuda_density_fitting_jk_plan_from_source(
                  device, &raw_source, 1, nbf, naux, metrics, strategy.metric_relative_threshold,
                  tiles.auxiliary_tile, tiles.ao_pair_tile, &raw_plan, diagnostic.fitted, detail),
              detail);
      cuda_df.reset(raw_plan);
      if (!diagnostic.fitted.empty())
        diagnostic.device_bytes += diagnostic.fitted[0].device_resident_bytes;
    }
    auto provider = [&](const FockTermSpec& term) -> std::optional<CudaFockProviderView> {
      if (!term.present) return {};
      return term.approximation == FockApproximation::Exact
                 ? CudaFockProviderView(cuda_exact.get())
                 : CudaFockProviderView(cuda_df.get(), *fitted);
    };
    cuda_view.emplace(strategy, diagnostic.nbf, diagnostic.ncoord, provider(strategy.spec.coulomb),
                      provider(strategy.spec.exchange));
  }
};

PreparedFockPlan::PreparedFockPlan(const core::System& system, const core::System* auxiliary,
                                   ResolvedFockBuild strategy, int device, std::size_t budget)
    : impl_(std::make_unique<Impl>(system, auxiliary, strategy, device, budget)) {}
PreparedFockPlan::~PreparedFockPlan() = default;
const ResolvedFockBuild& PreparedFockPlan::strategy() const noexcept {
  return impl_->diagnostic.strategy;
}
const core::System& PreparedFockPlan::system() const noexcept { return impl_->orbital; }
const integrals::IntegralData& PreparedFockPlan::one_electron() const noexcept {
  return impl_->one_electron();
}
const DensityFittingScfData* PreparedFockPlan::cpu_fitted_data() const noexcept {
  return impl_->cpu_view && impl_->fitted ? &*impl_->fitted : nullptr;
}
const FockPreparationDiagnostic& PreparedFockPlan::diagnostic() const noexcept {
  return impl_->diagnostic;
}
DirectJkMatrices PreparedFockPlan::build(const std::vector<double>& density,
                                         const std::vector<double>& beta) const {
  return impl_->cpu_view ? impl_->cpu_view->build(density, beta)
                         : impl_->cuda_view->build(density, beta);
}
std::vector<double> PreparedFockPlan::energy_derivative(const std::vector<double>& density,
                                                        const std::vector<double>& beta) const {
  return impl_->cpu_view ? impl_->cpu_view->energy_derivative(density, beta)
                         : impl_->cuda_view->energy_derivative(density, beta);
}
bool PreparedFockPlan::matches(const core::System& orbital, const core::System* auxiliary,
                               const ResolvedFockBuild& strategy, int device,
                               std::size_t budget) const noexcept {
  if (impl_->diagnostic.strategy != strategy || !same_system(impl_->orbital, orbital)) return false;
  if (strategy.backend == FockBackend::Cuda &&
      (device != impl_->device_id || budget != impl_->requested_budget ||
       execution_variant(strategy) != impl_->diagnostic.variant))
    return false;
  return !impl_->auxiliary || same_system(*impl_->auxiliary, auxiliary ? *auxiliary : orbital);
}
}  // namespace vibeqc::scf
