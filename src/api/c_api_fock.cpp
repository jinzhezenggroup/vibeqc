#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "vibeqc/fock.h"

#if VIBEQC_HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "runtime/resource_cuda.cuh"
#include "scf/cuda_direct_jk_device.hpp"
#endif

struct vibeqc_fock_plan {
  std::unique_ptr<vibeqc::scf::PreparedFockPlan> source;
  vibeqc::scf::FockBuildSpec requested;
  std::string detail;
};

struct vibeqc_rhf_response_resident {
  vibeqc::scf::PreparedFockPlan* parent{};
  std::string detail;
#if VIBEQC_HAS_CUDA
  vibeqc::scf::CudaDirectJkPlan* direct{};
  int device_id{-1};
  cudaStream_t stream{};
  cublasHandle_t blas{};
  void* allocation{};
  std::size_t allocation_bytes{};
  std::size_t nbf{}, nocc{}, nvirt{}, dimension{}, vector_slots{};
  double *slots{}, *coefficients{}, *energy_occ{}, *energy_virt{};
  double *density{}, *coulomb{}, *exchange{};
  double *transform_one{}, *transform_two{}, *gap_scratch{};
  std::vector<double> orbital_energies;
  bool reconstruction_ready{};
  int* numerical_error{};
  std::uint64_t h2d_bytes{}, d2h_bytes{}, synchronizations{}, operator_actions{}, blas_calls{};
#endif
};

/** Device-resident unrestricted response owner used by the shared UHF solver.
 * Alpha and beta rotation blocks share one lease arena, while the raw
 * unrestricted J/K buffers remain separate so the CUDA provider can preserve
 * spin coupling without a host intermediate. */
struct vibeqc_uhf_response_resident {
  vibeqc_fock_plan* parent{};
  std::string detail;
#if VIBEQC_HAS_CUDA
  vibeqc::scf::CudaDirectJkPlan* direct{};
  int device_id{-1};
  cudaStream_t stream{};
  cublasHandle_t blas{};
  void* allocation{};
  std::size_t allocation_bytes{};
  std::size_t nbf{}, nocc_alpha{}, nvirt_alpha{}, nocc_beta{}, nvirt_beta{}, dimension{},
      vector_slots{};
  std::size_t alpha_offset{}, beta_offset{};
  double *slots{}, *coefficients_alpha{}, *coefficients_beta{};
  double *energy_occ_alpha{}, *energy_virt_alpha{}, *energy_occ_beta{}, *energy_virt_beta{};
  double *density_alpha{}, *density_beta{}, *coulomb{}, *exchange_alpha{}, *exchange_beta{};
  double *transform_alpha{}, *transform_beta{}, *gap_alpha{}, *gap_beta{};
  int* numerical_error{};
  std::uint64_t h2d_bytes{}, d2h_bytes{}, synchronizations{}, operator_actions{}, blas_calls{};
#endif
};

namespace {
using namespace vibeqc::scf;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
FockTermSpec from_c(const vibeqc_fock_term& term) {
  require(term.present == 0 || term.present == 1, "Fock presence must be zero or one");
  return {term.present != 0, term.coefficient, static_cast<FockOperator>(term.op), term.omega,
          static_cast<FockApproximation>(term.approximation)};
}
vibeqc_fock_spec to_c(const FockBuildSpec& spec) {
  auto term = [](const FockTermSpec& t) {
    return vibeqc_fock_term{t.present ? 1 : 0, t.coefficient, static_cast<int32_t>(t.op), t.omega,
                            static_cast<int32_t>(t.approximation)};
  };
  return {sizeof(vibeqc_fock_spec),
          VIBEQC_ABI_VERSION,
          spec.version,
          static_cast<int32_t>(spec.spin),
          spec.derivative_order,
          term(spec.coulomb),
          term(spec.exchange)};
}
void finite(const std::vector<double>& data) {
  for (double x : data)
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite Fock output");
}
/** Output aliasing would make transactional publication ambiguous. Inputs
 * may alias outputs because they are copied before either provider executes. */
void disjoint_outputs(std::initializer_list<std::pair<const void*, std::size_t>> outputs) {
  std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
  for (const auto& [pointer, bytes] : outputs) {
    if (!pointer) continue;
    const auto first = reinterpret_cast<std::uintptr_t>(pointer);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - first,
            "Fock output range overflows address space");
    ranges.emplace_back(first, first + bytes);
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    require(ranges[i - 1].second <= ranges[i].first, "Fock output buffers overlap");
}
std::size_t buffer_bytes(uint64_t count) {
  require(count <= std::numeric_limits<std::size_t>::max() / sizeof(double),
          "Fock output size overflows address space");
  return count * sizeof(double);
}
}  // namespace

extern "C" vibeqc_status vibeqc_fock_plan_create(
    vibeqc_context* context, const vibeqc_system* system, const vibeqc_system* auxiliary,
    const vibeqc_fock_spec* spec, const vibeqc_fock_controls* controls, vibeqc_fock_plan** output) {
  if (output) *output = nullptr;
  if (!context || !system || !spec || !output) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(spec) ||
      (controls && !vibeqc::api::valid_descriptor(controls)))
    return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    require(context->state.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE ||
                context->state.executed_backend == VIBEQC_BACKEND_CUDA,
            "Fock plans require an explicit CPU or CUDA context");
    FockBuildSpec request;
    request.version = spec->spec_version;
    request.spin = static_cast<FockSpin>(spec->spin);
    request.derivative_order = spec->derivative_order;
    request.coulomb = from_c(spec->coulomb);
    request.exchange = from_c(spec->exchange);
    const auto backend = context->state.executed_backend == VIBEQC_BACKEND_CUDA ? FockBackend::Cuda
                                                                                : FockBackend::Cpu;
    const double screening = controls ? controls->screening_tolerance : 1e-12;
    const double requested_threshold = controls ? controls->metric_relative_threshold : 0.0;
    // Validate public controls even when resolution would discard an unused
    // DF cutoff. A malformed request must not depend on its selected provider.
    require(std::isfinite(screening) && screening >= 0.0,
            "Fock screening tolerance must be finite and nonnegative");
    require(std::isfinite(requested_threshold) && requested_threshold >= 0.0 &&
                requested_threshold < 1.0,
            "Fock metric cutoff must be finite and in [0, 1)");
    const double threshold = requested_threshold == 0.0 ? 1e-10 : requested_threshold;
    const auto budget = controls ? controls->device_budget_bytes : 0;
    require(budget <= std::numeric_limits<std::size_t>::max(), "Fock budget overflows size_t");
    auto plan = std::make_unique<vibeqc_fock_plan>();
    plan->requested = request;
    plan->source = std::make_unique<PreparedFockPlan>(
        system->data, auxiliary ? &auxiliary->data : nullptr,
        resolve_fock_build(request, backend, screening, threshold), context->state.device_id,
        budget);
    *output = plan.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}
extern "C" void vibeqc_fock_plan_destroy(vibeqc_fock_plan* plan) { delete plan; }
extern "C" const char* vibeqc_fock_plan_last_error(const vibeqc_fock_plan* plan) {
  return plan ? plan->detail.c_str() : "null Fock plan";
}

extern "C" vibeqc_status vibeqc_fock_plan_evaluate(vibeqc_fock_plan* plan, const double* density,
                                                   uint64_t density_count, const double* beta,
                                                   uint64_t beta_count,
                                                   vibeqc_fock_result* result) {
  if (!plan || !density || !result) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(result)) return VIBEQC_STATUS_ABI_MISMATCH;
  plan->detail.clear();
  try {
    const auto& source = *plan->source;
    const auto& strategy = source.strategy();
    const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
    const auto& ints = source.one_electron();
    const auto matrix = ints.nbf * ints.nbf;
    require(density_count == matrix && result->matrix_count == matrix &&
                (unrestricted ? beta && beta_count == matrix : beta_count == 0),
            "Fock density/output shape or spin mismatch");
    require(result->gradient ? strategy.spec.derivative_order == 1 &&
                                   result->gradient_count == source.system().atoms.size() * 3
                             : result->gradient_count == 0,
            "Fock gradient capability/count mismatch");
    const auto bytes = buffer_bytes(matrix);
    disjoint_outputs({{result->coulomb, bytes},
                      {result->exchange_alpha, bytes},
                      {result->exchange_beta, bytes},
                      {result->fock_alpha, bytes},
                      {result->fock_beta, bytes},
                      {result->gradient, buffer_bytes(result->gradient_count)},
                      {result, sizeof(*result)}});
    const std::vector<double> a(density, density + matrix);
    const std::vector<double> b =
        unrestricted ? std::vector<double>(beta, beta + matrix) : std::vector<double>{};
    // Derivative preflight runs before value execution; in particular a
    // nonsymmetric density cannot partially execute a generated DF response.
    const auto gradient = result->gradient ? source.energy_derivative(a, b) : std::vector<double>{};
    const auto jk = source.build(a, b);
    const auto fock = assemble_fock(strategy, ints.hcore, jk);
    const auto e2 = contract_fock_energy(strategy, jk, a, b);
    double e1 = 0.0;
    for (std::size_t i = 0; i < matrix; ++i)
      e1 += (a[i] + (unrestricted ? b[i] : 0.0)) * ints.hcore[i];
    if (!std::isfinite(e1) || !std::isfinite(e2) || !std::isfinite(ints.nuclear_repulsion))
      throw std::runtime_error("nonfinite Fock energy");
    for (const auto* values :
         {&jk.coulomb, &jk.exchange_alpha, &jk.exchange_beta, &fock.alpha, &fock.beta, &gradient})
      finite(*values);
    auto publish = [](double* target, const auto& values) {
      if (target && !values.empty()) std::copy(values.begin(), values.end(), target);
    };
    publish(result->coulomb, jk.coulomb);
    publish(result->exchange_alpha, jk.exchange_alpha);
    publish(result->exchange_beta, jk.exchange_beta);
    publish(result->fock_alpha, fock.alpha);
    publish(result->fock_beta, fock.beta);
    publish(result->gradient, gradient);
    result->energy_one_electron = e1;
    result->energy_two_electron = e2;
    result->nuclear_repulsion = ints.nuclear_repulsion;
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&plan->detail);
  }
}

extern "C" vibeqc_status vibeqc_fock_plan_solve(vibeqc_fock_plan* plan,
                                                const vibeqc_fock_scf_controls* controls,
                                                const double* initial_density,
                                                uint64_t initial_density_count,
                                                vibeqc_fock_scf_result* output) {
  if (!plan || !output) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(output) ||
      (controls && !vibeqc::api::valid_descriptor(controls)))
    return VIBEQC_STATUS_ABI_MISMATCH;
  plan->detail.clear();
  try {
    const auto& strategy = plan->source->strategy();
    const auto n = plan->source->one_electron().nbf;
    const auto count = n * n * (strategy.spec.spin == FockSpin::Unrestricted ? 2 : 1);
    const auto coordinates = 3 * plan->source->system().atoms.size();
    require(output->density ? output->density_count == count : output->density_count == 0,
            "SCF density output shape mismatch");
    require(output->forces
                ? output->force_count == coordinates && strategy.spec.derivative_order == 1
                : output->force_count == 0,
            "SCF force capability/count mismatch");
    require(initial_density ? initial_density_count == count : initial_density_count == 0,
            "SCF initial density shape mismatch");
    disjoint_outputs({{output->density, buffer_bytes(output->density_count)},
                      {output->forces, buffer_bytes(output->force_count)},
                      {output, sizeof(*output)}});
    ScfOptions options;
    options.resolved_fock_build = strategy;
    options.screening_tolerance = strategy.screening_tolerance;
    options.density_fitting_relative_threshold = strategy.metric_relative_threshold;
    options.compute_forces = output->forces != nullptr;
    options.strict_initial_density = true;
    if (controls) {
      require(controls->max_iterations > 0 &&
                  controls->max_iterations < std::numeric_limits<unsigned>::max() &&
                  controls->diis_history > 0 && std::isfinite(controls->energy_tolerance) &&
                  controls->energy_tolerance > 0 && std::isfinite(controls->density_tolerance) &&
                  controls->density_tolerance > 0,
              "SCF counts and finite tolerances must be positive");
      options.max_iterations = controls->max_iterations;
      options.diis_history = controls->diis_history;
      options.energy_tolerance = controls->energy_tolerance;
      options.density_tolerance = controls->density_tolerance;
    }
    const auto seed = initial_density
                          ? std::vector<double>(initial_density, initial_density + count)
                          : std::vector<double>{};
    const auto result =
        run_prepared_fock_strategy(*plan->source, options, initial_density ? &seed : nullptr);
    if (!result.converged) {
      plan->detail =
          "Fock SCF did not converge in " + std::to_string(result.iterations) + " iterations";
      return VIBEQC_STATUS_NOT_CONVERGED;
    }
    if (!std::isfinite(result.energy) || !std::isfinite(result.energy_change) ||
        !std::isfinite(result.density_rms) || result.density.size() != count ||
        result.forces.size() != (options.compute_forces ? coordinates : 0))
      throw std::runtime_error("invalid converged Fock SCF result");
    finite(result.density);
    finite(result.forces);
    if (output->density) std::copy(result.density.begin(), result.density.end(), output->density);
    if (output->forces) std::copy(result.forces.begin(), result.forces.end(), output->forces);
    output->energy = result.energy;
    output->energy_change = result.energy_change;
    output->density_rms = result.density_rms;
    output->iterations = result.iterations;
    output->initial_density_used = result.initial_density_used;
    output->fock_builds = result.fock_builds;
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&plan->detail);
  }
}

/** Private additive query preserves the public diagnostic struct's ABI.
 * Read the prepared variant, not the current environment: existing owners keep
 * their actual representation across later diagnostic selector changes.
 * Return 0 for dense, 1 for dual-owner packed, 2 for single-owner packed,
 * and -1 for an invalid handle.
 */
extern "C" int vibeqc_fock_plan_df_pair_storage_v1(const vibeqc_fock_plan* plan) {
  if (!plan) return -1;
  const auto storage = plan->source->diagnostic().variant.df_pair_storage;
  if (storage == DfPairStorage::SymmetricLowerSingle) return 2;
  return storage == DfPairStorage::SymmetricLower ? 1 : 0;
}

extern "C" vibeqc_status vibeqc_fock_plan_diagnostic(const vibeqc_fock_plan* plan,
                                                     vibeqc_fock_diagnostic* output) {
  if (!plan || !output) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(output)) return VIBEQC_STATUS_ABI_MISMATCH;
  const auto& info = plan->source->diagnostic();
  const bool cuda = info.strategy.backend == FockBackend::Cuda;
  vibeqc_fock_diagnostic out{};
  out.struct_size = sizeof(out);
  out.abi_version = VIBEQC_ABI_VERSION;
  out.requested = to_c(plan->requested);
  out.resolved = to_c(info.strategy.spec);
  out.backend = cuda ? VIBEQC_BACKEND_CUDA : VIBEQC_BACKEND_CPU_REFERENCE;
  out.resolved_schedule = static_cast<int32_t>(info.strategy.schedule);
  out.source_schedule = cuda                                      ? VIBEQC_FOCK_CUDA_INDEPENDENT
                        : info.strategy.metric_relative_threshold ? VIBEQC_FOCK_CPU_INDEPENDENT
                                                                  : VIBEQC_FOCK_CPU_REFERENCE;
  out.nbf = info.nbf;
  out.coordinate_count = plan->source->system().atoms.size() * 3;
  out.device_bytes = info.device_bytes;
  out.device_budget_bytes = info.device_budget_bytes;
  out.screening_tolerance = info.strategy.screening_tolerance;
  out.metric_relative_threshold = info.strategy.metric_relative_threshold;
  if (!info.fitted.empty()) {
    out.auxiliary_rank = info.fitted[0].effective_rank;
    out.auxiliary_tile = info.fitted[0].auxiliary_tile;
    out.df_streamed = info.fitted[0].streamed;
  } else if (const auto* fitted = plan->source->cpu_fitted_data()) {
    out.auxiliary_rank = fitted->three_center.effective_rank;
    out.auxiliary_tile = fitted->raw.naux;
  }
  const bool has_exact = (info.strategy.spec.coulomb.present &&
                          info.strategy.spec.coulomb.approximation == FockApproximation::Exact) ||
                         (info.strategy.spec.exchange.present &&
                          info.strategy.spec.exchange.approximation == FockApproximation::Exact);
  std::snprintf(out.direct_schedule, sizeof(out.direct_schedule), "%s",
                !has_exact ? "absent"
                : cuda     ? info.direct.schedule
                           : "cpu-reference-eri");
  std::snprintf(out.df_value_backend, sizeof(out.df_value_backend), "%s",
                !out.auxiliary_rank ? "absent"
                : cuda              ? info.fitted_source.value_backend
                                    : "cpu-reference");
  std::snprintf(out.df_value_mapping, sizeof(out.df_value_mapping), "%s",
                !out.auxiliary_rank ? "absent"
                : cuda              ? info.fitted_source.value_mapping
                                    : "dense");
  std::snprintf(out.df_response_mapping, sizeof(out.df_response_mapping), "%s",
                !out.auxiliary_rank || !info.strategy.spec.derivative_order ? "absent"
                : !cuda                                                     ? "cpu-reference"
                : info.variant.df_derivative_mapping                        ? "generated-serial"
                                                                            : "generated-atomic");
  if (cuda) {
    const auto provider = vibeqc::runtime::cuda_provider_name(info.variant.cuda_provider);
    const char* policy = info.variant.one_electron_value_capability_fallback ? "fallback"
                         : info.variant.one_electron_value_override          ? "override"
                                                                             : "auto";
    std::snprintf(out.one_electron_value_backend, sizeof(out.one_electron_value_backend),
                  "cuda-generated:%.*s:%s", static_cast<int>(provider.size()), provider.data(),
                  policy);
  } else {
    std::snprintf(out.one_electron_value_backend, sizeof(out.one_electron_value_backend), "%s",
                  "cpu-reference");
  }
  std::snprintf(out.one_electron_value_mapping, sizeof(out.one_electron_value_mapping), "%s",
                !cuda                                     ? "dense"
                : info.variant.one_electron_value_mapping ? "shell-warp"
                                                          : "pair-thread");
  // Prepared one-electron response owns the Dual pair-kernel matrices;
  // fused force-bridge environment selectors do not affect this source.
  std::snprintf(out.one_electron_response_mapping, sizeof(out.one_electron_response_mapping), "%s",
                !info.strategy.spec.derivative_order ? "absent"
                : cuda                               ? "dual-pair-thread"
                                                     : "cpu-reference");
  *output = out;
  return VIBEQC_STATUS_SUCCESS;
}

namespace {
#if VIBEQC_HAS_CUDA
void resident_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void resident_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("resident RHF response cuBLAS failure");
}
std::size_t resident_product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a) throw std::bad_alloc();
  return a * b;
}
std::size_t resident_sum(std::initializer_list<std::size_t> terms) {
  std::size_t value = 0;
  for (auto term : terms) {
    if (term > std::numeric_limits<std::size_t>::max() - value) throw std::bad_alloc();
    value += term;
  }
  return value;
}
double* resident_slot(vibeqc_rhf_response_resident* owner, std::uint32_t slot) {
  require(owner && slot < owner->vector_slots, "resident RHF response slot out of range");
  return owner->slots + static_cast<std::size_t>(slot) * owner->dimension;
}
const double* resident_slot(const vibeqc_rhf_response_resident* owner, std::uint32_t slot) {
  require(owner && slot < owner->vector_slots, "resident RHF response slot out of range");
  return owner->slots + static_cast<std::size_t>(slot) * owner->dimension;
}
template <class Function>
vibeqc_status resident_guard(vibeqc_rhf_response_resident* owner, Function function) {
  if (!owner) return VIBEQC_STATUS_INVALID_ARGUMENT;
  owner->detail.clear();
  try {
    resident_cuda(cudaSetDevice(owner->device_id));
    function();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&owner->detail);
  }
}
void resident_sync(vibeqc_rhf_response_resident* owner) {
  resident_cuda(cudaStreamSynchronize(owner->stream));
  ++owner->synchronizations;
}
#endif
}  // namespace

extern "C" vibeqc_status vibeqc_rhf_response_resident_create(
    vibeqc_fock_plan* plan, const double* coefficients, uint64_t coefficient_count,
    const double* orbital_energies, uint64_t energy_count, uint32_t nocc, uint32_t vector_slots,
    uint64_t device_budget_bytes, vibeqc_rhf_response_resident** output) {
  if (output) *output = nullptr;
  if (!plan || !coefficients || !orbital_energies || !output) return VIBEQC_STATUS_INVALID_ARGUMENT;
#if VIBEQC_HAS_CUDA
  plan->detail.clear();
  try {
    const auto& source = *plan->source;
    const auto& strategy = source.strategy();
    require(strategy.backend == vibeqc::scf::FockBackend::Cuda,
            "resident RHF response requires CUDA Fock plan");
    require(strategy.spec.spin == vibeqc::scf::FockSpin::Restricted &&
                strategy.spec.coulomb.present && strategy.spec.exchange.present &&
                strategy.spec.coulomb.approximation == vibeqc::scf::FockApproximation::Exact &&
                strategy.spec.exchange.approximation == vibeqc::scf::FockApproximation::Exact &&
                strategy.spec.coulomb.op == vibeqc::scf::FockOperator::FullRange &&
                strategy.spec.exchange.op == vibeqc::scf::FockOperator::FullRange &&
                strategy.spec.coulomb.coefficient == 1.0 &&
                strategy.spec.exchange.coefficient == -0.5 && strategy.screening_tolerance == 0.0,
            "resident RHF response requires exact unscreened conventional RHF J/K");
    auto* direct = source.cuda_direct_source();
    require(direct != nullptr, "resident RHF response direct CUDA source unavailable");
    const auto n = source.one_electron().nbf;
    require(n > 1 && nocc > 0 && nocc < n && vector_slots >= 8 && vector_slots <= 4096,
            "resident RHF response dimensions/slot count are invalid");
    require(coefficient_count == n * n && energy_count == n,
            "resident RHF response reference dimensions mismatch");
    for (std::size_t i = 0; i < coefficient_count; ++i)
      require(std::isfinite(coefficients[i]), "nonfinite resident RHF coefficients");
    for (std::size_t i = 0; i < energy_count; ++i)
      require(std::isfinite(orbital_energies[i]), "nonfinite resident RHF orbital energies");

    const auto o = static_cast<std::size_t>(nocc);
    const auto v = n - o;
    const auto dim = resident_product(o, v);
    const auto matrix = resident_product(n, n);
    const auto cublas_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    require(n <= cublas_limit && o <= cublas_limit && v <= cublas_limit && dim <= cublas_limit &&
                matrix <= cublas_limit,
            "resident RHF response dimensions exceed cuBLAS int limits");
    const auto transform = resident_product(n, o);
    const auto occupied_matrix = resident_product(o, o);
    const auto virtual_matrix = resident_product(v, v);
    const auto slot_values = resident_product(static_cast<std::size_t>(vector_slots), dim);
    const auto doubles = resident_sum(
        {slot_values, matrix, occupied_matrix, virtual_matrix, 3 * matrix, 2 * transform, dim});
    const auto bytes = resident_sum({resident_product(doubles, sizeof(double)), sizeof(int)});
    require(device_budget_bytes > 0 && bytes <= device_budget_bytes,
            "resident RHF response device budget is insufficient");

    auto owner = std::make_unique<vibeqc_rhf_response_resident>();
    owner->parent = plan->source.get();
    owner->direct = direct;
    owner->device_id = vibeqc::scf::cuda_direct_jk_device(direct);
    owner->stream = vibeqc::scf::cuda_direct_jk_stream(direct);
    owner->nbf = n;
    owner->nocc = o;
    owner->nvirt = v;
    owner->dimension = dim;
    owner->vector_slots = vector_slots;
    owner->allocation_bytes = bytes;
    owner->orbital_energies.assign(orbital_energies, orbital_energies + energy_count);

    resident_cuda(cudaSetDevice(owner->device_id));
    resident_blas(cublasCreate(&owner->blas));
    try {
      resident_blas(cublasSetStream(owner->blas, owner->stream));
      resident_blas(cublasSetPointerMode(owner->blas, CUBLAS_POINTER_MODE_HOST));
      resident_cuda(vibeqc::runtime::resource_cuda_malloc(&owner->allocation, bytes));
      auto* cursor = static_cast<double*>(owner->allocation);
      owner->slots = cursor;
      cursor += slot_values;
      owner->coefficients = cursor;
      cursor += matrix;
      owner->energy_occ = cursor;
      cursor += occupied_matrix;
      owner->energy_virt = cursor;
      cursor += virtual_matrix;
      owner->density = cursor;
      cursor += matrix;
      owner->coulomb = cursor;
      cursor += matrix;
      owner->exchange = cursor;
      cursor += matrix;
      owner->transform_one = cursor;
      cursor += transform;
      owner->transform_two = cursor;
      cursor += transform;
      owner->gap_scratch = cursor;
      cursor += dim;
      owner->numerical_error = reinterpret_cast<int*>(cursor);

      std::vector<double> column_major(matrix);
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
          column_major[column * n + row] = coefficients[row * n + column];
      std::vector<double> energy_occ(occupied_matrix, 0.0);
      std::vector<double> energy_virt(virtual_matrix, 0.0);
      for (std::size_t i = 0; i < o; ++i) energy_occ[i * o + i] = orbital_energies[i];
      for (std::size_t a = 0; a < v; ++a) energy_virt[a * v + a] = orbital_energies[o + a];
      resident_cuda(cudaMemcpyAsync(owner->coefficients, column_major.data(),
                                    matrix * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_occ, energy_occ.data(),
                                    occupied_matrix * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_virt, energy_virt.data(),
                                    virtual_matrix * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      owner->h2d_bytes = (matrix + occupied_matrix + virtual_matrix) * sizeof(double);
      resident_sync(owner.get());
    } catch (...) {
      if (owner->allocation) {
        (void)cudaStreamSynchronize(owner->stream);
        (void)vibeqc::runtime::resource_cuda_free(owner->allocation);
        owner->allocation = nullptr;
      }
      if (owner->blas) {
        (void)cublasDestroy(owner->blas);
        owner->blas = nullptr;
      }
      throw;
    }
    *output = owner.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&plan->detail);
  }
#else
  (void)coefficient_count;
  (void)energy_count;
  (void)nocc;
  (void)vector_slots;
  (void)device_budget_bytes;
  plan->detail = "resident RHF response requires a CUDA-enabled library";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" void vibeqc_rhf_response_resident_destroy(vibeqc_rhf_response_resident* owner) {
  if (!owner) return;
#if VIBEQC_HAS_CUDA
  if (owner->device_id >= 0) (void)cudaSetDevice(owner->device_id);
  if (owner->stream) (void)cudaStreamSynchronize(owner->stream);
  if (owner->allocation) (void)vibeqc::runtime::resource_cuda_free(owner->allocation);
  if (owner->blas) (void)cublasDestroy(owner->blas);
#endif
  delete owner;
}

extern "C" const char* vibeqc_rhf_response_resident_last_error(
    const vibeqc_rhf_response_resident* owner) {
  return owner ? owner->detail.c_str() : "null resident RHF response owner";
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_get_diagnostic(
    const vibeqc_rhf_response_resident* owner,
    vibeqc_rhf_response_resident_diagnostic* diagnostic) {
  if (!owner || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(diagnostic)) return VIBEQC_STATUS_ABI_MISMATCH;
#if VIBEQC_HAS_CUDA
  vibeqc_rhf_response_resident_diagnostic out{};
  out.struct_size = sizeof(out);
  out.abi_version = VIBEQC_ABI_VERSION;
  out.nbf = owner->nbf;
  out.nocc = owner->nocc;
  out.nvirt = owner->nvirt;
  out.dimension = owner->dimension;
  out.vector_slots = owner->vector_slots;
  out.owned_device_bytes = owner->allocation_bytes;
  out.h2d_bytes = owner->h2d_bytes;
  out.d2h_bytes = owner->d2h_bytes;
  out.synchronizations = owner->synchronizations;
  out.operator_actions = owner->operator_actions;
  out.blas_calls = owner->blas_calls;
  out.device_id = owner->device_id;
  *diagnostic = out;
  return VIBEQC_STATUS_SUCCESS;
#else
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_upload(vibeqc_rhf_response_resident* owner,
                                                             uint32_t slot, const double* values,
                                                             uint64_t count) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(values && count == owner->dimension, "resident RHF upload shape mismatch");
    for (std::size_t i = 0; i < count; ++i)
      require(std::isfinite(values[i]), "resident RHF upload requires finite vectors");
    resident_cuda(cudaMemcpyAsync(resident_slot(owner, slot), values,
                                  owner->dimension * sizeof(double), cudaMemcpyHostToDevice,
                                  owner->stream));
    owner->h2d_bytes += owner->dimension * sizeof(double);
    resident_sync(owner);
  });
#else
  (void)owner;
  (void)slot;
  (void)values;
  (void)count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_download(vibeqc_rhf_response_resident* owner,
                                                               uint32_t slot, double* values,
                                                               uint64_t count) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(values && count == owner->dimension, "resident RHF download shape mismatch");
    resident_cuda(cudaMemcpyAsync(values, resident_slot(owner, slot),
                                  owner->dimension * sizeof(double), cudaMemcpyDeviceToHost,
                                  owner->stream));
    owner->d2h_bytes += owner->dimension * sizeof(double);
    resident_sync(owner);
    for (std::size_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite resident RHF vector");
  });
#else
  (void)owner;
  (void)slot;
  (void)values;
  (void)count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_zero(vibeqc_rhf_response_resident* owner,
                                                           uint32_t slot) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    resident_cuda(cudaMemsetAsync(resident_slot(owner, slot), 0, owner->dimension * sizeof(double),
                                  owner->stream));
  });
#else
  (void)owner;
  (void)slot;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_copy(vibeqc_rhf_response_resident* owner,
                                                           uint32_t destination, uint32_t source) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    resident_blas(cublasDcopy(owner->blas, static_cast<int>(owner->dimension),
                              resident_slot(owner, source), 1, resident_slot(owner, destination),
                              1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)destination;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_scale(vibeqc_rhf_response_resident* owner,
                                                            uint32_t slot, double alpha) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(std::isfinite(alpha), "resident RHF scale must be finite");
    resident_blas(cublasDscal(owner->blas, static_cast<int>(owner->dimension), &alpha,
                              resident_slot(owner, slot), 1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)slot;
  (void)alpha;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_axpy(vibeqc_rhf_response_resident* owner,
                                                           uint32_t destination, double alpha,
                                                           uint32_t source) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(std::isfinite(alpha), "resident RHF axpy coefficient must be finite");
    resident_blas(cublasDaxpy(owner->blas, static_cast<int>(owner->dimension), &alpha,
                              resident_slot(owner, source), 1, resident_slot(owner, destination),
                              1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)destination;
  (void)alpha;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_dot(vibeqc_rhf_response_resident* owner,
                                                          uint32_t left, uint32_t right,
                                                          double* value) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(value, "resident RHF dot requires output");
    resident_blas(cublasDdot(owner->blas, static_cast<int>(owner->dimension),
                             resident_slot(owner, left), 1, resident_slot(owner, right), 1, value));
    ++owner->blas_calls;
    owner->d2h_bytes += sizeof(double);
    resident_sync(owner);
    if (!std::isfinite(*value)) throw std::runtime_error("nonfinite resident RHF dot");
  });
#else
  (void)owner;
  (void)left;
  (void)right;
  (void)value;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_norm(vibeqc_rhf_response_resident* owner,
                                                           uint32_t slot, double* value) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(value, "resident RHF norm requires output");
    resident_blas(cublasDnrm2(owner->blas, static_cast<int>(owner->dimension),
                              resident_slot(owner, slot), 1, value));
    ++owner->blas_calls;
    owner->d2h_bytes += sizeof(double);
    resident_sync(owner);
    if (!std::isfinite(*value)) throw std::runtime_error("nonfinite resident RHF norm");
  });
#else
  (void)owner;
  (void)slot;
  (void)value;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_apply(vibeqc_rhf_response_resident* owner,
                                                            uint32_t destination, uint32_t source) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    require(destination != source, "resident RHF operator source/output must be distinct");
    owner->reconstruction_ready = false;
    const auto n = static_cast<int>(owner->nbf);
    const auto o = static_cast<int>(owner->nocc);
    const auto v = static_cast<int>(owner->nvirt);
    const auto dim = static_cast<int>(owner->dimension);
    const double one = 1.0, zero = 0.0, two = 2.0, half = -0.5, minus = -1.0;
    const auto* x = resident_slot(owner, source);
    auto* y = resident_slot(owner, destination);
    const auto* c_occ = owner->coefficients;
    const auto* c_virt = owner->coefficients + owner->nbf * owner->nocc;

    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, o, v, &one, c_virt, n, x, v,
                              &zero, owner->transform_one, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two,
                              owner->transform_one, n, c_occ, n, &zero, owner->density, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two, c_occ, n,
                              owner->transform_one, n, &one, owner->density, n));
    ++owner->blas_calls;

    auto spec = owner->parent->strategy().spec;
    spec.derivative_order = 0;
    std::string detail;
    const auto status = vibeqc::scf::enqueue_cuda_direct_jk_device(
        owner->direct, spec, owner->density, nullptr, owner->nbf * owner->nbf, owner->coulomb,
        owner->exchange, nullptr, owner->numerical_error, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "resident direct J/K enqueue failed" : detail);

    resident_blas(cublasDcopy(owner->blas, n * n, owner->coulomb, 1, owner->density, 1));
    ++owner->blas_calls;
    resident_blas(cublasDaxpy(owner->blas, n * n, &half, owner->exchange, 1, owner->density, 1));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, o, n, &one, owner->density,
                              n, c_occ, n, &zero, owner->transform_two, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_T, CUBLAS_OP_N, v, o, n, &one, c_virt, n,
                              owner->transform_two, n, &zero, y, v));
    ++owner->blas_calls;

    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, v, o, v, &one,
                              owner->energy_virt, v, x, v, &zero, owner->gap_scratch, v));
    ++owner->blas_calls;
    resident_blas(cublasDaxpy(owner->blas, dim, &one, owner->gap_scratch, 1, y, 1));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, v, o, o, &one, x, v,
                              owner->energy_occ, o, &zero, owner->gap_scratch, v));
    ++owner->blas_calls;
    resident_blas(cublasDaxpy(owner->blas, dim, &minus, owner->gap_scratch, 1, y, 1));
    ++owner->blas_calls;

    int numerical_error = 0;
    resident_cuda(cudaMemcpyAsync(&numerical_error, owner->numerical_error, sizeof(int),
                                  cudaMemcpyDeviceToHost, owner->stream));
    owner->d2h_bytes += sizeof(int);
    resident_sync(owner);
    if (numerical_error) throw std::runtime_error("nonfinite resident RHF direct J/K action");
    ++owner->operator_actions;
  });
#else
  (void)owner;
  (void)destination;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_reconstruct_v1(
    vibeqc_rhf_response_resident* owner, uint32_t solution_slot, const double* frozen_mo,
    uint64_t frozen_count, const double* overlap_mo, uint64_t overlap_count) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    owner->reconstruction_ready = false;
    const auto n = owner->nbf, o = owner->nocc, v = owner->nvirt, matrix = n * n;
    require(frozen_mo && overlap_mo && frozen_count == matrix && overlap_count == matrix,
            "resident RHF reconstruction matrix shape mismatch");
    for (std::size_t i = 0; i < matrix; ++i)
      require(std::isfinite(frozen_mo[i]) && std::isfinite(overlap_mo[i]),
              "resident RHF reconstruction requires finite matrices");
    const auto* x = resident_slot(owner, solution_slot);
    const auto* c_occ = owner->coefficients;
    const double one = 1.0, zero = 0.0, two = 2.0, half = -0.5;
    std::vector<double> mo1(n * o), hs(n * o);
    for (std::size_t i = 0; i < o; ++i) {
      const double ei = owner->orbital_energies[i];
      for (std::size_t row = 0; row < n; ++row) {
        const double sij = overlap_mo[row * n + i];
        mo1[i * n + row] = -0.5 * sij;
        double value = frozen_mo[row * n + i] - sij * ei;
        if (row < o) value += (-0.5 * sij) * (owner->orbital_energies[row] - ei);
        hs[i * n + row] = value;
      }
    }

    resident_cuda(cudaMemcpyAsync(owner->transform_one, mo1.data(), mo1.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, owner->stream));
    owner->h2d_bytes += mo1.size() * sizeof(double);
    for (std::size_t i = 0; i < o; ++i) {
      resident_blas(cublasDaxpy(owner->blas, static_cast<int>(v), &one, x + i * v, 1,
                                owner->transform_one + i * n + o, 1));
      ++owner->blas_calls;
    }
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, o, n, &one,
                              owner->coefficients, n, owner->transform_one, n, &zero,
                              owner->transform_two, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two,
                              owner->transform_two, n, c_occ, n, &zero, owner->density, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two, c_occ, n,
                              owner->transform_two, n, &one, owner->density, n));
    ++owner->blas_calls;

    auto spec = owner->parent->strategy().spec;
    spec.derivative_order = 0;
    std::string detail;
    const auto status = vibeqc::scf::enqueue_cuda_direct_jk_device(
        owner->direct, spec, owner->density, nullptr, matrix, owner->coulomb, owner->exchange,
        nullptr, owner->numerical_error, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "resident reconstruction J/K failed" : detail);

    resident_blas(cublasDscal(owner->blas, static_cast<int>(matrix), &half, owner->exchange, 1));
    ++owner->blas_calls;
    resident_blas(cublasDaxpy(owner->blas, static_cast<int>(matrix), &one, owner->coulomb, 1,
                              owner->exchange, 1));
    ++owner->blas_calls;
    resident_cuda(cudaMemcpyAsync(owner->transform_one, hs.data(), hs.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, owner->stream));
    owner->h2d_bytes += hs.size() * sizeof(double);
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, o, n, &one, owner->exchange,
                              n, c_occ, n, &zero, owner->coulomb, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, o, n, &one,
                              owner->coefficients, n, owner->coulomb, n, &one, owner->transform_one,
                              n));
    ++owner->blas_calls;
    for (std::size_t i = 0; i < o; ++i) {
      const double ei = owner->orbital_energies[i];
      resident_blas(
          cublasDscal(owner->blas, static_cast<int>(n), &ei, owner->transform_two + i * n, 1));
      ++owner->blas_calls;
    }
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two,
                              owner->transform_two, n, c_occ, n, &zero, owner->coulomb, n));
    ++owner->blas_calls;
#if VIBEQC_CUDA_PROVIDER_CUMETAL
    // CuMetal does not expose cublasDgeam. Preserve the same column-major A + A^T
    // operation with its supported Level-1 surface, without changing the NVIDIA path.
    resident_blas(
        cublasDcopy(owner->blas, static_cast<int>(matrix), owner->coulomb, 1, owner->exchange, 1));
    ++owner->blas_calls;
    for (std::size_t column = 0; column < n; ++column) {
      resident_blas(cublasDaxpy(owner->blas, static_cast<int>(n), &one, owner->coulomb + column,
                                static_cast<int>(n), owner->exchange + column * n, 1));
      ++owner->blas_calls;
    }
#else
    resident_blas(cublasDgeam(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, &one, owner->coulomb, n,
                              &one, owner->coulomb, n, owner->exchange, n));
    ++owner->blas_calls;
#endif

    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, o, o, &one, c_occ, n,
                              owner->transform_one, n, &zero, owner->transform_two, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, o, &two,
                              owner->transform_two, n, c_occ, n, &one, owner->exchange, n));
    ++owner->blas_calls;
    resident_blas(
        cublasDcopy(owner->blas, static_cast<int>(matrix), owner->exchange, 1, owner->coulomb, 1));
    ++owner->blas_calls;
    int numerical_error = 0;
    resident_cuda(cudaMemcpyAsync(&numerical_error, owner->numerical_error, sizeof(int),
                                  cudaMemcpyDeviceToHost, owner->stream));
    owner->d2h_bytes += sizeof(int);
    resident_sync(owner);
    if (numerical_error) throw std::runtime_error("nonfinite resident RHF reconstruction J/K");
    owner->reconstruction_ready = true;
  });
#else
  (void)owner;
  (void)solution_slot;
  (void)frozen_mo;
  (void)frozen_count;
  (void)overlap_mo;
  (void)overlap_count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" const double* vibeqc_rhf_response_resident_reconstructed_weights_device_v1(
    const vibeqc_rhf_response_resident* owner) {
#if VIBEQC_HAS_CUDA
  return owner && owner->reconstruction_ready ? owner->density : nullptr;
#else
  (void)owner;
  return nullptr;
#endif
}

extern "C" vibeqc_status vibeqc_rhf_response_resident_download_reconstruction_v1(
    vibeqc_rhf_response_resident* owner, double* density_derivative, uint64_t density_count,
    double* energy_weighted_density_derivative, uint64_t energy_count) {
#if VIBEQC_HAS_CUDA
  return resident_guard(owner, [&] {
    const auto count = owner->nbf * owner->nbf;
    require(owner->reconstruction_ready, "resident RHF reconstruction is unavailable");
    require(density_derivative && energy_weighted_density_derivative && density_count == count &&
                energy_count == count,
            "resident RHF reconstruction download shape mismatch");
    resident_cuda(cudaMemcpyAsync(density_derivative, owner->density, count * sizeof(double),
                                  cudaMemcpyDeviceToHost, owner->stream));
    resident_cuda(cudaMemcpyAsync(energy_weighted_density_derivative, owner->coulomb,
                                  count * sizeof(double), cudaMemcpyDeviceToHost, owner->stream));
    owner->d2h_bytes += 2 * count * sizeof(double);
    resident_sync(owner);
  });
#else
  (void)owner;
  (void)density_derivative;
  (void)density_count;
  (void)energy_weighted_density_derivative;
  (void)energy_count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

namespace {
#if VIBEQC_HAS_CUDA
double* uhf_resident_slot(vibeqc_uhf_response_resident* owner, std::uint32_t slot) {
  require(owner && slot < owner->vector_slots, "resident UHF response slot out of range");
  return owner->slots + static_cast<std::size_t>(slot) * owner->dimension;
}
const double* uhf_resident_slot(const vibeqc_uhf_response_resident* owner, std::uint32_t slot) {
  require(owner && slot < owner->vector_slots, "resident UHF response slot out of range");
  return owner->slots + static_cast<std::size_t>(slot) * owner->dimension;
}
template <class Function>
vibeqc_status uhf_resident_guard(vibeqc_uhf_response_resident* owner, Function function) {
  if (!owner) return VIBEQC_STATUS_INVALID_ARGUMENT;
  owner->detail.clear();
  try {
    resident_cuda(cudaSetDevice(owner->device_id));
    function();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&owner->detail);
  }
}
void uhf_resident_sync(vibeqc_uhf_response_resident* owner) {
  resident_cuda(cudaStreamSynchronize(owner->stream));
  ++owner->synchronizations;
}
#endif
}  // namespace

extern "C" vibeqc_status vibeqc_uhf_response_resident_create(
    vibeqc_fock_plan* plan, const double* coefficients_alpha, uint64_t coefficients_alpha_count,
    const double* orbital_energies_alpha, uint64_t orbital_energies_alpha_count,
    uint32_t nocc_alpha, const double* coefficients_beta, uint64_t coefficients_beta_count,
    const double* orbital_energies_beta, uint64_t orbital_energies_beta_count, uint32_t nocc_beta,
    uint32_t vector_slots, uint64_t device_budget_bytes, vibeqc_uhf_response_resident** output) {
  if (output) *output = nullptr;
  if (!plan || !coefficients_alpha || !orbital_energies_alpha || !coefficients_beta ||
      !orbital_energies_beta || !output)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
#if VIBEQC_HAS_CUDA
  plan->detail.clear();
  try {
    const auto& source = *plan->source;
    const auto& strategy = source.strategy();
    require(strategy.backend == vibeqc::scf::FockBackend::Cuda,
            "resident UHF response requires CUDA Fock plan");
    require(strategy.spec.spin == vibeqc::scf::FockSpin::Unrestricted &&
                strategy.spec.coulomb.present && strategy.spec.exchange.present &&
                strategy.spec.coulomb.approximation == vibeqc::scf::FockApproximation::Exact &&
                strategy.spec.exchange.approximation == vibeqc::scf::FockApproximation::Exact &&
                strategy.spec.coulomb.op == vibeqc::scf::FockOperator::FullRange &&
                strategy.spec.exchange.op == vibeqc::scf::FockOperator::FullRange &&
                strategy.spec.coulomb.coefficient == 1.0 &&
                strategy.spec.exchange.coefficient == -1.0 && strategy.screening_tolerance == 0.0,
            "resident UHF response requires exact unscreened conventional UHF J/K");
    auto* direct = source.cuda_direct_source();
    require(direct != nullptr, "resident UHF response direct CUDA source unavailable");
    const auto n = source.one_electron().nbf;
    require(n > 1 && nocc_alpha > 0 && nocc_alpha < n && nocc_beta < n && vector_slots >= 8 &&
                vector_slots <= 4096,
            "resident UHF response dimensions/slot count are invalid");
    const auto matrix = resident_product(n, n);
    require(coefficients_alpha_count == matrix && coefficients_beta_count == matrix &&
                orbital_energies_alpha_count == n && orbital_energies_beta_count == n,
            "resident UHF response reference dimensions mismatch");
    for (std::size_t i = 0; i < matrix; ++i)
      require(std::isfinite(coefficients_alpha[i]) && std::isfinite(coefficients_beta[i]),
              "nonfinite resident UHF coefficients");
    for (std::size_t i = 0; i < n; ++i)
      require(std::isfinite(orbital_energies_alpha[i]) && std::isfinite(orbital_energies_beta[i]),
              "nonfinite resident UHF orbital energies");
    const auto oa = static_cast<std::size_t>(nocc_alpha), va = n - oa;
    const auto ob = static_cast<std::size_t>(nocc_beta), vb = n - ob;
    const auto alpha_dim = resident_product(oa, va), beta_dim = resident_product(ob, vb);
    const auto dimension = resident_sum({alpha_dim, beta_dim});
    const auto cublas_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    require(n <= cublas_limit && oa <= cublas_limit && va <= cublas_limit && ob <= cublas_limit &&
                vb <= cublas_limit && alpha_dim <= cublas_limit && beta_dim <= cublas_limit &&
                dimension <= cublas_limit && matrix <= cublas_limit,
            "resident UHF response dimensions exceed cuBLAS int limits");
    const auto slots = resident_product(static_cast<std::size_t>(vector_slots), dimension);
    const auto doubles =
        resident_sum({slots, 2 * matrix, 2 * matrix, 2 * matrix, matrix, resident_product(n, oa),
                      resident_product(n, ob), resident_product(oa, oa), resident_product(va, va),
                      resident_product(ob, ob), resident_product(vb, vb), alpha_dim, beta_dim});
    const auto bytes = resident_sum({resident_product(doubles, sizeof(double)), sizeof(int)});
    require(device_budget_bytes > 0 && bytes <= device_budget_bytes,
            "resident UHF response device budget is insufficient");

    auto owner = std::make_unique<vibeqc_uhf_response_resident>();
    owner->parent = plan;
    owner->direct = direct;
    owner->device_id = vibeqc::scf::cuda_direct_jk_device(direct);
    owner->stream = vibeqc::scf::cuda_direct_jk_stream(direct);
    owner->nbf = n;
    owner->nocc_alpha = oa;
    owner->nvirt_alpha = va;
    owner->nocc_beta = ob;
    owner->nvirt_beta = vb;
    owner->dimension = dimension;
    owner->vector_slots = vector_slots;
    owner->alpha_offset = 0;
    owner->beta_offset = alpha_dim;
    owner->allocation_bytes = bytes;
    resident_cuda(cudaSetDevice(owner->device_id));
    // Staging must outlive the inner catch: any later upload or sync may fail
    // while earlier asynchronous copies still borrow these host buffers.
    std::vector<double> alpha_column_major(matrix), beta_column_major(matrix);
    std::vector<double> eoa(oa * oa, 0.0), eva(va * va, 0.0), eob(ob * ob, 0.0), evb(vb * vb, 0.0);
    resident_blas(cublasCreate(&owner->blas));
    try {
      resident_blas(cublasSetStream(owner->blas, owner->stream));
      resident_blas(cublasSetPointerMode(owner->blas, CUBLAS_POINTER_MODE_HOST));
      resident_cuda(vibeqc::runtime::resource_cuda_malloc(&owner->allocation, bytes));
      auto* cursor = static_cast<double*>(owner->allocation);
      owner->slots = cursor;
      cursor += slots;
      owner->coefficients_alpha = cursor;
      cursor += matrix;
      owner->coefficients_beta = cursor;
      cursor += matrix;
      owner->energy_occ_alpha = cursor;
      cursor += oa * oa;
      owner->energy_virt_alpha = cursor;
      cursor += va * va;
      owner->energy_occ_beta = cursor;
      cursor += ob * ob;
      owner->energy_virt_beta = cursor;
      cursor += vb * vb;
      owner->density_alpha = cursor;
      cursor += matrix;
      owner->density_beta = cursor;
      cursor += matrix;
      owner->coulomb = cursor;
      cursor += matrix;
      owner->exchange_alpha = cursor;
      cursor += matrix;
      owner->exchange_beta = cursor;
      cursor += matrix;
      owner->transform_alpha = cursor;
      cursor += n * oa;
      owner->transform_beta = cursor;
      cursor += n * ob;
      owner->gap_alpha = cursor;
      cursor += alpha_dim;
      owner->gap_beta = cursor;
      cursor += beta_dim;
      owner->numerical_error = reinterpret_cast<int*>(cursor);

      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column) {
          alpha_column_major[column * n + row] = coefficients_alpha[row * n + column];
          beta_column_major[column * n + row] = coefficients_beta[row * n + column];
        }
      for (std::size_t i = 0; i < oa; ++i) eoa[i * oa + i] = orbital_energies_alpha[i];
      for (std::size_t a = 0; a < va; ++a) eva[a * va + a] = orbital_energies_alpha[oa + a];
      for (std::size_t i = 0; i < ob; ++i) eob[i * ob + i] = orbital_energies_beta[i];
      for (std::size_t a = 0; a < vb; ++a) evb[a * vb + a] = orbital_energies_beta[ob + a];
      resident_cuda(cudaMemcpyAsync(owner->coefficients_alpha, alpha_column_major.data(),
                                    matrix * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->coefficients_beta, beta_column_major.data(),
                                    matrix * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_occ_alpha, eoa.data(),
                                    eoa.size() * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_virt_alpha, eva.data(),
                                    eva.size() * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_occ_beta, eob.data(), eob.size() * sizeof(double),
                                    cudaMemcpyHostToDevice, owner->stream));
      resident_cuda(cudaMemcpyAsync(owner->energy_virt_beta, evb.data(),
                                    evb.size() * sizeof(double), cudaMemcpyHostToDevice,
                                    owner->stream));
      owner->h2d_bytes =
          (2 * matrix + eoa.size() + eva.size() + eob.size() + evb.size()) * sizeof(double);
      uhf_resident_sync(owner.get());
    } catch (...) {
      if (owner->allocation) {
        (void)cudaStreamSynchronize(owner->stream);
        (void)vibeqc::runtime::resource_cuda_free(owner->allocation);
        owner->allocation = nullptr;
      }
      if (owner->blas) {
        (void)cublasDestroy(owner->blas);
        owner->blas = nullptr;
      }
      throw;
    }
    *output = owner.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&plan->detail);
  }
#else
  (void)coefficients_alpha_count;
  (void)orbital_energies_alpha_count;
  (void)nocc_alpha;
  (void)coefficients_beta_count;
  (void)orbital_energies_beta_count;
  (void)nocc_beta;
  (void)vector_slots;
  (void)device_budget_bytes;
  plan->detail = "resident UHF response requires a CUDA-enabled library";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" void vibeqc_uhf_response_resident_destroy(vibeqc_uhf_response_resident* owner) {
  if (!owner) return;
#if VIBEQC_HAS_CUDA
  if (owner->device_id >= 0) (void)cudaSetDevice(owner->device_id);
  if (owner->stream) (void)cudaStreamSynchronize(owner->stream);
  if (owner->allocation) (void)vibeqc::runtime::resource_cuda_free(owner->allocation);
  if (owner->blas) (void)cublasDestroy(owner->blas);
#endif
  delete owner;
}

extern "C" const char* vibeqc_uhf_response_resident_last_error(
    const vibeqc_uhf_response_resident* owner) {
  return owner ? owner->detail.c_str() : "null resident UHF response owner";
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_get_diagnostic(
    const vibeqc_uhf_response_resident* owner,
    vibeqc_uhf_response_resident_diagnostic* diagnostic) {
  if (!owner || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
#if VIBEQC_HAS_CUDA
  if (diagnostic->struct_size < sizeof(*diagnostic) ||
      diagnostic->abi_version != VIBEQC_ABI_VERSION)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  vibeqc_uhf_response_resident_diagnostic out{};
  out.struct_size = sizeof(out);
  out.abi_version = VIBEQC_ABI_VERSION;
  out.nbf = owner->nbf;
  out.nocc_alpha = owner->nocc_alpha;
  out.nvirt_alpha = owner->nvirt_alpha;
  out.nocc_beta = owner->nocc_beta;
  out.nvirt_beta = owner->nvirt_beta;
  out.dimension = owner->dimension;
  out.vector_slots = owner->vector_slots;
  out.owned_device_bytes = owner->allocation_bytes;
  out.h2d_bytes = owner->h2d_bytes;
  out.d2h_bytes = owner->d2h_bytes;
  out.synchronizations = owner->synchronizations;
  out.operator_actions = owner->operator_actions;
  out.blas_calls = owner->blas_calls;
  out.device_id = owner->device_id;
  *diagnostic = out;
  return VIBEQC_STATUS_SUCCESS;
#else
  (void)diagnostic;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_upload(vibeqc_uhf_response_resident* owner,
                                                             uint32_t slot, const double* values,
                                                             uint64_t count) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(values && count == owner->dimension, "resident UHF upload shape mismatch");
    for (std::size_t i = 0; i < owner->dimension; ++i)
      require(std::isfinite(values[i]), "resident UHF upload requires finite vectors");
    resident_cuda(cudaMemcpyAsync(uhf_resident_slot(owner, slot), values,
                                  owner->dimension * sizeof(double), cudaMemcpyHostToDevice,
                                  owner->stream));
    owner->h2d_bytes += owner->dimension * sizeof(double);
    uhf_resident_sync(owner);
  });
#else
  (void)owner;
  (void)slot;
  (void)values;
  (void)count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_download(vibeqc_uhf_response_resident* owner,
                                                               uint32_t slot, double* values,
                                                               uint64_t count) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(values && count == owner->dimension, "resident UHF download shape mismatch");
    resident_cuda(cudaMemcpyAsync(values, uhf_resident_slot(owner, slot),
                                  owner->dimension * sizeof(double), cudaMemcpyDeviceToHost,
                                  owner->stream));
    owner->d2h_bytes += owner->dimension * sizeof(double);
    uhf_resident_sync(owner);
    for (std::size_t i = 0; i < owner->dimension; ++i)
      require(std::isfinite(values[i]), "nonfinite resident UHF vector");
  });
#else
  (void)owner;
  (void)slot;
  (void)values;
  (void)count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_zero(vibeqc_uhf_response_resident* owner,
                                                           uint32_t slot) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    resident_cuda(cudaMemsetAsync(uhf_resident_slot(owner, slot), 0,
                                  owner->dimension * sizeof(double), owner->stream));
    uhf_resident_sync(owner);
  });
#else
  (void)owner;
  (void)slot;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_copy(vibeqc_uhf_response_resident* owner,
                                                           uint32_t destination, uint32_t source) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(destination != source, "resident UHF copy source/output must be distinct");
    resident_blas(cublasDcopy(owner->blas, static_cast<int>(owner->dimension),
                              uhf_resident_slot(owner, source), 1,
                              uhf_resident_slot(owner, destination), 1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)destination;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_scale(vibeqc_uhf_response_resident* owner,
                                                            uint32_t slot, double alpha) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(std::isfinite(alpha), "resident UHF scale must be finite");
    resident_blas(cublasDscal(owner->blas, static_cast<int>(owner->dimension), &alpha,
                              uhf_resident_slot(owner, slot), 1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)slot;
  (void)alpha;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_axpy(vibeqc_uhf_response_resident* owner,
                                                           uint32_t destination, double alpha,
                                                           uint32_t source) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(std::isfinite(alpha), "resident UHF axpy coefficient must be finite");
    resident_blas(cublasDaxpy(owner->blas, static_cast<int>(owner->dimension), &alpha,
                              uhf_resident_slot(owner, source), 1,
                              uhf_resident_slot(owner, destination), 1));
    ++owner->blas_calls;
  });
#else
  (void)owner;
  (void)destination;
  (void)alpha;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_dot(vibeqc_uhf_response_resident* owner,
                                                          uint32_t left, uint32_t right,
                                                          double* value) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(value, "resident UHF dot requires output");
    resident_blas(cublasDdot(owner->blas, static_cast<int>(owner->dimension),
                             uhf_resident_slot(owner, left), 1, uhf_resident_slot(owner, right), 1,
                             value));
    ++owner->blas_calls;
    owner->d2h_bytes += sizeof(double);
    uhf_resident_sync(owner);
    require(std::isfinite(*value), "nonfinite resident UHF dot");
  });
#else
  (void)owner;
  (void)left;
  (void)right;
  (void)value;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_norm(vibeqc_uhf_response_resident* owner,
                                                           uint32_t slot, double* value) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(value, "resident UHF norm requires output");
    resident_blas(cublasDnrm2(owner->blas, static_cast<int>(owner->dimension),
                              uhf_resident_slot(owner, slot), 1, value));
    ++owner->blas_calls;
    owner->d2h_bytes += sizeof(double);
    uhf_resident_sync(owner);
    require(std::isfinite(*value), "nonfinite resident UHF norm");
  });
#else
  (void)owner;
  (void)slot;
  (void)value;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

extern "C" vibeqc_status vibeqc_uhf_response_resident_apply(vibeqc_uhf_response_resident* owner,
                                                            uint32_t destination, uint32_t source) {
#if VIBEQC_HAS_CUDA
  return uhf_resident_guard(owner, [&] {
    require(destination != source, "resident UHF operator source/output must be distinct");
    const auto n = static_cast<int>(owner->nbf);
    const auto oa = static_cast<int>(owner->nocc_alpha), va = static_cast<int>(owner->nvirt_alpha);
    const auto ob = static_cast<int>(owner->nocc_beta), vb = static_cast<int>(owner->nvirt_beta);
    const double one = 1.0, zero = 0.0, minus = -1.0;
    const auto* x = uhf_resident_slot(owner, source);
    auto* y = uhf_resident_slot(owner, destination);
    const auto* xa = x + owner->alpha_offset;
    const auto* xb = x + owner->beta_offset;
    auto* ya = y + owner->alpha_offset;
    auto* yb = y + owner->beta_offset;
    const auto* ca_occ = owner->coefficients_alpha;
    const auto* ca_virt = owner->coefficients_alpha + owner->nbf * owner->nocc_alpha;
    const auto* cb_occ = owner->coefficients_beta;
    const auto* cb_virt = owner->coefficients_beta + owner->nbf * owner->nocc_beta;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, oa, va, &one, ca_virt, n,
                              xa, va, &zero, owner->transform_alpha, n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, oa, &one,
                              owner->transform_alpha, n, ca_occ, n, &zero, owner->density_alpha,
                              n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, oa, &one, ca_occ, n,
                              owner->transform_alpha, n, &one, owner->density_alpha, n));
    ++owner->blas_calls;
    if (ob > 0) {
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, ob, vb, &one, cb_virt, n,
                                xb, vb, &zero, owner->transform_beta, n));
      ++owner->blas_calls;
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, ob, &one,
                                owner->transform_beta, n, cb_occ, n, &zero, owner->density_beta,
                                n));
      ++owner->blas_calls;
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, ob, &one, cb_occ, n,
                                owner->transform_beta, n, &one, owner->density_beta, n));
      ++owner->blas_calls;
    } else {
      resident_cuda(cudaMemsetAsync(owner->density_beta, 0,
                                    owner->nbf * owner->nbf * sizeof(double), owner->stream));
    }
    auto spec = owner->parent->source->strategy().spec;
    spec.derivative_order = 0;
    std::string detail;
    const auto status = vibeqc::scf::enqueue_cuda_direct_jk_device(
        owner->direct, spec, owner->density_alpha, owner->density_beta, owner->nbf * owner->nbf,
        owner->coulomb, owner->exchange_alpha, owner->exchange_beta, owner->numerical_error,
        detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "resident UHF direct J/K enqueue failed" : detail);
    resident_blas(cublasDcopy(owner->blas, n * n, owner->coulomb, 1, owner->density_alpha, 1));
    ++owner->blas_calls;
    resident_blas(
        cublasDaxpy(owner->blas, n * n, &minus, owner->exchange_alpha, 1, owner->density_alpha, 1));
    ++owner->blas_calls;
    resident_blas(cublasDcopy(owner->blas, n * n, owner->coulomb, 1, owner->density_beta, 1));
    ++owner->blas_calls;
    resident_blas(
        cublasDaxpy(owner->blas, n * n, &minus, owner->exchange_beta, 1, owner->density_beta, 1));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, oa, n, &one,
                              owner->density_alpha, n, ca_occ, n, &zero, owner->transform_alpha,
                              n));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_T, CUBLAS_OP_N, va, oa, n, &one, ca_virt, n,
                              owner->transform_alpha, n, &zero, ya, va));
    ++owner->blas_calls;
    if (ob > 0) {
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, n, ob, n, &one,
                                owner->density_beta, n, cb_occ, n, &zero, owner->transform_beta,
                                n));
      ++owner->blas_calls;
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_T, CUBLAS_OP_N, vb, ob, n, &one, cb_virt, n,
                                owner->transform_beta, n, &zero, yb, vb));
      ++owner->blas_calls;
    }
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, va, oa, va, &one,
                              owner->energy_virt_alpha, va, xa, va, &zero, owner->gap_alpha, va));
    ++owner->blas_calls;
    resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, va, oa, oa, &one, xa, va,
                              owner->energy_occ_alpha, oa, &zero, owner->transform_alpha, va));
    ++owner->blas_calls;
    resident_blas(
        cublasDaxpy(owner->blas, va * oa, &minus, owner->transform_alpha, 1, owner->gap_alpha, 1));
    ++owner->blas_calls;
    resident_blas(cublasDaxpy(owner->blas, va * oa, &one, owner->gap_alpha, 1, ya, 1));
    ++owner->blas_calls;
    if (ob > 0) {
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, vb, ob, vb, &one,
                                owner->energy_virt_beta, vb, xb, vb, &zero, owner->gap_beta, vb));
      ++owner->blas_calls;
      resident_blas(cublasDgemm(owner->blas, CUBLAS_OP_N, CUBLAS_OP_N, vb, ob, ob, &one, xb, vb,
                                owner->energy_occ_beta, ob, &zero, owner->transform_beta, vb));
      ++owner->blas_calls;
      resident_blas(
          cublasDaxpy(owner->blas, vb * ob, &minus, owner->transform_beta, 1, owner->gap_beta, 1));
      ++owner->blas_calls;
      resident_blas(cublasDaxpy(owner->blas, vb * ob, &one, owner->gap_beta, 1, yb, 1));
      ++owner->blas_calls;
    }
    int numerical_error = 0;
    resident_cuda(cudaMemcpyAsync(&numerical_error, owner->numerical_error, sizeof(int),
                                  cudaMemcpyDeviceToHost, owner->stream));
    owner->d2h_bytes += sizeof(int);
    uhf_resident_sync(owner);
    if (numerical_error) throw std::runtime_error("nonfinite resident UHF direct J/K action");
    ++owner->operator_actions;
  });
#else
  (void)owner;
  (void)destination;
  (void)source;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
