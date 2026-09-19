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
  vibeqc_fock_plan* parent{};
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
 * Return 0 for dense, 1 for packed lower pairs, and -1 for an invalid handle.
 */
extern "C" int vibeqc_fock_plan_df_pair_storage_v1(const vibeqc_fock_plan* plan) {
  if (!plan) return -1;
  return plan->source->diagnostic().variant.df_pair_storage == DfPairStorage::SymmetricLower ? 1
                                                                                             : 0;
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
  std::snprintf(out.one_electron_value_backend, sizeof(out.one_electron_value_backend), "%s",
                cuda ? "cuda-generated" : "cpu-reference");
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
    owner->parent = plan;
    owner->direct = direct;
    owner->device_id = vibeqc::scf::cuda_direct_jk_device(direct);
    owner->stream = vibeqc::scf::cuda_direct_jk_stream(direct);
    owner->nbf = n;
    owner->nocc = o;
    owner->nvirt = v;
    owner->dimension = dim;
    owner->vector_slots = vector_slots;
    owner->allocation_bytes = bytes;

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

    auto spec = owner->parent->source->strategy().spec;
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
