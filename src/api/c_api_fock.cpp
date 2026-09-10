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

struct vibeqc_fock_plan {
  std::unique_ptr<vibeqc::scf::PreparedFockPlan> source;
  vibeqc::scf::FockBuildSpec requested;
  std::string detail;
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
  context->last_detail.clear();
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
