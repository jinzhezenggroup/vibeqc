#include "methods/rccsd_method.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cc/df_source.hpp"
#include "cc/rccsdt_force.hpp"
#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"
#include "methods/correlated_cuda_source.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "posthf/source_reuse_schedule_generated.hpp"
#include "runtime/execution_context.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"
#if GENERATIVEQC_HAS_CUDA
#include "generated_direct_resident_psss_schedule.cuh"
#include "scf/cuda/df_source_domain.hpp"
#endif

namespace generativeqc::methods::detail {
namespace {

std::vector<std::size_t> range(std::size_t begin, std::size_t end) {
  std::vector<std::size_t> result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}

generativeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

cc::SolverOptions cc_options(const generativeqc_method_descriptor& d, std::size_t budget) {
  cc::SolverOptions options;
  options.max_bytes = budget;
  if (d.ccsd_max_iterations) options.max_iterations = d.ccsd_max_iterations;
  options.diis_size = d.ccsd_diis_history;
  if (d.ccsd_energy_tolerance) options.energy_tolerance = d.ccsd_energy_tolerance;
  if (d.ccsd_residual_tolerance) options.residual_tolerance = d.ccsd_residual_tolerance;
  if (d.ccsd_denominator_threshold) options.denominator_threshold = d.ccsd_denominator_threshold;
  options.damping = d.ccsd_damping;
  options.level_shift = d.ccsd_level_shift;
  cc::validate_options(options);
  return options;
}

scf::ScfOptions reference_options(const generativeqc_method_descriptor& d, std::size_t budget) {
  if (!std::isfinite(d.energy_tolerance) || d.energy_tolerance < 0 ||
      !std::isfinite(d.density_tolerance) || d.density_tolerance < 0)
    throw std::invalid_argument("invalid RCCSD reference convergence threshold");
  scf::ScfOptions options;
  options.max_iterations = d.max_iterations ? d.max_iterations : 100;
  options.diis_history = d.diis_history ? d.diis_history : 8;
  options.energy_tolerance = d.energy_tolerance > 0 ? std::min(d.energy_tolerance, 1e-11) : 1e-11;
  options.density_tolerance =
      d.density_tolerance > 0 ? std::min(d.density_tolerance, 1e-11) : 1e-11;
  options.screening_tolerance = 0;
  options.compute_forces = false;
  options.export_physical_reference = true;
  options.reference_memory_budget_bytes = budget;
  options.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  return options;
}

std::size_t correlation_budget(const generativeqc_method_descriptor& d) {
  const auto requested = d.correlation_memory_budget_bytes;
  if (requested > static_cast<std::uint64_t>(INT64_MAX) ||
      requested > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("RCCSD budget exceeds numeric capacity");
  return requested ? static_cast<std::size_t>(requested) : 256ULL << 20;
}

void validate_descriptor(const generativeqc_method_descriptor& d,
                         const runtime::ExecutionContext& execution) {
  if (d.screening_tolerance != 0)
    throw std::invalid_argument(
        "canonical RCCSD requires unscreened integrals (screening_tolerance=0)");
  if (d.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD density-fitted reference/integrals are not implemented");
  if (d.density_fitting_auxiliary_basis)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "conventional RCCSD does not accept an auxiliary basis");
  if (d.precision_mode != GENERATIVEQC_PRECISION_FP64 &&
      d.precision_mode != GENERATIVEQC_PRECISION_AUTO)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unknown floating-point precision mode");
  if (d.precision_mode != GENERATIVEQC_PRECISION_FP64)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "RCCSD requires FP64 precision");
  if (d.ccsd_frozen_core != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD frozen-core references are not implemented");
  if (execution.backend() != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
      execution.backend() != GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD requires an explicit CPU or CUDA backend");
}

std::vector<double> fock_mo(const hf::PhysicalReference& ref) {
  const auto n = ref.nbf;
  std::vector<double> scratch(n * n), result(n * n);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu)
        value += ref.fock[mu * n + nu] * ref.coefficients[nu * n + q];
      scratch[mu * n + q] = value;
    }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < n; ++mu)
        value += ref.coefficients[mu * n + p] * scratch[mu * n + q];
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD MO Fock matrix");
      result[p * n + q] = value;
    }
  return result;
}

#if GENERATIVEQC_HAS_CUDA
void attach_df_source(cc::Problem& problem, cc::DFSourceResult&& fitted, posthf::ProviderWork& work,
                      generativeqc_tensor::Metrics& metrics) {
  problem.naux = fitted.naux;
  problem.df_bov = std::move(fitted.bov);
  problem.df_bvv = std::move(fitted.bvv);
  problem.ovov = std::move(fitted.ovov);
  problem.ovvo = std::move(fitted.ovvo);
  problem.oovv = std::move(fitted.oovv);
  problem.ovoo = std::move(fitted.ovoo);
  problem.oooo = std::move(fitted.oooo);
  problem.provider_peak_bytes = fitted.numeric_capacity_bytes;
  problem.provider_host_bytes = fitted.host_output_bytes;
  metrics.owned_device_bytes =
      std::max<std::uint64_t>(metrics.owned_device_bytes, fitted.device_capacity_bytes);
  ++work.source_scans;
  work.source_reads += fitted.source_rows;
  work.source_values += fitted.source_values;
  work.transform_fmas += fitted.transform_summands + fitted.block_summands;
  work.transform_stages += fitted.transform_gemms + fitted.block_gemms;
  work.mo_blocks += 5;
  ++work.cuda_transform_calls;
  work.h2d_bytes += fitted.coefficient_h2d_bytes;
  work.d2h_bytes += fitted.factor_block_d2h_bytes;
  work.provider_seconds += fitted.total_seconds;
  work.source_seconds += fitted.source_seconds;
}
#endif

std::size_t retained_reference_bytes(const hf::PhysicalReference& ref) {
  std::size_t result = 0;
  const std::vector<double>* arrays[] = {
      &ref.overlap, &ref.hcore,           &ref.fock, &ref.coefficients, &ref.orbital_energies,
      &ref.density, &ref.weighted_density};
  for (const auto* values : arrays)
    result = posthf::checked_add(result, posthf::checked_mul(values->capacity(), sizeof(double)));
  return result;
}

cc::Problem build_problem(const integrals::ElectronInteractionSource& source,
                          const hf::PhysicalReference& ref, const cc::SolverOptions& options,
                          bool cuda, int device, posthf::ProviderWork& provider_work,
                          generativeqc_tensor::Metrics& provider_metrics,
                          const core::System* correlation_auxiliary = nullptr) {
  // A failed optional source may already have performed real work. Preserve
  // cumulative diagnostics, but validate the compiler schedule for this attempt.
  const auto initial_work = provider_work;
  cc::Problem p;
  p.nocc = ref.nocc;
  p.reference_retained_bytes = retained_reference_bytes(ref);
  p.nvir = ref.nbf - ref.nocc;
  p.reference_energy = ref.energy;
  const auto o = p.nocc, v = p.nvir, n = ref.nbf;
  if (!o || !v || ref.orbital_energies.size() != n)
    throw std::invalid_argument("invalid RCCSD canonical reference dimensions");

  const auto fmo = fock_mo(ref);
  p.foo.resize(o * o);
  p.fov.resize(o * v);
  p.fvv.resize(v * v);
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = 0; j < o; ++j) p.foo[i * o + j] = fmo[i * n + j];
    for (std::size_t a = 0; a < v; ++a) p.fov[i * v + a] = fmo[i * n + o + a];
  }
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = 0; b < v; ++b) p.fvv[a * v + b] = fmo[(o + a) * n + o + b];

  p.d1.resize(o * v);
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      const double physical = ref.orbital_energies[i] - ref.orbital_energies[o + a];
      minimum = std::min(minimum, std::abs(physical));
      if (!std::isfinite(physical) || physical >= 0.0 ||
          std::abs(physical) <= options.denominator_threshold)
        throw std::invalid_argument("near-zero or nonnegative physical RCCSD denominator");
      p.d1[i * v + a] = physical - options.level_shift;
    }
  const auto n2 = o * o * v * v;
  p.d2.resize(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const double physical = (ref.orbital_energies[i] - ref.orbital_energies[o + a]) +
                                  (ref.orbital_energies[j] - ref.orbital_energies[o + b]);
          minimum = std::min(minimum, std::abs(physical));
          if (!std::isfinite(physical) || physical >= 0.0 ||
              std::abs(physical) <= options.denominator_threshold)
            throw std::invalid_argument(
                "near-zero or nonnegative physical RCCSD doubles denominator");
          p.d2[((i * o + j) * v + a) * v + b] = physical - 2.0 * options.level_shift;
        }

  p.minimum_absolute_denominator = minimum;
  if (correlation_auxiliary) {
#if GENERATIVEQC_HAS_CUDA
    if (!cuda) throw std::invalid_argument("native molecular DF-CC source requires CUDA");
    // Account for the live Fock transform, split energy spectrum and problem
    // prefix while the DF source owns its device/host phase allocations.
    auto caller_elements = posthf::checked_add(fmo.capacity(), n);
    for (const auto* values : {&p.foo, &p.fov, &p.fvv, &p.d1, &p.d2})
      caller_elements = posthf::checked_add(caller_elements, values->capacity());
    // RawSource owns a basis copy while the caller's original system stays
    // alive. The new source already counts the supplied copy and auxiliary.
    const auto caller_bytes =
        posthf::checked_add(posthf::checked_mul(caller_elements, sizeof(double)),
                            posthf::source_capacity(source.orbital()));
    auto fitted = cc::build_df_source_cuda(source.orbital(), *correlation_auxiliary, ref,
                                           options.max_bytes, 1e-10, device, caller_bytes);
    attach_df_source(p, std::move(fitted), provider_work, provider_metrics);
    // RawSource is released before solve, but both caller systems and the
    // split orbital spectrum remain live beside the detached RHF reference.
    p.reference_retained_bytes = posthf::checked_add(
        p.reference_retained_bytes,
        posthf::checked_add(posthf::checked_mul(n, sizeof(double)),
                            posthf::checked_add(posthf::source_capacity(source.orbital()),
                                                posthf::source_capacity(*correlation_auxiliary))));
#else
    throw std::runtime_error("native molecular DF-CC source requires a CUDA build");
#endif
  } else {
    const auto occ = range(0, o), vir = range(o, n);
    const std::array<posthf::MOSlots, 7> requests{{
        {occ, vir, occ, vir},
        {occ, vir, vir, occ},
        {occ, occ, vir, vir},
        {occ, vir, vir, vir},
        {occ, vir, occ, occ},
        {occ, occ, occ, occ},
        {vir, vir, vir, vir},
    }};
    const std::array<std::array<std::size_t, 4>, 7> shapes{{
        {o, v, o, v},
        {o, v, v, o},
        {o, o, v, v},
        {o, v, v, v},
        {o, v, o, o},
        {o, o, o, o},
        {v, v, v, v},
    }};
    const std::array<std::vector<double>*, 7> targets{&p.ovov, &p.ovvo, &p.oovv, &p.ovvv,
                                                      &p.ovoo, &p.oooo, &p.vvvv};

    auto schedule_for = [&](const posthf::NativeBlockProvider& candidate) {
      const auto common_bytes = candidate.batch_bytes(shapes.front(), 0, cuda);
      std::vector<posthf::generated::SourceReuseRequest> schedule_requests;
      schedule_requests.reserve(requests.size());
      for (const auto& shape : shapes) {
        const auto single_bytes = candidate.batch_bytes(shape, 1, cuda);
        if (single_bytes < common_bytes)
          throw std::logic_error("RCCSD provider request accounting underflow");
        std::size_t output_elements = 1;
        for (const auto extent : shape)
          output_elements = posthf::checked_mul(output_elements, extent);
        schedule_requests.push_back(
            {single_bytes - common_bytes, posthf::checked_mul(output_elements, sizeof(double))});
      }
      return posthf::generated::ordered_source_reuse_plan(common_bytes, p.reference_retained_bytes,
                                                          options.max_bytes, schedule_requests);
    };

    posthf::NativeBlockProvider widest_provider(source, ref, options.max_bytes,
                                                std::numeric_limits<unsigned>::max(),
                                                posthf::AOTileDomain::Basis);
    const auto maximum_axis_tile = widest_provider.tile_shape()[0];
    std::vector<posthf::generated::SourceTileCandidate> tile_candidates;
    tile_candidates.reserve(maximum_axis_tile);
    for (std::size_t axis_tile = 1; axis_tile <= maximum_axis_tile; ++axis_tile) {
      try {
        posthf::NativeBlockProvider candidate(source, ref, options.max_bytes,
                                              static_cast<unsigned>(axis_tile),
                                              posthf::AOTileDomain::Basis);
        const auto candidate_reuse = schedule_for(candidate);
        tile_candidates.push_back({candidate.tile_shape()[0], candidate_reuse.batches.size(),
                                   candidate_reuse.peak_bytes});
      } catch (const std::length_error&) {
        continue;
      }
    }
    if (tile_candidates.empty())
      throw std::length_error("RCCSD MO provider exceeds numeric memory budget");
    const auto source_tile_plan = posthf::generated::select_source_tile(n, tile_candidates);
    posthf::NativeBlockProvider provider(source, ref, options.max_bytes,
                                         static_cast<unsigned>(source_tile_plan.axis_tile),
                                         posthf::AOTileDomain::Basis);
    const auto reuse = schedule_for(provider);

    std::size_t retained = 0;
    for (const auto& batch : reuse.batches) {
      std::vector<posthf::MOSlots> batch_requests;
      batch_requests.reserve(batch.end - batch.begin);
      for (std::size_t request = batch.begin; request < batch.end; ++request)
        batch_requests.push_back(requests[request]);
      auto outputs =
          provider.get_many(batch_requests, cuda, device, &provider_metrics, &provider_work);
      if (outputs.size() != batch_requests.size())
        throw std::runtime_error("RCCSD MO provider returned an invalid batch");
      for (std::size_t local = 0; local < outputs.size(); ++local) {
        const auto request = batch.begin + local;
        retained = posthf::checked_add(retained,
                                       posthf::checked_mul(outputs[local].size(), sizeof(double)));
        *targets[request] = std::move(outputs[local]);
      }
    }
    const auto source_scans = provider_work.source_scans - initial_work.source_scans;
    const auto source_reads = provider_work.source_reads - initial_work.source_reads;
    const auto source_values = provider_work.source_values - initial_work.source_values;
    if (source_scans != reuse.batches.size() || source_reads != source_tile_plan.source_reads)
      throw std::logic_error("RCCSD source-reuse execution disagrees with compiler schedule");
    const auto ao2 = posthf::checked_mul(n, n);
    const auto ao4 = posthf::checked_mul(ao2, ao2);
    const auto expected_source_values = posthf::checked_mul(ao4, source_scans);
    if (source_values != expected_source_values)
      throw std::logic_error("RCCSD AO source value count disagrees with compiler schedule");
    p.provider_peak_bytes = reuse.peak_bytes;
    p.provider_host_bytes = retained;
  }

  p.initial_t1.assign(o * v, 0.0);
  p.initial_t2.resize(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto t = ((i * o + j) * v + a) * v + b;
          const auto g = ((i * v + a) * o + j) * v + b;
          p.initial_t2[t] = p.ovov[g] / p.d2[t];
          if (!std::isfinite(p.initial_t2[t]))
            throw std::runtime_error("nonfinite RCCSD MP2-like initial amplitude");
        }
  return p;
}

RccsdNativeState execute_rccsd_prepared(
    runtime::ExecutionContext& execution, const core::System& system,
    const scf::ScfOptions& reference_options, const cc::SolverOptions& solver_options,
    std::size_t reference_capacity, scf::PreparedFockPlan* prepared_exact,
    const std::vector<double>* initial_density, bool* warm_start_fallback,
    std::unique_ptr<scf::PreparedFockPlan>* cuda_source_cache,
    const core::System* correlation_auxiliary = nullptr) {
  const char* allocation_stage = "HF reference";
  try {
    const bool cuda = execution.cuda_requested();
    if (warm_start_fallback) *warm_start_fallback = false;
    // A previous correlation source must not coexist with the next native
    // CUDA RHF arena. CPU keeps its qualified shared reference/source owner.
    if (cuda) {
      if (cuda_source_cache) cuda_source_cache->reset();
      prepared_exact = nullptr;
    }
    const auto reference_started = std::chrono::steady_clock::now();
    const auto run_reference = [&](const std::vector<double>* seed) {
      if (prepared_exact) {
        auto prepared_options = reference_options;
        prepared_options.resolved_fock_build = prepared_exact->strategy();
        return scf::run_prepared_fock_strategy(*prepared_exact, prepared_options, seed);
      }
      return cuda ? scf::run_rhf_cuda(system, reference_options, execution.device_id(), seed)
                  : scf::run_rhf(system, reference_options, seed);
    };
    scf::ScfResult hf;
    if (initial_density) {
      bool retried_cold = false;
      try {
        hf = run_reference(initial_density);
      } catch (...) {
        retried_cold = true;
        if (warm_start_fallback) *warm_start_fallback = true;
        hf = run_reference(nullptr);
      }
      if (!retried_cold && (!hf.converged || !hf.reference)) {
        hf = {};
        if (warm_start_fallback) *warm_start_fallback = true;
        hf = run_reference(nullptr);
      }
    } else {
      hf = run_reference(nullptr);
    }
    const double reference_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_started).count();
    if (!hf.converged || !hf.reference)
      throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED,
                        "HF did not converge; no RCCSD energy evaluated");
    const auto reference = hf.reference;
    hf.density.clear();
    hf.density.shrink_to_fit();
    const auto problem_started = std::chrono::steady_clock::now();
    std::size_t source_preparation_peak = 0;
#if GENERATIVEQC_HAS_CUDA
    if (cuda && cuda_source_cache && !correlation_auxiliary) {
      std::size_t primitives = 0, s_shells = 0, p_shells = 0;
      for (const auto& shell : system.shells) {
        primitives = posthf::checked_add(primitives, shell.primitives.size());
        s_shells += shell.angular_momentum == 0;
        p_shells += shell.angular_momentum == 1;
      }
      const auto n = molecule::ao_count(system);
      const auto device_bytes = scf::cuda_direct_jk_device_bytes(
          1, n, system.atoms.size(), system.shells.size(), primitives, 0);
      const auto source_plan = plan_correlated_cuda_source(
          n, molecule::cartesian_ao_count(system), system.atoms.size(), system.shells.size(),
          primitives, s_shells, p_shells, scf::cuda_execution::kResidentPsssThreads,
          posthf::source_capacity(system), retained_reference_bytes(*reference), device_bytes,
          solver_options.max_bytes);
      if (source_plan.admitted) {
        source_preparation_peak = source_plan.peak_bytes;
        try {
          auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
          // Admission above reserves value-only Direct storage. The default
          // Fock spec requests derivatives and would make this exact allowance
          // fail, silently routing every source tile through the host fallback.
          spec.derivative_order = 0;
          const auto strategy = scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 0.0);
          auto candidate = std::make_unique<scf::PreparedFockPlan>(
              system, nullptr, strategy, execution.device_id(), source_plan.device_bytes);
          *cuda_source_cache = std::move(candidate);
          prepared_exact = cuda_source_cache->get();
        } catch (const std::bad_alloc&) {
          // Optional Direct storage cannot displace the existing bounded
          // RawSource route. No partially prepared owner is published.
        }
      }
    }
#endif
    allocation_stage = "MO provider/problem";
    RccsdNativeState state;
    state.reference = reference;
    state.reference_energy_change = hf.energy_change;
    state.reference_density_rms = hf.density_rms;
    state.reference_iterations = static_cast<int>(hf.iterations);
    const auto o = reference->nocc;
    state.eps_o.assign(reference->orbital_energies.begin(),
                       reference->orbital_energies.begin() + static_cast<std::ptrdiff_t>(o));
    state.eps_v.assign(reference->orbital_energies.begin() + static_cast<std::ptrdiff_t>(o),
                       reference->orbital_energies.end());
    posthf::ProviderWork provider_work;
    generativeqc_tensor::Metrics provider_metrics{};
    std::unique_ptr<posthf::RawSource> raw_source;
    std::optional<scf::PreparedFockInteractionSourceView> prepared_source;
    const integrals::ElectronInteractionSource* source = nullptr;
    if (prepared_exact) {
      prepared_source.emplace(*prepared_exact);
      source = &*prepared_source;
    } else {
      raw_source = std::make_unique<posthf::RawSource>(system);
      source = raw_source.get();
    }
    const auto build = [&] {
      return build_problem(*source, *reference, solver_options, cuda, execution.device_id(),
                           provider_work, provider_metrics, correlation_auxiliary);
    };
    const auto retire_optional_source = [&] {
      if (!cuda || !cuda_source_cache || !*cuda_source_cache) return false;
      prepared_source.reset();
      cuda_source_cache->reset();
      prepared_exact = nullptr;
      return true;
    };
    try {
      state.problem = build();
    } catch (const std::length_error&) {
      if (!retire_optional_source()) throw;
      raw_source = std::make_unique<posthf::RawSource>(system);
      source = raw_source.get();
      state.problem = build();
    } catch (const std::bad_alloc&) {
      if (!retire_optional_source()) throw;
      raw_source = std::make_unique<posthf::RawSource>(system);
      source = raw_source.get();
      state.problem = build();
    }
    const double problem_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - problem_started).count();
    // Provider planning already charged this source. Subsequent CC/(T)/force
    // stages must additionally retain the prepared owner beside the reference.
    const auto prepared_retained = prepared_exact ? prepared_source->retained_numeric_bytes() : 0;
    if (prepared_exact)
      state.problem.reference_retained_bytes =
          posthf::checked_add(state.problem.reference_retained_bytes, prepared_retained);
    prepared_source.reset();
    raw_source.reset();
    allocation_stage = "CC resident solve";
    const auto solver_started = std::chrono::steady_clock::now();
    try {
      state.solved = cuda ? cc::solve_cuda(state.problem, solver_options, execution.device_id())
                          : cc::solve_cpu(state.problem, solver_options);
    } catch (const std::length_error&) {
      if (!retire_optional_source()) throw;
      state.problem.reference_retained_bytes -= prepared_retained;
      state.solved = cc::solve_cuda(state.problem, solver_options, execution.device_id());
    } catch (const std::bad_alloc&) {
      if (!retire_optional_source()) throw;
      state.problem.reference_retained_bytes -= prepared_retained;
      state.solved = cc::solve_cuda(state.problem, solver_options, execution.device_id());
    }
    const double solver_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solver_started).count();
    state.budget = solver_options.max_bytes;

    auto& diagnostic = state.diagnostic;
    diagnostic.struct_size = sizeof(diagnostic);
    diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
    diagnostic.reference_energy = reference->energy;
    diagnostic.reference_residual = reference->commutator_residual;
    diagnostic.minimum_absolute_denominator = state.problem.minimum_absolute_denominator;
    diagnostic.numeric_capacity_bytes =
        std::max({reference_capacity,
                  posthf::checked_add(
                      reference->numeric_capacity_bytes,
                      correlation_auxiliary ? posthf::source_capacity(*correlation_auxiliary) : 0),
                  source_preparation_peak, state.problem.provider_peak_bytes,
                  state.solved.diagnostic.numeric_capacity_bytes});
    diagnostic.mo_host_staging = cuda ? 1 : 0;
    diagnostic.correlation_owned_device_bytes = std::max<std::size_t>(
        provider_metrics.owned_device_bytes, state.solved.diagnostic.owned_device_bytes);
    diagnostic.correlation_provider_retained_bytes = state.problem.provider_host_bytes;
    diagnostic.mo_transfer_bytes =
        posthf::checked_add(provider_work.h2d_bytes, provider_work.d2h_bytes);
    diagnostic.host_to_device_ms = provider_metrics.input_ms;
    diagnostic.device_to_host_ms = provider_metrics.output_ms;
    diagnostic.transform_library_ms = provider_metrics.library_ms;
    diagnostic.tensor_kernel_ms = provider_metrics.kernel_ms;
    std::copy_n(cc::generated::iteration_equation_hash,
                std::min<std::size_t>(64, std::strlen(cc::generated::iteration_equation_hash)),
                diagnostic.equation_hash);
    diagnostic.ccsd_iterations = state.solved.diagnostic.iterations;
    diagnostic.ccsd_diis_restarts = state.solved.diagnostic.diis_restarts;
    diagnostic.ccsd_correlation_energy = state.solved.correlation_energy;
    diagnostic.ccsd_energy_change = state.solved.diagnostic.energy_change;
    diagnostic.ccsd_singles_residual_max = state.solved.diagnostic.r1_max;
    diagnostic.ccsd_doubles_residual_max = state.solved.diagnostic.r2_max;
    diagnostic.ccsd_replay_singles_residual_max = state.solved.diagnostic.replay_r1_max;
    diagnostic.ccsd_replay_doubles_residual_max = state.solved.diagnostic.replay_r2_max;
    diagnostic.ccsd_setup_h2d_bytes = state.solved.diagnostic.setup_h2d_bytes;
    diagnostic.ccsd_scalar_d2h_bytes = state.solved.diagnostic.scalar_d2h_bytes;
    diagnostic.ccsd_amplitude_d2h_bytes = state.solved.diagnostic.amplitude_d2h_bytes;
    diagnostic.ccsd_synchronizations = state.solved.diagnostic.synchronizations;
    auto& performance = state.performance;
    performance.reference_seconds = reference_seconds;
    performance.problem_seconds = problem_seconds;
    performance.provider_seconds = provider_work.provider_seconds;
    performance.source_seconds = provider_work.source_seconds;
    performance.solver_seconds = solver_seconds;
    performance.iteration_seconds = state.solved.diagnostic.iteration_seconds;
    performance.replay_seconds = state.solved.diagnostic.replay_seconds;
    performance.update_seconds = state.solved.diagnostic.update_seconds;
    performance.diis_seconds = state.solved.diagnostic.diis_seconds;
    performance.source_scans = provider_work.source_scans;
    performance.source_reads = provider_work.source_reads;
    performance.source_values = provider_work.source_values;
    performance.transform_fmas = provider_work.transform_fmas;
    performance.transform_stages = provider_work.transform_stages;
    performance.mo_blocks = provider_work.mo_blocks;
    performance.cuda_transform_calls = provider_work.cuda_transform_calls;
    performance.cuda_batch_calls = provider_work.cuda_batch_calls;
    performance.iteration_graph_calls = state.solved.diagnostic.iteration_graph_calls;
    performance.replay_graph_calls = state.solved.diagnostic.replay_graph_calls;
    performance.update_calls = state.solved.diagnostic.update_calls;
    performance.generated_error_checks = state.solved.diagnostic.generated_error_checks;
    performance.diis_gram_calls = state.solved.diagnostic.diis_gram_calls;
    performance.diis_coefficient_calls = state.solved.diagnostic.diis_coefficient_calls;
    performance.diis_combine_calls = state.solved.diagnostic.diis_combine_calls;
    if (cuda) {
      execution.observe_numeric_peak(runtime::ExecutionMemorySpace::Device,
                                     state.solved.diagnostic.owned_device_bytes);
    } else {
      execution.observe_numeric_peak(runtime::ExecutionMemorySpace::Host,
                                     static_cast<std::size_t>(diagnostic.numeric_capacity_bytes));
    }
    std::copy_n(cc::generated::replay_equation_hash,
                std::min<std::size_t>(64, std::strlen(cc::generated::replay_equation_hash)),
                diagnostic.ccsd_replay_equation_hash);

    state.result.energy = state.solved.total_energy;
    state.result.convergence = {
        state.solved.diagnostic.iterations, state.solved.diagnostic.energy_change,
        std::max(state.solved.diagnostic.r1_max, state.solved.diagnostic.r2_max),
        state.solved.converged()};
    state.result.executed_backend =
        cuda ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    return state;
  } catch (const std::length_error& error) {
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY, error.what());
  } catch (const std::bad_alloc&) {
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      std::string("RCCSD ") + allocation_stage + " allocation failed");
  }
}

class RccsdPrepared final : public PreparedCalculation {
 public:
  RccsdPrepared(Capabilities capabilities, runtime::ExecutionContext execution, core::System system,
                scf::ScfOptions reference_options, cc::SolverOptions solver_options,
                std::size_t reference_capacity)
      : capabilities_(capabilities),
        execution_(std::move(execution)),
        system_(std::move(system)),
        reference_options_(reference_options),
        solver_options_(solver_options),
        reference_capacity_(reference_capacity) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    // Execution updates the same tracker under this owner's mutex.
    std::lock_guard<std::mutex> lock(mutex_);
    return execution_.resources();
  }
  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_;
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_performance_;
  }
  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    last_performance_.reset();
  }

  Result execute(bool compute_forces) override {
    return execute_with_reference_seed(compute_forces, nullptr, nullptr, nullptr);
  }

  Result execute_with_reference_seed(bool compute_forces, const scf::HfWarmState* initial_state,
                                     bool* warm_start_fallback,
                                     std::optional<scf::HfWarmState>* retained_warm_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    last_performance_.reset();
    if (compute_forces && molecule::ao_count(system_) > 12)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "native RCCSD forces are qualified only through 12 AOs");

    const auto warm_capacity =
        warm_reference::reservation_bytes(system_, initial_state, retained_warm_state != nullptr);
    if (warm_capacity >= solver_options_.max_bytes ||
        reference_capacity_ > solver_options_.max_bytes - warm_capacity)
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                        "RCCSD warm state and reference exceed numeric memory budget");
    auto reference_options = reference_options_;
    auto solver_options = solver_options_;
    reference_options.reference_memory_budget_bytes -= warm_capacity;
    solver_options.max_bytes -= warm_capacity;

    if (!execution_.cuda_requested() && !cpu_exact_plan_) {
      const auto backend =
          execution_.cuda_requested() ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
      auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
      // RHF and the borrowed MO source consume values. The relaxed CC force
      // contracts derivatives later with its own admitted, final weights.
      spec.derivative_order = 0;
      const auto strategy =
          scf::resolve_fock_build(spec, backend, reference_options.screening_tolerance);
      cpu_exact_plan_ = std::make_unique<scf::PreparedFockPlan>(
          system_, nullptr, strategy, execution_.cuda_requested() ? execution_.device_id() : -1);
    }
    auto state = execute_rccsd_prepared(execution_, system_, reference_options, solver_options,
                                        reference_capacity_, cpu_exact_plan_.get(),
                                        initial_state ? &initial_state->density : nullptr,
                                        warm_start_fallback, &cpu_exact_plan_);
    state.external_reservation_bytes = warm_capacity;
    state.diagnostic.numeric_capacity_bytes =
        posthf::checked_add(state.diagnostic.numeric_capacity_bytes, warm_capacity);
    last_ = state.diagnostic;
    last_performance_ = state.performance;
    if (state.solved.status == cc::SolveStatus::NumericalFailure)
      throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, state.solved.reason);

    const auto retain_reference = [&] {
      if (retained_warm_state && state.reference)
        *retained_warm_state =
            warm_reference::capture(system_, *state.reference, state.reference_energy_change,
                                    state.reference_density_rms, state.reference_iterations);
    };

    if (!state.solved.converged() || !compute_forces) {
      if (state.solved.converged()) retain_reference();
      return state.result;
    }
    if (!state.reference)
      throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                        "RCCSD force owner lost the converged RHF reference");

    // Reuse the same prepared exact interaction owner for the force Hamiltonian.
    // CUDA sources publish raw ERI tiles directly into the existing bounded MO transform.
    std::unique_ptr<posthf::RawSource> force_raw_source;
    std::optional<scf::PreparedFockInteractionSourceView> force_prepared_source;
    const integrals::ElectronInteractionSource* force_source = nullptr;
    if (cpu_exact_plan_) {
      force_prepared_source.emplace(*cpu_exact_plan_);
      force_source = &*force_prepared_source;
    } else {
      force_raw_source = std::make_unique<posthf::RawSource>(system_);
      force_source = force_raw_source.get();
    }

    constexpr std::size_t kCudaDerivativeStageBudget = 64ULL << 20;
    auto force = execution_.cuda_requested()
                     ? cc::rccsd_force_cuda(system_, *force_source, *state.reference, state.problem,
                                            state.solved, state.eps_o, state.eps_v, state.budget,
                                            execution_.device_id(),
                                            std::min(state.budget, kCudaDerivativeStageBudget))
                     : cc::rccsd_force_cpu(system_, *force_source, *state.reference, state.problem,
                                           state.solved, state.eps_o, state.eps_v, state.budget);
    auto diagnostic = state.diagnostic;
    if (execution_.cuda_requested()) {
      if (!force.lambda.cuda_actions || !force.lambda.owned_device_bytes)
        throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "RCCSD CUDA force did not execute generated Lambda actions on device");
      if (!force.cuda_response_actions || !force.response_owned_device_bytes)
        throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "RCCSD CUDA force replayed Hamiltonian/orbital response on host");
      execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Device,
                                      force.lambda.owned_device_bytes);
      execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Device,
                                      force.response_owned_device_bytes);
      diagnostic.correlation_owned_device_bytes =
          std::max<std::uint64_t>(diagnostic.correlation_owned_device_bytes,
                                  std::max<std::uint64_t>(force.lambda.owned_device_bytes,
                                                          force.response_owned_device_bytes));
    }
    state.result.forces = std::move(force.forces);
    diagnostic.response_iterations = force.orbital_response.iterations;
    diagnostic.response_restarts = force.orbital_response.restarts;
    diagnostic.response_absolute_residual =
        std::max(force.orbital_response.residual_norm, force.independent_orbital_residual);
    diagnostic.response_relative_residual = force.orbital_response.relative_residual;
    diagnostic.response_workspace_bytes = force.orbital_response.workspace_bytes;
    diagnostic.measured_response_workspace_peak_bytes =
        force.orbital_response.measured_workspace_peak_bytes;
    diagnostic.response_workspace_allocation_count =
        force.orbital_response.workspace_allocation_count;
    const auto force_capacity =
        posthf::checked_add(force.numeric_capacity_bytes, state.external_reservation_bytes);
    diagnostic.planned_endpoint_peak_bytes =
        std::max<std::uint64_t>(diagnostic.numeric_capacity_bytes, force_capacity);
    diagnostic.force_provenance_flags = execution_.cuda_requested() ? 0xf : 0x7;
    diagnostic.numeric_capacity_bytes =
        std::max<std::uint64_t>(diagnostic.numeric_capacity_bytes, force_capacity);
    execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Host, force_capacity);
    execution_.observe_workspace_peak(runtime::ExecutionMemorySpace::Host,
                                      force.orbital_response.workspace_bytes);
    std::copy_n(force.response_operator_hash.c_str(),
                std::min<std::size_t>(64, force.response_operator_hash.size()),
                diagnostic.response_operator_hash);
    last_ = diagnostic;
    retain_reference();
    return state.result;
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::System system_;
  scf::ScfOptions reference_options_;
  cc::SolverOptions solver_options_;
  std::size_t reference_capacity_{};
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  std::optional<generativeqc_correlation_diagnostic> last_;
  std::optional<CcPerformanceDiagnostic> last_performance_;
  mutable std::mutex mutex_;
};

class RccsdPreparedBatch final : public PreparedBatch {
 public:
  RccsdPreparedBatch(Capabilities capabilities, core::ContextState& context,
                     std::vector<core::System> systems,
                     const generativeqc_method_descriptor& descriptor, bool warm_starts_enabled)
      : capabilities_(capabilities),
        execution_(context),
        context_(&context),
        systems_(std::move(systems)),
        warm_starts_enabled_(warm_starts_enabled),
        warm_states_(systems_.size()) {
    descriptor_ = descriptor;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_rccsd_calculation(capabilities_, *context_, system, descriptor_));
      owner_coordinates_.push_back(warm_reference::coordinates(system));
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }
  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }
  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("RCCSD batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& result = results[index];
      result.bucket_id = 0;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      result.calculation.executed_backend = execution_.backend();
      try {
        auto target = systems_[index];
        auto target_coordinates = warm_reference::coordinates(target);
        if (!coordinates.empty() && coordinates[index]) {
          if (!warm_reference::valid_coordinates(*coordinates[index], target))
            throw std::invalid_argument("invalid RCCSD batch item coordinates");
          target_coordinates = *coordinates[index];
          warm_reference::set_coordinates(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          auto candidate = prepare_rccsd_calculation(capabilities_, *context_, target, descriptor_);
          owners_[index] = std::move(candidate);
          owner_coordinates_[index] = std::move(target_coordinates);
        }
        const bool has_warm_state = warm_starts_enabled_ && warm_states_[index].has_value();
        result.warm_start_used = has_warm_state;
        bool warm_start_fallback = false;
        std::optional<scf::HfWarmState> next_warm_state;
        auto& owner = static_cast<RccsdPrepared&>(*owners_[index]);
        result.calculation = owner.execute_with_reference_seed(
            compute_forces, has_warm_state ? &*warm_states_[index] : nullptr, &warm_start_fallback,
            warm_starts_enabled_ && warm_start_updates_enabled_ ? &next_warm_state : nullptr);
        result.warm_start_fallback = warm_start_fallback;
        if (next_warm_state) warm_states_[index].swap(next_warm_state);
        result.status = result.calculation.convergence.converged
                            ? GENERATIVEQC_STATUS_SUCCESS
                            : GENERATIVEQC_STATUS_NOT_CONVERGED;
      } catch (...) {
        result.status = item_exception_status();
      }
    }
    return results;
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const override {
    if (index >= owners_.size())
      throw std::invalid_argument("correlation diagnostic batch index is out of range");
    return owners_[index]->correlation_diagnostic();
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic(
      std::size_t index) const override {
    if (index >= owners_.size())
      throw std::invalid_argument("CC performance diagnostic batch index is out of range");
    return owners_[index]->cc_performance_diagnostic();
  }

  void clear_warm_starts() override {
    for (auto& state : warm_states_) state.reset();
  }
  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    return posthf::checked_mul(n, n);
  }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    return warm_states_.at(index);
  }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_starts_enabled_ || states.size() != size())
      throw std::invalid_argument(
          "checkpoint restore requires a matching warm-enabled RCCSD batch");
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index])
        warm_reference::validate_checkpoint(systems_[index], *states[index], "RCCSD");
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index]) warm_states_[index].swap(states[index]);
  }
  void set_warm_start_updates(bool enabled) override { warm_start_updates_enabled_ = enabled; }
  runtime::ExecutionResourceSnapshot execution_resources(
      std::size_t index) const noexcept override {
    if (index >= owners_.size() || !owners_[index]) return {};
    return owners_[index]->execution_resources();
  }
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    return {};
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::ContextState* context_{};
  std::vector<core::System> systems_;
  bool warm_starts_enabled_{};
  bool warm_start_updates_enabled_{true};
  std::vector<std::optional<scf::HfWarmState>> warm_states_;
  generativeqc_method_descriptor descriptor_{};
  std::vector<std::unique_ptr<PreparedCalculation>> owners_;
  std::vector<std::vector<double>> owner_coordinates_;
};

}  // namespace

RccsdNativeState run_rccsd_native_state(
    runtime::ExecutionContext& execution, const core::System& system,
    const generativeqc_method_descriptor& descriptor,
    std::unique_ptr<scf::PreparedFockPlan>* prepared_exact_cache,
    const std::vector<double>* initial_density, bool* warm_start_fallback,
    std::size_t external_reservation_bytes, const core::System* correlation_auxiliary) {
  validate_descriptor(descriptor, execution);
  if (correlation_auxiliary && !execution.cuda_requested())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native molecular DF-CC source requires CUDA");
#if GENERATIVEQC_HAS_CUDA
  if (correlation_auxiliary) {
    std::string detail;
    if (!scf::cuda_execution::cuda_df_value_domain(system, *correlation_auxiliary, detail))
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, detail);
  }
#endif
  const auto budget = correlation_budget(descriptor);
  if (external_reservation_bytes >= budget)
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      "RCCSD warm state exhausts correlation memory budget");
  const auto phase_budget = budget - external_reservation_bytes;
  auto solver_options = cc_options(descriptor, phase_budget);
  auto reference = reference_options(descriptor, phase_budget);
  const auto auxiliary_reference_bytes =
      correlation_auxiliary ? posthf::source_capacity(*correlation_auxiliary) : 0;
  if (auxiliary_reference_bytes >= phase_budget)
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      "DF auxiliary basis exhausts the reference budget");
  reference.reference_memory_budget_bytes -= auxiliary_reference_bytes;
  const auto reference_capacity = posthf::checked_add(
      auxiliary_reference_bytes,
      posthf::rhf_reference_capacity(system, reference.diis_history,
                                     execution.backend() == GENERATIVEQC_BACKEND_CPU_REFERENCE));
  if (reference_capacity > phase_budget)
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      "RCCSD bounded RHF reference exceeds correlation memory budget");
  scf::PreparedFockPlan* prepared_exact = nullptr;
  if (prepared_exact_cache) {
    if (!execution.cuda_requested() && !*prepared_exact_cache) {
      const auto backend =
          execution.cuda_requested() ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
      auto spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
      // Match the value-only reference/source consumer; no coordinate-major
      // derivative tensor is needed or read during energy preparation.
      spec.derivative_order = 0;
      const auto strategy = scf::resolve_fock_build(spec, backend, reference.screening_tolerance);
      *prepared_exact_cache = std::make_unique<scf::PreparedFockPlan>(
          system, nullptr, strategy, execution.cuda_requested() ? execution.device_id() : -1);
    }
    prepared_exact = prepared_exact_cache->get();
  }
  auto state = execute_rccsd_prepared(
      execution, system, reference, solver_options, reference_capacity, prepared_exact,
      initial_density, warm_start_fallback, prepared_exact_cache, correlation_auxiliary);
  state.external_reservation_bytes = external_reservation_bytes;
  state.diagnostic.numeric_capacity_bytes =
      posthf::checked_add(state.diagnostic.numeric_capacity_bytes, external_reservation_bytes);
  return state;
}

generativeqc_status validate_rccsd_system(generativeqc_method, const core::System& system,
                                          std::string& detail) {
  if (!system.ecp_terms.empty()) {
    detail = "RCCSD with ECP is not implemented";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    detail = "RCCSD reference/provider validation supports shells through f";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (system.multiplicity != 1 || system.electron_count <= 0 || system.electron_count % 2) {
    detail = "RCCSD supports real closed-shell all-electron RHF references only";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (static_cast<std::size_t>(system.electron_count / 2) >= molecule::ao_count(system)) {
    detail = "RCCSD requires a nonempty virtual orbital space";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor) {
  runtime::ExecutionContext execution(context);
  validate_descriptor(descriptor, execution);
  const auto budget = correlation_budget(descriptor);
  auto solver_options = cc_options(descriptor, budget);
  auto reference = reference_options(descriptor, budget);
  const auto reference_capacity = posthf::rhf_reference_capacity(
      system, reference.diis_history, execution.backend() == GENERATIVEQC_BACKEND_CPU_REFERENCE);
  if (reference_capacity > budget)
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      "RCCSD bounded RHF reference exceeds correlation memory budget");
  return std::make_unique<RccsdPrepared>(capabilities, std::move(execution), system, reference,
                                         solver_options, reference_capacity);
}

std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities& capabilities,
                                                   core::ContextState& context,
                                                   std::vector<core::System> systems,
                                                   const generativeqc_method_descriptor& descriptor,
                                                   generativeqc_batch_flags flags) {
  constexpr auto supported_flags =
      static_cast<generativeqc_batch_flags>(GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  if ((flags & ~supported_flags) != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD prepared batches support the warm-start compatibility flag only; "
                      "shell/eigensolver profiling is unavailable");
  // The generic Python/C++ batch API enables its warm-start compatibility bit
  // by default.  RCCSD accepts that ABI contract but deliberately starts every
  // solve from the deterministic MP2-like amplitudes: dimensions alone do not
  // establish orbital compatibility, so geometry changes never reuse T1/T2.
  if (systems.empty())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "RCCSD batch is empty");
  const auto nbf = molecule::ao_count(systems.front());
  const auto nocc = static_cast<std::size_t>(systems.front().electron_count / 2);
  for (const auto& system : systems) {
    if (molecule::ao_count(system) != nbf ||
        static_cast<std::size_t>(system.electron_count / 2) != nocc)
      throw MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "RCCSD prepared batch requires one homogeneous (nocc,nvir) shape; split ragged groups");
  }
  return std::make_unique<RccsdPreparedBatch>(capabilities, context, std::move(systems), descriptor,
                                              (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0);
}

}  // namespace generativeqc::methods::detail
