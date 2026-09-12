/** Private development bridge for the HF/post-HF interface (schema 1).
 * These symbols are deliberately separate from method registration and the
 * public versioned C API. A source owns a deep copy of the native system.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "posthf/capacity.hpp"
#include "posthf/mp2_energy.hpp"
#include "posthf/raw_source.hpp"
#include "scf/density_fitting.hpp"
#include "scf/mean_field.hpp"
#include "scf/proposal_bridge.hpp"
#include "scf/proposals.hpp"
#include "vibeqc/vibeqc.h"

using vibeqc::posthf::RawSource;
namespace {
/**
 * Own one bounded CUDA DF source and its streamed RHF J/K plan.
 *
 * The public post-HF bridge owns this object across many density-response
 * actions.  Destroying it releases both the transferred integral source and
 * the plan exactly once, including on partial construction failure.
 */
struct PostHfRhfJkPlan {
  std::size_t nbf{};
  std::size_t naux{};
  vibeqc::scf::CudaDensityFittingIntegralSource* source{};
  vibeqc::scf::CudaDensityFittingJkPlan* plan{};

  ~PostHfRhfJkPlan() {
    if (plan) vibeqc::scf::destroy_cuda_density_fitting_jk_plan(plan);
    if (source) vibeqc::scf::destroy_cuda_density_fitting_integral_source(source);
  }
};

template <class F>
int guarded(char* error, std::size_t size, F&& fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const std::exception& ex) {
    if (error && size) std::snprintf(error, size, "%s", ex.what());
    return 1;
  } catch (...) {
    if (error && size) std::snprintf(error, size, "unknown post-HF failure");
    return 1;
  }
}
vibeqc::scf::PhysicalReference supplied_reference(const RawSource& source, const double* arrays,
                                                  std::size_t elements, double energy) {
  const auto n = source.nbf();
  const auto count = vibeqc::posthf::checked_add(
      vibeqc::posthf::checked_mul(5, vibeqc::posthf::checked_mul(n, n)), n);
  if (!arrays || count != elements || !std::isfinite(energy) ||
      source.orbital().multiplicity != 1 || source.orbital().electron_count % 2)
    throw std::invalid_argument("invalid supplied reference");
  vibeqc::scf::PhysicalReference ref;
  ref.nbf = n;
  ref.nocc = source.orbital().electron_count / 2;
  ref.energy = energy;
  const double* cursor = arrays;
  for (auto* m : {&ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients, &ref.density}) {
    m->assign(cursor, cursor + n * n);
    cursor += n * n;
  }
  ref.orbital_energies.assign(cursor, cursor + n);
  vibeqc::scf::validate_physical_reference(ref);
  return ref;
}
}  // namespace
extern "C" {
/** Reuse the SCF metric threshold and square symmetric inverse convention. */
VIBEQC_API int vibeqc_posthf_metric_v1(const double* metric, std::size_t n, double threshold,
                                       double* inverse_root, double* diagnostics, char* error,
                                       std::size_t size) {
  return guarded(error, size, [&] {
    if (!metric || !inverse_root || !diagnostics || !n || n > SIZE_MAX / n)
      throw std::invalid_argument("invalid post-HF metric");
    const auto factor = vibeqc::scf::factor_density_fitting_metric(
        std::vector<double>(metric, metric + n * n), n, threshold);
    std::copy(factor.inverse_square_root.begin(), factor.inverse_square_root.end(), inverse_root);
    diagnostics[0] = factor.effective_rank;
    diagnostics[1] = factor.absolute_threshold;
    diagnostics[2] = factor.condition_number;
  });
}
VIBEQC_API int vibeqc_posthf_source_create_v1(const vibeqc_system* orbital,
                                              const vibeqc_system* auxiliary, void** out,
                                              char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null source output");
    *out = nullptr;
    if (!orbital) throw std::invalid_argument("null orbital system");
    *out = new RawSource(orbital->data, auxiliary ? &auxiliary->data : nullptr);
  });
}
VIBEQC_API void vibeqc_posthf_source_destroy_v1(void* source) {
  delete static_cast<RawSource*>(source);
}
VIBEQC_API int vibeqc_posthf_source_read_v1(void* source, int op, const std::size_t* begin,
                                            const std::size_t* count, double* out,
                                            std::size_t elements, char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !begin || !count) throw std::invalid_argument("null raw source request");
    std::array<std::size_t, 4> b{}, c{};
    std::copy_n(begin, 4, b.begin());
    std::copy_n(count, 4, c.begin());
    static_cast<RawSource*>(source)->read(static_cast<RawSource::Operator>(op), b, c, out,
                                          elements);
  });
}
/** Run the existing RHF implementation, exporting only independently owned
 * density and scalar diagnostics. Snapshot canonicalization is a separate
 * checked operation; an SCF failure never returns a usable density.
 */
VIBEQC_API int vibeqc_posthf_rhf_density_v1(void* source, int backend, int device,
                                            unsigned max_iterations, double tolerance, int df,
                                            double metric_threshold, double* density,
                                            std::size_t elements, double* scalars, char* error,
                                            std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !density || !scalars || (backend != 0 && backend != 1) || (df != 0 && df != 1))
      throw std::invalid_argument("invalid RHF export request");
    const auto& raw = *static_cast<RawSource*>(source);
    if (raw.orbital().multiplicity != 1 || raw.orbital().electron_count % 2)
      throw std::invalid_argument("snapshot export supports closed-shell RHF only");
    if (elements != raw.nbf() * raw.nbf())
      throw std::invalid_argument("density output size mismatch");
    vibeqc::scf::ScfOptions options;
    options.max_iterations = max_iterations;
    options.energy_tolerance = tolerance;
    options.density_tolerance = tolerance;
    options.screening_tolerance = 0;
    options.density_fitting_relative_threshold = metric_threshold;
    vibeqc::scf::ScfResult result;
    if (df) {
      result = backend
                   ? vibeqc::scf::run_rhf_density_fitting_cuda(raw.orbital(), raw.auxiliary(),
                                                               options, device)
                   : vibeqc::scf::run_rhf_density_fitting(raw.orbital(), raw.auxiliary(), options);
    } else {
      result = backend ? vibeqc::scf::run_rhf_cuda(raw.orbital(), options, device)
                       : vibeqc::scf::run_rhf(raw.orbital(), options);
    }
    if (!result.converged || result.density.size() != elements)
      throw std::runtime_error("HF failed or did not converge; no reference exported");
    std::copy(result.density.begin(), result.density.end(), density);
    scalars[0] = result.energy;
    scalars[1] = result.energy_change;
    scalars[2] = result.density_rms;
    scalars[3] = result.iterations;
  });
}
/** Run the existing UHF implementation and export alpha/beta AO densities.
 * As with the RHF bridge, snapshot canonicalization remains a separately
 * checked host operation and a failed SCF cannot yield reusable state.
 */
VIBEQC_API int vibeqc_posthf_uhf_density_v1(void* source, int backend, int device,
                                            unsigned max_iterations, double tolerance, int df,
                                            double metric_threshold, double* density,
                                            std::size_t elements, double* scalars, char* error,
                                            std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !density || !scalars || (backend != 0 && backend != 1) || (df != 0 && df != 1))
      throw std::invalid_argument("invalid UHF export request");
    const auto& raw = *static_cast<RawSource*>(source);
    if (elements != 2 * raw.nbf() * raw.nbf())
      throw std::invalid_argument("UHF alpha/beta density output size mismatch");
    vibeqc::scf::ScfOptions options;
    options.max_iterations = max_iterations;
    options.energy_tolerance = tolerance;
    options.density_tolerance = tolerance;
    options.screening_tolerance = 0;
    options.density_fitting_relative_threshold = metric_threshold;
    vibeqc::scf::ScfResult result;
    if (df) {
      result = backend
                   ? vibeqc::scf::run_uhf_density_fitting_cuda(raw.orbital(), raw.auxiliary(),
                                                               options, device)
                   : vibeqc::scf::run_uhf_density_fitting(raw.orbital(), raw.auxiliary(), options);
    } else {
      result = backend ? vibeqc::scf::run_uhf_cuda(raw.orbital(), options, device)
                       : vibeqc::scf::run_uhf(raw.orbital(), options);
    }
    if (!result.converged || result.density.size() != elements)
      throw std::runtime_error("UHF failed or did not converge; no reference exported");
    std::copy(result.density.begin(), result.density.end(), density);
    scalars[0] = result.energy;
    scalars[1] = result.energy_change;
    scalars[2] = result.density_rms;
    scalars[3] = result.iterations;
  });
}
/** Opt-in small-system NUM01 diagnostic execution using the existing HF source.
 * Unlike a converged post-HF export, this preserves failed-solve scalar records.
 * RHF density has one spin-summed block; UHF has alpha then beta. It neither
 * registers a method nor changes production defaults. Host arrays and the
 * N<=12 audit boundary are explicit; CUDA callers must own a GPU allocation.
 */
int vibeqc_accuracy_hf_probe_v1(void* source, int method, int backend, int device,
                                unsigned max_iterations, unsigned diis_history,
                                double energy_tolerance, double density_tolerance,
                                double screening_tolerance, int df, double metric_threshold,
                                double* density, std::size_t density_elements, double* forces,
                                std::size_t force_elements, double* scalars,
                                std::size_t scalar_elements, char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !density || !forces || !scalars || scalar_elements != 5 ||
        (method != VIBEQC_METHOD_RHF && method != VIBEQC_METHOD_UHF) ||
        (backend != 0 && backend != 1) || (df != 0 && df != 1) || !max_iterations ||
        !std::isfinite(energy_tolerance) || energy_tolerance <= 0 ||
        !std::isfinite(density_tolerance) || density_tolerance <= 0 ||
        !std::isfinite(screening_tolerance) || screening_tolerance < 0 ||
        !std::isfinite(metric_threshold) || metric_threshold <= 0 || metric_threshold >= 1)
      throw std::invalid_argument("invalid HF accuracy probe controls");
    const auto& raw = *static_cast<RawSource*>(source);
    const std::size_t spins = method == VIBEQC_METHOD_RHF ? 1 : 2;
    if (!raw.nbf() || raw.nbf() > 12 || raw.naux() > 24 ||
        density_elements != spins * raw.nbf() * raw.nbf() ||
        force_elements != 3 * raw.orbital().atoms.size())
      throw std::invalid_argument("HF accuracy probe supports at most 12 orbital/24 auxiliary AOs");
    if (method == VIBEQC_METHOD_RHF &&
        (raw.orbital().multiplicity != 1 || raw.orbital().electron_count % 2))
      throw std::invalid_argument("RHF accuracy probe requires a closed-shell source");
    std::fill_n(density, density_elements, std::numeric_limits<double>::quiet_NaN());
    std::fill_n(forces, force_elements, std::numeric_limits<double>::quiet_NaN());
    vibeqc::scf::ScfOptions options;
    options.max_iterations = max_iterations;
    options.diis_history = diis_history;
    options.energy_tolerance = energy_tolerance;
    options.density_tolerance = density_tolerance;
    options.screening_tolerance = screening_tolerance;
    options.density_fitting_relative_threshold = metric_threshold;
    vibeqc::scf::ScfResult result;
    if (method == VIBEQC_METHOD_RHF) {
      if (df) {
        result =
            backend ? vibeqc::scf::run_rhf_density_fitting_cuda(raw.orbital(), raw.auxiliary(),
                                                                options, device)
                    : vibeqc::scf::run_rhf_density_fitting(raw.orbital(), raw.auxiliary(), options);
      } else {
        result = backend ? vibeqc::scf::run_rhf_cuda(raw.orbital(), options, device)
                         : vibeqc::scf::run_rhf(raw.orbital(), options);
      }
    } else {
      if (df) {
        result =
            backend ? vibeqc::scf::run_uhf_density_fitting_cuda(raw.orbital(), raw.auxiliary(),
                                                                options, device)
                    : vibeqc::scf::run_uhf_density_fitting(raw.orbital(), raw.auxiliary(), options);
      } else {
        result = backend ? vibeqc::scf::run_uhf_cuda(raw.orbital(), options, device)
                         : vibeqc::scf::run_uhf(raw.orbital(), options);
      }
    }
    scalars[0] = result.energy;
    scalars[1] = result.energy_change;
    scalars[2] = result.density_rms;
    scalars[3] = result.iterations;
    scalars[4] = result.converged ? 1 : 0;
    if (!result.converged) return;
    if (result.density.size() != density_elements || result.forces.size() != force_elements)
      throw std::runtime_error("HF probe returned inconsistent scientific-state dimensions");
    std::copy(result.density.begin(), result.density.end(), density);
    std::copy(result.forces.begin(), result.forces.end(), forces);
  });
}

/** Complete CPU reference solve with per-call hooks. The explicit small-system
 * boundary matches NUM01's diagnostic bridge. Production device-resident loops
 * cannot silently enter this callback path. Scalars preserve nonconvergence;
 * final densities are reusable only after the caller checks convergence and
 * re-evaluates the target operator at the destination state.
 */
int vibeqc_scf_solve_v1(void* source, int method, int multiplicity, int df, unsigned max_iterations,
                        unsigned diis_history, double energy_tolerance, double density_tolerance,
                        double metric_threshold, const double* initial_density,
                        ScfProposeV1 propose, ScfObserveV1 observe, double* density,
                        std::size_t elements, double* forces, std::size_t force_elements,
                        double* scalars, std::size_t scalar_elements, char* error,
                        std::size_t size) {
  return guarded(error, size, [&] {
    using namespace vibeqc::scf;
    if (!source || !density || !forces || !scalars || scalar_elements != 6 ||
        (method != VIBEQC_METHOD_RHF && method != VIBEQC_METHOD_UHF) || (df != 0 && df != 1) ||
        !max_iterations || max_iterations > 10000 || diis_history > 100 ||
        !std::isfinite(energy_tolerance) || energy_tolerance <= 0 ||
        !std::isfinite(density_tolerance) || density_tolerance <= 0 ||
        !std::isfinite(metric_threshold) || metric_threshold <= 0 || metric_threshold >= 1)
      throw std::invalid_argument("invalid SCF proposal bridge controls");
    const auto& raw = *static_cast<RawSource*>(source);
    auto system = raw.orbital();
    if (multiplicity < 1 || multiplicity - 1 > system.electron_count ||
        (system.electron_count - (multiplicity - 1)) % 2 ||
        (method == VIBEQC_METHOD_RHF && multiplicity != 1))
      throw std::invalid_argument("inconsistent SCF spin populations");
    system.multiplicity = multiplicity;
    const std::size_t spins = method == VIBEQC_METHOD_RHF ? 1 : 2;
    if (!raw.nbf() || raw.nbf() > 12 || raw.naux() > 24 ||
        elements != spins * raw.nbf() * raw.nbf() || force_elements != 3 * system.atoms.size())
      throw std::invalid_argument(
          "SCF proposal bridge supports at most 12 orbital/24 auxiliary AOs");
    std::fill_n(density, elements, std::numeric_limits<double>::quiet_NaN());
    std::fill_n(forces, force_elements, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> initial;
    if (initial_density) initial.assign(initial_density, initial_density + elements);
    ScfOptions options;
    options.max_iterations = max_iterations;
    options.diis_history = diis_history;
    options.energy_tolerance = energy_tolerance;
    options.density_tolerance = density_tolerance;
    options.screening_tolerance = 0;
    options.density_fitting_relative_threshold = metric_threshold;
    ScfHooks hooks;
    const auto view = [](const ScfSnapshot& s) {
      return ScfSnapshotViewV1{s.generation,        s.iteration,      s.nbf,
                               s.electrons.size(),  s.fock_builds,    s.electrons.data(),
                               s.occupation_weight, s.energy,         s.residual_rms,
                               s.density.data(),    s.fock.data(),    s.residual.data(),
                               s.overlap.data(),    s.baseline.data()};
    };
    if (propose)
      hooks.propose = [&](const ScfSnapshot& s) {
        const auto v = view(s);
        ScfProposal p;
        p.density.resize(elements, std::numeric_limits<double>::quiet_NaN());
        p.representation = static_cast<ProposalRepresentation>(
            propose(&v, p.density.data(), &p.generation, &p.iteration));
        return p;
      };
    if (observe)
      hooks.observe = [&](const ScfSnapshot& s, const ProposalDecision& d) {
        const auto v = view(s);
        const ScfDecisionViewV1 dv{static_cast<int>(d.action),
                                   d.reason.c_str(),
                                   d.trials,
                                   d.fraction,
                                   d.energy,
                                   d.residual_rms,
                                   d.inference_seconds,
                                   d.validation_seconds,
                                   d.operator_seconds};
        observe(&v, &dv);
      };
    options.hooks = propose || observe ? &hooks : nullptr;
    options.strict_initial_density = true;
    const auto* seed = initial_density ? &initial : nullptr;
    ScfResult result;
    if (method == VIBEQC_METHOD_RHF)
      result = df ? run_rhf_density_fitting(system, raw.auxiliary(), options, seed)
                  : run_rhf(system, options, seed);
    else
      result = df ? run_uhf_density_fitting(system, raw.auxiliary(), options, seed)
                  : run_uhf(system, options, seed);
    scalars[0] = result.energy;
    scalars[1] = result.energy_change;
    scalars[2] = result.density_rms;
    scalars[3] = result.iterations;
    scalars[4] = result.converged ? 1 : 0;
    scalars[5] = result.fock_builds;
    if (result.density.size() == elements)
      std::copy(result.density.begin(), result.density.end(), density);
    if (result.forces.size() == force_elements)
      std::copy(result.forces.begin(), result.forces.end(), forces);
  });
}
/**
 * Prepare a reusable streamed CUDA DF RHF J/K plan.
 *
 * This is a development bridge for response operators.  The source must own
 * an auxiliary basis; a missing auxiliary basis fails closed instead of
 * silently switching to a different Hamiltonian.
 */
int vibeqc_posthf_rhf_jk_plan_create_v1(void* source, int device, double threshold, void** out,
                                        double* diagnostics, char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !out || !diagnostics)
      throw std::invalid_argument("invalid RHF J/K plan request");
    *out = nullptr;
    if (!(threshold > 0.0) || !(threshold < 1.0) || !std::isfinite(threshold))
      throw std::invalid_argument("RHF J/K metric threshold must be in (0,1)");
    const auto& raw = *static_cast<RawSource*>(source);
    if (raw.naux() == 0U) throw std::runtime_error("CUDA DF response requires an auxiliary basis");
    auto prepared = std::make_unique<PostHfRhfJkPlan>();
    std::vector<vibeqc::core::System> orbital_systems{raw.orbital()};
    std::vector<vibeqc::core::System> auxiliary_systems{raw.auxiliary()};
    std::vector<double> metrics;
    std::size_t nbf = 0U;
    std::size_t naux = 0U;
    std::string detail;
    vibeqc_status status = vibeqc::scf::create_cuda_density_fitting_integral_source(
        device, orbital_systems, auxiliary_systems, &prepared->source, metrics, nbf, naux, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "CUDA DF source preparation failed" : detail);
    std::vector<vibeqc::scf::CudaDensityFittingMetricDiagnostic> plan_diagnostics;
    status = vibeqc::scf::create_cuda_density_fitting_jk_plan_from_source(
        device, &prepared->source, 1U, nbf, naux, metrics, threshold, 0U, 0U, &prepared->plan,
        plan_diagnostics, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "CUDA DF J/K plan preparation failed" : detail);
    prepared->nbf = nbf;
    prepared->naux = naux;
    diagnostics[0] = static_cast<double>(nbf);
    diagnostics[1] = static_cast<double>(naux);
    diagnostics[2] = plan_diagnostics.empty()
                         ? 0.0
                         : static_cast<double>(plan_diagnostics.front().device_resident_bytes);
    diagnostics[3] = plan_diagnostics.empty()
                         ? 0.0
                         : static_cast<double>(plan_diagnostics.front().peak_device_bytes);
    diagnostics[4] = plan_diagnostics.empty()
                         ? 0.0
                         : static_cast<double>(plan_diagnostics.front().host_resident_bytes);
    diagnostics[5] = threshold;
    *out = prepared.release();
  });
}

int vibeqc_posthf_rhf_jk_plan_execute_v1(void* plan, const double* density, std::size_t elements,
                                         double* coulomb, double* exchange, char* error,
                                         std::size_t size) {
  return guarded(error, size, [&] {
    auto* prepared = static_cast<PostHfRhfJkPlan*>(plan);
    if (!prepared || !prepared->plan || !density || !coulomb || !exchange ||
        elements != prepared->nbf * prepared->nbf)
      throw std::invalid_argument("invalid RHF J/K plan execution request");
    std::vector<double> density_vector(density, density + elements);
    std::vector<double> coulomb_vector;
    std::vector<double> exchange_vector;
    std::string detail;
    const vibeqc_status status = vibeqc::scf::execute_cuda_density_fitting_rhf_jk(
        prepared->plan, density_vector, coulomb_vector, exchange_vector, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "CUDA DF J/K execution failed" : detail);
    if (coulomb_vector.size() != elements || exchange_vector.size() != elements)
      throw std::runtime_error("CUDA DF J/K output size mismatch");
    std::copy(coulomb_vector.begin(), coulomb_vector.end(), coulomb);
    std::copy(exchange_vector.begin(), exchange_vector.end(), exchange);
  });
}

void vibeqc_posthf_rhf_jk_plan_destroy_v1(void* plan) {
  delete static_cast<PostHfRhfJkPlan*>(plan);
}

/** Private owned reference export for native consumer validation. Buffer order
 * is S,h,F,C,D
 * (each row-major n*n), then epsilon[n]. It uses the same
 * reference-only HF driver as the method
 * adapter, never the 12-AO exporter. */
VIBEQC_API int vibeqc_posthf_reference_v1(void* source, int backend, int device,
                                          unsigned iterations, double tolerance, std::size_t budget,
                                          double* arrays, std::size_t elements, double* scalars,
                                          char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !arrays || !scalars || (backend != 0 && backend != 1) || !iterations ||
        !std::isfinite(tolerance) || tolerance <= 0)
      throw std::invalid_argument("invalid bounded reference request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto n = raw.nbf();
    const auto expected = vibeqc::posthf::checked_add(
        vibeqc::posthf::checked_mul(5, vibeqc::posthf::checked_mul(n, n)), n);
    if (elements != expected) throw std::invalid_argument("reference export size mismatch");
    vibeqc::scf::ScfOptions options;
    options.max_iterations = iterations;
    options.energy_tolerance = tolerance;
    options.density_tolerance = tolerance;
    options.screening_tolerance = 0;
    options.export_physical_reference = true;
    options.reference_memory_budget_bytes = budget;
    const auto result = backend ? vibeqc::scf::run_rhf_cuda(raw.orbital(), options, device)
                                : vibeqc::scf::run_rhf(raw.orbital(), options);
    if (!result.converged || !result.reference)
      throw std::runtime_error("HF failed or bounded reference export is unsupported by backend");
    const auto& r = *result.reference;
    auto* cursor = arrays;
    for (const auto* a :
         {&r.overlap, &r.hcore, &r.fock, &r.coefficients, &r.density, &r.orbital_energies})
      cursor = std::copy(a->begin(), a->end(), cursor);
    scalars[0] = r.energy;
    scalars[1] = result.iterations;
    scalars[2] = r.commutator_residual;
    scalars[3] = r.canonical_density_drift;
    scalars[4] = r.eigen_residual;
    scalars[5] = static_cast<double>(r.numeric_capacity_bytes);
  });
}

/** Private identical-orbital validation entry to the production native consumer.
 * It is not a
 * public MP2 method or a substitute for the HF-to-MP2 route. */
VIBEQC_API int vibeqc_posthf_mp2_energy_v1(void* source, int backend, int device,
                                           const double* arrays, std::size_t elements,
                                           double hf_energy, std::size_t budget, double threshold,
                                           unsigned tile, double* out, char* error,
                                           std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !out || (backend != 0 && backend != 1))
      throw std::invalid_argument("invalid MP2 validation request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto ref = supplied_reference(raw, arrays, elements, hf_energy);
    const auto result =
        vibeqc::mp2::conventional_energy(ref, raw, budget, threshold, tile, backend == 1, device);
    out[0] = result.opposite_spin;
    out[1] = result.same_spin;
    out[2] = result.minimum_denominator;
    out[3] = result.numeric_capacity_bytes;
    out[4] = result.tiles;
  });
}
/** Explicit CG10 slot order for direct native CUDA/CPU layout verification. */
VIBEQC_API int vibeqc_posthf_mo_block_v1(void* source, int backend, int device,
                                         const double* arrays, std::size_t elements,
                                         double hf_energy, const std::size_t* shape,
                                         const std::size_t* slots, std::size_t budget, double* out,
                                         std::size_t output_elements, char* error,
                                         std::size_t size) {
  return guarded(error, size, [&] {
    if (!source || !out || !shape || !slots || (backend != 0 && backend != 1))
      throw std::invalid_argument("invalid MO validation request");
    const auto& raw = *static_cast<RawSource*>(source);
    const auto ref = supplied_reference(raw, arrays, elements, hf_energy);
    vibeqc::posthf::MOSlots request;
    std::size_t count = 1;
    for (unsigned k = 0; k < 4; ++k) {
      if (!shape[k] || shape[k] > ref.nbf)
        throw std::invalid_argument("invalid MO validation shape");
      request[k].assign(slots, slots + shape[k]);
      slots += shape[k];
      count = vibeqc::posthf::checked_mul(count, shape[k]);
    }
    if (count != output_elements) throw std::invalid_argument("MO validation output size mismatch");
    const vibeqc::posthf::NativeBlockProvider provider(raw, ref, budget);
    const auto values = provider.get(request, backend == 1, device);
    std::copy(values.begin(), values.end(), out);
  });
}
}
