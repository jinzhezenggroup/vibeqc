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
#include "posthf/raw_source.hpp"
#include "scf/density_fitting.hpp"
#include "scf/mean_field.hpp"

using vibeqc::posthf::RawSource;
namespace {
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
}  // namespace
extern "C" {
/** Reuse the SCF metric threshold and square symmetric inverse convention. */
int vibeqc_posthf_metric_v1(const double* metric, std::size_t n, double threshold,
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
int vibeqc_posthf_source_create_v1(const vibeqc_system* orbital, const vibeqc_system* auxiliary,
                                   void** out, char* error, std::size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null source output");
    *out = nullptr;
    if (!orbital) throw std::invalid_argument("null orbital system");
    *out = new RawSource(orbital->data, auxiliary ? &auxiliary->data : nullptr);
  });
}
void vibeqc_posthf_source_destroy_v1(void* source) { delete static_cast<RawSource*>(source); }
int vibeqc_posthf_source_read_v1(void* source, int op, const std::size_t* begin,
                                 const std::size_t* count, double* out, std::size_t elements,
                                 char* error, std::size_t size) {
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
int vibeqc_posthf_rhf_density_v1(void* source, int backend, int device, unsigned max_iterations,
                                 double tolerance, int df, double metric_threshold, double* density,
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
}
