#ifndef VIBEQC_SCF_DF_RESPONSE_WEIGHTS_HPP
#define VIBEQC_SCF_DF_RESPONSE_WEIGHTS_HPP

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "runtime/strided_range.hpp"

namespace vibeqc::scf {

/** A symmetric density contribution to .5*cJ*rho^T M+ rho - cK*Q:M+.
 * Coefficients may have either sign. Signed Fock weights map to cJ and
 * cK=-0.5*FockExchange; absent terms use zero and skip their response work. */
struct DensityFittingDensityResponse {
  std::span<const double> density;
  double coulomb_coefficient{}, exchange_coefficient{};
};

/** Numeric host scratch, excluding borrowed inputs and callback-owned storage. */
struct DensityFittingResponseWeightResources {
  std::size_t host_peak_bytes{}, auxiliary_tile{}, value_slices{}, weight_tiles{};
};

/** Derive bounded external A/M weights from one or more HF density terms.
 *
 * read_values(P, output) supplies the full row-major AO matrix A[:,:,P].
 * consume(kind, range, weights) consumes a transient weight span;
 * it must finish reading that span before returning. kind=0 is full dense
 * A[mu,nu,P], kind=1 is full dense M[P,Q]. Element k maps to range.index(k).
 * Each bounded auxiliary block is submitted together so the consumer can
 * expose parallelism across both AO pairs and auxiliary functions.
 * These are positive energy derivatives with unit dense multiplicity.
 *
 * Only a bounded auxiliary block of AO matrices is retained. The metric
 * reverse chain uses the spectral Frechet map at the requested cutoff;
 * an unresolved rank crossing throws, and callbacks must propagate failures.
 * maximum_bytes bounds this adapter's numeric host scratch independently
 * of the value/derivative consumer. A zero auxiliary cap chooses automatically.
 */
DensityFittingResponseWeightResources contract_density_fitting_response_weights(
    std::size_t nbf, std::size_t naux, const std::vector<double>& metric,
    const std::vector<double>& inverse, std::span<const DensityFittingDensityResponse> terms,
    double relative_threshold, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    const std::function<void(std::size_t, std::span<double>)>& read_values,
    const std::function<void(unsigned, runtime::StridedRange, std::span<const double>)>& consume);

}  // namespace vibeqc::scf
#endif
