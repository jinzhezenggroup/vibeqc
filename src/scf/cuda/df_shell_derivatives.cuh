#pragma once

#include "scf/cuda/df_derivatives.cuh"

namespace vibeqc::scf {
/** Compact angular-class shell lists; storage grows with shells, never triples.
 * AO offsets are in the original public Cartesian/spherical order. A host
 * owner may narrow each class list to the shells intersecting an active panel.
 */
struct DfShellBasisView {
  DfDerivativeBasisView basis;
  const std::int32_t* shell_ids{};
  const std::int64_t* ao_offsets{};
  std::size_t begin[4]{}, count[4]{};
};

/** Ordered dense reference, folded dense shell pairs, or folded packed AOs.
 * packed stores W_ii once and W_ij+W_ji at i*(i+1)/2+j (i>j), auxiliary-major.
 * It therefore includes the off-diagonal multiplicity before AO expansion.
 */
enum class DfDerivativePairs { full, symmetric, packed };

/** Launch generated weighted derivatives on a complete auxiliary-major AO panel.
 * All pointers belong to the caller and remain live through its stream drain.
 * Optional device counters record visited/nonzero shell triples, nonzero
 * public weights, executed primitive products, Cartesian component work, and
 * public weight loads (including zeros), in that order.
 * They are diagnostic atomics and must be disabled in promotion timings.
 * full_domain selects all 64 s/p/d/f classes; false retains the original seven
 * non-SSS s/p classes for comparison. Variants 0/1/2 select the compiler's warp,
 * packed-warp and compact-subgroup schedules, with identical mathematics.
 */
cudaError_t launch_df_shell_derivative_panel(DfShellBasisView orbital, DfShellBasisView auxiliary,
                                             const double* positions, std::size_t auxiliary_begin,
                                             std::size_t auxiliary_count, const double* weights,
                                             double* gradient, unsigned long long* counters,
                                             cudaStream_t stream, bool full_domain = false,
                                             unsigned variant = 0,
                                             DfDerivativePairs pairs = DfDerivativePairs::full);
}  // namespace vibeqc::scf
