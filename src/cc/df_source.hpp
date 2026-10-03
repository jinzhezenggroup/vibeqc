#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/types.hpp"
#include "hf/reference.hpp"

namespace generativeqc::cc {

/** Supplied-reference DF integrals for the native CC solver's host-input contract.
 * Raw integrals, metric factorization, orbital transforms and retained block
 * contractions execute on CUDA. Host vectors are the explicit solver boundary;
 * they are never populated from CPU/PySCF/reference-oracle contractions.
 * No ovvv/vvvv block is materialized. This does not refit the RHF reference.
 */
struct DFSourceResult {
  std::size_t nocc{}, nvir{}, naux{};
  std::vector<double> boo, bov, bvv, ovov, ovvo, oovv, ovoo, oooo;
  std::size_t numeric_capacity_bytes{}, host_output_bytes{};
  // Conservative device reservation, including the shared metric owner's lazy
  // SCF allowance. This is an admission bound, not a measured allocation peak.
  std::size_t device_capacity_bytes{};
  std::size_t source_rows{}, source_values{}, transform_gemms{}, block_gemms{};
  std::size_t transform_summands{}, block_summands{};
  // These counters cover the explicit factor/CC-block boundary. The shared
  // metric owner separately stages its metric and scalar factorization audits.
  std::size_t coefficient_h2d_bytes{}, factor_block_d2h_bytes{};
  std::size_t metric_staging_bytes{}, metric_rank{};
  double metric_absolute_threshold{}, metric_condition_number{};
  double source_seconds{}, metric_seconds{}, transform_seconds{}, block_seconds{}, total_seconds{};
};

/** Build once from normalized orbital/auxiliary basis and a physical MO frame.
 * caller_bytes covers external live state in addition to the supplied reference
 * and systems. The complete source operation must fit maximum_bytes. Failures
 * return no partial result; there is no CPU numerical fallback. CUDA builds
 * provide this internal entry point while public DF-CC forces remain gated.
 */
DFSourceResult build_df_source_cuda(const core::System& orbital, const core::System& auxiliary,
                                    const hf::PhysicalReference& reference,
                                    std::size_t maximum_bytes, double metric_relative_threshold,
                                    int device, std::size_t caller_bytes = 0);

/** Borrowed physical factors and fixed-orbital Lagrangian cotangents.
 * All factors are Q-major. Boo/Bvv must be symmetric spatial-orbital pairs.
 * bar_bov/bar_bvv are virtual-only contributions, composed exactly once with
 * the five retained blocks. Arbitrary finite seeds are allowed: this map is
 * not itself a primal/Lambda convergence certification or a nuclear force.
 */
struct DFFactorResponseView {
  std::span<const double> boo, bov, bvv;
  std::span<const double> bar_ovov, bar_ovvo, bar_oovv, bar_ovoo, bar_oooo;
  std::span<const double> bar_bov, bar_bvv;
};

struct DFFactorResponseResult {
  // Dense Frobenius cotangents. Bov includes both ov/vo terms: a full symmetric
  // BMO embedding assigns half to ov and half to transposed vo.
  std::vector<double> boo, bov, bvv;
  std::size_t numeric_capacity_bytes{}, owned_device_bytes{}, h2d_bytes{}, d2h_bytes{};
  std::size_t contraction_terms{}, generated_kernels{};
};

/** Compose generated retained Gram and virtual factor derivatives on CUDA.
 * One stream owns every input/arena and drains before any host publication or
 * exception releases a host destination. Admission counts all borrowed view
 * values, device storage and detached outputs. caller_bytes additionally
 * charges all other live owners and capacity beyond the supplied spans.
 */
DFFactorResponseResult pullback_df_factors_cuda(std::size_t nocc, std::size_t nvir,
                                                std::size_t naux, DFFactorResponseView inputs,
                                                std::size_t maximum_bytes, int device,
                                                std::size_t caller_bytes = 0);

}  // namespace generativeqc::cc
