#ifndef VIBEQC_SCF_CUDA_WEIGHTED_ERI_HPP
#define VIBEQC_SCF_CUDA_WEIGHTED_ERI_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {

/** One bounded primitive contribution, independent of SCF density/state.
 *
 * kind=0 supplies an arbitrary Cartesian component and weights[0]. kind=1
 * supplies a complete canonical psss block with x/y/z weights; angular must
 * then contain {{1,0,0},{0,0,0},{0,0,0},{0,0,0}}. Exponents are positive,
 * positions are in Bohr, and weights already include primitive contraction
 * coefficients, public-basis pullbacks, and any explicit orbit folding.
 *
 * Records can be streamed across calls. Their twelve nuclear derivatives
 * remain in independent shell-slot order until the caller scatters them to
 * physical atoms; two slots on the same atom must be added, never identified
 * before differentiation. Raw diagnostics use kind=0 with unit weight.
 */
struct CudaWeightedEriPrimitive {
  std::uint32_t kind{};
  std::uint32_t output_tile{};
  std::uint32_t angular[4][3]{};
  double exponents[4]{};
  double centers[4][3]{};
  double weights[3]{};
};
static_assert(sizeof(CudaWeightedEriPrimitive) == 208);

/** Weighted integral scalar and its four shell-center derivatives. */
struct CudaWeightedEriResult {
  double value{};
  double center[4][3]{};
};
static_assert(sizeof(CudaWeightedEriResult) == 13 * sizeof(double));

/** Explicit allocations owned by this call; excludes caller records/runtime. */
struct CudaWeightedEriDiagnostic {
  std::size_t host_peak_bytes{};
  std::size_t device_peak_bytes{};
  std::size_t primitive_capacity{};
  std::size_t generated_records{};
  std::size_t reference_records{};
};

/** Contract unscreened records with bounded host/device numeric storage.
 *
 * The budget includes returned host results, device results, and one reused
 * primitive upload buffer. Caller-owned inputs and CUDA context/implicit
 * kernel stacks are excluded. Large input spans are uploaded in chunks; no
 * four-index derivative tensor or HF density is formed. Empty input produces
 * zero output tiles. generated=false evaluates every record through the
 * retained Hermite/Dual3 primitive oracle, preserving the same weight semantics.
 */
vibeqc_status contract_cuda_weighted_eri_primitives(
    int device_id, const CudaWeightedEriPrimitive* records, std::size_t record_count,
    std::size_t tile_count, std::size_t memory_budget_bytes, bool generated,
    std::vector<CudaWeightedEriResult>& output, CudaWeightedEriDiagnostic& diagnostic,
    std::string& detail);

}  // namespace vibeqc::scf
#endif
