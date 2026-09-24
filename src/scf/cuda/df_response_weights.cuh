#ifndef VIBEQC_SCF_CUDA_DF_RESPONSE_WEIGHTS_CUH
#define VIBEQC_SCF_CUDA_DF_RESPONSE_WEIGHTS_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <span>

#include "runtime/strided_range.hpp"
#include "scf/cuda_df_gradient.hpp"

namespace vibeqc::scf {

/** Preserve the provider status across the bridge's stream-draining cleanup. */
struct CudaDfResponseBlasFailure {
  cublasStatus_t status;
};

/** One transient response panel consumed on the producer stream.
 * Kinds 0/1/2 retain dense three-center, metric and packed-weight semantics.
 * A packed kind-2 panel may additionally carry an exact occupied-response
 * factorization. In that case weights contains the non-exchange contribution
 * (currently Coulomb), while coefficients/projected/rank/coefficient describe
 * W_ex = coefficient * C U C^T. projected is auxiliary-major: each public
 * auxiliary column owns one column-major nbf-by-rank C*U block. Pointers stay
 * valid until the next producer operation on the same stream.
 */
struct CudaDfResponsePanel {
  unsigned kind{};
  runtime::StridedRange range{};
  std::size_t count{};
  const double* weights{};
  const double* coefficients{};
  const double* projected{};
  std::size_t rank{};
  double coefficient{};

  [[nodiscard]] bool factorized_exchange() const noexcept {
    return kind == 2 && coefficients && projected && rank;
  }
};

using CudaDfResponseConsumer = std::function<void(const CudaDfResponsePanel&)>;

/** Device scratch in doubles, excluding borrowed densities and metric factors.
 * The caller validates size products before using this allocation-free interface.
 * There are four metric matrices, three AO matrices, two auxiliary blocks and
 * two charge vectors per density term. The caller bounds both panels; a tile
 * equal to naux covers the complete tensor and is charged at its literal size.
 */
std::size_t cuda_df_response_workspace_elements(std::size_t n, std::size_t a, std::size_t terms,
                                                std::size_t tile);

/** Produce HF A/M response weights entirely on the caller's CUDA stream.
 * Densities are packed row-major in the same order as the host coefficient
 * descriptors. read_values(P, device_output) generates one public AO slice;
 * consume receives transient device weights and must enqueue its read on the
 * same stream before returning. Both callbacks propagate launch failures.
 *
 * The rank-deficient spectral Frechet map includes retained/discarded subspace
 * motion, using the exact eigensystem and cutoff of the forward transform.
 * Owner-verified full rank permits inverse-applied factors before quadratic
 * products when complete buffers or the fitted-panel callback are available.
 * The owner must reject unresolved rank crossings before calling this routine.
 * blas is the plan-owned host-scalar handle already bound to stream. The
 * serial_metric_dot ablation retains the original fixed-order dot kernel.
 * blas_products selects parallel charge GEMV and density GEMM on that same
 * handle. The scalar route retains its original ordering for comparison; both
 * routes form the complete metric reverse map and moving auxiliary response.
 * Return the first CUDA launch error; the caller drains before freeing scratch.
 * A cuBLAS failure throws CudaDfResponseBlasFailure so its allocation status
 * survives cleanup and cannot trigger an unrequested numerical fallback.
 *
 * packed_pairs requires admitted occupied factors. It emits kind=2 panels
 * with folded lower AO pairs: W_ii, W_ij+W_ji for i>j. Kinds 0/1 retain the
 * dense three-center/metric contracts. Optional public auxiliary shell offsets
 * trim packed panels to whole shells when the cap permits; a smaller explicit
 * cap still splits a shell. Offsets and callbacks outlive this synchronous call.
 * packed_block_rows bounds each rectangular AO expansion block (clipped to n);
 * larger blocks trade extra diagonal-block arithmetic for fewer BLAS launches.
 * Optional read_fitted(begin,count,output) supplies inverse-applied dense
 * auxiliary-major panels from an owner-validated resident forward tensor or
 * regenerated raw AO-pair blocks. It uses the same stream and permits strict
 * full-rank response with a smaller owned tile. Its callback may reuse the
 * otherwise dead workspace[2*a*a:4*a*a] metric scratch; bar_M lives separately
 * at workspace[a*a:2*a*a]. Count all repeated projections and raw generation.
 * single_fitted_tensor instead reserves one complete owned A/B tensor and one
 * bounded weight panel: add (a-tile)*n*n elements to the usual workspace size.
 * It applies the inverse in place through disjoint AO-pair batches, generating
 * raw values once. This option requires full rank, no borrow and no read_fitted.
 * streamed_occupied instead describes bridge-owned projected-factor scratch
 * plus owner-validated canonical coefficients. Raw slices feed charges and all
 * spin projections in one pass. Its raw buffer holds one AO matrix, staging
 * holds sum(rank^2)*a, and exchange holds max(max(rank^2)*a,tile*n*n).
 * It requires full rank and excludes borrowed tensors and fitted-panel reads.
 * Once the bridge-owned capacity gate admits that owner, packed_pairs uses the
 * same shell offsets/block bound as resident occupied response; factorized
 * exchange additionally requires the one-term packed contract.
 */
cudaError_t contract_cuda_df_response_weights(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, bool serial_metric_dot, bool blas_products,
    const std::function<void(std::size_t, double*)>& read_values,
    const CudaDfResponseConsumer& consume, const CudaDfResponseBuffers* borrowed = nullptr,
    std::span<const double> raw_host = {}, bool packed_pairs = false,
    std::span<const std::int64_t> auxiliary_shell_offsets = {}, std::size_t packed_block_rows = 256,
    const std::function<void(std::size_t, std::size_t, double*)>& read_fitted = {},
    bool single_fitted_tensor = false, const CudaDfResponseBuffers* streamed_occupied = nullptr,
    bool factorized_exchange = false);

}  // namespace vibeqc::scf
#endif
