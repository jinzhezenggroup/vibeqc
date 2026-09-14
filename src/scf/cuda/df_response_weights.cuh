#ifndef VIBEQC_SCF_CUDA_DF_RESPONSE_WEIGHTS_CUH
#define VIBEQC_SCF_CUDA_DF_RESPONSE_WEIGHTS_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <functional>
#include <span>

#include "runtime/strided_range.hpp"
#include "scf/cuda_df_gradient.hpp"

namespace vibeqc::scf {

/** Preserve the provider status across the bridge's stream-draining cleanup. */
struct CudaDfResponseBlasFailure {
  cublasStatus_t status;
};

/** Device scratch in doubles, excluding borrowed densities and metric factors.
 * The caller validates size products before using this allocation-free interface.
 * There are four metric matrices, three AO matrices, two auxiliary blocks and
 * two charge vectors per density term. No complete three-center tensor is used.
 */
std::size_t cuda_df_response_workspace_elements(std::size_t n, std::size_t a, std::size_t terms,
                                                std::size_t tile);

/** Produce HF A/M response weights entirely on the caller's CUDA stream.
 * Densities are packed row-major in the same order as the host coefficient
 * descriptors. read_values(P, device_output) generates one public AO slice;
 * consume receives transient device weights and must enqueue its read on the
 * same stream before returning. Both callbacks propagate launch failures.
 *
 * The spectral Frechet map includes retained/discarded subspace motion, using
 * the exact eigensystem and cutoff that defined the plan's metric transform.
 * The owner must reject unresolved rank crossings before calling this routine.
 * blas is the plan-owned host-scalar handle already bound to stream. The
 * serial_metric_dot ablation retains the original fixed-order dot kernel.
 * Return the first CUDA launch error; the caller drains before freeing scratch.
 * A cuBLAS failure throws CudaDfResponseBlasFailure so its allocation status
 * survives cleanup and cannot trigger an unrequested numerical fallback.
 */
cudaError_t contract_cuda_df_response_weights(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, bool serial_metric_dot,
    const std::function<void(std::size_t, double*)>& read_values,
    const std::function<void(unsigned, runtime::StridedRange, std::size_t, const double*)>&
        consume);

}  // namespace vibeqc::scf
#endif
