#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>

namespace generativeqc::posthf {

/** Borrowed immutable device inputs for B[p,q,Q] = C[mu,p] C[nu,q] A[mu,nu,P] W[P,Q].
 * All arrays are FP64, row-major and auxiliary contiguous. The caller owns
 * their lifetime through the synchronous response call. W is an independent
 * input here; the metric owner subsequently applies its fixed-rank rule.
 */
struct CudaDFMOSourceResponseView {
  std::size_t nbf{}, naux{};
  const double *coefficients{}, *inverse_root{}, *bar_whitened{};
};

using CudaDFSourceRead = std::function<void(std::size_t, double*, cudaStream_t)>;
using CudaDFSourceConsume = std::function<void(std::size_t, const double*, cudaStream_t)>;
using CudaDFSourceFinish = std::function<void(const double*, const double*, cudaStream_t)>;

struct CudaDFMOSourceResponseDiagnostic {
  std::size_t source_rows{}, source_values{}, output_rows{}, output_values{};
  std::size_t gemms{}, contraction_summands{}, owned_device_bytes{}, numeric_capacity_bytes{};
  std::size_t scalar_d2h_bytes{};
};

/** Execute the compiler-owned source VJP with bounded CUDA row callbacks.
 * Read(mu,row,stream) must supply the SAME raw [nu,P] row on both passes.
 * Consume receives the full unit-weight raw-A cotangent, without assuming
 * symmetric input coordinates. Finish receives bar_C and bar_W; a physical
 * consumer applies source pair symmetry and the shared metric rule as needed.
 *
 * Every callback enqueues on stream and may only borrow its row until the next
 * callback. Callback outputs remain provisional until this function succeeds;
 * exceptions and nonfinite arithmetic never certify partial derivatives.
 * The supplied BLAS handle must already use this stream and host scalar mode.
 * The call drains the stream on success and exceptions before freeing scratch.
 * Numeric admission includes these borrowed device inputs and all scratch;
 * caller_bytes additionally covers source/callback storage and other live owners.
 */
CudaDFMOSourceResponseDiagnostic pullback_df_mo_source_cuda(
    CudaDFMOSourceResponseView view, int device, cudaStream_t stream, cublasHandle_t blas,
    const CudaDFSourceRead& read, const CudaDFSourceConsume& consume,
    const CudaDFSourceFinish& finish, std::size_t maximum_bytes, std::size_t caller_bytes = 0);

}  // namespace generativeqc::posthf
