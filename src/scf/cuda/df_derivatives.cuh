#ifndef VIBEQC_SCF_CUDA_DF_DERIVATIVES_CUH
#define VIBEQC_SCF_CUDA_DF_DERIVATIVES_CUH
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "runtime/cuda_gaussian_products.cuh"

namespace vibeqc::scf {
/** Non-owning normalized public-AO expansions; atom indices share one geometry. */
using DfDerivativeBasisView = runtime::cuda_gaussian_products::BasisView;

/** Add a full row-major response-weight tile directly into atomic gradients.
 * kind=0: A[mu,nu,P], flattened ((mu*nbf+nu)*naux+P); kind=1: M[P,Q].
 * Every dense element is counted once, with no hidden symmetry multiplicity.
 * Weights are fixed external Lagrangian inputs. Nuclear, one-electron and
 * Pulay terms belong to other consumers. schedule=1 is deterministic serial
 * traversal; schedule=0 owns one dense element per thread with atomic sums.
 * stride maps weight k to offset+k*stride, allowing auxiliary-major HF
 * weights without a full dense A-weight transpose.
 * This launch allocates nothing and uses the caller's owning stream.
 */
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView orbital,
                                      DfDerivativeBasisView auxiliary, const double* positions,
                                      unsigned kind, std::size_t offset, std::size_t count,
                                      const double* weights, unsigned schedule, double* gradient,
                                      cudaStream_t stream, std::size_t stride = 1);
}  // namespace vibeqc::scf
#endif
