#ifndef VIBEQC_SCF_CUDA_DF_DERIVATIVES_CUH
#define VIBEQC_SCF_CUDA_DF_DERIVATIVES_CUH
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "runtime/cuda_gaussian_products.cuh"
#include "runtime/strided_range.hpp"

namespace vibeqc::scf {
/** Non-owning normalized public-AO expansions; atom indices share one geometry. */
using DfDerivativeBasisView = runtime::cuda_gaussian_products::BasisView;

/** Add a full row-major response-weight tile directly into atomic gradients.
 * kind=0: A[mu,nu,P], flattened ((mu*nbf+nu)*naux+P); kind=1: M[P,Q].
 * Every dense element is counted once, with no hidden symmetry multiplicity.
 * Weights are fixed external Lagrangian inputs. Nuclear, one-electron and
 * Pulay terms belong to other consumers. schedule=1 is deterministic serial
 * traversal; schedule=0 owns one dense element per thread with atomic sums.
 * range maps weight k to range.index(begin+k), allowing whole auxiliary-major
 * HF blocks without a full dense A-weight transpose. begin retains row phase
 * when the upload capacity splits a block inside an AO row.
 * This launch allocates nothing and uses the caller's owning stream.
 */
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView orbital,
                                      DfDerivativeBasisView auxiliary, const double* positions,
                                      unsigned kind, runtime::StridedRange range, std::size_t count,
                                      const double* weights, unsigned schedule, double* gradient,
                                      cudaStream_t stream, std::size_t begin = 0);
}  // namespace vibeqc::scf
#endif
