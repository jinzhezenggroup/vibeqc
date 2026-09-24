#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace vibeqc::scf::cuda_execution {
/** Compiler-generated full symmetric reconstruction in borrowed column-major
 * buffers. These launches allocate nothing and use the owning ordinary stream.
 * Invalid overlap spectra/metric identities set the caller's sticky error flag. */
void form_overlap_weights(cudaStream_t stream, std::size_t n, double* eigenvalues, int* invalid);
void form_weighted_projector(cudaStream_t stream, std::size_t n, unsigned spins,
                             const double* vectors, const double* weights, double* output);
void form_occupation_weights(cudaStream_t stream, std::size_t n, unsigned spins, int alpha,
                             int beta, double* weights);
void check_overlap_metric(cudaStream_t stream, std::size_t n, const double* metric, int* invalid);
}  // namespace vibeqc::scf::cuda_execution
