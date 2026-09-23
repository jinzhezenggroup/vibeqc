// Allocation-free adapter. Scalar policy and projector algebra are generated
// from the shared matrix-function and SCF tensor owners.
#include <algorithm>

#include "generated_mean_field_setup.cuh"
#include "scf/cuda/mean_field_setup.hpp"

namespace vibeqc::scf::cuda_execution {
namespace {
unsigned blocks(std::size_t elements) {
  return static_cast<unsigned>(std::min(std::size_t{65535}, 1 + (elements - 1) / 128));
}
}  // namespace

void form_overlap_weights(cudaStream_t stream, std::size_t n, double* values, int* invalid) {
  generated::overlap_spectral_weights<<<blocks(n), 128, 0, stream>>>(n, values, invalid);
}
void form_weighted_projector(cudaStream_t stream, std::size_t n, unsigned spins,
                             const double* vectors, const double* weights, double* output) {
  generated::weighted_projector<<<blocks(spins * n * n), 128, 0, stream>>>(n, spins, vectors,
                                                                           weights, output);
}
void form_occupation_weights(cudaStream_t stream, std::size_t n, unsigned spins, int alpha,
                             int beta, double* weights) {
  generated::occupation_weights<<<blocks(spins * n), 128, 0, stream>>>(n, spins, alpha, beta,
                                                                       weights);
}
void check_overlap_metric(cudaStream_t stream, std::size_t n, const double* metric, int* invalid) {
  generated::check_overlap_identity<<<blocks(n * n), 128, 0, stream>>>(n, metric, invalid);
}
}  // namespace vibeqc::scf::cuda_execution
