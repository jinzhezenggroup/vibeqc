#pragma once

#include <cuda_runtime.h>

#include "generated_triples_fock_response_cpu.hpp"

namespace generativeqc::cc::generated {

/** Device views borrowed from the admitted two-page resolvent owner.
 * Returned outputs borrow the arena until the next generated call. The owner
 * preserves vector pages separately and fences every generated error check.
 */
struct TriplesFockCudaState {
  std::size_t o{}, v{}, q{};
  TriplesResolventInputs inputs;
  TriplesFockMomentInputs moments;
  double* arena{};
  int* error{};
  cudaStream_t stream{};
};

TriplesResolventOutputs run_triples_resolvent_cuda(TriplesFockCudaState&);
ParameterOutput run_triples_oo_moment_cuda(TriplesFockCudaState&);
ParameterOutput run_triples_vv_moment_cuda(TriplesFockCudaState&);
std::size_t triples_resolvent_cuda_kernel_count();
std::size_t triples_oo_moment_cuda_kernel_count();
std::size_t triples_vv_moment_cuda_kernel_count();

}  // namespace generativeqc::cc::generated
