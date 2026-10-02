#pragma once

#include <cuda_runtime.h>

#include "generated_rccsd_cpu.hpp"

namespace generativeqc::cc::generated {

/** Borrowed device views for one runtime-indexed triples-response page.
 * The owner validates dimensions, sizes the arena with the shared generated
 * planner, and fences the stream before reusing controls or releasing storage.
 * Returned pointers borrow arena slots until the next page is submitted.
 */
struct TriplesResponseCudaState {
  std::size_t o{}, v{}, q{};
  TriplesResponseInputs inputs;
  double* arena{};
  int* error{};
  cudaStream_t stream{};
};

TriplesResponseOutputs run_triples_response_cuda(TriplesResponseCudaState& state);
std::size_t triples_response_cuda_kernels_per_page();

}  // namespace generativeqc::cc::generated
