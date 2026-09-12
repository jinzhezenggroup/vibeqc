#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "scf/cuda_weighted_eri.hpp"

namespace vibeqc::scf::cuda_execution {

/** Launch the existing weighted primitive consumer on borrowed arrays. */
void launch_weighted_eri_reference_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream,
                                          const CudaWeightedEriPrimitive* records,
                                          std::size_t count, bool generated,
                                          CudaWeightedEriResult* output);

/** Launch the existing weighted primitive consumer on borrowed arrays. */
void launch_weighted_eri_generated_psss_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream,
                                               const CudaWeightedEriPrimitive* records,
                                               std::size_t count, CudaWeightedEriResult* output);

}  // namespace vibeqc::scf::cuda_execution
