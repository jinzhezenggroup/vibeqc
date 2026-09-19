#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Explicit reference/performance exception; generated derivatives own production. */
void launch_one_electron_force_cooperative_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream, DeviceBatch batch,
                                                  const std::int32_t* pair_first,
                                                  const std::int32_t* pair_second,
                                                  std::size_t pair_count, const double* density,
                                                  const double* weighted_density,
                                                  const std::uint8_t* active, double* forces);

}  // namespace vibeqc::scf::cuda_execution
