#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the retained reference response or nuclear launch unchanged. */
void launch_build_cuda_one_electron_derivatives_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const std::int32_t* pair_first, const std::int32_t* pair_second, std::size_t pair_count,
    std::int64_t derivative_coordinate, double* overlap, double* hcore);

/** Forward the retained reference response or nuclear launch unchanged. */
void launch_build_cuda_nuclear_repulsion_kernel(bool derivative, dim3 grid, dim3 block,
                                                std::size_t shared_bytes, cudaStream_t stream,
                                                DeviceBatch batch,
                                                std::int64_t derivative_coordinate,
                                                double* nuclear_repulsion);

}  // namespace vibeqc::scf::cuda_execution
