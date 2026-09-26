#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_fock_direct_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* hcore, const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_uhf_fock_direct_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* hcore, const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock);

}  // namespace vibeqc::scf::cuda_execution
