#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_eri_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                             DeviceBatch batch, double* eri);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                              std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                              const double* eri, const double* density, const std::uint8_t* active,
                              double* fock);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* hcore, const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock);

}  // namespace vibeqc::scf::cuda_execution
