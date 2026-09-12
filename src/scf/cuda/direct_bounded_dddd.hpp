#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_bounded_direct_dddd_streaming_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, bool force, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    std::uint32_t* bra_head, DeviceShellClassProfileEntry* profile,
    unsigned long long* fp64_work_count);

}  // namespace vibeqc::scf::cuda_execution
