#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_bounded_direct_shell_quartet_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile);

}  // namespace vibeqc::scf::cuda_execution
