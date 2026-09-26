#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_two_electron_force_psss_resident_bra_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, const PsssResidentTask* resident_tasks,
    const std::uint32_t* resident_ket_pairs, std::size_t resident_task_count,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, bool force_density_product_screening,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask);

/** Resolve host spin/precision while retaining compile-time angular dispatch. */
void dispatch_angular_force_quartets(
    bool unrestricted, cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, const PsssResidentTask* psss_resident_tasks,
    const std::uint32_t* psss_resident_ket_pairs, std::size_t psss_resident_task_count,
    std::size_t resident_psss_bra_primitive_pairs, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask);

}  // namespace vibeqc::scf::cuda_execution
