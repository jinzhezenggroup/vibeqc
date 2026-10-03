#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/direct_block_domain.hpp"

namespace generativeqc::scf::cuda_execution {

enum class DirectRangeOperator : std::uint32_t {
  Full = 0,
  Long = 1,
  Short = 2,
  RshSources = 3,
  FullSources = 4
};

/** Force-output fallback; purpose selects screening semantics, not the scientific output. */
/** Method-neutral force variant. Coefficients multiply the Coulomb and exchange
 * density contractions without changing topology, screening, or recurrence.
 * separate_sources requires two total_atoms*3 output channels, [J', K']. */
void launch_bounded_direct_shell_quartet_kernel_scaled(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile,
    double coulomb_coefficient, double exchange_coefficient, bool separate_sources = false,
    detail::BoundedDirectBlockDomain block_domain = {});

/** Range-separated exchange derivative on the same bounded shell scheduler.
 * Full-range Schwarz bounds remain a conservative gate for SR/LR operators. */
void launch_bounded_direct_range_exchange_force_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DirectRangeOperator radial_operator, double omega,
    double exchange_coefficient);

/** Range-separated positive K through the same bounded shell scheduler.
 * Full-range Schwarz and density bounds remain conservative for SR/LR values. */
void launch_bounded_direct_range_exchange_fock_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DirectRangeOperator radial_operator, double omega);

/** Fused [J', SR-K', LR-K'] output over one bounded shell traversal. */
void launch_bounded_direct_rsh_force_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* source_forces,
    unsigned long long* global_cursor, double omega, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient);

void launch_bounded_direct_shell_quartet_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile);

/** Consume only shell classes outside covered_shell_class_mask and publish one
 * raw source: J when coulomb_only, positive K when exchange_only. */
void launch_bounded_direct_fock_source_shell_quartet_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    std::uint64_t covered_shell_class_mask, const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, bool coulomb_only, bool exchange_only);

/** Consume only Fock registry gaps through the bounded hierarchical dispatcher. */
void launch_bounded_direct_fock_shell_quartet_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    unsigned long long* global_cursor);

}  // namespace generativeqc::scf::cuda_execution
