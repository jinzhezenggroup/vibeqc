#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_compact_bounded_exact_class_force_wave_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer, unsigned shell_class,
    unsigned high_pair_class, unsigned low_pair_class, double screening_tolerance,
    std::uint64_t page_begin, std::uint32_t page_capacity, std::uint32_t bra_ordinal_begin,
    std::uint32_t bra_ordinal_end, bool same_pair_class, GeneratedShellTask* tasks,
    std::uint32_t* task_count, std::uint32_t* bra_head, const std::uint32_t* overflow,
    bool force_execution, std::uint32_t* signature_counts, const std::uint32_t* signature_offsets);

}  // namespace vibeqc::scf::cuda_execution
