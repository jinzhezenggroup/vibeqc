#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_contract_bounded_exact_low_order_force_page_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer, unsigned shell_class,
    unsigned high_pair_class, unsigned low_pair_class, double screening_tolerance,
    std::uint64_t page_begin, std::uint32_t page_capacity, std::uint32_t bra_ordinal_begin,
    std::uint32_t bra_ordinal_end, bool same_pair_class, const double* schwarz_bounds,
    const double* density, double* forces, std::uint32_t* bra_head);

}  // namespace vibeqc::scf::cuda_execution
