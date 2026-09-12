#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_force_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

// Retained direct bounded contraction contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** Runtime angular dispatch used only by the generic large-topology path. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_fock_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock, std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FOCK_CASE(order)                                                       \
  case order:                                                                                 \
    contract_fock_direct_quartet_subtile<Unrestricted, order>(                                \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, fock, \
        nullptr, subtile, lane);                                                              \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FOCK_CASE(0);
    VIBEQC_BOUNDED_FOCK_CASE(1);
    VIBEQC_BOUNDED_FOCK_CASE(2);
    VIBEQC_BOUNDED_FOCK_CASE(3);
    VIBEQC_BOUNDED_FOCK_CASE(4);
    VIBEQC_BOUNDED_FOCK_CASE(5);
    VIBEQC_BOUNDED_FOCK_CASE(6);
    VIBEQC_BOUNDED_FOCK_CASE(7);
    VIBEQC_BOUNDED_FOCK_CASE(8);
    VIBEQC_BOUNDED_FOCK_CASE(9);
    VIBEQC_BOUNDED_FOCK_CASE(10);
    VIBEQC_BOUNDED_FOCK_CASE(11);
    VIBEQC_BOUNDED_FOCK_CASE(12);
    default:
      break;
  }
#undef VIBEQC_BOUNDED_FOCK_CASE
}

template <bool Unrestricted>
__device__ inline __noinline__ void contract_bounded_direct_force_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces, std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FORCE_CASE(order)                                                        \
  case order:                                                                                   \
    contract_two_electron_force_quartet_subtile<Unrestricted, order>(                           \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, forces, \
        0U, subtile, lane);                                                                     \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FORCE_CASE(0);
    VIBEQC_BOUNDED_FORCE_CASE(1);
    VIBEQC_BOUNDED_FORCE_CASE(2);
    VIBEQC_BOUNDED_FORCE_CASE(3);
    VIBEQC_BOUNDED_FORCE_CASE(4);
    VIBEQC_BOUNDED_FORCE_CASE(5);
    VIBEQC_BOUNDED_FORCE_CASE(6);
    VIBEQC_BOUNDED_FORCE_CASE(7);
    VIBEQC_BOUNDED_FORCE_CASE(8);
    VIBEQC_BOUNDED_FORCE_CASE(9);
    VIBEQC_BOUNDED_FORCE_CASE(10);
    VIBEQC_BOUNDED_FORCE_CASE(11);
    VIBEQC_BOUNDED_FORCE_CASE(12);
    default:
      break;
  }
#undef VIBEQC_BOUNDED_FORCE_CASE
}

}  // namespace vibeqc::scf::cuda_execution
