#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::dft {
/** Borrowed device buffers for one current local-dense task, ABI version 1.
 *
 * Arrays are FP64. ao is [jet,point,active_ao]; features has the fixed
 * [13,point] layout (rho,grad_xyz,tau per spin, then sigma_aa/ab/bb), but only
 * slots selected by the owner's immutable feature mask are meaningful. ao_ids
 * maps local columns into the global density/potential domain. Consumers enqueue on stream,
 * write both spin local_potential matrices [2,active_ao,active_ao], then use
 * the owner's scatter call. No feature/jet download is needed.
 *
 * The owner must remain locked and alive throughout consumption. Density
 * replacement, another task or closure invalidates this view. generation is
 * checked again by scatter. Pointers must never be retained beyond the lease.
 */
struct GridTaskView {
  std::uint64_t version{}, generation{};
  std::size_t npoint{}, nao{}, nactive{}, jets{};
  const std::size_t* ao_ids{};
  const double *points{}, *ao{}, *features{};
  double *local_potential{}, *potential{};
  cudaStream_t stream{};
  int* error{};
};

/** Immutable packed AO source owned by the same CUDA grid plan.
 *
 * This view has no task generation because geometry/basis data are frozen for
 * the owner's lifetime. Consumers may borrow it only while the owner remains
 * alive and must enqueue on stream when combining it with task views.
 */
struct GridBasisView {
  std::uint64_t version{};
  std::size_t natom{}, nprimitive{}, nao{};
  const double* basis{};
  cudaStream_t stream{};
};
}  // namespace vibeqc::dft
