#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::dft {
/** Borrowed device buffers for one current local-dense task, ABI version 1.
 *
 * Arrays are FP64. ao is [jet,point,active_ao]; features is [13,point]
 * (rho,grad_xyz,tau per spin, then sigma_aa/ab/bb). ao_ids maps local columns
 * into the global density/potential domain. Consumers enqueue on stream,
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
}  // namespace vibeqc::dft
