#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d4_reference.hpp"

namespace vibeqc::dft::dispersion {

struct D4CudaBatch {
  std::uint32_t systems{};
  std::uint32_t total_atoms{};
  const std::uint32_t* offsets{};
  const std::int32_t* atomic_numbers{};
  const double* coordinates{};
  const double* charges{};
  const std::uint8_t* active{};
};

struct D4CudaResult {
  D4Status* statuses{};
  double* energies{};   // [systems, 2] => two-body, ATM
  double* gradients{};  // [total_atoms, 3], dE/dR at fixed supplied charges
  double* dedq{};       // [total_atoms], dE/dq
};

inline constexpr std::size_t d4_cuda_workspace_elements(std::size_t total_atoms) {
  return 27u * total_atoms;
}

// Launch a bounded, block-cooperative fixed-charge D4 batch. One CUDA block owns
// each ragged molecule while its lanes share pair/triple work. Failed/inactive
// members publish zero outputs and never poison successful peers. A malformed
// shared offset partition invalidates the whole batch before member work.
cudaError_t launch_d4_fixed_charge_batched_cuda(const D4CudaBatch& batch,
                                                const D4Parameters& parameters, D4Tables tables,
                                                double* workspace, std::size_t workspace_elements,
                                                const D4CudaResult& result,
                                                cudaStream_t stream = nullptr);

}  // namespace vibeqc::dft::dispersion
