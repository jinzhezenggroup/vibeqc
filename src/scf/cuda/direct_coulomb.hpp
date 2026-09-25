#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/topology.hpp"

namespace vibeqc::scf::cuda_execution {

/** Optional geometry owner for the generated pure-J consumer. It borrows the
 * direct provider's stream and public basis metadata, and owns bounded shell
 * topology, Cartesian transforms and scratch. No quartet list is materialized.
 */
struct GeneratedCoulombPlan {
  DeviceBatch batch{};
  cudaStream_t stream{};
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  std::uint64_t class_mask{};
  unsigned worker_blocks{};
  double screening{};
  double *density{}, *coulomb{}, *temporary{}, *total_density{}, *zero{}, *schwarz{},
      *shell_bounds{};
  std::uint8_t* active{};
  std::uint32_t* heads{};
  GeneratedShellPairStream* topology{};
  ~GeneratedCoulombPlan();
};

/** Unsupported angular classes or insufficient optional capacity return null.
 * CUDA execution failures propagate; only allocation failure selects fallback.
 */
std::unique_ptr<GeneratedCoulombPlan> prepare_generated_coulomb(const HostBatch& host,
                                                                DeviceBatch borrowed,
                                                                cudaStream_t stream, int device,
                                                                double screening,
                                                                std::size_t budget);

/** Enqueue raw J from total spin density. Inputs and result use public AO order.
 * The same stream owns every transform, scatter and projection; no host copies.
 */
cudaError_t enqueue_generated_coulomb(GeneratedCoulombPlan& plan, const double* density,
                                      const double* beta, double* coulomb);

}  // namespace vibeqc::scf::cuda_execution
