#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

#include "scf/cuda/direct_coulomb.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda_direct_jk.hpp"

namespace vibeqc::scf {

/** Own one exact public-AO provider source and its density/output scratch.
 * Kernel consumers borrow the packed view; the plan drains its stream before
 * releasing buffers. Layout and explicit budget accounting remain unchanged.
 */
struct CudaDirectJkPlan {
  int device_id{-1};
  cuda_execution::DeviceBatch batch{};
  cudaStream_t stream{};
  unsigned derivative_order{};
  std::size_t matrix_elements{}, coordinates_per_item{}, coordinate_elements{};
  double screening_tolerance{};
  double *density{}, *beta{}, *coulomb{}, *alpha_exchange{}, *beta_exchange{}, *bounds{},
      *derivative{};
  int* numerical_failure{};
  std::vector<void*> allocations;
  std::size_t device_bytes{};
  CudaDirectJkDiagnostic diagnostic{};
  std::unique_ptr<cuda_execution::GeneratedCoulombPlan> generated_coulomb;
  ~CudaDirectJkPlan();
};

}  // namespace vibeqc::scf
