#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"

// Cartesian powers and packed/public AO mappings shared by retained
// integral evaluators. No queue scheduling policy belongs in this layer.
namespace vibeqc::scf::cuda_execution {

struct Angular {
  unsigned x;
  unsigned y;
  unsigned z;
};

__device__ inline unsigned angular_axis(const Angular& angular, int axis) {
  return axis == 0 ? angular.x : (axis == 1 ? angular.y : angular.z);
}

__device__ inline void add_angular_axis(Angular& angular, int axis, int delta) {
  unsigned* value = axis == 0 ? &angular.x : (axis == 1 ? &angular.y : &angular.z);
  *value = static_cast<unsigned>(static_cast<int>(*value) + delta);
}

__device__ inline unsigned angular_total(const Angular& angular) {
  return angular.x + angular.y + angular.z;
}

__device__ inline bool is_s_function(const Angular& angular) { return angular_total(angular) == 0; }

__device__ inline Angular ao_angular(const DeviceBatch& batch, std::int64_t ao, unsigned term = 0) {
  const std::size_t offset = (static_cast<std::size_t>(ao) * kMaximumAoExpansionTerms + term) * 3;
  return {batch.ao_term_angular[offset], batch.ao_term_angular[offset + 1],
          batch.ao_term_angular[offset + 2]};
}

__device__ inline double ao_term_coefficient(const DeviceBatch& batch, std::int64_t ao,
                                             unsigned term) {
  return batch.ao_term_coefficients[static_cast<std::size_t>(ao) * kMaximumAoExpansionTerms + term];
}

__device__ inline Angular direct_ao_angular(const DeviceBatch& batch, std::int64_t ao) {
  const std::size_t offset = static_cast<std::size_t>(ao) * 3;
  return {batch.direct_ao_angular[offset], batch.direct_ao_angular[offset + 1],
          batch.direct_ao_angular[offset + 2]};
}

/** Map one canonical direct AO to its component within the owning shell. */
__device__ inline bool direct_shell_component_index(const DeviceBatch& batch, std::int64_t base,
                                                    std::int32_t shell, std::int32_t ao,
                                                    unsigned& component) {
  const std::int64_t shell_begin = batch.shell_direct_ao_offsets[shell];
  const std::int64_t local_begin = shell_begin - base;
  if (local_begin < 0 || ao < local_begin) return false;
  component = static_cast<unsigned>(ao - local_begin);
  return true;
}

template <typename Scalar>
__device__ inline Scalar vec_axis(const Vec3<Scalar>& vector, int axis) {
  return axis == 0 ? vector.x : (axis == 1 ? vector.y : vector.z);
}

}  // namespace vibeqc::scf::cuda_execution
