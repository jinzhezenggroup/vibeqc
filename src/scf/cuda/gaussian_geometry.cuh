#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Borrowed atomic coordinates and Gaussian product geometry; derivative
// seeds retain the packed batch coordinate convention.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Vec3<Scalar> atom_position(const DeviceBatch& batch, std::int64_t atom,
                                             std::int64_t derivative_coordinate) {
  const std::int64_t base = atom * 3;
  if constexpr (std::is_same_v<Scalar, Dual3>) {
    // Any coordinate belonging to the requested atom denotes the combined
    // x/y/z seed. Negative coordinates continue to mean value-only mode.
    const bool differentiated = derivative_coordinate >= 0 && derivative_coordinate / 3 == atom;
    return {
        {batch.positions[base], differentiated ? 1.0 : 0.0, 0.0, 0.0},
        {batch.positions[base + 1], 0.0, differentiated ? 1.0 : 0.0, 0.0},
        {batch.positions[base + 2], 0.0, 0.0, differentiated ? 1.0 : 0.0},
    };
  } else {
    return {
        scalar<Scalar>(batch.positions[base], derivative_coordinate == base ? 1.0 : 0.0),
        scalar<Scalar>(batch.positions[base + 1], derivative_coordinate == base + 1 ? 1.0 : 0.0),
        scalar<Scalar>(batch.positions[base + 2], derivative_coordinate == base + 2 ? 1.0 : 0.0),
    };
  }
}

template <typename Scalar>
__device__ inline Scalar distance_squared(const Vec3<Scalar>& first, const Vec3<Scalar>& second) {
  const Scalar dx = first.x - second.x;
  const Scalar dy = first.y - second.y;
  const Scalar dz = first.z - second.z;
  return dx * dx + dy * dy + dz * dz;
}

template <typename Scalar>
__device__ inline Vec3<Scalar> product_center(double alpha, const Vec3<Scalar>& first, double beta,
                                              const Vec3<Scalar>& second) {
  using Real = EvaluationReal<Scalar>;
  const Real alpha_value{alpha};
  const Real beta_value{beta};
  const Real exponent = alpha_value + beta_value;
  return {(alpha_value * first.x + beta_value * second.x) / exponent,
          (alpha_value * first.y + beta_value * second.y) / exponent,
          (alpha_value * first.z + beta_value * second.z) / exponent};
}

}  // namespace vibeqc::scf::cuda_execution
