#pragma once

#include <cstddef>
#include <cstdint>

#include "scf/generated_shell_task.hpp"

namespace vibeqc::scf::cuda_execution {

/** Borrowed device basis view and primitive-pair storage ABI. The owning plan outlives every
 * submitted kernel. */
template <typename Scalar>
struct Vec3 {
  Scalar x;
  Scalar y;
  Scalar z;
};

/** Geometry and contraction data shared by every quartet using a shell pair. */
struct PrimitivePairData {
  double exponent_sum;
  double reduced_exponent;
  Vec3<double> product_center;
  double weighted_coefficient;
  double first_product_scale;
  double second_product_scale;
};

static_assert(sizeof(PrimitivePairData) == 8 * sizeof(double));
static_assert(sizeof(PrimitivePairData) == sizeof(detail::GeneratedPrimitivePairData));
static_assert(alignof(PrimitivePairData) == alignof(detail::GeneratedPrimitivePairData));

struct DeviceBatch {
  std::int32_t batch_size;
  std::int32_t nbf;
  // Direct shell quartets always use normalized Cartesian source AOs. This
  // equals nbf for Cartesian public bases and is larger for spherical d/f.
  std::int32_t direct_nbf;
  std::int64_t total_atoms;
  std::int64_t total_shells;
  std::int64_t total_shell_pairs;
  std::int64_t total_shell_quartets;
  std::int64_t total_shell_pair_blocks;
  std::int64_t total_shell_pair_block_quartets;
  const std::int64_t* atom_offsets;
  const std::int32_t* atom_systems;
  const std::int32_t* atomic_numbers;
  const double* positions;
  const std::int64_t* system_shell_offsets;
  const std::int32_t* shell_atoms;
  const std::uint8_t* shell_angular;
  const std::int64_t* shell_ao_offsets;
  const std::int64_t* shell_direct_ao_offsets;
  const std::int64_t* shell_primitive_offsets;
  const std::int64_t* system_shell_pair_offsets;
  const std::int64_t* system_shell_quartet_offsets;
  const std::int64_t* system_shell_pair_block_offsets;
  const std::int64_t* system_shell_pair_block_quartet_offsets;
  const std::int32_t* shell_pair_systems;
  const std::int32_t* shell_pair_first;
  const std::int32_t* shell_pair_second;
  const std::int64_t* shell_pair_primitive_offsets;
  const PrimitivePairData* shell_primitive_pairs;
  // Every target AO refers back to one physical shell and carries up to three
  // normalized Cartesian expansion terms. Cartesian AOs use one term; real
  // spherical d/f AOs use the sparse solid-harmonic combinations.
  const std::int32_t* ao_shells;
  const std::uint8_t* ao_term_counts;
  const std::uint8_t* ao_term_angular;
  const double* ao_term_coefficients;
  const std::int32_t* direct_ao_shells;
  const std::uint8_t* direct_ao_angular;
  const double* direct_ao_coefficients;
  // Column-major C with public AO rows and Cartesian source AO columns:
  // phi_public = C * phi_cartesian.
  const double* ao_to_direct_transform;
  const double* primitive_exponents;
  const double* primitive_coefficients;
  const std::int32_t* occupied;
};

}  // namespace vibeqc::scf::cuda_execution
