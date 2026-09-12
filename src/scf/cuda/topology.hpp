#pragma once

#include <vector>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda_batch.hpp"

namespace vibeqc::scf::cuda_execution {

/** Owned host topology used to prepare a borrowed device basis view. Packing preserves
 * Cartesian/public AO ordering. */
struct HostBatch {
  // Only populated for an ECP batch; owns normalized scientific inputs used by
  // device ECP integration. Coordinates remain part of the geometry key.
  std::vector<core::System> ecp_systems;
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t spin_count{1};
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atom_systems;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<std::int64_t> system_shell_offsets;
  std::vector<std::int32_t> shell_atoms;
  std::vector<std::uint8_t> shell_angular;
  std::vector<std::int64_t> shell_ao_offsets;
  std::vector<std::int64_t> shell_direct_ao_offsets;
  std::vector<std::int64_t> shell_primitive_offsets;
  std::vector<std::int64_t> system_shell_pair_offsets;
  std::vector<std::int64_t> system_shell_quartet_offsets;
  std::vector<std::int64_t> system_shell_pair_block_offsets;
  std::vector<std::int64_t> system_shell_pair_block_quartet_offsets;
  std::vector<std::int32_t> shell_pair_systems;
  std::vector<std::int32_t> shell_pair_first;
  std::vector<std::int32_t> shell_pair_second;
  std::vector<std::int64_t> shell_pair_primitive_offsets;
  std::vector<PsssResidentTask> psss_resident_tasks;
  std::vector<std::uint32_t> psss_resident_ket_pairs;
  std::vector<std::int32_t> ao_shells;
  std::vector<std::uint8_t> ao_term_counts;
  std::vector<std::uint8_t> ao_term_angular;
  std::vector<double> ao_term_coefficients;
  std::vector<std::int32_t> direct_ao_shells;
  std::vector<std::uint8_t> direct_ao_angular;
  std::vector<double> direct_ao_coefficients;
  std::vector<double> ao_to_direct_transform;
  std::vector<double> primitive_exponents;
  std::vector<double> primitive_coefficients;
  std::vector<std::int32_t> occupied;
  std::vector<std::uint8_t> warm_mask;
  std::vector<double> warm_density;
};

/** Pack homogeneous topology and optional spin-resolved warm densities in input order. */
bool pack_host_batch(const std::vector<core::System>& systems,
                     const std::vector<const std::vector<double>*>& initial_densities,
                     HostBatch& host, bool unrestricted = false);

/** Compare immutable topology; coordinates and warm state are checked separately by replay. */
bool same_topology(const HostBatch& first, const HostBatch& second);

/** Describe the compacted PPPS queue in the same descriptor order consumed on device. */
CudaPppsQueueProfile build_ppps_queue_profile(const HostBatch& host,
                                              const std::vector<std::uint32_t>& descriptor_counts,
                                              const std::vector<std::uint32_t>& ordered_signatures,
                                              unsigned multiprocessor_count);

}  // namespace vibeqc::scf::cuda_execution
