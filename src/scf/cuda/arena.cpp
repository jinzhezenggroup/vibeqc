#include "scf/cuda/arena.hpp"

#include "runtime/bounded_workspace.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

bool make_layout(std::size_t batch_size, std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                 std::size_t shell_count, std::size_t shell_pair_count,
                 std::size_t shell_pair_block_count, std::size_t bounded_generated_task_capacity,
                 std::size_t shell_pair_primitive_count, std::size_t psss_resident_task_count,
                 std::size_t psss_resident_ket_pair_count, std::size_t shell_quartet_tile_count,
                 std::size_t fp32_shell_quartet_tile_count,
                 std::size_t generated_shell_task_capacity,
                 std::size_t ppps_resident_ket_task_capacity,
                 std::size_t generic_order5_tile_capacity, std::size_t primitives,
                 std::size_t diis_history, std::size_t eigensolver_profile_capacity,
                 std::size_t spin_count, bool persistent_eri, bool transformed_direct,
                 bool shell_class_profiling, bool inactive_eigensolver_profiling,
                 bool bounded_fock_class_timing, bool bounded_direct_streaming,
                 bool mixed_precision_fock, ArenaLayout& layout) {
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrices = 0;
  std::size_t spin_matrices = 0;
  std::size_t eris = 0;
  std::size_t aos = 0;
  std::size_t direct_aos = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrices = 0;
  std::size_t direct_spin_matrices = 0;
  std::size_t transform_elements = 0;
  std::size_t transform_temporaries = 0;
  std::size_t ppps_signature_elements = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  if (!vibeqc::runtime::checked_multiply(nbf, nbf, matrix_size) ||
      !vibeqc::runtime::checked_multiply(matrix_size, matrix_size, eri_size) ||
      !vibeqc::runtime::checked_multiply(batch_size, matrix_size, matrices) ||
      !vibeqc::runtime::checked_multiply(matrices, spin_count, spin_matrices) ||
      !vibeqc::runtime::checked_multiply(batch_size, nbf, aos) ||
      !vibeqc::runtime::checked_multiply(batch_size, direct_nbf, direct_aos) ||
      !vibeqc::runtime::checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !vibeqc::runtime::checked_multiply(batch_size, direct_matrix_size, direct_matrices) ||
      !vibeqc::runtime::checked_multiply(direct_matrices, spin_count, direct_spin_matrices) ||
      !vibeqc::runtime::checked_multiply(aos, direct_nbf, transform_elements) ||
      !vibeqc::runtime::checked_multiply(transform_elements, spin_count, transform_temporaries) ||
      !vibeqc::runtime::checked_multiply(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count, kPppsSignatureBucketCount,
          ppps_signature_elements) ||
      !vibeqc::runtime::checked_add(nbf, 1, nbf_plus_one) ||
      !vibeqc::runtime::checked_multiply(nbf, nbf_plus_one, pair_product))
    return false;
  const std::size_t pair_count = pair_product / 2;
  if (persistent_eri && !vibeqc::runtime::checked_multiply(batch_size, eri_size, eris)) {
    return false;
  }
  std::size_t history_matrices = 0;
  std::size_t diis_dimension = 0;
  std::size_t diis_linear_elements = 0;
  if (!vibeqc::runtime::checked_multiply(spin_matrices, diis_history, history_matrices) ||
      !vibeqc::runtime::checked_add(diis_history, 1, diis_dimension) ||
      !vibeqc::runtime::checked_multiply(diis_dimension, diis_dimension, diis_linear_elements) ||
      !vibeqc::runtime::checked_multiply(diis_linear_elements, batch_size, diis_linear_elements))
    return false;
  vibeqc::runtime::WorkspaceLayout workspace;
  ArenaLayout made{};
  if (!workspace.append<std::int64_t>(batch_size + 1, made.atom_offsets) ||
      !workspace.append<std::int32_t>(atoms, made.atom_systems) ||
      !workspace.append<std::int32_t>(atoms, made.atomic_numbers) ||
      !workspace.append<double>(atoms * 3, made.positions) ||
      !workspace.append<std::int64_t>(batch_size + 1, made.system_shell_offsets) ||
      !workspace.append<std::int32_t>(shell_count, made.shell_atoms) ||
      !workspace.append<std::uint8_t>(shell_count, made.shell_angular) ||
      !workspace.append<std::int64_t>(shell_count + 1, made.shell_ao_offsets) ||
      !workspace.append<std::int64_t>(shell_count + 1, made.shell_direct_ao_offsets) ||
      !workspace.append<std::int64_t>(shell_count + 1, made.shell_primitive_offsets) ||
      !workspace.append<std::int64_t>(batch_size + 1, made.system_shell_pair_offsets) ||
      !workspace.append<std::int64_t>(batch_size + 1, made.system_shell_quartet_offsets) ||
      !workspace.append<std::int64_t>(batch_size + 1, made.system_shell_pair_block_offsets) ||
      !workspace.append<std::int64_t>(batch_size + 1,
                                      made.system_shell_pair_block_quartet_offsets) ||
      !workspace.append<std::int32_t>(shell_pair_count, made.shell_pair_systems) ||
      !workspace.append<std::int32_t>(shell_pair_count, made.shell_pair_first) ||
      !workspace.append<std::int32_t>(shell_pair_count, made.shell_pair_second) ||
      !workspace.append<std::int64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : shell_pair_count + 1,
          made.shell_pair_primitive_offsets) ||
      !workspace.append<PrimitivePairData>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0
                                                                     : shell_pair_primitive_count,
          made.shell_primitive_pairs) ||
      !workspace.append<PsssResidentTask>(psss_resident_task_count, made.psss_resident_tasks) ||
      !workspace.append<std::uint32_t>(psss_resident_ket_pair_count,
                                       made.psss_resident_ket_pairs) ||
      !workspace.append<std::int32_t>(aos, made.ao_shells) ||
      !workspace.append<std::uint8_t>(aos, made.ao_term_counts) ||
      !workspace.append<std::uint8_t>(aos * kMaximumAoExpansionTerms * 3, made.ao_term_angular) ||
      !workspace.append<double>(aos * kMaximumAoExpansionTerms, made.ao_term_coefficients) ||
      !workspace.append<std::int32_t>(direct_aos, made.direct_ao_shells) ||
      !workspace.append<std::uint8_t>(direct_aos * 3, made.direct_ao_angular) ||
      !workspace.append<double>(direct_aos, made.direct_ao_coefficients) ||
      !workspace.append<double>(transformed_direct ? transform_elements : 0,
                                made.ao_to_direct_transform) ||
      !workspace.append<double>(primitives, made.primitive_exponents) ||
      !workspace.append<double>(primitives, made.primitive_coefficients) ||
      !workspace.append<std::int32_t>(batch_size * spin_count, made.occupied) ||
      !workspace.append<std::uint8_t>(batch_size, made.warm_mask) ||
      !workspace.append<std::uint32_t>(mixed_precision_fock ? batch_size : 0,
                                       made.mixed_item_census) ||
      !workspace.append<double>(spin_matrices, made.warm_density) ||
      !workspace.append<std::uint8_t>(batch_size, made.warm_invalid) ||
      !workspace.append<double>(matrices, made.overlap) ||
      !workspace.append<double>(matrices, made.hcore) ||
      !workspace.append<double>(eris, made.eri) ||
      !workspace.append<double>(
          persistent_eri ? 0 : (transformed_direct ? direct_matrices : matrices),
          made.schwarz_bounds) ||
      !workspace.append<double>(transformed_direct ? direct_spin_matrices : 0,
                                made.direct_density) ||
      !workspace.append<double>(transformed_direct ? direct_spin_matrices : 0, made.direct_fock) ||
      !workspace.append<double>(transformed_direct ? transform_temporaries : 0,
                                made.direct_transform_temporary) ||
      !workspace.append<double>(persistent_eri ? 0 : shell_pair_count, made.shell_pair_bounds) ||
      !workspace.append<ShellPairDensityBounds>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : shell_pair_count,
          made.shell_pair_density_bounds) ||
      !workspace.append<std::uint32_t>(bounded_direct_streaming ? shell_pair_count : 0,
                                       made.bounded_direct_shell_pair_order) ||
      !workspace.append<std::uint32_t>(bounded_direct_streaming ? shell_pair_count : 0,
                                       made.bounded_stream_shell_pair_order) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectShellPairClassCount * (batch_size + 1) : 0,
          made.bounded_stream_pair_class_offsets) ||
      !workspace.append<GeneratedShellPairStream>(bounded_direct_streaming ? 1 : 0,
                                                  made.bounded_stream_topology) ||
      !workspace.append<double>(bounded_direct_streaming ? shell_pair_block_count : 0,
                                made.bounded_direct_shell_pair_block_bounds) ||
      !workspace.append<double>(bounded_direct_streaming ? batch_size : 0,
                                made.bounded_direct_system_density_bounds) ||
      !workspace.append<double>(
          bounded_direct_streaming ? batch_size * detail::kDirectShellPairClassCount : 0,
          made.bounded_direct_system_pair_density_bounds) ||
      !workspace.append<GeneratedShellTask>(
          bounded_direct_streaming ? bounded_generated_task_capacity : 0,
          made.bounded_direct_generated_tasks) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_direct_generated_task_counts) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount + 1 : 0,
          made.bounded_direct_generated_task_offsets) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount + 1 : 0,
          made.bounded_direct_generated_retry_task_offsets) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_direct_generated_task_heads) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_direct_generated_overflow) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_direct_generated_retry_mask) ||
      !workspace.append<std::uint32_t>(bounded_direct_streaming ? 1 : 0,
                                       made.bounded_direct_generated_retry_any) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
          made.bounded_force_signature_counts) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
          made.bounded_force_signature_offsets) ||
      !workspace.append<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureScanBlockCount : 0,
          made.bounded_force_signature_block_offsets) ||
      !workspace.append<std::uint64_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_fock_class_timer_starts) ||
      !workspace.append<std::uint64_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_fock_class_timer_elapsed) ||
      !workspace.append<std::uint32_t>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_fock_class_timer_launches) ||
      !workspace.append<unsigned long long>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_fock_fp64_work_counts) ||
      !workspace.append<unsigned long long>(
          bounded_fock_class_timing ? detail::kDirectQuartetShellClassCount : 0,
          made.bounded_fock_fp32_work_counts) ||
      !workspace.append<std::uint32_t>(persistent_eri || bounded_direct_streaming
                                           ? 0
                                           : detail::kDirectQuartetAngularOrderCount + 1,
                                       made.active_shell_quartet_tile_offsets) ||
      !workspace.append<std::uint32_t>(
          persistent_eri || bounded_direct_streaming ? 0 : detail::kDirectQuartetAngularOrderCount,
          made.active_shell_quartet_tile_counts) ||
      !workspace.append<ActiveShellQuartetTile>(
          persistent_eri || bounded_direct_streaming ? 0 : shell_quartet_tile_count,
          made.active_shell_quartet_tiles) ||
      !workspace.append<std::uint32_t>(
          mixed_precision_fock ? detail::kDirectQuartetAngularOrderCount + 1 : 0,
          made.fp32_shell_quartet_tile_offsets) ||
      !workspace.append<std::uint32_t>(
          mixed_precision_fock ? detail::kDirectQuartetAngularOrderCount : 0,
          made.fp32_shell_quartet_tile_counts) ||
      !workspace.append<ActiveShellQuartetTile>(
          mixed_precision_fock ? fp32_shell_quartet_tile_count : 0,
          made.fp32_shell_quartet_tiles) ||
      !workspace.append<DeviceShellClassProfileEntry>(
          shell_class_profiling ? detail::kDirectQuartetShellClassCount : 0,
          made.shell_class_profile) ||
      !workspace.append<std::uint32_t>(
          shell_quartet_tile_count == 0 ? 0 : kPersistentFockAngularOrderCount,
          made.persistent_fock_task_heads) ||
      !workspace.append<std::uint32_t>(mixed_precision_fock ? kPersistentFockAngularOrderCount : 0,
                                       made.fp32_persistent_fock_task_heads) ||
      !workspace.append<std::uint32_t>(
          shell_quartet_tile_count == 0 ? 0 : kPersistentForceAngularOrderCount,
          made.persistent_force_task_heads) ||
      !workspace.append<GeneratedShellTask>(generated_shell_task_capacity,
                                            made.generated_shell_tasks) ||
      !workspace.append<std::uint8_t>(
          generated_shell_task_capacity == 0 ? 0 : shell_quartet_tile_count,
          made.generated_shell_classes) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount + 1,
          made.generated_shell_task_offsets) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount,
          made.generated_shell_task_counts) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount,
          made.generated_shell_task_write_counts) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : detail::kDirectQuartetShellClassCount,
          made.generated_shell_task_heads) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : kLowOrderSignatureElementCount,
          made.generated_low_order_signature_counts) ||
      !workspace.append<std::uint32_t>(
          generated_shell_task_capacity == 0 ? 0 : kLowOrderSignatureElementCount,
          made.generated_low_order_signature_offsets) ||
      !workspace.append<GeneratedPppsResidentTask>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
          made.generated_ppps_resident_tasks) ||
      !workspace.append<std::uint32_t>(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
                                       made.generated_ppps_resident_bra_counts) ||
      !workspace.append<std::uint32_t>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count + 1,
          made.generated_ppps_resident_bra_offsets) ||
      !workspace.append<std::uint32_t>(ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
                                       made.generated_ppps_resident_bra_write_counts) ||
      !workspace.append<std::uint32_t>(ppps_signature_elements,
                                       made.generated_ppps_resident_signature_counts) ||
      !workspace.append<std::uint32_t>(ppps_signature_elements,
                                       made.generated_ppps_resident_signature_offsets) ||
      !workspace.append<std::uint32_t>(shell_class_profiling ? ppps_resident_ket_task_capacity : 0,
                                       made.generated_ppps_resident_signatures) ||
      !workspace.append<std::uint64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : 1,
          made.generated_fock_shell_class_mask) ||
      !workspace.append<std::uint64_t>(mixed_precision_fock ? 1 : 0,
                                       made.generated_mixed_fock_shell_class_mask) ||
      !workspace.append<ActiveShellQuartetTile>(generic_order5_tile_capacity,
                                                made.generic_order5_tiles) ||
      !workspace.append<std::uint32_t>(generic_order5_tile_capacity == 0 ? 0 : 1,
                                       made.generic_order5_tile_count) ||
      !workspace.append<std::int32_t>(pair_count, made.ao_pair_first) ||
      !workspace.append<std::int32_t>(pair_count, made.ao_pair_second) ||
      !workspace.append<double>(batch_size, made.nuclear_repulsion) ||
      !workspace.append<double>(matrices, made.orthogonalizer) ||
      !workspace.append<double>(spin_matrices, made.temporary) ||
      !workspace.append<double>(spin_matrices, made.eigensystem) ||
      !workspace.append<double>(spin_matrices, made.coefficients) ||
      !workspace.append<double>(batch_size * spin_count * nbf, made.eigenvalues) ||
      !workspace.append<double>(spin_matrices, made.density) ||
      !workspace.append<double>(spin_matrices, made.next_density) ||
      !workspace.append<double>(spin_matrices, made.fock) ||
      !workspace.append<double>(spin_matrices, made.residual) ||
      !workspace.append<double>(spin_matrices, made.weighted_density) ||
      !workspace.append<double>(spin_count == 2 ? matrices : 0, made.total_density) ||
      !workspace.append<double>(spin_count == 2 ? matrices : 0, made.total_weighted_density) ||
      !workspace.append<double>(history_matrices, made.fock_history) ||
      !workspace.append<double>(history_matrices, made.residual_history) ||
      !workspace.append<double>(diis_linear_elements, made.diis_linear_system) ||
      !workspace.append<double>(batch_size * diis_dimension, made.diis_coefficients) ||
      !workspace.append<std::uint32_t>(batch_size, made.diis_count) ||
      !workspace.append<std::uint32_t>(batch_size, made.diis_head) ||
      !workspace.append<double>(batch_size, made.energy) ||
      !workspace.append<double>(batch_size, made.previous_energy) ||
      !workspace.append<double>(batch_size, made.energy_change) ||
      !workspace.append<double>(batch_size, made.density_rms) ||
      !workspace.append<double>(atoms * 3, made.forces) ||
      !workspace.append<std::uint8_t>(batch_size, made.active) ||
      !workspace.append<std::uint8_t>(batch_size, made.converged) ||
      !workspace.append<std::uint8_t>(batch_size, made.failed) ||
      !workspace.append<std::uint8_t>(batch_size, made.final_fock_reuse_mask) ||
      !workspace.append<std::uint32_t>(1, made.final_fock_rebuild_count) ||
      !workspace.append<std::uint8_t>(batch_size * spin_count, made.spin_active) ||
      !workspace.append<std::uint32_t>(batch_size, made.iterations) ||
      !workspace.append<int>(batch_size * spin_count, made.solver_info) ||
      !workspace.append<std::uint32_t>(inactive_eigensolver_profiling ? 1 : 0,
                                       made.inactive_eigensolver_profile_count) ||
      !workspace.append<DeviceInactiveEigensolverProfileEntry>(
          inactive_eigensolver_profiling ? eigensolver_profile_capacity : 0,
          made.inactive_eigensolver_profile) ||
      !workspace.append<std::uint64_t>(bounded_direct_streaming ? 1 : 0,
                                       made.bounded_direct_cursor))
    return false;
  made.bytes = workspace.bytes();
  layout = made;
  return true;
}

}  // namespace vibeqc::scf::cuda_execution
