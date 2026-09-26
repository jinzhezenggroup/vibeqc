#pragma once

#include <cstddef>

namespace vibeqc::scf::cuda_execution {

/** Offsets for one persistent SCF arena. Optional regions have zero length when their execution
 * mode is disabled. */
struct ArenaLayout {
  std::size_t bytes{};
  std::size_t atom_offsets{};
  std::size_t atom_systems{};
  std::size_t atomic_numbers{};
  std::size_t positions{};
  std::size_t system_shell_offsets{};
  std::size_t shell_atoms{};
  std::size_t shell_angular{};
  std::size_t shell_ao_offsets{};
  std::size_t shell_direct_ao_offsets{};
  std::size_t shell_primitive_offsets{};
  std::size_t system_shell_pair_offsets{};
  std::size_t system_shell_quartet_offsets{};
  std::size_t system_shell_pair_block_offsets{};
  std::size_t system_shell_pair_block_quartet_offsets{};
  std::size_t shell_pair_systems{};
  std::size_t shell_pair_first{};
  std::size_t shell_pair_second{};
  std::size_t shell_pair_primitive_offsets{};
  std::size_t shell_primitive_pairs{};
  std::size_t psss_resident_tasks{};
  std::size_t psss_resident_ket_pairs{};
  std::size_t ao_shells{};
  std::size_t ao_term_counts{};
  std::size_t ao_term_angular{};
  std::size_t ao_term_coefficients{};
  std::size_t direct_ao_shells{};
  std::size_t direct_ao_angular{};
  std::size_t direct_ao_coefficients{};
  std::size_t ao_to_direct_transform{};
  std::size_t primitive_exponents{};
  std::size_t primitive_coefficients{};
  std::size_t occupied{};
  std::size_t warm_mask{};
  /** Per-item mixed-precision census: zero keeps that item in the FP64 lists. */
  std::size_t mixed_item_census{};
  std::size_t warm_density{};
  // Setup-only flags for rejecting an external warm density before graph
  // capture.  They are deliberately separate from `failed`, whose lifetime
  // spans the SCF graph and denotes numerical solver failures.
  std::size_t warm_invalid{};
  std::size_t overlap{};
  std::size_t hcore{};
  std::size_t eri{};
  std::size_t schwarz_bounds{};
  std::size_t direct_density{};
  std::size_t direct_fock{};
  std::size_t direct_transform_temporary{};
  std::size_t shell_pair_bounds{};
  std::size_t shell_pair_density_bounds{};
  std::size_t bounded_direct_shell_pair_order{};
  std::size_t bounded_stream_shell_pair_order{};
  std::size_t bounded_stream_pair_class_offsets{};
  std::size_t bounded_stream_topology{};
  std::size_t bounded_direct_shell_pair_block_bounds{};
  std::size_t bounded_direct_system_density_bounds{};
  std::size_t bounded_direct_system_pair_density_bounds{};
  std::size_t bounded_direct_generated_tasks{};
  std::size_t bounded_direct_generated_task_counts{};
  std::size_t bounded_direct_generated_task_offsets{};
  std::size_t bounded_direct_generated_retry_task_offsets{};
  std::size_t bounded_direct_generated_task_heads{};
  std::size_t bounded_direct_generated_overflow{};
  std::size_t bounded_direct_generated_retry_mask{};
  std::size_t bounded_direct_generated_retry_any{};
  std::size_t bounded_force_signature_counts{};
  std::size_t bounded_force_signature_offsets{};
  std::size_t bounded_force_signature_block_offsets{};
  std::size_t bounded_fock_class_timer_starts{};
  std::size_t bounded_fock_class_timer_elapsed{};
  std::size_t bounded_fock_class_timer_launches{};
  std::size_t bounded_fock_fp64_work_counts{};
  std::size_t bounded_fock_fp32_work_counts{};
  std::size_t active_shell_quartet_tile_offsets{};
  std::size_t active_shell_quartet_tile_counts{};
  std::size_t active_shell_quartet_tiles{};
  std::size_t fp32_shell_quartet_tile_offsets{};
  std::size_t fp32_shell_quartet_tile_counts{};
  std::size_t fp32_shell_quartet_tiles{};
  std::size_t shell_class_profile{};
  std::size_t persistent_fock_task_heads{};
  std::size_t fp32_persistent_fock_task_heads{};
  std::size_t persistent_force_task_heads{};
  std::size_t generated_shell_tasks{};
  std::size_t generated_shell_classes{};
  std::size_t generated_shell_task_offsets{};
  std::size_t generated_shell_task_counts{};
  std::size_t generated_shell_task_write_counts{};
  std::size_t generated_shell_task_heads{};
  std::size_t generated_low_order_signature_counts{};
  std::size_t generated_low_order_signature_offsets{};
  std::size_t generated_ppps_resident_tasks{};
  std::size_t generated_ppps_resident_bra_counts{};
  std::size_t generated_ppps_resident_bra_offsets{};
  std::size_t generated_ppps_resident_bra_write_counts{};
  std::size_t generated_ppps_resident_signature_counts{};
  std::size_t generated_ppps_resident_signature_offsets{};
  std::size_t generated_ppps_resident_signatures{};
  std::size_t generated_fock_shell_class_mask{};
  std::size_t generated_mixed_fock_shell_class_mask{};
  std::size_t generic_order5_tiles{};
  std::size_t generic_order5_tile_count{};
  std::size_t ao_pair_first{};
  std::size_t ao_pair_second{};
  std::size_t nuclear_repulsion{};
  std::size_t orthogonalizer{};
  std::size_t temporary{};
  std::size_t eigensystem{};
  std::size_t coefficients{};
  std::size_t eigenvalues{};
  std::size_t density{};
  std::size_t next_density{};
  std::size_t fock{};
  std::size_t residual{};
  std::size_t weighted_density{};
  std::size_t total_density{};
  std::size_t total_weighted_density{};
  std::size_t fock_history{};
  std::size_t residual_history{};
  std::size_t diis_linear_system{};
  std::size_t diis_coefficients{};
  std::size_t diis_count{};
  std::size_t diis_head{};
  std::size_t energy{};
  std::size_t previous_energy{};
  std::size_t energy_change{};
  std::size_t density_rms{};
  std::size_t forces{};
  std::size_t active{};
  std::size_t converged{};
  std::size_t failed{};
  std::size_t final_fock_reuse_mask{};
  std::size_t final_fock_rebuild_count{};
  std::size_t spin_active{};
  std::size_t iterations{};
  std::size_t solver_info{};
  std::size_t inactive_eigensolver_profile_count{};
  std::size_t inactive_eigensolver_profile{};
  std::size_t bounded_direct_cursor{};
};

/** Build offsets transactionally; publish a layout only if all sizes and alignment operations fit.
 */
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
                 bool mixed_precision_fock, ArenaLayout& layout);

/** Borrow an array from the already allocated and validated arena. */
template <typename T>
T* arena_pointer(void* arena, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(arena) + offset);
}

}  // namespace vibeqc::scf::cuda_execution
