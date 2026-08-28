#ifndef VIBEQC_SCF_AOT_SHELL_REGISTRY_HPP
#define VIBEQC_SCF_AOT_SHELL_REGISTRY_HPP

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstddef>

namespace vibeqc::scf::generated {

/** Runtime identity of the AOT bundle selected for one CUDA device. */
struct ProfileInfo {
  const char* name;
  const char* target_architecture;
  int compute_capability_major;
  int compute_capability_minor;
  bool tuned;
  bool portable;
  bool compatible;
};

/** Launch and bucketing metadata scoped to the selected runtime profile. */
struct ShellKernelMetadata {
  const char* name;
  unsigned shell_class;
  unsigned angular_order;
  unsigned block_threads;
  unsigned consumer_mask;
  unsigned component_tile;
};

/** Cache profile selection for a device during CUDA context initialization. */
void select_profile_for_device(
    int device_id, int compute_capability_major,
    int compute_capability_minor) noexcept;

/** Return the profile selected for the current CUDA device. */
const ProfileInfo& selected_profile() noexcept;

/** Return force-kernel metadata for the selected profile. */
const ShellKernelMetadata* selected_shell_kernels(
    std::size_t& count) noexcept;

/** Return Fock-kernel metadata for the selected profile. */
const ShellKernelMetadata* selected_fock_shell_kernels(
    std::size_t& count) noexcept;

/** Return the enabled exact-class force mask for the selected profile. */
std::uint64_t enabled_shell_class_mask() noexcept;

/** Return the enabled exact-class Fock mask for the selected profile. */
std::uint64_t enabled_fock_shell_class_mask() noexcept;

/** Launch one generated persistent force worker by exact shell-class index. */
cudaError_t launch_shell_class(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/** Launch one generated persistent Fock worker by exact shell-class index. */
cudaError_t launch_shell_class_fock(
    unsigned shell_class, cudaStream_t stream, bool unrestricted,
    unsigned worker_blocks, const void* tasks,
    const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, const std::uint32_t* task_count,
    std::uint32_t* task_head) noexcept;

/**
 * Launch the optional canonical ppps resident-bra force worker.
 *
 * ``resident_tasks`` contains one descriptor per block.  Each descriptor
 * names a cached p-p bra pair and a contiguous range of grouped p-s ket
 * tasks.  The opaque pointers keep this public ABI independent of the
 * profile-scoped generated CUDA types; AOT wrappers validate their layout
 * against ``generated_shell_task.hpp`` before launching.  Implementations
 * return ``cudaErrorNotSupported`` when the selected profile has no resident
 * route, and ``cudaErrorInvalidValue`` when no profile is selected or the
 * descriptor count cannot be represented by a CUDA grid.
 */
cudaError_t launch_ppps_resident(
    cudaStream_t stream, bool unrestricted, const void* resident_tasks,
    const void* ket_tasks, const std::int64_t* primitive_pair_offsets,
    const void* primitive_pairs, const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    unsigned block_threads, std::size_t task_count) noexcept;

}  // namespace vibeqc::scf::generated

#endif
