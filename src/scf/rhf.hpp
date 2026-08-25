#ifndef QCE_SCF_RHF_HPP
#define QCE_SCF_RHF_HPP

#include "core/types.hpp"
#include "scf/direct_task_layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qce::scf {

/**
 * Run closed-shell RHF and assemble its variational analytic gradient.
 *
 * `initial_density`, when non-null, must contain an AO density matrix for the
 * same basis topology. The solver copies it and never mutates caller storage.
 */
core::ScfResult run_rhf(const core::System& system,
                        const core::ScfOptions& options,
                        const std::vector<double>* initial_density = nullptr);

/**
 * Run spin-unrestricted HF and assemble its variational analytic gradient.
 *
 * The retained warm-start density is packed as alpha then beta AO matrices.
 * Multiplicity defines N_alpha - N_beta = multiplicity - 1.
 */
core::ScfResult run_uhf(const core::System& system,
                        const core::ScfOptions& options,
                        const std::vector<double>* initial_density = nullptr);

/**
 * Run RHF through the native CUDA scientific path.
 *
 * Geometry-dependent integrals, SCF matrices, eigensolves, convergence state,
 * final energy, and analytic forces remain device resident until publication.
 * The current validated envelope is the same contracted Cartesian s-p-d-f basis
 * accepted by the public basis validator; the CPU implementation remains an
 * independent numerical oracle rather than an implicit CUDA fallback.
 */
core::ScfResult run_rhf_cuda(const core::System& system,
                             const core::ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density = nullptr);

/** Execute UHF through the native CUDA scientific path. */
core::ScfResult run_uhf_cuda(const core::System& system,
                             const core::ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density = nullptr);

struct RhfBucketItem {
  qce_status status{QCE_STATUS_INTERNAL_ERROR};
  core::ScfResult scf;
};

struct CudaRhfBucketPlan;

/** Final-density work retained by CUDA direct screening for one shell class. */
struct CudaRhfShellClassProfileEntry {
  std::uint64_t shell_quartets{};
  std::uint64_t tiles{};
  std::uint64_t ao_quartets{};
  std::uint64_t primitive_quartets{};
};

using CudaRhfShellClassProfile =
    std::array<CudaRhfShellClassProfileEntry,
               detail::kDirectQuartetShellClassCount>;

/** Internal diagnostics for the immutable CUDA basis topology layout. */
struct CudaRhfBasisLayoutStats {
  std::size_t system_count{};
  std::size_t shell_count{};
  std::size_t shell_pair_count{};
  std::size_t shell_quartet_count{};
  std::size_t ao_count{};
  std::size_t unique_primitive_count{};
  std::size_t expanded_primitive_references{};
  std::size_t device_basis_bytes{};
};

/**
 * Inspect the basis packing used by CUDA fixed-topology plans.
 *
 * This is an internal diagnostic API, not part of the stable public C ABI.
 * It performs no GPU allocation and is used to prove that contracted
 * primitives are retained once per shell while public AOs use bounded sparse
 * component metadata and direct J/K owns one Cartesian-source transform.
 */
CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(
    const std::vector<core::System>& systems);

/**
 * Execute compatible RHF systems through one device-resident CUDA batch.
 * Systems share a matrix-size bucket but retain independent convergence and
 * failure state.
 */
std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

/** Execute through a reusable fixed-topology CUDA allocation/Graph owner. */
std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

/** Copy the most recent profile owned by a reusable CUDA bucket plan. */
bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan* plan,
    CudaRhfShellClassProfile& profile) noexcept;

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept;

}  // namespace qce::scf

#endif
