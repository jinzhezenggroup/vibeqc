#ifndef VIBEQC_SCF_CUDA_BATCH_HPP
#define VIBEQC_SCF_CUDA_BATCH_HPP

#include "core/types.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc::scf {

struct RhfBucketItem {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  ScfResult scf;
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

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(
    const std::vector<core::System>& systems);

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling = false);

bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan* plan,
    CudaRhfShellClassProfile& profile) noexcept;

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept;

}  // namespace vibeqc::scf

#endif
