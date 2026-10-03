#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"
#include "scf/direct_block_domain.hpp"

namespace generativeqc::scf::cuda_execution {

struct ShellPairDensityBounds;

enum class DirectCoulombRange : std::uint32_t { Full = 0, Long = 1, Short = 2 };

/** A borrowed angular-bucket view; null pointers select dense canonical work. */
struct CanonicalPairRows {
  const std::int32_t* order{};
  const std::uint64_t* prefix{};
};

/** Query CUB's explicit sort/scan storage without allocating on the device. */
cudaError_t canonical_pair_workspace(int pairs, int segments, int largest_bucket,
                                     std::size_t& bytes);

/** Sort native FP64 Schwarz keys within each item/angular bucket on this stream. */
cudaError_t prepare_canonical_pair_order(cudaStream_t stream, DeviceBatch batch,
                                         const std::int32_t* pairs, const double* bounds,
                                         int pair_count, int segment_count,
                                         const int* segment_offsets, double* input_keys,
                                         std::int32_t* input_order, double* sorted_keys,
                                         std::int32_t* sorted_order, void* workspace,
                                         std::size_t workspace_bytes);

/** Prefix only quartets admitted by the original product comparison.
 * The same-bucket triangle is defined in the sorted order, preserving symmetry.
 */
cudaError_t prepare_canonical_pair_rows(cudaStream_t stream, const double* sorted_keys,
                                        std::size_t first_begin, std::size_t first_count,
                                        std::size_t second_begin, std::size_t second_count,
                                        bool same_bucket, double screening, std::uint64_t* prefix,
                                        void* workspace, std::size_t workspace_bytes);

// Shared one-output-owner reduction width, unchanged from the direct source.
constexpr unsigned kIndependentJkThreads = 32;

/** Device-only validity reduction for the allocation-free provider seam. */
void launch_independent_jk_finite_kernel(cudaStream_t stream, const double* values,
                                         std::size_t count, int* failure);

/** Exact unscreened public-AO ERI tile for prepared interaction-source consumers.
 * The output is row-major [i,j,k,l] with l fastest. Caller owns storage/stream.
 */
void launch_independent_eri_tile(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                 const std::array<std::size_t, 4>& begin,
                                 const std::array<std::size_t, 4>& count, std::size_t elements,
                                 double* eri);

/** Materialize one validated system's complete public-AO ERI tensor.
 * The caller owns an nbf^4 output and its stream. No cache is retained here;
 * partial AO tiles must use the independent rectangular tile producer.
 * The implementation shares the resident producer's compiler-owned schedule.
 */
void launch_build_eri_system_orbits(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                    std::size_t elements, double* eri);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_bounds_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, DeviceBatch batch, double* bounds,
                                         int* failure, bool cartesian = false);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, DeviceBatch batch, std::size_t system_begin,
                                  bool want_j, bool want_k, bool unrestricted, bool mixed_j,
                                  DirectCoulombRange exchange_range, double exchange_omega,
                                  double screening, const double* bounds, const double* density,
                                  const double* beta, double* j_out, double* ka_out, double* kb_out,
                                  std::uint64_t* mixed_coulomb_work_count);

/** Consume one angular-homogeneous block of symmetry-unique public-AO ERIs.
 * Inputs and outputs use the compiler's interleaved spin scatter ABI. No
 * four-index tensor or device-to-host staging is retained. */
void launch_canonical_jk_kernel(cudaStream_t stream, DeviceBatch batch, bool cartesian,
                                std::int32_t system, unsigned angular_order,
                                const std::int32_t* pairs, CanonicalPairRows rows,
                                std::size_t first_begin, std::size_t first_count,
                                std::size_t second_begin, std::size_t second_count,
                                bool same_bucket, bool want_j, bool want_k, bool unrestricted,
                                DirectCoulombRange exchange_range, double exchange_omega,
                                double screening, const double* bounds, const double* density,
                                double* coulomb, double* exchange, std::uint64_t* work_count);

/** Preserve the exact public-AO consumer launch and borrowed allocations. */
void launch_independent_jk_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, double cj, double ck,
    bool unrestricted, DirectCoulombRange exchange_range, double exchange_omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out);

/** Fuse Coulomb and split-range exchange derivatives in one quartet traversal.
 * Output is source-major [J, short-range K, long-range K].
 */
void launch_independent_rsh_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, std::size_t source_stride,
    double cj, double short_ck, double long_ck, bool unrestricted, double omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out);

/** Reuse the canonical geometry schedule and the existing RSH derivative algebra. */
void launch_canonical_rsh_derivative_kernel(cudaStream_t stream, DeviceBatch batch, bool cartesian,
                                            std::int32_t system, unsigned angular_order,
                                            const std::int32_t* pairs, CanonicalPairRows rows,
                                            std::size_t first_begin, std::size_t first_count,
                                            std::size_t second_begin, std::size_t second_count,
                                            bool same_bucket, std::size_t source_stride, double cj,
                                            double short_ck, double long_ck, bool unrestricted,
                                            double omega, double screening, const double* bounds,
                                            const double* density, const double* beta, double* out,
                                            std::uint64_t* work_count);

/** Provider-facing shell derivative seam. Queue/numerical ownership remains in
 * the Direct consumer layer; host source owners borrow only this launch ABI.
 * Output owns two total_atoms*3 channels, weighted Coulomb then exchange. */
void launch_bounded_shell_energy_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    double coulomb_coefficient, double exchange_coefficient,
    detail::BoundedDirectBlockDomain block_domain = {});

/** SR/LR exchange derivative through the same bounded shell scheduler.
 * The full-range Schwarz/density bounds remain conservative for both ranges. */
void launch_bounded_shell_range_exchange_derivative(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DirectCoulombRange range, double omega, double exchange_coefficient);

/** Fill one raw SR/LR positive-K source through the bounded shell dispatcher. */
void launch_bounded_shell_range_exchange_source(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* output, unsigned long long* cursor,
    DirectCoulombRange range, double omega);

/** Fill one raw full-range source for classes not covered by generated/native
 * consumers, reusing the Direct-HF bounded shell dispatcher. */
void launch_bounded_shell_fock_source(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    std::uint64_t covered_shell_class_mask, const std::uint32_t* class_state,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    unsigned long long* cursor, bool coulomb_only, bool exchange_only);

/** Fused [J', SR-K', LR-K'] over one screened shell traversal. */
void launch_bounded_shell_rsh_derivatives(
    bool unrestricted, unsigned worker_blocks, cudaStream_t stream, DeviceBatch batch,
    double screening, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint32_t* class_state, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* source_forces, unsigned long long* cursor, double omega,
    double coulomb_coefficient, double short_exchange_coefficient,
    double long_exchange_coefficient);

}  // namespace generativeqc::scf::cuda_execution
