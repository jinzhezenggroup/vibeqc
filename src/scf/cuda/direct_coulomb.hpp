#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/direct_block_schedule.hpp"

namespace generativeqc::scf::cuda_execution {

struct ShellPairDensityBounds;

/** Optional geometry owner for the generated pure-J consumer. It borrows the
 * direct provider's stream and public basis metadata, and owns bounded shell
 * topology, Cartesian transforms and scratch. No quartet list is materialized.
 */
struct GeneratedCoulombPlan {
  DeviceBatch batch{};
  cudaStream_t stream{};
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  std::uint64_t class_mask{}, value_class_mask{};
  /** True only when generated/native streaming value consumers cover every
   * present shell class without the bounded higher-l fallback. */
  bool value_capability{true};
  unsigned worker_blocks{};
  double screening{};
  double *density{}, *coulomb{}, *temporary{}, *total_density{}, *zero{}, *schwarz{},
      *shell_bounds{};
  std::uint8_t* active{};
  std::uint32_t* heads{};
  const std::uint32_t* pair_order{};
  const std::uint32_t* pair_class_offsets{};
  GeneratedShellPairStream* topology{};
  ~GeneratedCoulombPlan();
};

/** Unsupported angular classes or insufficient optional capacity return null.
 * CUDA execution failures propagate; only allocation failure selects fallback.
 */
std::unique_ptr<GeneratedCoulombPlan> prepare_generated_coulomb(
    const HostBatch& host, DeviceBatch borrowed, cudaStream_t stream, int device, double screening,
    std::size_t budget, bool allow_bounded_shell_fallback = false,
    detail::BoundedDirectHostSchedule* bounded_schedule = nullptr);

/** Enqueue raw J from total spin density. Inputs and result use public AO order.
 * The same stream owns every transform, scatter and projection; no host copies.
 */
cudaError_t enqueue_generated_coulomb(GeneratedCoulombPlan& plan, const double* density,
                                      const double* beta, double* coulomb);

/** Optional raw-K owner layered on the generated-J geometry/topology owner.
 * Direct CUDA plans prefer complete generated/native value coverage when the
 * optional device budget admits it. Bounded through-f values require an
 * explicit qualification opt-in on CudaDirectJkPlan. Density screening uses
 * the same shell-pair reductions as Direct HF. The bounded lease supports
 * strict-FP64 SR/LR value K; generated streaming classes remain full-range.
 */
struct GeneratedExchangePlan {
  std::unique_ptr<GeneratedCoulombPlan> shared;
  std::vector<void*> allocations;
  std::size_t device_bytes{}, host_preparation_bytes{};
  double *public_spin{}, *direct_spin{}, *direct_exchange{};
  double *density_temporary{}, *fock_temporary{}, *public_exchange{};
  ShellPairDensityBounds* shell_pair_density_bounds{};
  double *system_density_bounds{}, *system_pair_density_bounds{};
  std::uint32_t* heads{};
  GeneratedShellPairStream* topology{};
  // Optional bounded Direct-HF lease. Value fallback and stationary forces
  // share immutable shell topology, screening metadata and one cursor.
  bool force_capability{}, bounded_value_capability{};
  const std::uint32_t* bounded_pair_order{};
  /** Optional geometry-live row index; owned by allocations, never by a call. */
  detail::BoundedDirectBlockDomain bounded_block_domain{};
  std::uint32_t* bounded_value_overflow{};
  double *shell_pair_block_bounds{}, *force{};
  unsigned long long* force_cursor{};
  ~GeneratedExchangePlan();
};

/** Prepare the generated J+full-range-K owner within one explicit budget.
 * Through-f callers may retain HF's bounded shell fallback for value and force
 * classes not owned by generated/native streaming consumers. */
std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange(
    const HostBatch& host, DeviceBatch borrowed, cudaStream_t stream, int device, double screening,
    std::size_t budget, bool force_capability = false, bool allow_bounded_shell_fallback = false);

/** Enqueue a complete raw J through generated/native classes plus the bounded
 * higher-l shell fallback owned by the exchange plan. */
cudaError_t enqueue_generated_coulomb(GeneratedExchangePlan& plan, bool unrestricted,
                                      const double* alpha, const double* beta, double* coulomb);

/** Enqueue positive raw K in public AO order. UHF returns independent alpha/beta
 * matrices. The caller owns output buffers on the same device/stream.
 */
cudaError_t enqueue_generated_exchange(GeneratedExchangePlan& plan, bool unrestricted,
                                       const double* alpha, const double* beta,
                                       double* alpha_exchange, double* beta_exchange,
                                       DirectCoulombRange range = DirectCoulombRange::Full,
                                       double omega = 0.0);

/** Enqueue primary full-range J/K plus one SR/LR correction after a
 * single public-to-Cartesian spin-density transform and shell-density-bound
 * reduction. Through-f bounded shell value capability is required. */
cudaError_t enqueue_generated_rsh_values(GeneratedExchangePlan& plan, bool unrestricted,
                                         const double* alpha, const double* beta, double* coulomb,
                                         double* full_alpha_exchange, double* full_beta_exchange,
                                         double* range_alpha_exchange, double* range_beta_exchange,
                                         DirectCoulombRange range, double omega);

/** Execute separate full-range J' and K' fixed-density energy derivatives
 * through the retained shell topology. Output is source-major [J,K], each
 * containing 3*atom_count energy-gradient values. The public AO density stays
 * resident; this routine transforms it to the owner's Cartesian basis once. */
cudaError_t execute_generated_full_range_energy_derivatives(
    GeneratedExchangePlan& plan, bool unrestricted, const double* alpha, const double* beta,
    double coulomb_coefficient, double exchange_coefficient, std::vector<double>& derivatives);

/** Stationary RSH sources [J(full), K(short), K(long)] through one retained
 * shell owner, one public-to-Cartesian density transform and one bounded shell
 * traversal. */
cudaError_t execute_generated_rsh_energy_derivatives(GeneratedExchangePlan& plan, bool unrestricted,
                                                     const double* alpha, const double* beta,
                                                     double coulomb_coefficient,
                                                     double short_exchange_coefficient,
                                                     double long_exchange_coefficient, double omega,
                                                     std::vector<double>& derivatives);

}  // namespace generativeqc::scf::cuda_execution
