#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_ks_final_state.hpp"
#include "dft/grid.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/types.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::dft {

/** Explicit component ownership for composition into #203. Provider/context
 * overhead and host quadrature preparation remain distinct from the native
 * iteration arena. No independent user-level memory allowance is introduced. */
struct CudaKsResources {
  std::size_t state_device_bytes{}, xc_device_bytes{}, provider_device_bytes{};
  std::size_t retained_host_numeric_bytes{};
};

/** Counts owned transport, not an estimate from the iteration count. One-
 * electron provider setup includes its existing explicit host export; its
 * preparation diagnostics are reported separately by PreparedFockPlan. */
struct CudaKsTransfers {
  std::uint64_t setup_h2d_bytes{}, density_h2d_bytes{}, scalar_d2h_bytes{}, matrix_d2h_bytes{};
  std::uint64_t final_state_d2h_bytes{}, final_state_reads{};
  std::uint64_t synchronizations{}, iterations{};
  /** Number of subsequent proposals using the CPU-compatible stationary-cycle
   * shift; cumulative across replays, independent of transfer counts. */
  std::uint64_t occupation_stabilized_proposals{};
};

/** Exact state-arena size from the allocator's own typed layout. This query
 * performs no CUDA call and allocates no matrices or other numeric buffers. */
std::size_t cuda_ks_state_bytes(std::size_t nao, unsigned spins, unsigned diis_history);

/** Native ordinary-stream LDA/PBE RKS/UKS trajectory. The borrowed common
 * Fock plan must outlive it. Model/grid/functional identity is immutable;
 * changing it requires a new owner. Initial guesses/normalization and grid
 * preparation are explicit host setup, with no CPU XC or matrix export in
 * an iteration. Final output is a separate, measured operation.
 *
 * Split enqueue/finish operations let a native ragged batch enqueue all
 * active item streams before reading their small scalar records. Each owner
 * isolates pending/active/failed/converged and last-good warm states. */
class CudaKsPlan {
 public:
  CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis, const MolecularGrid& grid,
             const scf::ScfOptions& options, bool pbe, std::size_t tile_points = 256);
  ~CudaKsPlan();
  CudaKsPlan(const CudaKsPlan&) = delete;
  CudaKsPlan& operator=(const CudaKsPlan&) = delete;

  /** Start fresh DIIS/history. A null seed reuses a compatible last-good
   * device density when requested; explicit seeds are normalized per spin. */
  void begin(const std::vector<double>* initial_density = nullptr, bool reuse_warm = true);
  bool active() const noexcept;
  bool pending() const noexcept;
  bool failed() const noexcept;
  void enqueue_iteration();
  /** Resolve a submitted iteration; returns true while another is needed. */
  bool finish_iteration();
  /** Terminal result; density export is optional and never used in an iteration. */
  scf::ScfResult result(bool export_density = true);
  /** Energy-only adapters leave the final density resident by disabling export. */
  scf::ScfResult run(const std::vector<double>* initial_density = nullptr, bool reuse_warm = true,
                     bool export_density = true);
  /** Export only the last converged state for a changed-geometry rebuild. */
  std::vector<double> warm_density();
  /** Freeze replacement without disabling reuse of the last-good density. */
  void set_warm_start_updates(bool enabled) noexcept;
  /** Forget the seed without downloading or changing the current result. */
  void clear_warm_start() noexcept;
  /** Revoke final-state eligibility without changing warm-start ownership. */
  void invalidate_final_state() noexcept;
  /** Read-only host eligibility query. It performs no CUDA call or transfer. */
  vibeqc_status final_state_token(CudaKsFinalStateToken& token, std::string& detail) const;
  /** Export a detached, strictly validated current physical state. Exact-token
   * comparison
   * precedes transfer; eligibility is rechecked before publication.
   * W is built only when
   * explicitly requested. */
  vibeqc_status read_final_state(const CudaKsFinalStateToken& expected,
                                 bool compute_weighted_density, VerifiedKsFinalState& state,
                                 std::string& detail);
  const CudaKsResources& resources() const noexcept;
  CudaKsTransfers transfers() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vibeqc::dft
