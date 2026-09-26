#ifndef VIBEQC_XTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define VIBEQC_XTB_RUNTIME_GFN2_CPU_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpu_dispatch/features.hpp"
#include "runtime/types.hpp"

namespace vibeqc::xtb::detail {

struct Gfn2CpuOrbitalSnapshot {
  std::int64_t orbital_count = 0;
  double electron_count = 0.0;
  double alpha_electron_count = 0.0;
  double beta_electron_count = 0.0;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_primitive_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::uint8_t> angular_momenta;
  std::vector<double> primitive_exponents;
  std::vector<double> primitive_coefficients;
  std::vector<double> overlap;
  std::vector<double> coefficients;
  std::vector<double> occupations;
};

// Retains molecular topology and numerical workspaces across synchronous calls.
// Every call resets SCC from the SAD state; VibeQC has no public warm-start API.
class Gfn2CpuExecutionCache {
 public:
  explicit Gfn2CpuExecutionCache(CpuIsa cpu_isa = CpuIsa::kBaseline);
  ~Gfn2CpuExecutionCache();

  Gfn2CpuExecutionCache(const Gfn2CpuExecutionCache&) = delete;
  Gfn2CpuExecutionCache& operator=(const Gfn2CpuExecutionCache&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend vibeqc_xtb_status_t execute_restricted_gfn2_cpu(
      Gfn2CpuExecutionCache& cache, const vibeqc_xtb_batch_t& batch,
      const vibeqc_xtb_compute_options_t& options, vibeqc_xtb_batch_result_t& result,
      std::string& error);
  friend vibeqc_xtb_status_t copy_restricted_gfn2_orbital_snapshot_cpu(
      Gfn2CpuExecutionCache& cache, Gfn2CpuOrbitalSnapshot& snapshot, std::string& error);
};

/*
 * Execute one already descriptor-validated host request.
 *
 * Inputs are copied before numerical work so under-aligned C buffers remain
 * well-defined and caller mutations cannot race an in-flight synchronous
 * call. Requested outputs and result flags are committed only after every
 * batch member reaches either a successful or documented terminal state.
 */
vibeqc_xtb_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                                const vibeqc_xtb_batch_t& batch,
                                                const vibeqc_xtb_compute_options_t& options,
                                                vibeqc_xtb_batch_result_t& result,
                                                std::string& error);

/* Copy the last converged one-system restricted GFN2 orbital state.
 * This is an internal bridge for SCF initialization, not a public xTB result
 * descriptor. Coefficients remain in the native GFN2 spherical AO order. */
vibeqc_xtb_status_t copy_restricted_gfn2_orbital_snapshot_cpu(
    Gfn2CpuExecutionCache& cache, Gfn2CpuOrbitalSnapshot& snapshot, std::string& error);

}  // namespace vibeqc::xtb::detail

#endif  // VIBEQC_XTB_RUNTIME_GFN2_CPU_EXECUTION_HPP
