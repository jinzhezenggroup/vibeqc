#ifndef VIBEQC_SCF_FOCK_PREPARED_HPP
#define VIBEQC_SCF_FOCK_PREPARED_HPP

#include <memory>

#include "scf/cuda_fock_provider.hpp"

namespace vibeqc::scf {

/** Backend variants frozen into the prepared source; mathematical identity
 * remains in ResolvedFockBuild::spec. These switches never authorize DF. */
struct FockExecutionVariant {
  unsigned one_electron_value_mapping{}, df_value_mapping{}, df_derivative_mapping{};
  bool operator==(const FockExecutionVariant&) const = default;
};

/** Source preparation diagnostics, separate from the semantic request and
 * subsequent SCF convergence controls. Counts describe explicit native
 * buffers; CUDA modules, driver/library-private storage and recurrence stacks
 * are outside these counts. A source-backed DF plan retains bounded tiles.
 */
struct FockPreparationDiagnostic {
  ResolvedFockBuild strategy;
  std::size_t nbf{}, ncoord{}, device_bytes{}, device_budget_bytes{};
  CudaDirectJkDiagnostic direct;
  FockExecutionVariant variant;
  CudaDensityFittingSourceDiagnostic fitted_source;
  std::vector<CudaDensityFittingMetricDiagnostic> fitted;
};

/** Immutable geometry/semantic owner for the common Fock provider views.
 * CPU numerical sources use the existing reference integrals; CUDA uses
 * the existing direct evaluator and source-backed DF tile generator. The
 * owner also supplies one-electron data to mean-field consumers, avoiding
 * duplicated preparation in SCF and fixed-density endpoints. It is not
 * concurrently reentrant because CUDA scratch and streams are shared.
 *
 * A positive device budget bounds explicit provider buffers; zero selects
 * the same 256 MiB allowance as the independent CUDA SCF route. CPU storage
 * retains its existing reference representation and outer resource policy.
 */
class PreparedFockPlan {
 public:
  PreparedFockPlan(const core::System& orbital, const core::System* auxiliary,
                   ResolvedFockBuild strategy, int device_id = -1,
                   std::size_t device_budget_bytes = 0);
  ~PreparedFockPlan();
  PreparedFockPlan(const PreparedFockPlan&) = delete;
  PreparedFockPlan& operator=(const PreparedFockPlan&) = delete;

  const ResolvedFockBuild& strategy() const noexcept;
  const core::System& system() const noexcept;
  const integrals::IntegralData& one_electron() const noexcept;
  /** CPU accounting borrows the actual DF owner, including its AO data. */
  const DensityFittingScfData* cpu_fitted_data() const noexcept;
  const FockPreparationDiagnostic& diagnostic() const noexcept;
  DirectJkMatrices build(const std::vector<double>& density,
                         const std::vector<double>& beta = {}) const;
  std::vector<double> energy_derivative(const std::vector<double>& density,
                                        const std::vector<double>& beta = {}) const;
  /** Exact comparison of immutable source inputs and execution controls.
   * Convergence thresholds, DIIS history and warm density are deliberately
   * excluded: they do not alter the prepared mathematical operator. */
  bool matches(const core::System& orbital, const core::System* auxiliary,
               const ResolvedFockBuild& strategy, int device_id,
               std::size_t device_budget_bytes) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vibeqc::scf
#endif
