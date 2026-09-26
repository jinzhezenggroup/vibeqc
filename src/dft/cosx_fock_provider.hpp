#ifndef VIBEQC_DFT_COSX_FOCK_PROVIDER_HPP
#define VIBEQC_DFT_COSX_FOCK_PROVIDER_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "dft/cuda_cosx.hpp"
#include "dft/grid.hpp"
#include "scf/fock_prepared.hpp"

namespace vibeqc::dft {

struct CosxFockPreparationDiagnostic {
  scf::ResolvedFockBuild strategy{};
  scf::FockPreparationDiagnostic coulomb{};
  CudaCosxStagingDiagnostic exchange{};
  CudaCosxMolecularDerivativeDiagnostic derivative{};
  std::size_t device_bytes{}, derivative_peak_device_bytes{}, device_budget_bytes{}, tile_points{};
};

/** DFT-owned prepared composition of an existing J provider and CUDA COSX K.
 *
 * The mixed ResolvedFockBuild remains the mathematical authority. The SCF
 * layer never imports DFT/COSX implementation details: this adapter depends
 * downward on PreparedFockPlan for J and returns the shared DirectJkMatrices
 * consumed by ordinary Fock/energy assembly.
 */
class PreparedCosxFockPlan {
 public:
  PreparedCosxFockPlan(const core::System& orbital, const core::System* auxiliary,
                       scf::ResolvedFockBuild strategy, std::size_t tile_points, int device_id,
                       std::size_t device_budget_bytes = 0);
  ~PreparedCosxFockPlan();
  PreparedCosxFockPlan(const PreparedCosxFockPlan&) = delete;
  PreparedCosxFockPlan& operator=(const PreparedCosxFockPlan&) = delete;

  const scf::ResolvedFockBuild& strategy() const noexcept;
  const core::System& system() const noexcept;
  const integrals::IntegralData& one_electron() const noexcept;
  const MolecularGrid& grid() const noexcept;
  const CosxFockPreparationDiagnostic& diagnostic() const noexcept;

  scf::DirectJkMatrices build(const std::vector<double>& density,
                              const std::vector<double>& beta = {});
  /** Fixed-density two-electron molecular gradient using exactly the prepared
   * RI-J/COSX-K semantics. One-electron/Pulay/nuclear terms remain method-owned. */
  std::vector<double> energy_derivative(const std::vector<double>& density,
                                        const std::vector<double>& beta = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vibeqc::dft

#endif
