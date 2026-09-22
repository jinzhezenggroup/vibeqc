#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "core/types.hpp"
#include "dft/cosx_reference.hpp"

namespace vibeqc::dft {

struct CudaCosxStagingDiagnostic {
  std::size_t nbf{}, npoint{}, tile_points{};
  std::size_t grid_device_bytes{}, cosx_device_bytes{}, device_bytes{}, device_budget_bytes{};
  std::size_t esp_tile_elements{}, ao_tile_elements{};
  bool ao_on_device{}, esp_on_device{}, assembly_on_device{};
};

struct CudaCosxMolecularDerivativeDiagnostic {
  std::size_t nbf{}, npoint{}, natom{}, tile_points{};
  std::size_t grid_device_bytes{}, derivative_device_bytes{}, device_bytes{};
  std::size_t esp_tile_elements{}, ao_jet_elements{}, coordinate_elements{};
  bool bounded_tiling{}, atomic_coordinate_reduction{};
};

/** Pure resource estimate for the bounded native candidate. */
CudaCosxStagingDiagnostic cuda_cosx_staging_diagnostic(const core::System& system,
                                                       std::size_t npoint, std::size_t tile_points);

/** Bounded CUDA correctness primitive for the explicit-point COSX derivative.
 *
 * Returns point-major xyz dE_x/dR_point while density, quadrature weights,
 * Gaussian centers and basis data are held fixed. This is not a molecular
 * nuclear gradient and is intentionally not wired into SCF force dispatch.
 */
std::vector<double> cuda_cosx_point_derivative_reference(const core::System& system,
                                                         std::span<const double> points_xyz,
                                                         std::span<const double> weights,
                                                         std::span<const double> density,
                                                         CosxDensityConvention convention,
                                                         std::size_t tile_points, int device);

/** Pure resource estimate for the bounded complete molecular COSX derivative. */
CudaCosxMolecularDerivativeDiagnostic cuda_cosx_molecular_derivative_diagnostic(
    const MolecularGrid& grid, std::size_t tile_points);

/** Complete fixed-density molecular derivative of the materialized COSX model.
 *
 * The CUDA path contracts AO-center, ESP-center and owner-point responses
 * directly into 3*Natom coordinates per tile. Becke partition-weight motion is
 * contracted on the host from one scalar sensitivity per already-materialized
 * grid point. No coordinate-major ESP or point-derivative tensor is retained.
 */
std::vector<double> cuda_cosx_molecular_energy_derivative(const MolecularGrid& grid,
                                                          std::span<const double> density,
                                                          CosxDensityConvention convention,
                                                          std::size_t tile_points, int device);

class CudaCosxStagingPlan {
 public:
  CudaCosxStagingPlan(const core::System& system, std::span<const double> points_xyz,
                      std::span<const double> weights, std::size_t tile_points, int device,
                      std::size_t device_budget_bytes = 0);
  ~CudaCosxStagingPlan();
  CudaCosxStagingPlan(const CudaCosxStagingPlan&) = delete;
  CudaCosxStagingPlan& operator=(const CudaCosxStagingPlan&) = delete;

  CosxReferenceResult build(std::span<const double> density, CosxDensityConvention convention);
  const CudaCosxStagingDiagnostic& diagnostic() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vibeqc::dft
