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
