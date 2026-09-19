#pragma once

#include <cstddef>
#include <memory>
#include <span>

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
