#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::dft::nlc {

enum class Vv10Variant : std::int32_t { vv10 = 1, rvv10 = 2 };

struct Vv10Parameters {
  Vv10Variant variant{Vv10Variant::vv10};
  double b{};
  double c{};
  double coefficient{1.0};
  bool operator==(const Vv10Parameters&) const = default;
};

struct Vv10ResourceUsage {
  std::uint64_t workspace_bytes{};
  std::uint64_t host_workspace_bytes{};
  std::uint64_t device_workspace_bytes{};
  std::uint64_t maximum_bytes{};
  std::uint64_t pair_evaluations{};
  std::uint64_t tiles{};
  std::uint32_t point_count{};
  std::uint32_t tile_points{};
};

class Vv10Plan {
 public:
  static std::unique_ptr<Vv10Plan> prepare(vibeqc_backend backend, int device_id,
                                           std::uint32_t point_count, std::uint32_t tile_points,
                                           Vv10Parameters parameters, std::uint64_t maximum_bytes,
                                           std::string& detail, vibeqc_status& status);

  [[nodiscard]] vibeqc_backend backend() const noexcept { return backend_; }
  [[nodiscard]] int device_id() const noexcept { return device_id_; }
  [[nodiscard]] const Vv10Parameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] const Vv10ResourceUsage& resources() const noexcept { return resources_; }

  vibeqc_status execute(std::span<const double> coordinates, std::span<const double> weights,
                        std::span<const double> density, std::span<const double> density_gradient,
                        double& energy, std::span<double> vrho, std::span<double> vsigma,
                        std::span<double> point_derivative, std::span<double> weight_derivative,
                        std::string& detail);

 private:
  Vv10Plan(vibeqc_backend backend, int device_id, Vv10Parameters parameters,
           Vv10ResourceUsage resources)
      : backend_(backend), device_id_(device_id), parameters_(parameters), resources_(resources) {}

  vibeqc_backend backend_{};
  int device_id_{};
  Vv10Parameters parameters_{};
  Vv10ResourceUsage resources_{};
  std::vector<double> omega_;
  std::vector<double> kappa_;
  std::vector<double> weighted_density_;
  std::vector<double> domega_drho_;
  std::vector<double> domega_dsigma_;
  std::vector<double> dkappa_drho_;
};

#if VIBEQC_HAS_CUDA
void execute_vv10_cuda(const double* coordinates, const double* weights, const double* density,
                       const double* density_gradient, std::size_t point_count,
                       std::size_t tile_points, Vv10Parameters parameters, int device_id,
                       double& energy, double* vrho, double* vsigma, double* point_derivative,
                       double* weight_derivative);
#endif

}  // namespace vibeqc::dft::nlc
