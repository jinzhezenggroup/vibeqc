#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

#if VIBEQC_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

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
struct Vv10CudaDeviceLayout {
  std::size_t point_count{};
  std::size_t tile_points{};
  std::size_t workspace_bytes{};
  bool features{};
  bool geometry{};
};

/** Exact caller-owned workspace for resident CUDA VV10/rVV10 execution.
 * Input/output arrays and the numerical-error slot are caller-owned and are
 * not included in workspace_bytes.
 */
Vv10CudaDeviceLayout vv10_cuda_device_layout(std::size_t point_count, std::size_t tile_points,
                                             bool features, bool geometry);

/** Enqueue one resident fixed-grid evaluation on the caller stream.
 * All scientific inputs and outputs are device-resident. No allocation,
 * host transfer or synchronization occurs here. energy may alias workspace
 * because it is published only after all pair kernels have consumed scratch.
 */
void enqueue_vv10_cuda_device(const Vv10CudaDeviceLayout& layout, Vv10Parameters parameters,
                              int device_id, cudaStream_t stream, const double* coordinates,
                              const double* weights, const double* density,
                              const double* density_gradient, void* workspace,
                              std::size_t workspace_bytes, double* energy, double* vrho,
                              double* vsigma, double* point_derivative, double* weight_derivative,
                              int* numerical_error);

/** Apply the molecular fixed-grid domain policy without changing point extent.
 * Inputs are validated before screening. rho<threshold becomes zero weight,
 * rho=1 and grad-rho=0 so inactive points disappear from both pair domains
 * while local scales remain finite. No allocation, transfer or fence occurs.
 */
void enqueue_vv10_molecular_domain_cuda(cudaStream_t stream, std::size_t point_count,
                                        double density_threshold, const double* weights,
                                        const double* density, const double* density_gradient,
                                        double* effective_weights, double* effective_density,
                                        double* effective_density_gradient, int* numerical_error);

void execute_vv10_cuda(const double* coordinates, const double* weights, const double* density,
                       const double* density_gradient, std::size_t point_count,
                       std::size_t tile_points, Vv10Parameters parameters, int device_id,
                       double& energy, double* vrho, double* vsigma, double* point_derivative,
                       double* weight_derivative);
#endif

}  // namespace vibeqc::dft::nlc
