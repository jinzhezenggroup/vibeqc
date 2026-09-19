#include <stdexcept>
#include <utility>

#include "dft/grid.hpp"
#include "scf/cuda_fock_provider.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/cuda_cosx.hpp"
#endif

namespace vibeqc::scf {
namespace {

#if VIBEQC_HAS_CUDA
dft::GridSpec grid_spec(const FockCosxSpec& spec) {
  dft::GridSpec grid;
  grid.version = spec.grid_version;
  grid.radial_points = spec.radial_points;
  grid.angular_polar = spec.angular_polar;
  grid.angular_azimuth = spec.angular_azimuth;
  grid.partition_iterations = spec.partition_iterations;
  grid.coincident_tolerance = spec.coincident_tolerance;
  grid.element_radii = spec.element_radii;
  return grid;
}

class CosxExchangeAdapter final : public CudaSeminumericalExchangeProvider {
 public:
  CosxExchangeAdapter(const core::System& system, const FockCosxSpec& spec, std::size_t tile_points,
                      int device, std::size_t max_device_bytes)
      : grid_(system, grid_spec(spec)),
        plan_(system, grid_.points(), grid_.weights(), tile_points, device, max_device_bytes) {
    const auto& source = plan_.diagnostic();
    diagnostic_.nbf = source.nbf;
    diagnostic_.ncoord = source.ncoord;
    diagnostic_.npoint = source.npoint;
    diagnostic_.tile_points = source.tile_points;
    diagnostic_.grid_device_bytes = source.grid_device_bytes;
    diagnostic_.exchange_device_bytes = source.cosx_device_bytes;
    diagnostic_.device_bytes = source.device_bytes;
    diagnostic_.esp_tile_elements = source.esp_tile_elements;
    diagnostic_.ao_tile_elements = source.ao_tile_elements;
    diagnostic_.ao_on_device = source.ao_on_device;
    diagnostic_.esp_on_device = source.esp_on_device;
    diagnostic_.assembly_on_device = source.assembly_on_device;
  }

  const CudaSeminumericalExchangeDiagnostic& diagnostic() const noexcept override {
    return diagnostic_;
  }

  std::vector<double> build_exchange(std::span<const double> density, bool spin_resolved) override {
    auto result = plan_.build(density, spin_resolved ? dft::CosxDensityConvention::spin_resolved
                                                     : dft::CosxDensityConvention::rhf_spin_summed);
    return std::move(result.exchange);
  }

 private:
  dft::MolecularGrid grid_;
  dft::CudaCosxStagingPlan plan_;
  CudaSeminumericalExchangeDiagnostic diagnostic_;
};
#endif

}  // namespace

std::unique_ptr<CudaSeminumericalExchangeProvider> make_cuda_seminumerical_exchange_provider(
    const core::System& system, const FockCosxSpec& spec, std::size_t tile_points, int device,
    std::size_t max_device_bytes) {
#if VIBEQC_HAS_CUDA
  return std::make_unique<CosxExchangeAdapter>(system, spec, tile_points, device, max_device_bytes);
#else
  (void)system;
  (void)spec;
  (void)tile_points;
  (void)device;
  (void)max_device_bytes;
  throw std::invalid_argument("CUDA seminumerical exchange provider is not built");
#endif
}

}  // namespace vibeqc::scf
