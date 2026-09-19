#ifndef VIBEQC_SCF_CUDA_FOCK_PROVIDER_HPP
#define VIBEQC_SCF_CUDA_FOCK_PROVIDER_HPP

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "core/types.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/fock_provider.hpp"

namespace vibeqc::scf {

/** Method-neutral contract for a prepared CUDA seminumerical exchange owner.
 * DFT/COSX code owns grid construction and its concrete CUDA staging plan; the
 * shared SCF layer sees only raw exchange matrices and explicit resource facts.
 */
struct CudaSeminumericalExchangeDiagnostic {
  std::size_t nbf{}, ncoord{}, npoint{}, tile_points{};
  std::size_t grid_device_bytes{}, exchange_device_bytes{}, device_bytes{};
  std::size_t esp_tile_elements{}, ao_tile_elements{};
  bool ao_on_device{}, esp_on_device{}, assembly_on_device{};
};

class CudaSeminumericalExchangeProvider {
 public:
  virtual ~CudaSeminumericalExchangeProvider() = default;
  virtual const CudaSeminumericalExchangeDiagnostic& diagnostic() const noexcept = 0;
  virtual std::vector<double> build_exchange(std::span<const double> density,
                                             bool spin_resolved) = 0;
};

/** Construct the concrete COSX adapter without importing DFT implementation
 * types into the common SCF prepared-owner boundary. */
std::unique_ptr<CudaSeminumericalExchangeProvider> make_cuda_seminumerical_exchange_provider(
    const core::System& system, const FockCosxSpec& spec, std::size_t tile_points, int device,
    std::size_t max_device_bytes);

/** Borrow one item in an existing direct, DF, or COSX CUDA plan. The enclosing
 * geometry cache owns every handle and immutable DF data object and must
 * outlive this view. Item identity includes its plan, batch index, and data.
 * Rebinding is required after geometry/basis/cutoff changes; the view owns
 * neither integrals nor a separate cache. Calls on a shared plan serialize.
 *
 * Raw contractions accept row-major nonsymmetric densities. The generated
 * DF response currently requires symmetric densities, preflighted for both
 * providers before either derivative executes. Only requested matrices and
 * compact gradients cross back to the host.
 */
class CudaFockProviderView {
 public:
  static constexpr FockBackend backend = FockBackend::Cuda;
  explicit CudaFockProviderView(CudaDirectJkPlan* exact, std::size_t item = 0);
  CudaFockProviderView(CudaDensityFittingJkPlan* fitted, const DensityFittingScfData& data,
                       std::size_t item = 0);
  CudaFockProviderView(CudaDensityFittingJkPlan*, DensityFittingScfData&&,
                       std::size_t = 0) = delete;
  explicit CudaFockProviderView(CudaSeminumericalExchangeProvider* exchange);
  FockApproximation approximation() const;
  std::size_t nbf() const;
  std::size_t ncoord() const;
  bool operator==(const CudaFockProviderView&) const = default;

 private:
  template <class>
  friend class BasicFockPlanView;
  void validate(const ResolvedFockBuild& strategy) const;
  void validate_density(const std::vector<double>& density, const std::vector<double>& beta,
                        bool derivative) const;
  DirectJkMatrices build(FockBuildSpec spec, const std::vector<double>& density,
                         const std::vector<double>& beta) const;
  std::vector<double> derivative(FockBuildSpec spec, const std::vector<double>& density,
                                 const std::vector<double>& beta) const;
  CudaDirectJkPlan* exact_{};
  CudaDensityFittingJkPlan* fitted_{};
  CudaSeminumericalExchangeProvider* seminumerical_exchange_{};
  const DensityFittingScfData* data_{};
  std::size_t item_{};
};

/** CPU and CUDA providers share all selection, preflight, and pair composition
 * rules. Backend-specific views wrap existing numerical implementations. */
using CudaFockPlanView = BasicFockPlanView<CudaFockProviderView>;

}  // namespace vibeqc::scf
#endif
