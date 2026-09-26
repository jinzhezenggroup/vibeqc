#ifndef VIBEQC_SCF_CUDA_FOCK_PROVIDER_HPP
#define VIBEQC_SCF_CUDA_FOCK_PROVIDER_HPP

#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/fock_provider.hpp"

namespace vibeqc::scf {

/** Borrow one item in an existing direct or DF CUDA plan. The enclosing
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
  const DensityFittingScfData* data_{};
  std::size_t item_{};
};

/** CPU and CUDA providers share all selection, preflight, and pair composition
 * rules. Backend-specific views wrap existing numerical implementations. */
using CudaFockPlanView = BasicFockPlanView<CudaFockProviderView>;

}  // namespace vibeqc::scf
#endif
