#ifndef VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#define VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "runtime/strided_range.hpp"
#include "scf/df_response_weights.hpp"
namespace vibeqc::scf {
struct CudaDensityFittingIntegralSource;
struct CudaDensityFittingJkPlan;
/** Bind the prepared owner's immutable source after copying that source into
 * a new single-system plan. The caller owns these host arrays for the plan's
 * lifetime; geometry/basis changes create a new owner. Arbitrary tensor-plan
 * callers do not invoke this contract and keep the explicit upload adapter.
 */
void bind_cuda_density_fitting_response_source(CudaDensityFittingJkPlan* plan,
                                               const core::System& orbital,
                                               const core::System& auxiliary,
                                               std::span<const double> raw) noexcept;
/** Borrowed device factors for one fixed-geometry metric. Eigenvectors use
 * cuSOLVER column-major order; X=M^(-1/2) is symmetric. All pointers and the
 * retained cutoff belong to the same plan and outlive its owning stream work.
 */
struct CudaDfMetricView {
  const double* inverse_square_root{};
  const double* eigenvectors{};
  const double* eigenvalues{};
  double relative_threshold{};
};
/** Borrow of one canonical column-major C, with D=density_scale*C*C^T.
 * The plan owner validates the method token, exact density and device
 * generation before constructing this view. A zero-rank spin has no pointer.
 */
struct CudaDfOccupiedResponseFactor {
  const double* coefficients{};
  std::size_t rank{};
  double density_scale{};
};
/** Immutable FP64 raw [Q,mu,nu] view borrowed from one resident value plan.
 * Strides are in doubles. owner_identity is the process-unique immutable
 * geometry/orbital/auxiliary/metric identity also used for occupied factors;
 * rebuilding that owner invalidates the view even if addresses are recycled.
 * Construction also checks the prepared source's immutable raw/atom/shell
 * allocation bindings and orbital/auxiliary representation identities.
 * The metric eigensystem and cutoff record the forward rank policy, but raw
 * values include ALL metric directions. The pointer is plan-owned, live until
 * stream drain, and never denotes a streamed panel or transformed B. No new
 * allocation is associated with this view.
 */
struct CudaDfRawTensorView {
  const double* data{};
  std::size_t nbf{}, naux{}, auxiliary_stride{}, row_stride{}, column_stride{1};
  std::uint64_t owner_identity{};
  CudaDfMetricView metric{};
};
/** Exclusive, stream-ordered borrow from the existing J/K tensor allocation.
 * Two buffers remain mutable response scratch. The third is immutable when
 * resident_raw is present; otherwise it receives the explicit host upload.
 * The owner validates capacity and provenance before lending distinct buffers.
 * The plan's stream orders the last J/K use, force, and next SCF use, and the
 * synchronous bridge drains on success and failure before releasing the borrow.
 */
struct CudaDfResponseBuffers {
  double* staging_weights{};
  double* raw_auxiliary_major{};
  double* exchange_response{};
  std::size_t elements_per_buffer{};
  bool occupied_response{};
  std::array<CudaDfOccupiedResponseFactor, 3> occupied_factors{};
  CudaDfRawTensorView resident_raw{};
  bool batch_products{true};
  // Optional validated full-rank RHF U[mu,i,Q], Q contiguous. It aliases
  // staging_weights and is consumed before that allocation becomes mutable
  // response storage. occupied_factors[0] is its exact final canonical C.
  const double* final_occupied_projection{};
};
/** Owned numeric staging and explicit transfers, excluding caller weights/system data. */
struct DfGradientResources {
  std::size_t host_bytes{}, device_bytes{}, host_to_device_bytes{}, device_to_host_bytes{};
  std::size_t weight_tile_elements{}, tiles{}, uploads{}, stream_synchronizations{};
  std::size_t value_slices{}, auxiliary_weight_tile{};
  /** Bulk tensors and response weights, separately from metadata/density/final results. */
  std::size_t tensor_host_to_device_bytes{}, tensor_device_to_host_bytes{};
  std::size_t response_host_to_device_bytes{}, density_host_to_device_bytes{};
  std::size_t recomputed_value_bytes{}, device_response_bytes{};
  /** Already charged to the value plan, never added again to owned device_bytes. */
  std::size_t borrowed_device_bytes{};
  bool device_response{};
  /** True only after the owner validates and executes occupied response. */
  bool occupied_response{};
};
/** Synchronous bounded bridge for fixed full A[mu,nu,P] and M[P,Q] weights.
 * Orbital/auxiliary bases share the same physical atom coordinates but may
 * attach their shells to distinct atoms. Missing weight spans mean zero.
 * Each domain is tiled independently; maximum_bytes bounds host numeric
 * staging and device allocations separately. Output changes only on success.
 */
vibeqc_status execute_cuda_df_gradient(int device, const core::System& orbital,
                                       const core::System& auxiliary, std::span<const double> bar_a,
                                       std::span<const double> bar_m, unsigned schedule,
                                       std::size_t maximum_bytes, std::size_t maximum_tile_elements,
                                       std::vector<double>& gradient, std::string& detail,
                                       DfGradientResources* resources = nullptr);
/** Contract one caller-owned A/M weight tile into a detached partial gradient.
 * kind=0 maps
 * A[mu,nu,P], kind=1 maps M[P,Q]. The strided range addresses the
 * corresponding full row-major
 * tensor without padding a molecular weight.
 * This call owns only bounded metadata/device staging
 * and one O(Natom) result.
 */
vibeqc_status execute_cuda_df_gradient_tile(int device, const core::System& orbital,
                                            const core::System& auxiliary, unsigned kind,
                                            runtime::StridedRange range,
                                            std::span<const double> weights, unsigned schedule,
                                            std::size_t maximum_bytes,
                                            std::vector<double>& gradient, std::string& detail,
                                            DfGradientResources* resources = nullptr);
/** HF adapter using the same generic derivative consumer on the plan's stream.
 * Values are borrowed from raw_a when resident, or regenerated from source.
 * device_metric selects device response contractions, with source generation
 * or bounded uploads from raw_a. Without it, the compatibility raw-value
 * adapter reports its bounded host response staging.
 * Device response borrows blas_handle from the same plan, already bound to
 * stream with host scalar pointer mode; no handle or workspace is created here.
 * No coordinate-indexed A/M derivative tensors are formed or downloaded.
 * maximum_bytes bounds this bridge's numeric host/device scratch separately
 * from the caller's plan; an active resource ledger also enforces total device
 * ownership. Output is transactional and generated failures are returned.
 */
vibeqc_status execute_cuda_df_hf_gradient(
    int device, void* stream, CudaDensityFittingIntegralSource* source, std::size_t source_index,
    const core::System& orbital, const core::System& auxiliary, std::span<const double> raw_a,
    const std::vector<double>& metric, const std::vector<double>& inverse,
    std::span<const DensityFittingDensityResponse> terms, double relative_threshold,
    unsigned schedule, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    std::vector<double>& gradient, std::string& detail, DfGradientResources* resources = nullptr,
    const CudaDfMetricView* device_metric = nullptr, void* blas_handle = nullptr,
    const CudaDfResponseBuffers* borrowed = nullptr);
}  // namespace vibeqc::scf
#endif
