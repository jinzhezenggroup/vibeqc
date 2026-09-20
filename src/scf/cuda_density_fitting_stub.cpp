#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf {
vibeqc_status try_cuda_density_fitting_final_rhf_jk(CudaDensityFittingJkPlan*,
                                                    const CudaDfFinalStateToken&,
                                                    const std::vector<double>&,
                                                    std::vector<double>&, std::vector<double>&,
                                                    bool& used, std::string&, bool) {
  used = false;
  return VIBEQC_STATUS_SUCCESS;
}
void bind_cuda_density_fitting_response_source(CudaDensityFittingJkPlan*, const core::System&,
                                               const core::System&,
                                               std::span<const double>) noexcept {}
std::uint64_t cuda_density_fitting_solve_epoch(const CudaDensityFittingJkPlan*) noexcept {
  return 0;
}
vibeqc_status cuda_density_fitting_final_state_token(const CudaDensityFittingJkPlan*, std::size_t,
                                                     CudaDfFinalStateToken& token,
                                                     std::string& detail) {
  token = {};
  detail = "CUDA DF final-state snapshots are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
vibeqc_status read_cuda_density_fitting_final_state(CudaDensityFittingJkPlan*,
                                                    const CudaDfFinalStateToken&,
                                                    CudaDfFinalStateSnapshot& snapshot,
                                                    std::string& detail, bool) {
  snapshot = {};
  detail = "CUDA DF final-state snapshots are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
vibeqc_status solve_cuda_density_fitting_eigen(
    CudaDensityFittingJkPlan*, const std::vector<double>& matrix,
    const std::vector<double>* overlap, const std::vector<double>* orthogonalizer,
    std::vector<double>& values, std::vector<double>& coefficients,
    CudaDfEigenDiagnostic& diagnostic, std::string& detail, std::size_t) {
  diagnostic = {};
  if (df_eigen_outputs_alias(matrix, overlap, orthogonalizer, values, coefficients)) {
    detail = "ordinary CUDA DF eigen inputs and outputs must not alias";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  values.clear();
  coefficients.clear();
  detail = "ordinary CUDA DF eigen operations are unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status execute_cuda_density_fitting_generated_force_response(
    CudaDensityFittingJkPlan*, std::size_t, const core::System&, const core::System&,
    std::span<const double>, const std::vector<double>&,
    std::span<const DensityFittingDensityResponse>, unsigned, std::size_t, std::size_t,
    std::vector<double>&, std::string& detail, DfGradientResources*, const CudaDfFinalStateToken*) {
  detail = "CUDA DF generated response is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

CudaDensityFittingSourceDiagnostic cuda_density_fitting_integral_source_diagnostic(
    const CudaDensityFittingIntegralSource*) noexcept {
  return {};
}

vibeqc_status create_cuda_density_fitting_integral_source(int, const std::vector<core::System>&,
                                                          const std::vector<core::System>&,
                                                          CudaDensityFittingIntegralSource** source,
                                                          std::vector<double>& metrics,
                                                          std::size_t& nbf, std::size_t& naux,
                                                          std::string& detail) {
  if (source != nullptr) *source = nullptr;
  metrics.clear();
  nbf = 0;
  naux = 0;
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

void destroy_cuda_density_fitting_integral_source(CudaDensityFittingIntegralSource*) noexcept {}

std::size_t cuda_density_fitting_integral_source_device_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_host_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_host_peak_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_coordinate_count(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

CudaDensityFittingSourceCounters cuda_density_fitting_integral_source_counters(
    const CudaDensityFittingIntegralSource*) noexcept {
  return {};
}

CudaDensityFittingSourceCounters cuda_density_fitting_jk_plan_source_counters(
    const CudaDensityFittingJkPlan*) noexcept {
  return {};
}

bool cuda_density_fitting_integral_source_matches(const CudaDensityFittingIntegralSource*, int,
                                                  std::size_t, std::size_t, std::size_t) noexcept {
  return false;
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int, CudaDensityFittingIntegralSource**, std::size_t, std::size_t, std::size_t,
    const std::vector<double>&, double, std::size_t, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  diagnostics.clear();
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail);
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool, DfValueStorageOptions, std::size_t) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail);
}

vibeqc_status generate_cuda_density_fitting_transformed_tile(CudaDensityFittingIntegralSource*,
                                                             std::size_t, std::size_t, std::size_t,
                                                             std::size_t, std::size_t, std::int64_t,
                                                             const double*, void*, double*,
                                                             std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status generate_cuda_density_fitting_raw_tile(CudaDensityFittingIntegralSource*, std::size_t,
                                                     std::size_t, std::size_t, std::size_t,
                                                     std::size_t, std::int64_t, void*, double*,
                                                     std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status generate_cuda_density_fitting_metric_derivative_tile(
    CudaDensityFittingIntegralSource*, std::size_t, std::size_t, std::size_t, std::int64_t, void*,
    double*, std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan*) noexcept {
  return 0U;
}
void set_cuda_density_fitting_scf_value_budget(CudaDensityFittingJkPlan*, std::size_t) noexcept {}

void set_cuda_density_fitting_scf_diis_history(CudaDensityFittingJkPlan*, unsigned) noexcept {}

unsigned cuda_density_fitting_scf_diis_history(const CudaDensityFittingJkPlan*) noexcept {
  return 0;
}

std::size_t cuda_density_fitting_scf_value_budget(const CudaDensityFittingJkPlan*) noexcept {
  return 0;
}

bool cuda_density_fitting_scf_policy_matches(const CudaDensityFittingJkPlan*) noexcept {
  return false;
}
DfPairStorage cuda_density_fitting_pair_storage(const CudaDensityFittingJkPlan*) noexcept {
  return DfPairStorage::Dense;
}
bool cuda_density_fitting_jk_plan_matches(const CudaDensityFittingJkPlan*, std::size_t, std::size_t,
                                          std::size_t, double) noexcept {
  return false;
}

namespace {

vibeqc_status unavailable(CudaDensityFittingJkPlan** plan, std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace

vibeqc_status create_cuda_density_fitting_jk_plan(
    int, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, double, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail,
    std::size_t) {
  diagnostics.clear();
  return unavailable(plan, detail);
}

vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, double, std::size_t, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail,
    std::size_t) {
  diagnostics.clear();
  return unavailable(plan, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan*,
                                                  const std::vector<double>&, std::vector<double>&,
                                                  std::vector<double>&, std::string& detail,
                                                  JkTermSelection) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(CudaDensityFittingJkPlan*,
                                                  const std::vector<double>&,
                                                  const std::vector<double>&, std::vector<double>&,
                                                  std::vector<double>&, std::vector<double>&,
                                                  std::string& detail, JkTermSelection) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_item(CudaDensityFittingJkPlan*, std::size_t,
                                                       const std::vector<double>&,
                                                       std::vector<double>&, std::vector<double>&,
                                                       std::string& detail, JkTermSelection) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_item(CudaDensityFittingJkPlan*, std::size_t,
                                                       const std::vector<double>&,
                                                       const std::vector<double>&,
                                                       std::vector<double>&, std::vector<double>&,
                                                       std::vector<double>&, std::string& detail,
                                                       JkTermSelection) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_device(CudaDensityFittingJkPlan*, const double*,
                                                         double*, double*, std::string& detail,
                                                         JkTermSelection, FockMatrixLayout) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_device(CudaDensityFittingJkPlan*, const double*,
                                                         const double*, double*, double*, double*,
                                                         std::string& detail, JkTermSelection,
                                                         FockMatrixLayout) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<std::int32_t>&, const std::vector<double>&,
    unsigned, double, double, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<std::int32_t>&, const std::vector<double>&,
    unsigned, double, double, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail, const std::vector<double>&, unsigned) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<std::int32_t>&,
    const std::vector<std::int32_t>&, const std::vector<double>&, unsigned, double, double,
    std::vector<double>&, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<std::int32_t>&,
    const std::vector<std::int32_t>&, const std::vector<double>&, unsigned, double, double,
    std::vector<double>&, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail, const std::vector<double>&, unsigned) {
  return unavailable(nullptr, detail);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan*) noexcept {}

DensityFactorIdentity cuda_density_fitting_factor_identity(const CudaDensityFittingJkPlan*,
                                                           std::size_t, std::uint64_t,
                                                           std::uint64_t) noexcept {
  return {};
}

vibeqc_status execute_cuda_density_fitting_occupied_exchange(
    CudaDensityFittingJkPlan*, const std::vector<double>&, DensityFactorSpin,
    std::span<const CudaOccupiedDensityInput>, std::vector<double>&, std::vector<std::uint8_t>&,
    std::string& detail) {
  return unavailable(nullptr, detail);
}

solver::PhysicalFockFrame evaluate_cuda_density_fitting_final_fock(
    CudaDensityFittingJkPlan*, const solver::FinalStateIdentity&,
    const std::vector<reference::Matrix>&, const reference::Matrix&) {
  throw std::runtime_error("CUDA physical Fock evaluation is unavailable");
}
solver::FinalStateOperations cuda_density_fitting_final_state_operations(
    CudaDensityFittingJkPlan*) {
  throw std::runtime_error("CUDA final-state validation is unavailable");
}
bool validate_cuda_density_fitting_eigen_frame(CudaDensityFittingJkPlan*, const reference::Matrix&,
                                               const reference::Matrix*, const reference::Matrix&,
                                               const reference::Matrix&,
                                               solver::EigenFrameDiagnostic&, std::string& detail) {
  detail = "CUDA eigenframe validation is unavailable";
  return false;
}
}  // namespace vibeqc::scf
