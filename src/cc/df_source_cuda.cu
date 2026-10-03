#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "cc/df_source.hpp"
#include "df_mo_source_generated.hpp"
#include "generated_df_cc_source_cuda.cuh"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"

namespace generativeqc::cc {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
std::size_t bytes(std::size_t count) { return checked_mul(count, sizeof(double)); }
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check_status(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
void admit(std::size_t count, std::size_t budget) {
  if (count > budget) throw std::length_error("native CUDA DF-CC source exceeds numeric budget");
}
struct SourceDelete {
  void operator()(scf::CudaDensityFittingIntegralSource* p) const noexcept {
    scf::destroy_cuda_density_fitting_integral_source(p);
  }
};
struct PlanDelete {
  void operator()(scf::CudaDensityFittingJkPlan* p) const noexcept {
    scf::destroy_cuda_density_fitting_jk_plan(p);
  }
};

// This bounds the shared streamed J/K owner before its metric setup. It includes
// its conservative lazy SCF reservations even though this consumer never runs
// SCF on that owner. Validate against its reported peak before proceeding.
std::size_t metric_setup_bound(std::size_t n, std::size_t q, std::size_t source_device,
                               std::size_t source_host_peak) {
  auto count = checked_add(checked_mul(64, checked_mul(n, n)), checked_mul(8, checked_mul(n, q)));
  count = checked_add(count, checked_mul(8, checked_mul(q, q)));
  count = checked_add(count, checked_mul(8, checked_add(n, q)));
  auto peak = checked_add(bytes(count), 1ULL << 20);
  peak = checked_add(peak, checked_mul(4, scf::df_eigen_workspace_allowance(n)));
  peak = checked_add(peak, checked_mul(2, scf::df_eigen_workspace_allowance(q)));
  return checked_add(peak, checked_add(source_device, checked_mul(2, source_host_peak)));
}
}  // namespace

DFSourceResult build_df_source_cuda(const core::System& orbital, const core::System& auxiliary,
                                    const hf::PhysicalReference& ref, std::size_t budget,
                                    double relative_threshold, int device,
                                    std::size_t caller_bytes) {
  const auto started = Clock::now();
  const auto n = ref.nbf, o = ref.nocc, q = molecule::ao_count(auxiliary);
  if (!n || !o || o >= n || !q || molecule::ao_count(orbital) != n || device < 0 || !budget ||
      !std::isfinite(relative_threshold) || relative_threshold <= 0 || relative_threshold >= 1 ||
      ref.coefficients.size() != checked_mul(n, n) ||
      !std::all_of(
          ref.coefficients.begin(), ref.coefficients.end(),
          [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("invalid native CUDA DF-CC source request");
  const auto v = n - o;
  const auto layout = generated::df_source::source_layout(o, v, q);
  // Every packed BLAS dimension is checked before any source is generated.
  const auto int_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (std::max({n, q, layout.row_values, layout.matrix_values}) > int_limit)
    throw std::length_error("native CUDA DF-CC source exceeds BLAS indexing");
  const auto work = posthf::generated::df_mo_source_work(n, q);
  auto external = checked_add(caller_bytes, posthf::source_capacity(orbital));
  external = checked_add(external, posthf::source_capacity(auxiliary));
  for (const auto* values : {&ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients,
                             &ref.orbital_energies, &ref.density, &ref.weighted_density})
    external = checked_add(external, bytes(values->capacity()));
  const auto execution_payload =
      std::max({layout.transform_bytes, layout.packing_bytes, layout.blocks_bytes});
  admit(checked_add(external, execution_payload), budget);
  runtime::CudaDeviceScope device_scope(device);
  DFSourceResult result;
  result.nocc = o;
  result.nvir = v;
  result.naux = q;
  result.host_output_bytes = bytes(layout.output_values);

  auto stage = Clock::now();
  scf::CudaDensityFittingIntegralSource* raw_source = nullptr;
  std::vector<double> metric;
  std::size_t source_n = 0, source_q = 0;
  std::string detail;
  const auto source_status = scf::create_cuda_density_fitting_integral_source(
      device, {orbital}, {auxiliary}, &raw_source, metric, source_n, source_q, detail);
  std::unique_ptr<scf::CudaDensityFittingIntegralSource, SourceDelete> source(raw_source);
  check_status(source_status, detail);
  if (source_n != n || source_q != q || metric.size() != checked_mul(q, q))
    throw std::runtime_error("native CUDA DF-CC source dimensions changed");
  const auto placement = scf::cuda_density_fitting_integral_source_diagnostic(source.get());
  if (std::strcmp(placement.value_backend, "generated_rys") != 0 ||
      !placement.public_transform_on_device)
    throw std::runtime_error("native CUDA DF-CC requires generated device DF integrals");
  const auto source_device = scf::cuda_density_fitting_integral_source_device_bytes(source.get());
  const auto source_host = scf::cuda_density_fitting_integral_source_host_peak_bytes(source.get());
  // The existing source API reports its setup peak after construction. Reject
  // it before publishing a handle or allocating metric/transform consumers.
  const auto source_peak = checked_add(
      external, checked_add(source_device, checked_add(source_host, bytes(metric.capacity()))));
  admit(source_peak, budget);
  const auto setup_bound =
      checked_add(external, metric_setup_bound(n, q, source_device, source_host));
  admit(setup_bound, budget);
  result.source_seconds = elapsed(stage);

  stage = Clock::now();
  scf::CudaDensityFittingJkPlan* raw_plan = nullptr;
  auto* transferred = source.release();
  std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
  const auto plan_status = scf::create_cuda_density_fitting_jk_plan_from_source(
      device, &transferred, 1, n, q, metric, relative_threshold, q, n, &raw_plan, diagnostics,
      detail, false);
  // The shared factory consumes the source on every outcome. Establish RAII
  // before inspecting status so any published plan is also failure-safe.
  std::unique_ptr<scf::CudaDensityFittingJkPlan, PlanDelete> owner(raw_plan);
  check_status(plan_status, detail);
  if (!owner || !owner->streamed || !owner->integral_source || !owner->inverse_square_roots ||
      diagnostics.size() != 1)
    throw std::runtime_error("native CUDA DF-CC requires the streamed metric owner");
  auto& plan = *owner;
  const auto& diag = diagnostics.front();
  const auto setup_peak =
      checked_add(external, checked_add(diag.peak_device_bytes, diag.peak_host_bytes));
  if (setup_peak > setup_bound)
    throw std::logic_error("shared DF metric owner exceeded source setup bound");
  admit(setup_peak, budget);
  result.metric_rank = diag.effective_rank;
  result.metric_absolute_threshold = diag.absolute_threshold;
  result.metric_condition_number = diag.condition_number;
  result.metric_staging_bytes = checked_mul(2, bytes(metric.size()));
  std::vector<double>().swap(metric);
  const auto fixed =
      checked_add(external, checked_add(diag.device_resident_bytes, diag.host_resident_bytes));
  const auto execution_peak = checked_add(fixed, execution_payload);
  admit(execution_peak, budget);
  result.numeric_capacity_bytes = std::max({source_peak, setup_bound, execution_peak});
  const auto device_payload = std::max({layout.transform_bytes, layout.packing_bytes,
                                        layout.blocks_bytes - result.host_output_bytes});
  result.device_capacity_bytes =
      std::max({source_device, diag.peak_device_bytes,
                checked_add(diag.device_resident_bytes, device_payload)});
  result.metric_seconds = elapsed(stage);

  const auto stream = plan.stream;
  // Buffers are declared after the metric owner: all stream-dependent storage
  // drains and dies before its source/BLAS/stream owner, including exceptions.
  runtime::OwnedCudaBuffer<double> bmo(device, layout.source_values, stream);
  auto gemm = [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k,
                  const double* a, const double* b, double* output) {
    const double one = 1, zero = 0;
    const auto status = cublasDgemm(
        plan.blas, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
        static_cast<int>(m), static_cast<int>(columns), static_cast<int>(k), &one, a,
        static_cast<int>(ta == 'N' ? m : k), b, static_cast<int>(tb == 'N' ? k : columns), &zero,
        output, static_cast<int>(m));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("native CUDA DF-CC source GEMM failed");
  };
  stage = Clock::now();
  {
    runtime::OwnedCudaBuffer<double> transformed(device, layout.source_values, stream);
    runtime::OwnedCudaBuffer<double> coefficients(device, layout.matrix_values, stream);
    runtime::OwnedCudaBuffer<double> row(device, layout.row_values, stream);
    runtime::cuda_resource_check(cudaMemcpyAsync(coefficients.get(), ref.coefficients.data(),
                                                 bytes(layout.matrix_values),
                                                 cudaMemcpyHostToDevice, stream));
    result.coefficient_h2d_bytes = bytes(layout.matrix_values);
    posthf::generated::transform_df_mo_source(
        n, q, coefficients.get(), plan.inverse_square_roots, bmo.get(), transformed.get(),
        [&](std::size_t mu) {
          check_status(scf::generate_cuda_density_fitting_raw_tile(
                           plan.integral_source, 0, mu * n, n, 0, q, -1, stream, row.get(), detail),
                       detail);
          ++result.source_rows;
          return row.get();
        },
        [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k, const double* a,
            const double* b, double* output) {
          gemm(ta, tb, m, columns, k, a, b, output);
          ++result.transform_gemms;
          result.transform_summands =
              checked_add(result.transform_summands, checked_mul(checked_mul(m, columns), k));
        });
    runtime::cuda_resource_check(cudaStreamSynchronize(stream));
  }
  result.source_values = checked_mul(result.source_rows, layout.row_values);
  if (result.source_values != work.raw_values || result.transform_gemms != work.gemms ||
      result.transform_summands != work.contraction_summands)
    throw std::logic_error("native DF source execution differs from compiler work");
  result.transform_seconds = elapsed(stage);

  stage = Clock::now();
  runtime::OwnedCudaBuffer<double> arena(device, layout.packing_values, stream);
  runtime::OwnedCudaBuffer<int> error(device, 1, stream);
  generated::df_source::CudaState state{o, v, q, bmo.get(), arena.get(), error.get(), stream};
  const auto factors = generated::df_source::pack_cuda(state);
  int failed = 0;
  runtime::cuda_resource_check(
      cudaMemcpyAsync(&failed, error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  runtime::cuda_resource_check(cudaStreamSynchronize(stream));
  if (failed) throw std::runtime_error("nonfinite native CUDA DF-CC source factors");
  bmo.reset();
  runtime::OwnedCudaBuffer<double> block(device, layout.largest_block, stream);
  auto download = [&](std::vector<double>& values, const double* device_values, std::size_t count) {
    values.resize(count);
    runtime::cuda_resource_check(cudaMemcpyAsync(values.data(), device_values, bytes(count),
                                                 cudaMemcpyDeviceToHost, stream));
    runtime::cuda_resource_check(cudaStreamSynchronize(stream));
    if (!std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite native CUDA DF-CC integral block");
    result.factor_block_d2h_bytes = checked_add(result.factor_block_d2h_bytes, bytes(count));
  };
  download(result.bov, factors.bov, checked_mul(q, checked_mul(o, v)));
  download(result.bvv, factors.bvv, checked_mul(q, checked_mul(v, v)));
  auto block_gemm = [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k,
                        const double* a, const double* b, double* output) {
    gemm(ta, tb, m, columns, k, a, b, output);
    ++result.block_gemms;
    result.block_summands =
        checked_add(result.block_summands, checked_mul(checked_mul(m, columns), k));
  };
  generated::df_source::build_ovov(o, v, q, factors, block.get(), block_gemm);
  download(result.ovov, block.get(), generated::df_source::ovov_elements(o, v));
  generated::df_source::build_ovvo(o, v, q, factors, block.get(), block_gemm);
  download(result.ovvo, block.get(), generated::df_source::ovvo_elements(o, v));
  generated::df_source::build_oovv(o, v, q, factors, block.get(), block_gemm);
  download(result.oovv, block.get(), generated::df_source::oovv_elements(o, v));
  generated::df_source::build_ovoo(o, v, q, factors, block.get(), block_gemm);
  download(result.ovoo, block.get(), generated::df_source::ovoo_elements(o, v));
  generated::df_source::build_oooo(o, v, q, factors, block.get(), block_gemm);
  download(result.oooo, block.get(), generated::df_source::oooo_elements(o, v));
  if (result.factor_block_d2h_bytes != result.host_output_bytes)
    throw std::logic_error("native DF-CC publication size differs from compiler layout");
  result.block_seconds = elapsed(stage);
  result.total_seconds = elapsed(started);
  return result;
}
}  // namespace generativeqc::cc
