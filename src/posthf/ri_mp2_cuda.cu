#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "posthf/capacity.hpp"
#include "posthf/mp2_schedule_generated.hpp"
#include "posthf/ri_mp2_cuda.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"

namespace vibeqc::mp2 {
namespace {

constexpr unsigned kThreads = 256;

void cuda_check(cudaError_t status, const char* what) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

std::size_t bytes(std::size_t elements) { return posthf::checked_mul(elements, sizeof(double)); }

std::size_t elements2(std::size_t a, std::size_t b) { return posthf::checked_mul(a, b); }

std::size_t elements3(std::size_t a, std::size_t b, std::size_t c) {
  return posthf::checked_mul(elements2(a, b), c);
}

struct SourceOwner {
  scf::CudaDensityFittingIntegralSource* pointer{};
  ~SourceOwner() { scf::destroy_cuda_density_fitting_integral_source(pointer); }
  SourceOwner() = default;
  SourceOwner(const SourceOwner&) = delete;
  SourceOwner& operator=(const SourceOwner&) = delete;
};

struct PlanOwner {
  scf::CudaDensityFittingJkPlan* pointer{};
  ~PlanOwner() { scf::destroy_cuda_density_fitting_jk_plan(pointer); }
  PlanOwner() = default;
  PlanOwner(const PlanOwner&) = delete;
  PlanOwner& operator=(const PlanOwner&) = delete;
};

struct EventPair {
  runtime::OwnedCudaEvent begin;
  runtime::OwnedCudaEvent end;
  explicit EventPair(int device) : begin(device), end(device) {}
  template <class Function>
  double measure(cudaStream_t stream, Function&& function) {
    begin.record(stream);
    function();
    end.record(stream);
    end.synchronize();
    return end.elapsed_since(begin);
  }
};

__global__ void reduce_ri_mp2_block(const double* g, const double* exchange, std::size_t g_stride,
                                    std::size_t exchange_stride, std::size_t a_count,
                                    std::size_t b_count, std::size_t no, std::size_t nv,
                                    std::size_t i, std::size_t j_begin, std::size_t a_begin,
                                    std::size_t b_begin, const double* orbital_energies,
                                    bool same_virtual_block, double* pair_sums) {
  const std::size_t local_j = blockIdx.x;
  const std::size_t j = j_begin + local_j;
  const double* direct = g + local_j * g_stride;
  const double* swapped = same_virtual_block ? direct : exchange + local_j * exchange_stride;
  double os = 0.0;
  double ss = 0.0;
  double os_correction = 0.0;
  double ss_correction = 0.0;
  const std::size_t count = a_count * b_count;
  for (std::size_t element = threadIdx.x; element < count; element += blockDim.x) {
    const std::size_t a = element % a_count;
    const std::size_t b = element / a_count;
    const double direct_value = direct[a + a_count * b];
    const double exchange_value =
        same_virtual_block ? direct[b + b_count * a] : swapped[b + b_count * a];
    const double denominator = orbital_energies[i] + orbital_energies[j] -
                               orbital_energies[no + a_begin + a] -
                               orbital_energies[no + b_begin + b];
    const double os_value = direct_value * direct_value / denominator;
    const double ss_value = direct_value * (direct_value - exchange_value) / denominator;
    const double os_adjusted = os_value - os_correction;
    const double os_next = os + os_adjusted;
    os_correction = (os_next - os) - os_adjusted;
    os = os_next;
    const double ss_adjusted = ss_value - ss_correction;
    const double ss_next = ss + ss_adjusted;
    ss_correction = (ss_next - ss) - ss_adjusted;
    ss = ss_next;
  }
  __shared__ double os_shared[kThreads];
  __shared__ double ss_shared[kThreads];
  os_shared[threadIdx.x] = os;
  ss_shared[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      os_shared[threadIdx.x] += os_shared[threadIdx.x + offset];
      ss_shared[threadIdx.x] += ss_shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const std::size_t pair = i * no + j;
    pair_sums[2 * pair] += os_shared[0];
    pair_sums[2 * pair + 1] += ss_shared[0];
  }
}

__global__ void reduce_pair_sums(const double* pair_sums, std::size_t pairs, double* result) {
  double os = 0.0;
  double ss = 0.0;
  double os_correction = 0.0;
  double ss_correction = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const double os_adjusted = pair_sums[2 * pair] - os_correction;
    const double os_next = os + os_adjusted;
    os_correction = (os_next - os) - os_adjusted;
    os = os_next;
    const double ss_adjusted = pair_sums[2 * pair + 1] - ss_correction;
    const double ss_next = ss + ss_adjusted;
    ss_correction = (ss_next - ss) - ss_adjusted;
    ss = ss_next;
  }
  __shared__ double os_shared[kThreads];
  __shared__ double ss_shared[kThreads];
  os_shared[threadIdx.x] = os;
  ss_shared[threadIdx.x] = ss;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      os_shared[threadIdx.x] += os_shared[threadIdx.x + offset];
      ss_shared[threadIdx.x] += ss_shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    result[0] = os_shared[0];
    result[1] = ss_shared[0];
  }
}

std::size_t fixed_capacity(const posthf::RawSource& source, std::size_t n, std::size_t no,
                           std::size_t nv, std::size_t na, std::size_t source_device,
                           std::size_t source_host_peak) {
  const auto matrix = elements2(n, n);
  const auto metric = elements2(na, na);
  const auto row = elements2(n, na);
  const auto reference = posthf::checked_add(posthf::checked_mul(5, matrix), n);
  auto total = bytes(reference);
  total = posthf::checked_add(total, posthf::source_capacity(source.orbital()));
  total = posthf::checked_add(total, posthf::source_capacity(source.auxiliary()));
  total = posthf::checked_add(total, source_device);
  total = posthf::checked_add(total, source_host_peak);
  // Streamed DF plan: seven n*n matrices, four row-sized panels, three metric
  // matrices at setup peak, two auxiliary vectors, and one bounded eigensolver
  // allowance on each side of the host/device accounting boundary.
  total = posthf::checked_add(total, bytes(posthf::checked_mul(7, matrix)));
  total = posthf::checked_add(total, bytes(posthf::checked_mul(4, row)));
  total = posthf::checked_add(total, bytes(posthf::checked_mul(3, metric)));
  total = posthf::checked_add(total, bytes(posthf::checked_mul(3, na)));
  total = posthf::checked_add(total, scf::df_eigen_workspace_allowance(na));
  total = posthf::checked_add(total, scf::df_eigen_workspace_allowance(na));
  // Persistent coefficient/energy uploads and pair/result storage.
  total = posthf::checked_add(total, bytes(elements2(n, no)));
  total = posthf::checked_add(total, bytes(elements2(n, nv)));
  total = posthf::checked_add(total, bytes(n));
  total = posthf::checked_add(total, bytes(posthf::checked_mul(2, elements2(no, no))));
  total = posthf::checked_add(total, 2 * sizeof(double));
  return total;
}

}  // namespace

RiMp2CudaBlockPlan plan_ri_mp2_cuda_blocks(std::size_t fixed, std::size_t budget, std::size_t n,
                                           std::size_t no, std::size_t nv, std::size_t na) {
  const auto plan = generated::ri_mp2_residency_plan(fixed, budget, n, no, nv, na);
  return {plan.virtual_block, plan.j_batch, plan.peak_bytes, plan.full_resident};
}

RiMp2CudaEnergy density_fitted_energy_cuda(const scf::PhysicalReference& ref,
                                           const posthf::RawSource& source, std::size_t budget,
                                           double metric_relative_threshold, int device) {
  const std::size_t n = ref.nbf;
  const std::size_t no = ref.nocc;
  const std::size_t nv = n - no;
  const std::size_t na = source.naux();
  if (device < 0 || !n || !no || !nv || !na)
    throw std::invalid_argument("invalid CUDA RI-MP2 dimensions");

  runtime::CudaDeviceScope device_scope(device);
  SourceOwner source_owner;
  std::vector<double> metric;
  std::size_t source_n = 0, source_na = 0;
  std::string detail;
  const auto source_status = scf::create_cuda_density_fitting_integral_source(
      device, {source.orbital()}, {source.auxiliary()}, &source_owner.pointer, metric, source_n,
      source_na, detail);
  if (source_status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (source_status != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "CUDA RI-MP2 DF source setup failed" : detail);
  if (source_n != n || source_na != na)
    throw std::runtime_error("CUDA RI-MP2 DF source dimensions changed during setup");

  const std::size_t source_device =
      scf::cuda_density_fitting_integral_source_device_bytes(source_owner.pointer);
  const std::size_t source_host_peak =
      scf::cuda_density_fitting_integral_source_host_peak_bytes(source_owner.pointer);
  const auto preflight = fixed_capacity(source, n, no, nv, na, source_device, source_host_peak);
  if (preflight > budget)
    throw std::length_error("CUDA RI-MP2 DF source/factorization exceeds numeric memory budget");

  PlanOwner plan_owner;
  std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
  auto* transferred_source = source_owner.pointer;
  source_owner.pointer = nullptr;
  const auto plan_status = scf::create_cuda_density_fitting_jk_plan_from_source(
      device, &transferred_source, 1, n, na, metric, metric_relative_threshold, na, n,
      &plan_owner.pointer, diagnostics, detail, false);
  // The source-backed plan consumes/destroys the transferred handle on every outcome.
  transferred_source = nullptr;
  if (plan_status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (plan_status != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "CUDA RI-MP2 metric factorization failed" : detail);
  auto& plan = *plan_owner.pointer;
  if (!plan.streamed || !plan.integral_source || !plan.inverse_square_roots || plan.nbf != n ||
      plan.naux != na)
    throw std::runtime_error("CUDA RI-MP2 did not receive the expected streamed DF factor owner");
  if (diagnostics.size() != 1)
    throw std::runtime_error("CUDA RI-MP2 DF plan did not publish one resource diagnostic");

  const auto matrix = elements2(n, n);
  const auto reference_elements = posthf::checked_add(posthf::checked_mul(5, matrix), n);
  auto caller_bytes = bytes(reference_elements);
  caller_bytes = posthf::checked_add(caller_bytes, posthf::source_capacity(source.orbital()));
  caller_bytes = posthf::checked_add(caller_bytes, posthf::source_capacity(source.auxiliary()));

  const auto& plan_diagnostic = diagnostics.front();
  auto plan_setup_peak =
      posthf::checked_add(plan_diagnostic.peak_device_bytes, plan_diagnostic.peak_host_bytes);
  plan_setup_peak = posthf::checked_add(plan_setup_peak, caller_bytes);
  if (plan_setup_peak > budget)
    throw std::length_error("CUDA RI-MP2 metric-plan setup exceeds numeric memory budget");

  // The setup metric copy is no longer a correlation owner after the plan has
  // retained its eigensystem/inverse square root.
  metric.clear();
  metric.shrink_to_fit();

  auto execution_fixed = caller_bytes;
  execution_fixed = posthf::checked_add(execution_fixed, plan_diagnostic.device_resident_bytes);
  execution_fixed = posthf::checked_add(execution_fixed, plan_diagnostic.host_resident_bytes);
  // Compact occupied/virtual coefficients coexist once on host and once on
  // device; orbital energies add one device copy. The generated-row buffer,
  // pair accumulator and final device/host scalars are independent of B block size.
  execution_fixed = posthf::checked_add(execution_fixed, bytes(posthf::checked_mul(2, matrix)));
  execution_fixed = posthf::checked_add(execution_fixed, bytes(n));
  execution_fixed = posthf::checked_add(execution_fixed, bytes(elements2(n, na)));
  execution_fixed =
      posthf::checked_add(execution_fixed, bytes(posthf::checked_mul(2, elements2(no, no))));
  execution_fixed = posthf::checked_add(execution_fixed, 4 * sizeof(double));

  const auto blocking = plan_ri_mp2_cuda_blocks(execution_fixed, budget, n, no, nv, na);
  const auto virtual_block = blocking.virtual_block;
  const auto j_batch = blocking.j_batch;
  const bool full_resident = blocking.full_resident;
  const std::size_t planned_peak = std::max(plan_setup_peak, blocking.peak_bytes);

  const auto stream = plan.stream;
  runtime::cuda_trace::TraceOperation trace("ri_mp2_energy", stream,
                                            {1, n, na, true, !full_resident});
  runtime::cuda_trace::trace_counter("ri_mp2_virtual_block", virtual_block);
  runtime::cuda_trace::trace_counter("ri_mp2_virtual_blocks",
                                     (nv + virtual_block - 1) / virtual_block);
  runtime::cuda_trace::trace_counter("ri_mp2_j_batch", j_batch);
  runtime::cuda_trace::trace_counter("ri_mp2_preflight_bytes", preflight);
  runtime::cuda_trace::trace_counter("ri_mp2_plan_setup_peak_bytes", plan_setup_peak);
  runtime::cuda_trace::trace_counter("ri_mp2_execution_fixed_bytes", execution_fixed);
  runtime::cuda_trace::trace_counter("ri_mp2_planned_peak_bytes", planned_peak);

  std::vector<double> cocc(elements2(n, no));
  std::vector<double> cvir(elements2(n, nv));
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t i = 0; i < no; ++i) cocc[mu + n * i] = ref.coefficients[mu * n + i];
    for (std::size_t a = 0; a < nv; ++a) cvir[mu + n * a] = ref.coefficients[mu * n + no + a];
  }

  runtime::OwnedCudaBuffer<double> cocc_device(device, cocc.size(), stream);
  runtime::OwnedCudaBuffer<double> cvir_device(device, cvir.size(), stream);
  runtime::OwnedCudaBuffer<double> eps_device(device, ref.orbital_energies.size(), stream);
  runtime::OwnedCudaBuffer<double> row_device(device, elements2(n, na), stream);
  runtime::OwnedCudaBuffer<double> tmp_device(device, elements3(n, na, virtual_block), stream);
  runtime::OwnedCudaBuffer<double> first_b(device, elements3(na, no, virtual_block), stream);
  runtime::OwnedCudaBuffer<double> second_b;
  runtime::OwnedCudaBuffer<double> direct_device(
      device, elements3(j_batch, virtual_block, virtual_block), stream);
  runtime::OwnedCudaBuffer<double> exchange_device;
  if (!full_resident) {
    second_b.allocate(device, elements3(na, no, virtual_block), stream);
    exchange_device.allocate(device, elements3(j_batch, virtual_block, virtual_block), stream);
  }
  runtime::OwnedCudaBuffer<double> pair_device(device, posthf::checked_mul(2, elements2(no, no)),
                                               stream);
  runtime::OwnedCudaBuffer<double> result_device(device, 2, stream);

  EventPair timer(device);
  RiMp2CudaEnergy result;
  result.numeric_capacity_bytes = planned_peak;
  result.virtual_block = virtual_block;
  result.metrics.provider_retained_bytes =
      bytes(elements3(na, no, full_resident ? nv : posthf::checked_mul(2, virtual_block)));

  const std::size_t coefficient_bytes = bytes(posthf::checked_add(
      posthf::checked_add(cocc.size(), cvir.size()), ref.orbital_energies.size()));
  result.metrics.input_ms += timer.measure(stream, [&] {
    cuda_check(cudaMemcpyAsync(cocc_device.get(), cocc.data(), bytes(cocc.size()),
                               cudaMemcpyHostToDevice, stream),
               "upload RI-MP2 occupied coefficients");
    cuda_check(cudaMemcpyAsync(cvir_device.get(), cvir.data(), bytes(cvir.size()),
                               cudaMemcpyHostToDevice, stream),
               "upload RI-MP2 virtual coefficients");
    cuda_check(cudaMemcpyAsync(eps_device.get(), ref.orbital_energies.data(),
                               bytes(ref.orbital_energies.size()), cudaMemcpyHostToDevice, stream),
               "upload RI-MP2 orbital energies");
    cuda_check(cudaMemsetAsync(pair_device.get(), 0, bytes(pair_device.size()), stream),
               "clear RI-MP2 pair sums");
  });

  const double one = 1.0;
  const double zero = 0.0;
  std::size_t source_passes = 0;
  std::size_t source_row_generations = 0;
  std::size_t transform_gemms = 0;
  auto transform_block = [&](std::size_t begin, std::size_t count, double* output) {
    runtime::cuda_trace::TraceRegion region("ri_mp2_ao_to_mo", stream);
    ++source_passes;
    for (std::size_t mu = 0; mu < n; ++mu) {
      const auto status = scf::generate_cuda_density_fitting_transformed_tile(
          plan.integral_source, 0, mu * n, n, 0, na, -1, plan.inverse_square_roots, stream,
          row_device.get(), detail);
      if (status != VIBEQC_STATUS_SUCCESS)
        throw std::runtime_error(detail.empty() ? "CUDA RI-MP2 transformed DF row failed" : detail);
      blas_check(
          cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(na),
                      static_cast<int>(count), static_cast<int>(n), &one, row_device.get(),
                      static_cast<int>(na), cvir_device.get() + begin * n, static_cast<int>(n),
                      &zero, tmp_device.get() + mu * na * count, static_cast<int>(na)),
          "RI-MP2 virtual AO-to-MO transform");
      ++source_row_generations;
      ++transform_gemms;
    }
    const std::size_t rows = elements2(na, count);
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("CUDA RI-MP2 transformed B exceeds cuBLAS int32 indexing");
    blas_check(cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(rows),
                           static_cast<int>(no), static_cast<int>(n), &one, tmp_device.get(),
                           static_cast<int>(rows), cocc_device.get(), static_cast<int>(n), &zero,
                           output, static_cast<int>(rows)),
               "RI-MP2 occupied AO-to-MO transform");
    ++transform_gemms;
  };

  std::size_t energy_gemms = 0;
  std::size_t contribution_blocks = 0;
  const auto blocks = (nv + virtual_block - 1) / virtual_block;
  const auto transform_started = std::chrono::steady_clock::now();
  for (std::size_t a_block = 0; a_block < blocks; ++a_block) {
    const std::size_t a_begin = a_block * virtual_block;
    const std::size_t a_count = std::min(virtual_block, nv - a_begin);
    transform_block(a_begin, a_count, first_b.get());
    for (std::size_t b_block = 0; b_block < blocks; ++b_block) {
      const std::size_t b_begin = b_block * virtual_block;
      const std::size_t b_count = std::min(virtual_block, nv - b_begin);
      const bool same = a_block == b_block;
      double* b_values = first_b.get();
      if (!same) {
        transform_block(b_begin, b_count, second_b.get());
        b_values = second_b.get();
      }

      runtime::cuda_trace::TraceRegion energy_region("ri_mp2_energy_contraction", stream);
      const std::size_t a_stride = elements2(na, a_count);
      const std::size_t b_stride = elements2(na, b_count);
      const std::size_t g_stride = elements2(a_count, b_count);
      for (std::size_t i = 0; i < no; ++i) {
        for (std::size_t j_begin = 0; j_begin < no; j_begin += j_batch) {
          const std::size_t j_count = std::min(j_batch, no - j_begin);
          blas_check(cublasDgemmStridedBatched(
                         plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(a_count),
                         static_cast<int>(b_count), static_cast<int>(na), &one,
                         first_b.get() + i * a_stride, static_cast<int>(na), 0,
                         b_values + j_begin * b_stride, static_cast<int>(na),
                         static_cast<long long>(b_stride), &zero, direct_device.get(),
                         static_cast<int>(a_count), static_cast<long long>(g_stride),
                         static_cast<int>(j_count)),
                     "RI-MP2 direct fitted-integral batch");
          ++energy_gemms;
          if (!same) {
            blas_check(
                cublasDgemmStridedBatched(
                    plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(b_count),
                    static_cast<int>(a_count), static_cast<int>(na), &one, b_values + i * b_stride,
                    static_cast<int>(na), 0, first_b.get() + j_begin * a_stride,
                    static_cast<int>(na), static_cast<long long>(a_stride), &zero,
                    exchange_device.get(), static_cast<int>(b_count),
                    static_cast<long long>(g_stride), static_cast<int>(j_count)),
                "RI-MP2 exchange fitted-integral batch");
            ++energy_gemms;
          }
          reduce_ri_mp2_block<<<static_cast<unsigned>(j_count), kThreads, 0, stream>>>(
              direct_device.get(), same ? nullptr : exchange_device.get(), g_stride, g_stride,
              a_count, b_count, no, nv, i, j_begin, a_begin, b_begin, eps_device.get(), same,
              pair_device.get());
          cuda_check(cudaGetLastError(), "launch RI-MP2 energy reduction");
          contribution_blocks += j_count;
        }
      }
    }
  }
  // The outer wall/event scope is intentionally one stream-resident endpoint:
  // transformed B never crosses the host boundary.
  cuda_check(cudaStreamSynchronize(stream), "finish RI-MP2 transformed contractions");
  result.metrics.library_ms += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - transform_started)
                                   .count();

  runtime::cuda_trace::trace_counter("ri_mp2_source_passes", source_passes);
  runtime::cuda_trace::trace_counter("ri_mp2_source_row_generations", source_row_generations);
  runtime::cuda_trace::trace_counter("ri_mp2_ao_to_mo_gemms", transform_gemms);
  runtime::cuda_trace::trace_counter("ri_mp2_energy_gemms", energy_gemms);
  runtime::cuda_trace::trace_counter("ri_mp2_contribution_blocks", contribution_blocks);
  const auto b_storage_bytes = bytes(posthf::checked_add(first_b.size(), second_b.size()));
  runtime::cuda_trace::trace_counter("ri_mp2_resident_b_bytes", b_storage_bytes);

  reduce_pair_sums<<<1, kThreads, 0, stream>>>(pair_device.get(), elements2(no, no),
                                               result_device.get());
  cuda_check(cudaGetLastError(), "launch RI-MP2 final reduction");
  double host_result[2]{};
  result.metrics.output_ms += timer.measure(stream, [&] {
    cuda_check(cudaMemcpyAsync(host_result, result_device.get(), sizeof(host_result),
                               cudaMemcpyDeviceToHost, stream),
               "download RI-MP2 energy scalars");
  });
  if (!std::isfinite(host_result[0]) || !std::isfinite(host_result[1]))
    throw std::runtime_error("nonfinite CUDA RI-MP2 energy accumulation");
  result.opposite_spin = host_result[0];
  result.same_spin = host_result[1];
  result.source_passes = source_passes;
  result.logical_tiles = posthf::checked_mul(elements2(no, no), elements2(blocks, blocks));

  const std::size_t metric_bytes = bytes(elements2(na, na));
  const std::size_t metric_diagnostic_bytes = posthf::checked_add(bytes(na), sizeof(int));
  result.transfer_bytes = coefficient_bytes;
  result.transfer_bytes = posthf::checked_add(result.transfer_bytes, metric_bytes);  // source D2H
  result.transfer_bytes = posthf::checked_add(result.transfer_bytes, metric_bytes);  // plan H2D
  result.transfer_bytes = posthf::checked_add(result.transfer_bytes, metric_diagnostic_bytes);
  result.transfer_bytes = posthf::checked_add(result.transfer_bytes, bytes(na));  // scales H2D
  result.transfer_bytes = posthf::checked_add(result.transfer_bytes, sizeof(host_result));
  const auto metric_h2d_bytes = posthf::checked_add(metric_bytes, bytes(na));
  const auto total_h2d_bytes = posthf::checked_add(coefficient_bytes, metric_h2d_bytes);
  const auto total_d2h_bytes = posthf::checked_add(
      posthf::checked_add(metric_bytes, metric_diagnostic_bytes), sizeof(host_result));
  runtime::cuda_trace::trace_counter("ri_mp2_transfer_bytes", result.transfer_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_h2d_bytes", total_h2d_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_d2h_bytes", total_d2h_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_metric_d2h_bytes", metric_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_metric_diagnostic_d2h_bytes", metric_diagnostic_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_metric_h2d_bytes", metric_h2d_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_coefficient_h2d_bytes", coefficient_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_final_d2h_bytes", sizeof(host_result));

  const auto working_elements = posthf::checked_add(
      posthf::checked_add(posthf::checked_add(cocc_device.size(), cvir_device.size()),
                          posthf::checked_add(eps_device.size(), row_device.size())),
      posthf::checked_add(
          posthf::checked_add(tmp_device.size(), first_b.size()),
          posthf::checked_add(
              posthf::checked_add(second_b.size(), direct_device.size()),
              posthf::checked_add(exchange_device.size(),
                                  posthf::checked_add(pair_device.size(), result_device.size())))));
  const auto working_device_bytes = bytes(working_elements);
  const auto plan_device_bytes = plan_diagnostic.device_resident_bytes;
  result.metrics.owned_device_bytes =
      static_cast<std::uint64_t>(posthf::checked_add(plan_device_bytes, working_device_bytes));
  runtime::cuda_trace::trace_counter("ri_mp2_plan_device_resident_bytes", plan_device_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_plan_host_resident_bytes",
                                     plan_diagnostic.host_resident_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_plan_peak_device_bytes",
                                     plan_diagnostic.peak_device_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_plan_peak_host_bytes",
                                     plan_diagnostic.peak_host_bytes);
  runtime::cuda_trace::trace_counter("ri_mp2_working_device_bytes", working_device_bytes);
  result.metrics.device_ms = result.metrics.input_ms + result.metrics.output_ms +
                             result.metrics.library_ms + result.metrics.kernel_ms;
  return result;
}

}  // namespace vibeqc::mp2
