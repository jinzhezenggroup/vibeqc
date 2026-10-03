#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "scf/density_fitting.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"

namespace {
using namespace generativeqc::scf;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct DeviceMatrix {
  double* pointer{};
  explicit DeviceMatrix(const std::vector<double>& input) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), input.size() * sizeof(double)));
    const auto status =
        cudaMemcpy(pointer, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      cudaFree(pointer);
      check(status);
    }
  }
  ~DeviceMatrix() { cudaFree(pointer); }
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  void verify(const std::vector<double>& expected) const {
    std::vector<double> actual(expected.size());
    check(
        cudaMemcpy(actual.data(), pointer, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i)
      require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-12,
              "independent device provider matrix differs from CPU");
  }
};

struct DeviceCounter {
  std::uint64_t* pointer{};
  DeviceCounter() { check(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(*pointer))); }
  ~DeviceCounter() { cudaFree(pointer); }
  DeviceCounter(const DeviceCounter&) = delete;
  DeviceCounter& operator=(const DeviceCounter&) = delete;
  void set(std::uint64_t value) const {
    check(cudaMemcpy(pointer, &value, sizeof(value), cudaMemcpyHostToDevice));
  }
  std::uint64_t get() const {
    std::uint64_t value{};
    check(cudaMemcpy(&value, pointer, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
  }
};

/** Exercise the same resident boundary consumed by native KS: enqueue on the
 * provider stream, then explicitly export only at this reference boundary. */
void direct_device(CudaDirectJkPlan* plan, FockBuildSpec spec, const std::vector<double>& a,
                   const std::vector<double>& b, const std::vector<double>& expected_j,
                   const std::vector<double>& expected_a, const std::vector<double>& expected_b) {
  spec.derivative_order = 0;
  const bool unrestricted = spec.spin == FockSpin::Unrestricted;
  const std::vector<double> sentinel(a.size(), 123.0);
  DeviceMatrix da(a), db(b), j(sentinel), ka(sentinel), kb(sentinel), error({0.0});
  std::string detail;
  const auto status = enqueue_cuda_direct_jk_device(
      plan, spec, da.pointer, unrestricted ? db.pointer : nullptr, a.size(),
      spec.coulomb.present ? j.pointer : nullptr, spec.exchange.present ? ka.pointer : nullptr,
      spec.exchange.present && unrestricted ? kb.pointer : nullptr,
      reinterpret_cast<int*>(error.pointer), detail);
  require(status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
  const auto stream = cuda_direct_jk_stream(plan);
  int failure = -1;
  check(cudaMemcpyAsync(&failure, error.pointer, sizeof(failure), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  require(failure == 0, "valid resident J/K data failed numerical validation");
  j.verify(spec.coulomb.present ? expected_j : sentinel);
  ka.verify(spec.exchange.present ? expected_a : sentinel);
  kb.verify(spec.exchange.present && unrestricted ? expected_b : sentinel);
  da.verify(a);
  db.verify(b);
}

void direct_device_failures(CudaDirectJkPlan* plan, const std::vector<double>& a) {
  DeviceMatrix da(a), j(std::vector<double>(a.size())), error({0.0});
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.derivative_order = 0;
  spec.exchange.present = false;
  std::string detail;
  auto* failure_pointer = reinterpret_cast<int*>(error.pointer);
  require(enqueue_cuda_direct_jk_device(plan, spec, da.pointer, nullptr, a.size(), da.pointer,
                                        nullptr, nullptr, failure_pointer,
                                        detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "resident J/K accepted overlapping input/output");
  require(enqueue_cuda_direct_jk_device(plan, spec, da.pointer, nullptr, a.size() - 1, j.pointer,
                                        nullptr, nullptr, failure_pointer,
                                        detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "resident J/K accepted incomplete batch dimensions");
  auto invalid = a;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  check(cudaMemcpy(da.pointer, invalid.data(), invalid.size() * sizeof(double),
                   cudaMemcpyHostToDevice));
  const auto stream = cuda_direct_jk_stream(plan);
  for (bool recover : {false, true}) {
    if (recover)
      check(cudaMemcpy(da.pointer, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
    require(enqueue_cuda_direct_jk_device(plan, spec, da.pointer, nullptr, a.size(), j.pointer,
                                          nullptr, nullptr, failure_pointer,
                                          detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    int failure = -1;
    check(cudaMemcpyAsync(&failure, failure_pointer, sizeof(failure), cudaMemcpyDeviceToHost,
                          stream));
    check(cudaStreamSynchronize(stream));
    require(recover ? failure == 0 : failure != 0,
            "resident J/K failure state was lost or not reset");
  }
}

void direct_rsh_device(CudaDirectJkPlan* plan, FockBuildSpec correction,
                       const std::vector<double>& a, const std::vector<double>& b,
                       const std::vector<double>& expected_j,
                       const std::vector<double>& expected_full_a,
                       const std::vector<double>& expected_full_b,
                       const std::vector<double>& expected_range_a,
                       const std::vector<double>& expected_range_b) {
  const bool unrestricted = correction.spin == FockSpin::Unrestricted;
  auto primary = make_hf_fock_spec(correction.spin);
  primary.derivative_order = correction.derivative_order = 0;
  correction.coulomb.present = false;
  const std::vector<double> sentinel(a.size(), 123.0);
  DeviceMatrix da(a), db(b), j(sentinel), full_a(sentinel), full_b(sentinel), range_a(sentinel),
      range_b(sentinel), primary_error({123.0}), range_error({123.0});
  std::string detail;
  const auto enqueue = [&](double* full_output, double* range_output, int* range_failure) {
    return enqueue_cuda_direct_rsh_values_device(
        plan, primary, correction, da.pointer, unrestricted ? db.pointer : nullptr, a.size(),
        j.pointer, full_output, unrestricted ? full_b.pointer : nullptr, range_output,
        unrestricted ? range_b.pointer : nullptr, reinterpret_cast<int*>(primary_error.pointer),
        range_failure, detail);
  };
  auto* range_failure = reinterpret_cast<int*>(range_error.pointer);
  if (!plan->bounded_value_opt_in) {
    require(enqueue(full_a.pointer, range_a.pointer, range_failure) ==
                GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
            "fused RSH bypassed the default bounded-value policy");
    for (const auto* output : {&j, &full_a, &full_b, &range_a, &range_b}) output->verify(sentinel);
    primary_error.verify({123.0});
    range_error.verify({123.0});
  } else {
    require(enqueue(full_a.pointer, full_a.pointer, range_failure) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "fused RSH accepted overlapping full/range exchange outputs");
    require(
        enqueue(da.pointer, range_a.pointer, range_failure) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
        "fused RSH accepted overlapping density/output buffers");
    require(
        enqueue(full_a.pointer, range_a.pointer, reinterpret_cast<int*>(primary_error.pointer)) ==
            GENERATIVEQC_STATUS_INVALID_ARGUMENT,
        "fused RSH accepted overlapping numerical status buffers");
    require(enqueue(full_a.pointer, range_a.pointer, reinterpret_cast<int*>(range_a.pointer)) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "fused RSH accepted overlapping output/status buffers");
    require(enqueue(full_a.pointer, range_a.pointer, range_failure) == GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    check(cudaStreamSynchronize(cuda_direct_jk_stream(plan)));
    int first = -1, second = -1;
    check(cudaMemcpy(&first, primary_error.pointer, sizeof(first), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&second, range_error.pointer, sizeof(second), cudaMemcpyDeviceToHost));
    require(first == 0 && second == 0, "fused RSH failed numerical validation");
    j.verify(expected_j);
    full_a.verify(expected_full_a);
    full_b.verify(unrestricted ? expected_full_b : sentinel);
    range_a.verify(expected_range_a);
    range_b.verify(unrestricted ? expected_range_b : sentinel);
  }
  da.verify(a);
  db.verify(b);
}

void mixed_coulomb_work_census(bool through_f = false) {
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, through_f ? 3U : 0U, {{0.6, 1.0}}}};
  system.electron_count = 2;
  system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());

  const std::size_t n = generativeqc::molecule::ao_count(system), matrix = n * n;
  std::vector<double> density{0.9, 0.2, -0.1, 0.7};
  if (through_f) {
    density.resize(matrix);
    for (std::size_t index = 0; index < matrix; ++index) density[index] = (1.0 + 0.01 * index) / n;
  }
  const std::vector<double> zero_density(matrix, 0.0), sentinel(matrix, 123.0);
  require(density.size() == matrix, "mixed-work fixture density shape changed");

  const auto make_plan = [&](double screening) {
    CudaDirectJkPlan* raw{};
    CudaDirectJkDiagnostic diagnostic;
    require(create_cuda_direct_jk_plan(0, {system}, 0, screening, 32U * 1024U * 1024U, &raw,
                                       diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    return raw;
  };
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
      make_plan(0.0), &destroy_cuda_direct_jk_plan);
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> screened(
      make_plan(std::numeric_limits<double>::max()), &destroy_cuda_direct_jk_plan);
  if (through_f)
    require(plan->canonical_pairs != nullptr,
            "through-f mixed-work fixture must retain the competing FP64 canonical route");

  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.derivative_order = 0;
  spec.exchange.present = false;
  DeviceMatrix device_density(density), uninstrumented(sentinel), instrumented(sentinel),
      error({0.0});
  DeviceCounter count;
  auto* failure = reinterpret_cast<int*>(error.pointer);
  require(enqueue_cuda_direct_jk_device_mixed_j(plan.get(), spec, device_density.pointer, nullptr,
                                                matrix, uninstrumented.pointer, nullptr, nullptr,
                                                failure, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  count.set(999);
  require(enqueue_cuda_direct_jk_device_mixed_j(
              plan.get(), spec, device_density.pointer, nullptr, matrix, instrumented.pointer,
              nullptr, nullptr, failure, detail, count.pointer) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  check(cudaStreamSynchronize(cuda_direct_jk_stream(plan.get())));
  require(count.get() == matrix * matrix,
          "mixed Coulomb census did not count evaluated AO-ERI recurrences");
  std::vector<double> first(matrix), second(matrix);
  check(cudaMemcpy(first.data(), uninstrumented.pointer, matrix * sizeof(double),
                   cudaMemcpyDeviceToHost));
  check(cudaMemcpy(second.data(), instrumented.pointer, matrix * sizeof(double),
                   cudaMemcpyDeviceToHost));
  require(first == second, "mixed Coulomb instrumentation changed the J result");

  check(cudaMemcpy(device_density.pointer, zero_density.data(), matrix * sizeof(double),
                   cudaMemcpyHostToDevice));
  count.set(999);
  require(enqueue_cuda_direct_jk_device_mixed_j(
              plan.get(), spec, device_density.pointer, nullptr, matrix, instrumented.pointer,
              nullptr, nullptr, failure, detail, count.pointer) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  check(cudaStreamSynchronize(cuda_direct_jk_stream(plan.get())));
  require(count.get() == 0, "mixed Coulomb replay leaked its prior recurrence count");

  check(cudaMemcpy(device_density.pointer, density.data(), matrix * sizeof(double),
                   cudaMemcpyHostToDevice));
  count.set(999);
  require(enqueue_cuda_direct_jk_device_mixed_j(
              screened.get(), spec, device_density.pointer, nullptr, matrix, instrumented.pointer,
              nullptr, nullptr, failure, detail, count.pointer) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  check(cudaStreamSynchronize(cuda_direct_jk_stream(screened.get())));
  require(count.get() == 0, "screened mixed Coulomb recurrences were counted as executed");

  require(
      enqueue_cuda_direct_jk_device_mixed_j(
          plan.get(), spec, device_density.pointer, nullptr, matrix, instrumented.pointer, nullptr,
          nullptr, failure, detail, reinterpret_cast<std::uint64_t*>(device_density.pointer)) ==
          GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "mixed Coulomb census accepted a counter aliasing its density");
  require(enqueue_cuda_direct_jk_device_mixed_j(
              plan.get(), spec, device_density.pointer, nullptr, matrix, instrumented.pointer,
              nullptr, nullptr, failure, detail,
              reinterpret_cast<std::uint64_t*>(reinterpret_cast<char*>(count.pointer) + 1)) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "mixed Coulomb census accepted a misaligned counter");
}

void direct_eri_tile(CudaDirectJkPlan* plan, std::size_t item,
                     const std::vector<double>& expected_eri, std::size_t n) {
  require(plan != nullptr && n >= 2, "invalid raw ERI tile fixture");
  const std::array<std::size_t, 4> begin{0, 1, 0, 0};
  const std::array<std::size_t, 4> count{2, 1, 2, 2};
  const std::size_t elements = 8;
  std::vector<double> expected(elements);
  for (std::size_t local = 0; local < elements; ++local) {
    auto remainder = local;
    std::array<std::size_t, 4> index{};
    for (std::size_t reverse = 0; reverse < 4; ++reverse) {
      const auto axis = 3 - reverse;
      index[axis] = begin[axis] + remainder % count[axis];
      remainder /= count[axis];
    }
    const auto flat = ((index[0] * n + index[1]) * n + index[2]) * n + index[3];
    expected[local] = expected_eri[flat];
  }

  DeviceMatrix output(std::vector<double>(elements, 123.0));
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::string detail;
  const auto status = enqueue_cuda_direct_eri_tile(plan, item, begin, count, output.pointer,
                                                   elements, stream, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) {
    cudaStreamDestroy(stream);
    require(false, detail.c_str());
  }
  check(cudaStreamSynchronize(stream));
  check(cudaStreamDestroy(stream));
  output.verify(expected);
}

void prepared_interaction_source_device(const generativeqc::core::System& system,
                                        const std::vector<double>& expected_eri) {
  auto spec = make_hf_fock_spec(FockSpin::Restricted, FockApproximation::Exact);
  spec.derivative_order = 0;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cuda, 0.0);
  // Exercise the same exact value-only allowance as the correlated source
  // owner. A derivative request cannot fit and must not become a host fallback.
  const auto n = generativeqc::molecule::ao_count(system);
  std::size_t primitives = 0;
  for (const auto& shell : system.shells) primitives += shell.primitives.size();
  const auto allowance =
      cuda_direct_jk_device_bytes(1, n, system.atoms.size(), system.shells.size(), primitives, 0);
  PreparedFockPlan prepared(system, nullptr, strategy, 0, allowance);
  require(prepared.diagnostic().direct.device_bytes == allowance &&
              prepared.diagnostic().direct.derivative_order == 0,
          "value-only prepared source exceeded its queried device allowance");
  PreparedFockInteractionSourceView source(prepared);
  const auto op = generativeqc::integrals::ElectronInteractionOperator::eri;
  require(
      source.supports(op) && !source.supports_host_read(op) && source.supports_device_read(op, 0),
      "prepared CUDA interaction source advertised the wrong ERI memory spaces");

  require(source.nbf() == n, "value-only source changed the public AO basis");
  const std::array<std::size_t, 4> begin{0, 1, 0, 0};
  const std::array<std::size_t, 4> count{2, 1, 2, 2};
  constexpr std::size_t elements = 8;
  std::vector<double> expected(elements);
  for (std::size_t local = 0; local < elements; ++local) {
    auto remainder = local;
    std::array<std::size_t, 4> index{};
    for (std::size_t reverse = 0; reverse < 4; ++reverse) {
      const auto axis = 3 - reverse;
      index[axis] = begin[axis] + remainder % count[axis];
      remainder /= count[axis];
    }
    expected[local] = expected_eri[((index[0] * n + index[1]) * n + index[2]) * n + index[3]];
  }

  DeviceMatrix output(std::vector<double>(elements, 123.0));
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  source.read_device(op, begin, count,
                     {0, reinterpret_cast<void*>(stream), output.pointer, elements}, elements);
  check(cudaStreamSynchronize(stream));
  check(cudaStreamDestroy(stream));
  output.verify(expected);

  std::array<double, 1> host_sentinel{123.0};
  bool host_rejected = false;
  try {
    source.read(op, {0, 0, 0, 0}, {1, 1, 1, 1}, host_sentinel.data(), 1);
  } catch (const std::invalid_argument&) {
    host_rejected = true;
  }
  require(host_rejected && host_sentinel[0] == 123.0,
          "prepared CUDA interaction source silently published a host ERI");
}

void mixed_coulomb_preserves_strict_exchange() {
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
  system.electron_count = 2;
  system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  const std::size_t n = generativeqc::molecule::ao_count(system), matrix = n * n;
  const std::vector<double> density{0.9, 0.2, -0.1, 0.7};
  const std::vector<double> zeros(matrix, 0.0);
  CudaDirectJkPlan* raw{};
  CudaDirectJkDiagnostic diagnostic;
  require(create_cuda_direct_jk_plan(0, {system}, 0, 0.0, 32U * 1024U * 1024U, &raw, diagnostic,
                                     detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
      raw, &destroy_cuda_direct_jk_plan);
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.derivative_order = 0;
  DeviceMatrix device_density(density), strict_j(zeros), strict_k(zeros), mixed_j(zeros),
      mixed_k(zeros), error({0.0});
  DeviceCounter count;
  auto* failure = reinterpret_cast<int*>(error.pointer);
  require(enqueue_cuda_direct_jk_device(plan.get(), spec, device_density.pointer, nullptr, matrix,
                                        strict_j.pointer, strict_k.pointer, nullptr, failure,
                                        detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  check(cudaStreamSynchronize(cuda_direct_jk_stream(plan.get())));
  count.set(0);
  require(enqueue_cuda_direct_jk_device_mixed_j(plan.get(), spec, device_density.pointer, nullptr,
                                                matrix, mixed_j.pointer, mixed_k.pointer, nullptr,
                                                failure, detail,
                                                count.pointer) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  check(cudaStreamSynchronize(cuda_direct_jk_stream(plan.get())));
  require(count.get() > 0, "mixed-J hybrid did not execute mixed Coulomb work");
  std::vector<double> strict_exchange(matrix), mixed_exchange(matrix);
  check(cudaMemcpy(strict_exchange.data(), strict_k.pointer, matrix * sizeof(double),
                   cudaMemcpyDeviceToHost));
  check(cudaMemcpy(mixed_exchange.data(), mixed_k.pointer, matrix * sizeof(double),
                   cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < matrix; ++i)
    require(std::isfinite(strict_exchange[i]) && std::isfinite(mixed_exchange[i]) &&
                std::abs(strict_exchange[i] - mixed_exchange[i]) < 3e-12,
            "mixed-J hybrid changed strict FP64 exchange");
}

void direct_value_dispatch_selection() {
  const auto hybrid = direct_jk_value_dispatch(true, true, true, false);
  require(hybrid.generated_coulomb && !hybrid.generic_coulomb && hybrid.generic_exchange,
          "generated-capable hybrid J/K did not split J from exact K");

  const auto mixed = direct_jk_value_dispatch(true, true, true, true);
  require(!mixed.generated_coulomb && mixed.generic_coulomb && mixed.generic_exchange,
          "mixed-J hybrid escaped the generic J/K path");

  const auto fallback = direct_jk_value_dispatch(false, true, true, false);
  require(!fallback.generated_coulomb && fallback.generic_coulomb && fallback.generic_exchange,
          "missing generated capacity did not retain generic J/K fallback");

  const auto pure_j = direct_jk_value_dispatch(true, true, false, false);
  require(pure_j.generated_coulomb && !pure_j.generic_coulomb && !pure_j.generic_exchange,
          "pure J no longer selects the generated Coulomb consumer");

  const auto pure_k = direct_jk_value_dispatch(true, false, true, false);
  require(!pure_k.generated_coulomb && !pure_k.generic_coulomb && pure_k.generic_exchange,
          "K-only request selected an unrelated Coulomb consumer");

  const auto generated_hybrid = direct_jk_value_dispatch(true, true, true, true, false);
  require(generated_hybrid.generated_coulomb && generated_hybrid.generated_exchange &&
              !generated_hybrid.generic_coulomb && !generated_hybrid.generic_exchange,
          "qualified generated J/K did not stay fully generated");

  const auto generated_k = direct_jk_value_dispatch(true, true, false, true, false);
  require(!generated_k.generated_coulomb && generated_k.generated_exchange &&
              !generated_k.generic_coulomb && !generated_k.generic_exchange,
          "qualified K-only request did not select generated exchange");

  const auto generated_mixed = direct_jk_value_dispatch(true, true, true, true, true);
  require(!generated_mixed.generated_coulomb && !generated_mixed.generated_exchange &&
              generated_mixed.generic_coulomb && generated_mixed.generic_exchange,
          "mixed-J request incorrectly entered generated exchange route");
}

void device_selection() {
  const std::vector<double> metric{2.0, 0.1, 0.1, 1.3};
  const std::vector<double> tensor{1.3, 0.2, 0.3, -0.1, 0.3, -0.1, 0.8, 0.6};
  const std::vector<double> a{1.2, 0.31, -0.07, 0.7}, b{0.1, -0.05, 0.13, 0.4};
  const auto transformed = orthonormalize_density_fitting_three_center(
      tensor, 2, factor_density_fitting_metric(metric, 2));
  const auto rhf = build_density_fitting_rhf_jk(transformed, a);
  const auto uhf = build_density_fitting_uhf_jk(transformed, a, b);
  for (const auto layout : {FockMatrixLayout::RowMajor, FockMatrixLayout::ColumnMajor})
    for (std::size_t tile : {0U, 1U}) {
      CudaDensityFittingJkPlan* raw{};
      std::string detail;
      std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
      require(create_cuda_density_fitting_jk_plan_tiled(0, 1, 2, 2, metric, tensor, 1e-10, tile,
                                                        tile, &raw, diagnostics,
                                                        detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>
          plan(raw, &destroy_cuda_density_fitting_jk_plan);
      auto input_a = a, input_b = b;
      if (layout == FockMatrixLayout::ColumnMajor) {
        std::swap(input_a[1], input_a[2]);
        std::swap(input_b[1], input_b[2]);
      }
      DeviceMatrix da(input_a), db(input_b);
      for (bool j : {false, true})
        for (bool k : {false, true}) {
          const JkTermSelection terms{j, k};
          const std::vector<double> sentinel(4, 123.0);
          DeviceMatrix dj(sentinel), dka(sentinel), dkb(sentinel);
          require(execute_cuda_density_fitting_rhf_jk_device(
                      plan.get(), da.pointer, j ? dj.pointer : nullptr, k ? dka.pointer : nullptr,
                      detail, terms, layout) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());  // Test boundary observes the plan's nonblocking stream.
          dj.verify(j ? rhf.coulomb : sentinel);
          dka.verify(k ? rhf.exchange : sentinel);
          // Give unselected outputs valid sentinels this time: the service must
          // neither write them nor assume non-null means a term was requested.
          require(execute_cuda_density_fitting_uhf_jk_device(
                      plan.get(), da.pointer, db.pointer, dj.pointer, dka.pointer, dkb.pointer,
                      detail, terms, layout) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());
          dj.verify(j ? uhf.coulomb : sentinel);
          dka.verify(k ? uhf.alpha_exchange : sentinel);
          dkb.verify(k ? uhf.beta_exchange : sentinel);
        }
      require(execute_cuda_density_fitting_rhf_jk_device(plan.get(), da.pointer, nullptr, nullptr,
                                                         detail, {true, false}) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device J accepted");
      require(execute_cuda_density_fitting_uhf_jk_device(
                  plan.get(), da.pointer, db.pointer, nullptr, nullptr, nullptr, detail,
                  {false, true}) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device K accepted");
      da.verify(input_a);
      db.verify(input_b);
    }
}

std::vector<double> reference_exchange_from_eri(const std::vector<double>& eri,
                                                std::size_t dimension,
                                                const std::vector<double>& density) {
  std::vector<double> exchange(dimension * dimension);
  for (std::size_t first = 0; first < dimension; ++first)
    for (std::size_t second = 0; second < dimension; ++second)
      for (std::size_t third = 0; third < dimension; ++third)
        for (std::size_t fourth = 0; fourth < dimension; ++fourth)
          exchange[first * dimension + second] +=
              density[third * dimension + fourth] *
              eri[((first * dimension + third) * dimension + second) * dimension + fourth];
  return exchange;
}

std::vector<double> reference_range_exchange(const generativeqc::core::System& system,
                                             const std::vector<double>& density,
                                             generativeqc::integrals::CoulombRange range,
                                             double omega) {
  const auto eri = range == generativeqc::integrals::CoulombRange::Full
                       ? generativeqc::integrals::build_integrals(system, false, true).eri
                       : generativeqc::integrals::build_range_eri(system, range, omega);
  return reference_exchange_from_eri(eri, generativeqc::molecule::ao_count(system), density);
}

/** Deny each real optional allocation through the external resource ledger,
 * while leaving the provider's own budget unchanged. Check the usable fallback
 * and its retained charge, rather than only exercising shape-budget rejection.
 */
void spd_optional_allocation_fallback() {
  namespace runtime = generativeqc::runtime;
  struct LedgerScope {
    std::shared_ptr<runtime::DeviceResourceLedger> previous{runtime::active_device_resource_ledger};
    std::shared_ptr<runtime::DeviceResourceLedger> ledger{
        std::make_shared<runtime::DeviceResourceLedger>()};
    explicit LedgerScope(std::size_t limit) {
      ledger->limit = limit;
      ledger->device = 0;
      runtime::active_device_resource_ledger = ledger;
    }
    ~LedgerScope() { runtime::active_device_resource_ledger = previous; }
  };
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, 2, {{0.5, 1.0}}}};
  system.electron_count = 2;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  constexpr std::size_t budget = 64U << 20;
  using Plan = std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)>;
  std::vector<std::size_t> limits;
  std::size_t canonical_begin{}, rows_begin{}, metadata_begin{};
  {
    LedgerScope scope(budget);
    CudaDirectJkPlan* raw{};
    CudaDirectJkDiagnostic diagnostic;
    require(create_cuda_direct_jk_plan(0, {system}, 1, 0.0, budget, &raw, diagnostic, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    Plan plan(raw, &destroy_cuda_direct_jk_plan);
    require(plan->generated_exchange && plan->canonical_cartesian && plan->canonical_pairs &&
                plan->canonical_row_prefix && plan->batch.shell_ao_offsets,
            "allocation fault baseline lacks SPD/canonical/row/derivative storage");
    std::size_t prefix = plan->generated_exchange->device_bytes;
    for (std::size_t index = 0; index < plan->allocations.size(); ++index) {
      const auto* pointer = plan->allocations[index];
      if (pointer == plan->canonical_batch.shell_direct_ao_offsets) canonical_begin = index;
      if (pointer == plan->canonical_exchange) rows_begin = index + 1U;
      if (pointer == plan->batch.shell_ao_offsets) metadata_begin = index;
      std::lock_guard<std::mutex> lock(runtime::device_resource_mutex);
      const auto found = runtime::device_allocation_owners.find(plan->allocations[index]);
      require(found != runtime::device_allocation_owners.end(), "unregistered provider allocation");
      prefix += found->second.bytes;
      limits.push_back(prefix - 1U);
    }
    require(canonical_begin && canonical_begin < rows_begin && rows_begin < metadata_begin &&
                metadata_begin < limits.size() && prefix == diagnostic.device_bytes &&
                prefix == scope.ledger->live,
            "provider allocation inventory does not match its retained charge");
    plan.reset();
    require(scope.ledger->live == 0, "baseline provider leaked tracked device storage");
  }
  const auto n = generativeqc::molecule::ao_count(system);
  std::vector<double> density(n * n);
  for (std::size_t index = 0; index < density.size(); ++index)
    density[index] = std::cos(0.31 * (index / n) + 0.17 * (index % n)) / n;
  const std::array<std::vector<double>, 3> eri{
      generativeqc::integrals::build_integrals(system, false).eri,
      generativeqc::integrals::build_range_eri(system, generativeqc::integrals::CoulombRange::Short,
                                               0.37),
      generativeqc::integrals::build_range_eri(system, generativeqc::integrals::CoulombRange::Long,
                                               0.37)};
  for (std::size_t index = canonical_begin; index < limits.size(); ++index) {
    LedgerScope scope(limits[index]);
    CudaDirectJkPlan* raw{};
    CudaDirectJkDiagnostic diagnostic;
    require(create_cuda_direct_jk_plan(0, {system}, 1, 0.0, budget, &raw, diagnostic, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    Plan plan(raw, &destroy_cuda_direct_jk_plan);
    require(scope.ledger->rejected > 0 && detail.empty() &&
                diagnostic.device_bytes == scope.ledger->live &&
                diagnostic.device_bytes <= limits[index] && plan->generated_exchange &&
                plan->generated_exchange->force_capability &&
                direct_jk_generated_full_range_value_available(*plan),
            "optional allocation failure lost the generated owner or its resource accounting");
    require(bool(plan->canonical_pairs) == (index >= rows_begin) &&
                bool(plan->canonical_row_prefix) == (index >= metadata_begin),
            "optional allocation failure left a partial source or discarded its earlier owner");
    if (index < rows_begin)
      require(!plan->canonical_cartesian && !plan->canonical_transform &&
                  !plan->canonical_density && plan->canonical_pair_offsets.empty(),
              "failed canonical source retained stale availability metadata");
    if (index >= metadata_begin)
      require(!plan->batch.shell_ao_offsets && !plan->batch.shell_pair_first &&
                  !plan->batch.shell_pair_second && !plan->batch.total_shell_pairs,
              "failed derivative metadata retained dangling pointers");
    std::size_t op_index = 0;
    for (auto op : {FockOperator::FullRange, FockOperator::ShortRange, FockOperator::LongRange}) {
      auto spec = make_hf_fock_spec(FockSpin::Restricted);
      spec.coulomb.present = false;
      spec.exchange.op = op;
      spec.exchange.omega = op == FockOperator::FullRange ? 0.0 : 0.37;
      const auto expected = reference_exchange_from_eri(eri[op_index++], n, density);
      direct_device(plan.get(), spec, density, {}, {}, expected, {});
    }
    plan.reset();
    require(scope.ledger->live == 0, "failed optional preparation leaked tracked device storage");
  }
  // Through-f canonical preparation retains its established required behavior.
  system.shells[1].angular_momentum = 3;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  const auto minimum = cuda_direct_jk_device_bytes(1, generativeqc::molecule::ao_count(system), 2,
                                                   system.shells.size(), 2, 1);
  LedgerScope scope(minimum);
  CudaDirectJkPlan* raw{};
  CudaDirectJkDiagnostic diagnostic;
  const auto status =
      create_cuda_direct_jk_plan(0, {system}, 1, 0.0, budget, &raw, diagnostic, detail);
  Plan plan(raw, &destroy_cuda_direct_jk_plan);
  require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && !plan && scope.ledger->rejected > 0 &&
              scope.ledger->live == 0,
          "through-f required allocation failure was hidden or leaked storage");
}

/** SPD full-range coverage must not send SR/LR back to ordered AO^4 work.
 * Compare independently formed CPU ERIs at two geometries, including arbitrary
 * density orientation. Count actual source work separately from endpoint time.
 */
void spd_canonical_range_values() {
  for (unsigned angular : {0U, 1U, 2U}) {
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      generativeqc::core::System first;
      first.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
      first.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, angular, {{0.5, 1.0}}}};
      if (angular == 2U) first.shells.push_back({0, 1, {{0.7, 1.0}}});
      first.electron_count = 2;
      first.basis_representation = representation;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(first, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      auto second = first;
      second.atoms[1].position[2] += 0.17;
      const auto n = generativeqc::molecule::ao_count(first);
      const auto matrix = n * n;
      std::vector<double> alpha(2U * matrix), beta(2U * matrix);
      for (std::size_t index = 0; index < alpha.size(); ++index) {
        alpha[index] = std::cos(0.31 * (index / n) + 0.17 * (index % n)) / n;
        beta[index] = std::sin(0.23 * (index / n) - 0.37 * (index % n)) / n;
      }
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      constexpr std::size_t budget = 64U << 20;
      require(create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, budget, &raw, diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(plan->canonical_pairs && plan->canonical_cartesian && plan->canonical_row_prefix &&
                  plan->generated_exchange && plan->generated_exchange->force_capability &&
                  direct_jk_generated_full_range_value_available(*plan) &&
                  diagnostic.device_bytes <= budget,
              "SPD range source displaced generated full-range ownership or exceeded budget");
      std::size_t primitives = 0;
      for (const auto& shell : first.shells) primitives += 2U * shell.primitives.size();
      const auto minimum =
          cuda_direct_jk_device_bytes(2, n, 4, 2U * first.shells.size(), primitives, 1);
      // At the exact old generated-owner capacity, the new optional source
      // must disappear while the full-range value/force owner remains intact.
      const auto generated_budget = minimum + plan->generated_exchange->device_bytes;
      CudaDirectJkPlan* constrained_raw{};
      CudaDirectJkDiagnostic constrained_diagnostic;
      require(
          create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, generated_budget, &constrained_raw,
                                     constrained_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> constrained(
          constrained_raw, &destroy_cuda_direct_jk_plan);
      require(!constrained->canonical_pairs &&
                  direct_jk_generated_full_range_value_available(*constrained) &&
                  constrained->generated_exchange->force_capability &&
                  constrained_diagnostic.device_bytes == generated_budget,
              "optional canonical range storage displaced the constrained SPD owner");
      DeviceMatrix census(std::vector<double>(2U, 0.0));
      plan->canonical_work_count = reinterpret_cast<std::uint64_t*>(census.pointer);
      const auto source_n = static_cast<std::size_t>(plan->canonical_batch.nbf);
      const auto pairs = source_n * (source_n + 1U) / 2U;
      const auto quartets = 2U * pairs * (pairs + 1U) / 2U;
      const std::array<std::vector<double>, 2> full_eri{
          generativeqc::integrals::build_integrals(first, false).eri,
          generativeqc::integrals::build_integrals(second, false).eri};
      for (auto op : {FockOperator::FullRange, FockOperator::ShortRange, FockOperator::LongRange}) {
        constexpr double omega = 0.37;
        const auto range = op == FockOperator::ShortRange
                               ? generativeqc::integrals::CoulombRange::Short
                               : generativeqc::integrals::CoulombRange::Long;
        const std::array<std::vector<double>, 2> range_eri{
            op == FockOperator::FullRange
                ? full_eri[0]
                : generativeqc::integrals::build_range_eri(first, range, omega),
            op == FockOperator::FullRange
                ? full_eri[1]
                : generativeqc::integrals::build_range_eri(second, range, omega)};
        for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted})
          for (bool want_j : {false, true})
            for (bool want_k : {false, true}) {
              auto spec = make_hf_fock_spec(spin);
              spec.coulomb.present = want_j;
              spec.exchange.present = want_k;
              spec.exchange.op = op;
              spec.exchange.omega = op == FockOperator::FullRange ? 0.0 : omega;
              std::vector<double> expected_j, expected_a, expected_b;
              for (std::size_t item = 0; item < 2U; ++item) {
                const std::vector<double> a(alpha.begin() + item * matrix,
                                            alpha.begin() + (item + 1U) * matrix);
                const std::vector<double> b(beta.begin() + item * matrix,
                                            beta.begin() + (item + 1U) * matrix);
                if (want_j) {
                  auto j_spec = make_hf_fock_spec(spin);
                  j_spec.exchange.present = false;
                  const auto j = build_exact_direct_jk(
                      resolve_fock_build(j_spec, FockBackend::Cpu, 0.0), n, full_eri[item], a,
                      spin == FockSpin::Unrestricted ? b : std::vector<double>{});
                  expected_j.insert(expected_j.end(), j.coulomb.begin(), j.coulomb.end());
                }
                if (want_k) {
                  const auto ka = reference_exchange_from_eri(range_eri[item], n, a);
                  expected_a.insert(expected_a.end(), ka.begin(), ka.end());
                  if (spin == FockSpin::Unrestricted) {
                    const auto kb = reference_exchange_from_eri(range_eri[item], n, b);
                    expected_b.insert(expected_b.end(), kb.begin(), kb.end());
                  }
                }
              }
              direct_device(plan.get(), spec, alpha, beta, expected_j, expected_a, expected_b);
              std::array<std::uint64_t, 2> work{};
              check(cudaMemcpy(work.data(), plan->canonical_work_count, sizeof(work),
                               cudaMemcpyDeviceToHost));
              const bool canonical = want_k && op != FockOperator::FullRange;
              require(work[0] == (canonical ? quartets : 0U) &&
                          work[1] == (canonical ? quartets * (want_j ? 2U : 1U) : 0U),
                      "SPD generated/canonical selection disagrees with actual source work");
              direct_device(constrained.get(), spec, alpha, beta, expected_j, expected_a,
                            expected_b);
            }
      }
      std::cout << "{\"spd_source_aos\":" << source_n << ",\"batch\":2,\"lr_quartets\":" << quartets
                << ",\"device_bytes\":" << diagnostic.device_bytes
                << ",\"generated_only_bytes\":" << generated_budget << "}\n";
    }
  }
}

/** Qualify the through-f shell value owner and retain the canonical source as
 * the independent range/resource fallback. CPU integral matrices are an oracle
 * only; production never borrows them. */
void canonical_value_provider() {
  for (unsigned angular : {0U, 1U, 2U, 3U}) {
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      generativeqc::core::System first;
      first.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
      first.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, angular, {{0.5, 1.0}}}};
      if (angular == 2U) first.shells.push_back({0, 1, {{0.7, 1.0}}});
      if (angular != 3U) first.shells.push_back({0, 3, {{0.9, 1.0}}});
      first.electron_count = 2;
      first.basis_representation = representation;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(first, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      auto second = first;
      second.atoms[1].position[2] += 0.17;
      const auto dimension = generativeqc::molecule::ao_count(first);
      const auto matrix = dimension * dimension;
      std::vector<double> alpha(2U * matrix), beta(2U * matrix);
      for (std::size_t index = 0; index < alpha.size(); ++index) {
        alpha[index] =
            std::cos(0.31 * (index / dimension) + 0.17 * (index % dimension)) / dimension;
        beta[index] = std::sin(0.23 * (index / dimension) - 0.37 * (index % dimension)) / dimension;
      }
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, 64U << 20, &raw, diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(plan->canonical_pairs && plan->canonical_cartesian && plan->canonical_row_prefix &&
                  plan->canonical_pair_offsets.size() == 2 && plan->generated_exchange &&
                  plan->generated_exchange->bounded_value_capability &&
                  plan->generated_exchange->shared &&
                  !plan->generated_exchange->shared->value_capability,
              "through-f shell owner/canonical fallback was not prepared");
      require(!plan->bounded_value_opt_in &&
                  !direct_jk_generated_full_range_value_available(*plan) &&
                  std::string(diagnostic.schedule).find("canonical-cartesian-jk/") == 0,
              "through-f production values did not retain the canonical default");
      require(diagnostic.device_bytes <= 64U << 20,
              "canonical source exceeded its explicit budget");
      DeviceMatrix census(std::vector<double>(2U, 0.0));
      plan->canonical_work_count = reinterpret_cast<std::uint64_t*>(census.pointer);
      const std::array<std::vector<double>, 2> full_eri{
          generativeqc::integrals::build_integrals(first, false).eri,
          generativeqc::integrals::build_integrals(second, false).eri};
      for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
        for (auto op :
             {FockOperator::FullRange, FockOperator::LongRange, FockOperator::ShortRange}) {
          const auto range =
              op == FockOperator::FullRange
                  ? generativeqc::integrals::CoulombRange::Full
                  : (op == FockOperator::LongRange ? generativeqc::integrals::CoulombRange::Long
                                                   : generativeqc::integrals::CoulombRange::Short);
          constexpr double omega = 0.37;
          const std::array<std::vector<double>, 2> range_eri{
              op == FockOperator::FullRange
                  ? full_eri[0]
                  : generativeqc::integrals::build_range_eri(first, range, omega),
              op == FockOperator::FullRange
                  ? full_eri[1]
                  : generativeqc::integrals::build_range_eri(second, range, omega)};
          for (bool want_j : {false, true}) {
            for (bool want_k : {false, true}) {
              auto spec = make_hf_fock_spec(spin);
              spec.coulomb.present = want_j;
              spec.exchange.present = want_k;
              spec.exchange.op = op;
              spec.exchange.omega = op == FockOperator::FullRange ? 0.0 : omega;
              std::vector<double> expected_j, expected_alpha, expected_beta;
              for (std::size_t item = 0; item < 2U; ++item) {
                const std::vector<double> density_alpha(alpha.begin() + item * matrix,
                                                        alpha.begin() + (item + 1U) * matrix);
                const std::vector<double> density_beta(beta.begin() + item * matrix,
                                                       beta.begin() + (item + 1U) * matrix);
                if (want_j) {
                  auto coulomb_spec = make_hf_fock_spec(spin);
                  coulomb_spec.derivative_order = 0;
                  coulomb_spec.exchange.present = false;
                  const auto jk = build_exact_direct_jk(
                      resolve_fock_build(coulomb_spec, FockBackend::Cpu, 0.0), dimension,
                      full_eri[item], density_alpha,
                      spin == FockSpin::Unrestricted ? density_beta : std::vector<double>{});
                  expected_j.insert(expected_j.end(), jk.coulomb.begin(), jk.coulomb.end());
                }
                if (want_k) {
                  const auto reference_alpha =
                      reference_exchange_from_eri(range_eri[item], dimension, density_alpha);
                  expected_alpha.insert(expected_alpha.end(), reference_alpha.begin(),
                                        reference_alpha.end());
                  if (spin == FockSpin::Unrestricted) {
                    const auto reference_beta =
                        reference_exchange_from_eri(range_eri[item], dimension, density_beta);
                    expected_beta.insert(expected_beta.end(), reference_beta.begin(),
                                         reference_beta.end());
                  }
                }
              }
              for (bool bounded_opt_in : {false, true}) {
                plan->bounded_value_opt_in = bounded_opt_in;
                direct_device(plan.get(), spec, alpha,
                              spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                              expected_j, expected_alpha, expected_beta);
                if (plan->canonical_work_count) {
                  std::array<std::uint64_t, 2> work{};
                  check(cudaMemcpy(work.data(), plan->canonical_work_count, sizeof(work),
                                   cudaMemcpyDeviceToHost));
                  const auto source_dimension = static_cast<std::size_t>(plan->canonical_batch.nbf);
                  const auto pairs = source_dimension * (source_dimension + 1U) / 2U;
                  const auto quartets = diagnostic.batch_size * pairs * (pairs + 1U) / 2U;
                  const bool canonical_range_exchange = want_k && op != FockOperator::FullRange;
                  const bool canonical_route = (want_j || want_k) && !bounded_opt_in;
                  const auto radial_passes =
                      canonical_route ? (canonical_range_exchange && want_j ? 2U : 1U) : 0U;
                  require(work[0] == (canonical_route ? quartets : 0U) &&
                              work[1] == radial_passes * quartets,
                          "default/opt-in value selection disagrees with executed canonical work");
                }
                if (want_j && want_k && op != FockOperator::FullRange) {
                  std::vector<double> expected_full_alpha, expected_full_beta;
                  for (std::size_t item = 0; item < 2U; ++item) {
                    const std::vector<double> item_alpha(alpha.begin() + item * matrix,
                                                         alpha.begin() + (item + 1U) * matrix);
                    const auto full_alpha =
                        reference_exchange_from_eri(full_eri[item], dimension, item_alpha);
                    expected_full_alpha.insert(expected_full_alpha.end(), full_alpha.begin(),
                                               full_alpha.end());
                    if (spin == FockSpin::Unrestricted) {
                      const std::vector<double> item_beta(beta.begin() + item * matrix,
                                                          beta.begin() + (item + 1U) * matrix);
                      const auto full_beta =
                          reference_exchange_from_eri(full_eri[item], dimension, item_beta);
                      expected_full_beta.insert(expected_full_beta.end(), full_beta.begin(),
                                                full_beta.end());
                    }
                  }
                  direct_rsh_device(plan.get(), spec, alpha,
                                    spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                                    expected_j, expected_full_alpha, expected_full_beta,
                                    expected_alpha, expected_beta);
                }
                if (want_k && plan->canonical_transform) {
                  const auto* spans = plan->canonical_projection_spans;
                  require(spans, "shell-local projection inventory was not retained");
                  const bool bounded = plan->bounded_value_opt_in;
                  plan->bounded_value_opt_in = false;
                  plan->canonical_projection_spans = nullptr;
                  direct_device(plan.get(), spec, alpha,
                                spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                                expected_j, expected_alpha, expected_beta);
                  plan->canonical_projection_spans = spans;
                  plan->bounded_value_opt_in = bounded;
                }
              }
            }
          }
        }
      }
      direct_device_failures(plan.get(), alpha);
      std::size_t primitive_count = 0;
      for (const auto& shell : first.shells) primitive_count += 2U * shell.primitives.size();
      const auto minimum_budget = cuda_direct_jk_device_bytes(
          2, dimension, 4, 2U * first.shells.size(), primitive_count, 1);
      CudaDirectJkPlan* fallback_raw{};
      CudaDirectJkDiagnostic fallback_diagnostic;
      require(
          create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, minimum_budget, &fallback_raw,
                                     fallback_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> fallback(
          fallback_raw, &destroy_cuda_direct_jk_plan);
      require(!fallback->canonical_pairs && fallback_diagnostic.device_bytes == minimum_budget,
              "insufficient optional capacity did not retain the bounded generic fallback");
      const auto dense_budget = minimum_budget +
                                2U * dimension * (dimension + 1U) * sizeof(std::int32_t) +
                                12U * matrix * sizeof(double);
      CudaDirectJkPlan* dense_raw{};
      require(
          create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, dense_budget, &dense_raw,
                                     fallback_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> dense(
          dense_raw, &destroy_cuda_direct_jk_plan);
      require(dense->canonical_pairs && !dense->canonical_cartesian &&
                  !dense->canonical_row_prefix && fallback_diagnostic.device_bytes == dense_budget,
              "insufficient span capacity did not retain the dense canonical fallback");
      auto spec = make_hf_fock_spec(FockSpin::Restricted);
      std::vector<double> expected_j, expected_k;
      for (std::size_t item = 0; item < 2U; ++item) {
        const std::vector<double> item_density(alpha.begin() + item * matrix,
                                               alpha.begin() + (item + 1U) * matrix);
        const auto reference =
            build_exact_direct_jk(resolve_fock_build(spec, FockBackend::Cpu, 0.0), dimension,
                                  full_eri[item], item_density);
        expected_j.insert(expected_j.end(), reference.coulomb.begin(), reference.coulomb.end());
        expected_k.insert(expected_k.end(), reference.exchange_alpha.begin(),
                          reference.exchange_alpha.end());
      }
      require(!dense->bounded_value_opt_in,
              "dense canonical fallback unexpectedly opted into bounded values");
      direct_device(dense.get(), spec, alpha, {}, expected_j, expected_k, {});
    }
  }
}

/** Reuse HF's prepared one-electron consumer through f without new H2D work.
 * Independently contract CPU analytic S/H derivatives with symmetric D/W. */
void canonical_one_electron_reuse() {
  for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
    generativeqc::core::System system;
    system.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
    system.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, 3, {{0.5, 1.0}}}};
    system.electron_count = 2;
    system.basis_representation = representation;
    std::string detail;
    require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    const auto dimension = generativeqc::molecule::ao_count(system);
    const auto matrix = dimension * dimension;
    std::vector<double> density(matrix), weighted_density(matrix);
    for (std::size_t pair = 0; pair < matrix; ++pair) {
      const auto coordinate = pair / dimension + pair % dimension;
      density[pair] = std::cos(0.31 * coordinate) / dimension;
      weighted_density[pair] = std::sin(0.17 * coordinate) / dimension;
    }
    DeviceMatrix input(density), weights(weighted_density);
    for (double displacement : {0.0, 0.27}) {
      auto moved = system;
      moved.atoms[1].position[2] += displacement;
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {moved}, 1, 0.0, 32U << 20, &raw, diagnostic, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(plan->generated_exchange && plan->generated_exchange->force_capability &&
                  plan->generated_exchange->bounded_value_capability &&
                  plan->generated_exchange->shared &&
                  !plan->generated_exchange->shared->value_capability &&
                  plan->batch.shell_ao_offsets && plan->batch.shell_pair_first &&
                  plan->batch.shell_pair_second,
              "through-f owner did not retain bounded value/force shell state and one-electron "
              "metadata");
      auto value_spec = make_hf_fock_spec(FockSpin::Restricted);
      value_spec.derivative_order = 0;
      require(!direct_jk_generated_exchange_value_available(*plan, value_spec),
              "bounded through-f values were selected without explicit qualification");
      plan->bounded_value_opt_in = true;
      require(direct_jk_generated_exchange_value_available(*plan, value_spec),
              "bounded through-f shell owner was not admitted for full-range value exchange");
      value_spec.derivative_order = 1;
      require(!direct_jk_generated_exchange_value_available(*plan, value_spec),
              "derivative request incorrectly selected the zero-order value route");
      plan->bounded_value_opt_in = false;
      std::vector<double> shell_rsh, canonical_rsh;
      constexpr double omega = 0.3;
      require(execute_cuda_direct_shell_rsh_energy_derivatives_device(
                  plan.get(), FockSpin::Restricted, 1.0, -0.37, -0.61, omega, input.pointer,
                  nullptr, matrix, shell_rsh, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      require(execute_cuda_direct_rsh_energy_derivatives_device(
                  plan.get(), FockSpin::Restricted, 1.0, -0.37, -0.61, omega, input.pointer,
                  nullptr, matrix, canonical_rsh, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      require(shell_rsh.size() == canonical_rsh.size(),
              "through-f shell/canonical RSH source shapes differ");
      for (std::size_t coordinate = 0; coordinate < shell_rsh.size(); ++coordinate)
        require(std::abs(shell_rsh[coordinate] - canonical_rsh[coordinate]) < 3e-10,
                "through-f shell RSH derivative differs from the canonical source");
      const auto oracle = generativeqc::integrals::build_integrals(moved, true, false);
      std::vector<double> hcore, pulay;
      OneElectronGradientResources resources;
      require(execute_prepared_cuda_stationary_one_electron_pair(
                  plan.get(), input.pointer, weights.pointer, matrix,
                  6U * system.atoms.size() * sizeof(double), hcore, pulay, detail,
                  &resources) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      for (std::size_t coordinate = 0; coordinate < 3U * system.atoms.size(); ++coordinate) {
        double expected_hcore = 0.0, expected_pulay = 0.0;
        for (std::size_t pair = 0; pair < matrix; ++pair) {
          expected_hcore += density[pair] * oracle.hcore_derivative[coordinate * matrix + pair];
          expected_pulay -=
              weighted_density[pair] * oracle.overlap_derivative[coordinate * matrix + pair];
        }
        require(std::abs(hcore[coordinate] - expected_hcore) < 3e-11 &&
                    std::abs(pulay[coordinate] - expected_pulay) < 3e-11,
                "reused through-f HF one-electron gradient differs from independent CPU");
      }
      require(resources.device_bytes == 0 && resources.host_to_device_bytes == 0 &&
                  resources.host_numeric_bytes == 6U * system.atoms.size() * sizeof(double) &&
                  resources.device_to_host_bytes == resources.host_numeric_bytes &&
                  resources.stream_synchronizations == 1,
              "through-f prepared one-electron source restaged metadata or allocated device work");
      input.verify(density);
      weights.verify(weighted_density);
    }
  }
  std::cout
      << "CUDA through-f shell RSH + HF one-electron reuse/CPU gradient/zero-H2D gates PASS\n";
}

/** Boundary fixtures include empty rows, equal keys, underflow and exact products.
 * A division-based threshold or lower_bound on inclusive prefixes fails here. */
void canonical_screening_rows() {
  using namespace generativeqc::scf::cuda_execution;
  const std::vector<double> keys{
      4.0, 1.3, 0.7, 0.0, 3.0, 1.3, 1.0, std::numeric_limits<double>::denorm_min(), 0.0};
  DeviceMatrix input(keys), prefix(std::vector<double>(4U));
  std::size_t workspace_bytes = 0;
  check(canonical_pair_workspace(static_cast<int>(keys.size()), 2, 4, workspace_bytes));
  DeviceMatrix workspace(std::vector<double>((workspace_bytes + 7U) / 8U));
  const double boundary = 1.3 * 0.7;
  for (bool same_bucket : {false, true}) {
    const std::size_t second_begin = same_bucket ? 0U : 4U;
    const std::size_t second_count = same_bucket ? 4U : 5U;
    for (double screening :
         {0.0, boundary, std::nextafter(boundary, 0.0), std::nextafter(boundary, 2.0), 16.0,
          std::nextafter(16.0, 17.0), std::numeric_limits<double>::max()}) {
      auto* rows = reinterpret_cast<std::uint64_t*>(prefix.pointer);
      check(prepare_canonical_pair_rows(nullptr, input.pointer, 0, 4, second_begin, second_count,
                                        same_bucket, screening, rows, workspace.pointer,
                                        workspace_bytes));
      std::array<std::uint64_t, 4> actual{};
      check(cudaMemcpy(actual.data(), rows, sizeof(actual), cudaMemcpyDeviceToHost));
      std::uint64_t expected = 0;
      for (std::size_t row = 0; row < 4U; ++row) {
        for (std::size_t ket = 0; ket < (same_bucket ? row + 1U : second_count); ++ket)
          if (!(keys[row] * keys[second_begin + ket] < screening)) ++expected;
        require(actual[row] == expected, "screened row changed the exact FP64 product predicate");
      }
    }
  }
  std::cout << "CUDA screened-row product/equality/empty-row gates PASS\n";
}

/** Compare actual screened work and complete J/K with independent CPU ERIs.
 * Screened and dense canonical schedules must admit the same physical quartets. */
/** Independent CPU Cartesian oracle for the screened, projected public ERIs.
 * Construct the spherical expansion from molecule algebra, not native packed
 * transforms. Screening is deliberately applied before the four-index
 * projection: a Cartesian discrete-screening predicate is not a public one.
 */
std::vector<double> screened_cartesian_public_eri(const generativeqc::core::System& system,
                                                  const std::vector<double>& source,
                                                  const std::vector<double>& bounds,
                                                  double screening) {
  const auto dimension = generativeqc::molecule::ao_count(system);
  const auto source_dimension = generativeqc::molecule::cartesian_ao_count(system);
  const auto matrix = dimension * dimension;
  const auto source_matrix = source_dimension * source_dimension;
  std::vector<std::vector<std::pair<std::size_t, double>>> expansions;
  std::size_t shell_offset = 0;
  for (const auto& shell : system.shells) {
    const auto components = generativeqc::molecule::cartesian_components(shell.angular_momentum);
    for (const auto& expansion : generativeqc::molecule::ao_expansions(
             shell.angular_momentum, system.basis_representation)) {
      auto& terms = expansions.emplace_back();
      for (const auto& term : expansion) {
        const auto found = std::find(components.begin(), components.end(), term.component);
        require(found != components.end(), "CPU spherical component missing from Cartesian basis");
        terms.emplace_back(shell_offset + static_cast<std::size_t>(found - components.begin()),
                           term.coefficient);
      }
    }
    shell_offset += components.size();
  }
  std::vector<double> projected(matrix * matrix);
  for (std::size_t first = 0; first < dimension; ++first)
    for (std::size_t second = 0; second < dimension; ++second)
      for (std::size_t third = 0; third < dimension; ++third)
        for (std::size_t fourth = 0; fourth < dimension; ++fourth) {
          double value = 0.0;
          for (const auto& [source_first, first_coefficient] : expansions[first])
            for (const auto& [source_second, second_coefficient] : expansions[second])
              for (const auto& [source_third, third_coefficient] : expansions[third])
                for (const auto& [source_fourth, fourth_coefficient] : expansions[fourth]) {
                  const auto bra = source_first * source_dimension + source_second;
                  const auto ket = source_third * source_dimension + source_fourth;
                  if (bounds[bra] * bounds[ket] < screening) continue;
                  value += first_coefficient * second_coefficient * third_coefficient *
                           fourth_coefficient * source[bra * source_matrix + ket];
                }
          projected[(first * dimension + second) * matrix + third * dimension + fourth] = value;
        }
  return projected;
}

void canonical_screened_values() {
  for (double displacement : {0.0, 0.27}) {
    generativeqc::core::System system;
    system.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7 + displacement}}};
    system.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, 3, {{0.5, 1.0}}}};
    system.electron_count = 2;
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    std::string detail;
    require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    const auto dimension = generativeqc::molecule::ao_count(system);
    const auto matrix = dimension * dimension;
    auto cartesian = system;
    cartesian.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
    const auto source_dimension = generativeqc::molecule::ao_count(cartesian);
    const auto source_matrix = source_dimension * source_dimension;
    const auto full = generativeqc::integrals::build_integrals(cartesian, false).eri;
    std::vector<double> bounds(source_matrix), alpha(matrix), beta(matrix);
    for (std::size_t pair = 0; pair < source_matrix; ++pair)
      bounds[pair] = std::sqrt(std::abs(full[pair * source_matrix + pair]));
    for (std::size_t pair = 0; pair < matrix; ++pair) {
      alpha[pair] = std::cos(0.31 * (pair / dimension) + 0.17 * (pair % dimension)) / dimension;
      beta[pair] = std::sin(0.23 * (pair / dimension) - 0.37 * (pair % dimension)) / dimension;
    }
    for (double screening : {0.08, 0.9}) {
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      const auto status = create_cuda_direct_jk_plan(0, {system}, 0, screening, 32U << 20, &raw,
                                                     diagnostic, detail);
      require(status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(plan->canonical_cartesian && plan->canonical_row_prefix && plan->generated_exchange &&
                  plan->generated_exchange->bounded_value_capability &&
                  plan->generated_exchange->shared &&
                  !plan->generated_exchange->shared->value_capability,
              "screened canonical fallback and competing through-f shell owner were not prepared");
      // This oracle screens Cartesian AO pairs before public projection. The
      // opt-in shell provider applies a different shell+density predicate.
      // Select only this fixture's canonical fallback; canonical_value_provider
      // independently checks both default canonical and opt-in bounded values.
      plan->bounded_value_opt_in = false;
      require(!direct_jk_generated_full_range_value_available(*plan),
              "canonical screening fixture still selects a competing full-range shell source");
      DeviceMatrix census(std::vector<double>(2U));
      plan->canonical_work_count = reinterpret_cast<std::uint64_t*>(census.pointer);
      std::vector<std::size_t> pairs;
      for (std::size_t first = 0; first < source_dimension; ++first)
        for (std::size_t second = 0; second <= first; ++second)
          pairs.push_back(first * source_dimension + second);
      std::uint64_t admitted = 0;
      for (std::size_t bra = 0; bra < pairs.size(); ++bra)
        for (std::size_t ket = 0; ket <= bra; ++ket)
          if (!(bounds[pairs[bra]] * bounds[pairs[ket]] < screening)) ++admitted;
      require(admitted < pairs.size() * (pairs.size() + 1U) / 2U,
              "screened fixture did not reject any quartets");
      for (auto operation :
           {FockOperator::FullRange, FockOperator::ShortRange, FockOperator::LongRange}) {
        const auto range = operation == FockOperator::ShortRange
                               ? generativeqc::integrals::CoulombRange::Short
                               : generativeqc::integrals::CoulombRange::Long;
        const auto screened_full = screened_cartesian_public_eri(system, full, bounds, screening);
        const auto screened_range =
            operation == FockOperator::FullRange
                ? screened_full
                : screened_cartesian_public_eri(
                      system, generativeqc::integrals::build_range_eri(cartesian, range, 0.37),
                      bounds, screening);
        for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
          auto spec = make_hf_fock_spec(spin);
          auto j_spec = spec;
          j_spec.exchange.present = false;
          const auto reference = build_exact_direct_jk(
              resolve_fock_build(j_spec, FockBackend::Cpu, 0.0), dimension, screened_full, alpha,
              spin == FockSpin::Unrestricted ? beta : std::vector<double>{});
          const auto ka = reference_exchange_from_eri(screened_range, dimension, alpha);
          const auto kb = spin == FockSpin::Unrestricted
                              ? reference_exchange_from_eri(screened_range, dimension, beta)
                              : std::vector<double>{};
          spec.exchange.op = operation;
          spec.exchange.omega = operation == FockOperator::FullRange ? 0.0 : 0.37;
          direct_device(plan.get(), spec, alpha,
                        spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                        reference.coulomb, ka, kb);
          std::array<std::uint64_t, 2> work{};
          check(cudaMemcpy(work.data(), plan->canonical_work_count, sizeof(work),
                           cudaMemcpyDeviceToHost));
          require(work[0] == admitted &&
                      work[1] == admitted * (operation == FockOperator::FullRange ? 1U : 2U),
                  "screened canonical source still visited rejected quartets");
          const auto* prefix = plan->canonical_row_prefix;
          plan->canonical_row_prefix = nullptr;
          direct_device(plan.get(), spec, alpha,
                        spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                        reference.coulomb, ka, kb);
          plan->canonical_row_prefix = prefix;
        }
      }
    }
  }
  std::cout << "CUDA screened full/SR/LR matrices and work gates PASS\n";
}

/** Count executed candidates and radial evaluations at two larger dimensions.
 * The single-primitive fixture isolates scheduling, not the OMol25 endpoint.
 * Independent CPU value/force oracles remain in the separate numerical gates. */
/** Exercise order-two derivative seeds inside a Cartesian through-f owner.
 * The spectator f density is zero. An independent compact s+p / s+d CPU
 * system therefore supplies finite-difference energies without borrowing the
 * native projection or the full-range order-two shortcut being tested.
 */
void canonical_order_two_derivatives() {
  constexpr double step = 1e-4, omega = 0.3, short_coefficient = -0.37, long_coefficient = -0.61;
  for (unsigned angular : {1U, 2U})
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      generativeqc::core::System compact;
      compact.atoms = {{1, {0.1, -0.2, -0.8}}, {1, {0.3, 0.1, 0.7}}};
      compact.shells = {{0, 0, {{0.8, 1.0}}}, {1, angular, {{0.6, 1.0}}}};
      compact.electron_count = 2;
      compact.basis_representation = representation;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(compact, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      auto first = compact;
      first.shells.push_back({0, 3, {{0.7, 1.0}}});
      auto second = first;
      second.atoms[1].position[0] += 0.23;
      const auto dimension = generativeqc::molecule::ao_count(first);
      const auto compact_dimension = generativeqc::molecule::ao_count(compact);
      const auto matrix = dimension * dimension;
      std::vector<double> alpha(matrix), beta(matrix),
          compact_alpha(compact_dimension * compact_dimension), compact_beta(compact_alpha.size());
      for (std::size_t row = 0; row < compact_dimension; ++row)
        for (std::size_t column = 0; column < compact_dimension; ++column) {
          alpha[row * dimension + column] = compact_alpha[row * compact_dimension + column] =
              std::cos(0.3 * (row + column)) / compact_dimension;
          beta[row * dimension + column] = compact_beta[row * compact_dimension + column] =
              std::sin(0.4 * (row + column)) / (2 * compact_dimension);
        }
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      const auto status = create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, 32U << 20, &raw,
                                                     diagnostic, detail);
      require(status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(plan->canonical_cartesian, "order-two gate missed the Cartesian source");
      for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted})
        for (std::size_t item = 0; item < 2; ++item) {
          std::vector<double> fused;
          const auto derivative_status = execute_cuda_direct_rsh_energy_derivatives_item(
              plan.get(), item, spin, 1.0, short_coefficient, long_coefficient, omega, alpha,
              spin == FockSpin::Unrestricted ? beta : std::vector<double>{}, fused, detail);
          require(derivative_status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
          const auto energy = [&](double shift) {
            auto displaced = compact;
            displaced.atoms[1].position[0] += (item ? 0.23 : 0.0) + shift;
            const auto full = generativeqc::integrals::build_integrals(displaced, false).eri;
            auto spec = make_hf_fock_spec(spin);
            const auto jk = build_exact_direct_jk(
                resolve_fock_build(spec, FockBackend::Cpu, 0.0), compact_dimension, full,
                compact_alpha,
                spin == FockSpin::Unrestricted ? compact_beta : std::vector<double>{});
            std::array<double, 3> values{};
            for (std::size_t pair = 0; pair < compact_alpha.size(); ++pair)
              values[0] += 0.5 * jk.coulomb[pair] *
                           (compact_alpha[pair] +
                            (spin == FockSpin::Unrestricted ? compact_beta[pair] : 0.0));
            for (unsigned radial = 0; radial < 2U; ++radial) {
              const auto eri = generativeqc::integrals::build_range_eri(
                  displaced,
                  radial ? generativeqc::integrals::CoulombRange::Long
                         : generativeqc::integrals::CoulombRange::Short,
                  omega);
              const auto ka = reference_exchange_from_eri(eri, compact_dimension, compact_alpha);
              const auto kb = reference_exchange_from_eri(eri, compact_dimension, compact_beta);
              for (std::size_t pair = 0; pair < compact_alpha.size(); ++pair)
                values[radial + 1U] +=
                    0.5 * (radial ? long_coefficient : short_coefficient) *
                    (compact_alpha[pair] * ka[pair] +
                     (spin == FockSpin::Unrestricted ? compact_beta[pair] * kb[pair] : 0.0));
            }
            return values;
          };
          const auto plus = energy(step), minus = energy(-step);
          for (unsigned source = 0; source < 3U; ++source) {
            const auto finite_difference = (plus[source] - minus[source]) / (2 * step);
            require(std::abs(fused[source * 6U + 3U] - finite_difference) < 3e-8,
                    "Cartesian order-two derivative discarded a seed or changed spin weights");
          }
        }
    }
  std::cout << "CUDA Cartesian order-two derivative seed/CPU finite-difference gates PASS\n";
}

void canonical_work_census() {
  for (const std::size_t dimension : {58U, 116U}) {
    generativeqc::core::System system;
    system.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
    system.electron_count = 2;
    system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    for (std::size_t shell = 0; shell < dimension - 26U; ++shell)
      system.shells.push_back({static_cast<unsigned>(shell % 2U), 0, {{0.4 + 0.01 * shell, 1.0}}});
    for (unsigned angular : {1U, 2U, 3U})
      for (unsigned shell = 0; shell < 4U - angular; ++shell)
        system.shells.push_back({shell % 2U, angular, {{0.6 + 0.1 * shell, 1.0}}});
    std::string detail;
    const auto basis_status = generativeqc::molecule::validate_and_normalize(system, detail);
    require(basis_status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
    require(generativeqc::molecule::ao_count(system) == dimension, "census fixture AO drift");
    CudaDirectJkPlan* raw{};
    CudaDirectJkDiagnostic diagnostic;
    const auto preparation_status =
        create_cuda_direct_jk_plan(0, {system}, 0, 0.0, 64U << 20, &raw, diagnostic, detail);
    require(preparation_status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
    std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
        raw, &destroy_cuda_direct_jk_plan);
    require(plan->canonical_pairs && plan->canonical_cartesian && plan->generated_exchange &&
                plan->generated_exchange->bounded_value_capability &&
                plan->generated_exchange->shared &&
                !plan->generated_exchange->shared->value_capability,
            "through-f bounded shell value owner/canonical range source was not prepared");
    DeviceMatrix census(std::vector<double>(2U, 0.0));
    plan->canonical_work_count = reinterpret_cast<std::uint64_t*>(census.pointer);
    const auto matrix = dimension * dimension;
    std::vector<double> density(matrix);
    for (std::size_t index = 0; index < matrix; ++index)
      density[index] =
          std::cos(0.31 * (index / dimension) + 0.17 * (index % dimension)) / dimension;
    DeviceMatrix input(density), coulomb(std::vector<double>(matrix, 0.0)),
        exchange(std::vector<double>(matrix, 0.0)), error({0.0});
    const auto source_dimension = static_cast<std::size_t>(plan->canonical_batch.nbf);
    const auto pairs = source_dimension * (source_dimension + 1U) / 2U;
    const auto quartets = pairs * (pairs + 1U) / 2U;
    require(!plan->bounded_value_opt_in && !direct_jk_generated_full_range_value_available(*plan),
            "work census did not start with canonical production selection");
    for (bool bounded_opt_in : {false, true}) {
      plan->bounded_value_opt_in = bounded_opt_in;
      for (auto operation : {FockOperator::FullRange, FockOperator::LongRange}) {
        auto spec = make_hf_fock_spec(FockSpin::Restricted);
        spec.derivative_order = 0;
        spec.exchange.op = operation;
        spec.exchange.omega = operation == FockOperator::FullRange ? 0.0 : 0.37;
        spec.coulomb.present = operation == FockOperator::FullRange;
        check(cudaStreamSynchronize(plan->stream));
        const auto started = std::chrono::steady_clock::now();
        const auto enqueue_status = enqueue_cuda_direct_jk_device(
            plan.get(), spec, input.pointer, nullptr, matrix,
            spec.coulomb.present ? coulomb.pointer : nullptr, exchange.pointer, nullptr,
            reinterpret_cast<int*>(error.pointer), detail);
        require(enqueue_status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
        check(cudaStreamSynchronize(plan->stream));
        const auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::array<std::uint64_t, 2> work{};
        check(cudaMemcpy(work.data(), plan->canonical_work_count, sizeof(work),
                         cudaMemcpyDeviceToHost));
        int numerical_error{};
        check(cudaMemcpy(&numerical_error, error.pointer, sizeof(int), cudaMemcpyDeviceToHost));
        require(numerical_error == 0, "census source produced nonfinite values");
        // This fixture admits both full and positive-omega range value sources
        // to the bounded provider. Neither opt-in route visits canonical AOs.
        const bool bounded_route = bounded_opt_in;
        if (bounded_route) {
          require(work[0] == 0U && work[1] == 0U,
                  "bounded through-f value unexpectedly entered the canonical AO source");
        } else {
          require(work[0] == quartets && work[1] == quartets,
                  "canonical values repeated or omitted a candidate/radial evaluation");
        }
        std::cout << "{\"public_aos\":" << dimension << ",\"source_aos\":" << source_dimension
                  << ",\"operator\":\""
                  << (operation == FockOperator::FullRange ? "full-JK" : "long-K")
                  << "\",\"route\":\"" << (bounded_route ? "bounded-shell" : "canonical")
                  << "\",\"bounded_value_opt_in\":" << (bounded_opt_in ? "true" : "false")
                  << ",\"candidate_quartets\":" << work[0]
                  << ",\"evaluated_radial_eris\":" << work[1] << ",\"seconds\":" << seconds
                  << ",\"device_bytes\":" << diagnostic.device_bytes << "}\n";
      }
    }
  }
}

void range_exchange_provider() {
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.1, -0.8}}, {1, {0.2, -0.1, 0.7}}};
  system.shells = {{0, 0, {{0.9, 0.7}, {0.25, 0.3}}}, {1, 1, {{0.65, 1.0}}}};
  system.electron_count = 2;
  system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail.c_str());
  const std::size_t n = generativeqc::molecule::ao_count(system), matrix = n * n;
  std::vector<double> alpha(matrix), beta(matrix);
  for (std::size_t ij = 0; ij < matrix; ++ij) {
    alpha[ij] = std::sin(0.41 * (ij / n) - 0.27 * (ij % n)) / n;
    beta[ij] = std::cos(0.23 * (ij / n) + 0.31 * (ij % n)) / (2.0 * n);
  }

  CudaDirectJkPlan* raw{};
  CudaDirectJkDiagnostic diagnostic;
  require(create_cuda_direct_jk_plan(0, {system}, 0, 0.0, 32U * 1024U * 1024U, &raw, diagnostic,
                                     detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
      raw, &destroy_cuda_direct_jk_plan);

  constexpr double omega = 0.37;
  for (const auto [op, radial] :
       {std::pair{FockOperator::ShortRange, generativeqc::integrals::CoulombRange::Short},
        std::pair{FockOperator::LongRange, generativeqc::integrals::CoulombRange::Long}}) {
    const auto expected_alpha = reference_range_exchange(system, alpha, radial, omega);
    const auto expected_beta = reference_range_exchange(system, beta, radial, omega);
    for (const auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
      FockBuildSpec spec;
      spec.spin = spin;
      spec.derivative_order = 0;
      spec.coulomb.present = false;
      spec.exchange = {true, -0.31, op, omega, FockApproximation::Exact};
      std::vector<double> j, ka, kb;
      require(execute_cuda_direct_jk(plan.get(), spec, alpha,
                                     spin == FockSpin::Unrestricted ? beta : std::vector<double>{},
                                     j, ka, kb, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      require(j.empty() && ka.size() == expected_alpha.size() &&
                  kb.size() == (spin == FockSpin::Unrestricted ? expected_beta.size() : 0),
              "range-separated CUDA exchange returned the wrong spin matrix set");
      for (std::size_t i = 0; i < ka.size(); ++i)
        require(std::isfinite(ka[i]) && std::abs(ka[i] - expected_alpha[i]) < 2e-10,
                "range-separated CUDA alpha exchange differs from the CPU range-ERI oracle");
      for (std::size_t i = 0; i < kb.size(); ++i)
        require(std::isfinite(kb[i]) && std::abs(kb[i] - expected_beta[i]) < 2e-10,
                "range-separated CUDA beta exchange differs from the CPU range-ERI oracle");
    }
  }

  // Pick a positive threshold between actual full-range Schwarz products so
  // the qualification checks both retained and skipped range quartets.
  const auto full_eri = generativeqc::integrals::build_integrals(system, false, true).eri;
  std::vector<double> bounds(matrix), products;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k)
      bounds[i * n + k] = std::sqrt(std::abs(full_eri[((i * n + k) * n + i) * n + k]));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          products.push_back(bounds[i * n + k] * bounds[j * n + l]);
  const auto [minimum, maximum] = std::minmax_element(products.begin(), products.end());
  require(*minimum < *maximum, "range-screening fixture has no distinct Schwarz products");
  const double screening = 0.5 * (*minimum + *maximum);

  CudaDirectJkPlan* screened_raw{};
  CudaDirectJkDiagnostic screened_diagnostic;
  require(create_cuda_direct_jk_plan(0, {system}, 0, screening, 32U * 1024U * 1024U, &screened_raw,
                                     screened_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> screened(
      screened_raw, &destroy_cuda_direct_jk_plan);
  for (const auto [op, radial] :
       {std::pair{FockOperator::ShortRange, generativeqc::integrals::CoulombRange::Short},
        std::pair{FockOperator::LongRange, generativeqc::integrals::CoulombRange::Long}}) {
    FockBuildSpec range_spec;
    range_spec.spin = FockSpin::Restricted;
    range_spec.derivative_order = 0;
    range_spec.coulomb.present = false;
    range_spec.exchange = {true, -0.31, op, omega, FockApproximation::Exact};
    std::vector<double> j, ka, kb;
    require(execute_cuda_direct_jk(screened.get(), range_spec, alpha, {}, j, ka, kb, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail.c_str());
    const auto range_eri = generativeqc::integrals::build_range_eri(system, radial, omega);
    std::vector<double> screened_expected(matrix);
    std::size_t skipped = 0, retained = 0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < n; ++k)
          for (std::size_t l = 0; l < n; ++l) {
            if (bounds[i * n + k] * bounds[j * n + l] < screening) {
              ++skipped;
              continue;
            }
            ++retained;
            screened_expected[i * n + j] +=
                alpha[k * n + l] * range_eri[((i * n + k) * n + j) * n + l];
          }
    require(skipped > 0 && retained > 0, "range-screening fixture did not exercise both paths");
    const auto unscreened_expected = reference_range_exchange(system, alpha, radial, omega);
    double screening_effect = 0.0;
    for (std::size_t i = 0; i < matrix; ++i)
      screening_effect =
          std::max(screening_effect, std::abs(screened_expected[i] - unscreened_expected[i]));
    require(screening_effect > 1e-7, "range-screening fixture has no measurable skipped work");
    require(j.empty() && kb.empty() && ka.size() == screened_expected.size(),
            "screened CUDA range exchange returned the wrong matrix set");
    for (std::size_t i = 0; i < ka.size(); ++i)
      require(std::isfinite(ka[i]) && std::abs(ka[i] - screened_expected[i]) < 2e-10,
              "screened CUDA range exchange differs from the screened CPU oracle");
  }
}

/** Independent displaced CPU values check radial seeds and public AO ordering.
 * Use two geometries in one owner to catch atom-offset mistakes, both spin
 * contractions, and s/p/d/f including spherical expansions. No GPU value is
 * reused as the finite-difference oracle.
 */
void range_exchange_derivatives() {
  constexpr double omega = 0.3, coefficient = -0.37, step = 1e-4;
  for (unsigned angular : {0U, 1U, 2U, 3U}) {
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      generativeqc::core::System first;
      first.atoms = {{1, {0.1, -0.2, -0.8}}, {1, {0.3, 0.1, 0.7}}};
      first.shells = {{0, 0, {{0.8, 1.0}}}, {1, angular, {{0.6, 1.0}}}};
      first.electron_count = 2;
      first.basis_representation = representation;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(first, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      auto second = first;
      second.atoms[1].position[0] += 0.23;
      const std::size_t n = generativeqc::molecule::ao_count(first), matrix = n * n;
      std::vector<double> a(matrix), b(matrix), packed_a, packed_b;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          a[i * n + j] = std::cos(0.3 * i + 0.3 * j) / n;
          b[i * n + j] = std::sin(0.4 * i + 0.4 * j) / (2 * n);
        }
      packed_a = a;
      packed_a.insert(packed_a.end(), a.begin(), a.end());
      packed_b = b;
      packed_b.insert(packed_b.end(), b.begin(), b.end());
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, 1, 0.0, 32U << 20, &raw, diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      for (auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
        std::vector<std::vector<double>> gradients;
        for (auto [op, radial] :
             {std::pair{FockOperator::FullRange, generativeqc::integrals::CoulombRange::Full},
              std::pair{FockOperator::ShortRange, generativeqc::integrals::CoulombRange::Short},
              std::pair{FockOperator::LongRange, generativeqc::integrals::CoulombRange::Long}}) {
          FockBuildSpec spec;
          spec.spin = spin;
          spec.derivative_order = 1;
          spec.coulomb.present = false;
          spec.exchange = {true, coefficient, op,
                           radial == generativeqc::integrals::CoulombRange::Full ? 0.0 : omega};
          std::vector<double> gradient;
          require(execute_cuda_direct_energy_derivative(
                      plan.get(), spec, packed_a,
                      spin == FockSpin::Unrestricted ? packed_b : std::vector<double>{}, gradient,
                      detail) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          for (std::size_t item = 0; item < 2; ++item) {
            const auto energy = [&](double shift) {
              auto displaced = item == 0 ? first : second;
              displaced.atoms[1].position[0] += shift;
              const auto ka = reference_range_exchange(displaced, a, radial, omega);
              const auto kb = reference_range_exchange(displaced, b, radial, omega);
              double value = 0.0;
              for (std::size_t ij = 0; ij < matrix; ++ij)
                value += 0.5 * coefficient *
                         (a[ij] * ka[ij] + (spin == FockSpin::Unrestricted ? b[ij] * kb[ij] : 0.0));
              return value;
            };
            const double fd = (energy(step) - energy(-step)) / (2 * step);
            require(std::isfinite(gradient[item * 6 + 3]) &&
                        std::abs(gradient[item * 6 + 3] - fd) < 3e-8,
                    "CUDA radial derivative disagrees with displaced CPU range energy");
            for (std::size_t axis = 0; axis < 3; ++axis)
              require(std::abs(gradient[item * 6 + axis] + gradient[item * 6 + 3 + axis]) < 2e-11,
                      "CUDA radial derivative violates translation invariance");
          }
          gradients.push_back(std::move(gradient));
        }
        for (std::size_t item = 0; item < 2; ++item) {
          FockBuildSpec j_spec;
          j_spec.spin = spin;
          j_spec.derivative_order = 1;
          j_spec.coulomb = {true, 1.0};
          j_spec.exchange.present = false;
          std::vector<double> j_gradient, fused;
          require(execute_cuda_direct_energy_derivative_item(
                      plan.get(), item, j_spec, a,
                      spin == FockSpin::Unrestricted ? b : std::vector<double>{}, j_gradient,
                      detail) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          require(execute_cuda_direct_rsh_energy_derivatives_item(
                      plan.get(), item, spin, 1.0, coefficient, coefficient, omega, a,
                      spin == FockSpin::Unrestricted ? b : std::vector<double>{}, fused,
                      detail) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          require(fused.size() == 18 && j_gradient.size() == 6,
                  "fused CUDA RSH derivative returned the wrong source shape");
          // The public generic provider above can choose canonical derivatives.
          // Qualify the distinct bounded shell route used by molecular WB97M-V
          // as well, including its full-minus-LR reconstruction of the SR source.
          CudaDirectJkPlan* shell_raw{};
          CudaDirectJkDiagnostic shell_diagnostic;
          require(create_cuda_direct_jk_plan(0, {item == 0 ? first : second}, 1, 0.0, 64U << 20,
                                             &shell_raw, shell_diagnostic,
                                             detail) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> shell_plan(
              shell_raw, &destroy_cuda_direct_jk_plan);
          DeviceMatrix device_a(a), device_b(b);
          std::vector<double> shell;
          require(execute_cuda_direct_shell_rsh_energy_derivatives_device(
                      shell_plan.get(), spin, 1.0, coefficient, coefficient, omega,
                      device_a.pointer, spin == FockSpin::Unrestricted ? device_b.pointer : nullptr,
                      matrix, shell, detail) == GENERATIVEQC_STATUS_SUCCESS,
                  detail.c_str());
          require(shell.size() == fused.size(), "bounded shell RSH source shape changed");
          for (std::size_t coordinate = 0; coordinate < shell.size(); ++coordinate)
            require(std::isfinite(shell[coordinate]) &&
                        std::abs(shell[coordinate] - fused[coordinate]) < 3e-10,
                    "bounded shell RSH derivative differs from CPU-qualified canonical source");
          for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
            require(std::abs(fused[coordinate] - j_gradient[coordinate]) < 2e-11,
                    "fused CUDA RSH Coulomb derivative changed");
            require(std::abs(fused[6 + coordinate] - gradients[1][item * 6 + coordinate]) < 2e-11,
                    "fused CUDA short-range derivative changed");
            require(std::abs(fused[12 + coordinate] - gradients[2][item * 6 + coordinate]) < 2e-11,
                    "fused CUDA long-range derivative changed");
          }
        }
        for (std::size_t coordinate = 0; coordinate < 12; ++coordinate)
          require(std::abs(gradients[0][coordinate] - gradients[1][coordinate] -
                           gradients[2][coordinate]) < 2e-11,
                  "CUDA radial full derivative is not short plus long range");
      }
    }
  }
}

/** Four-center CPU displaced values qualify bounded LR accumulation.
 * d/p/s/s covers weighted low orders; d/p/p/s, d/d/p/s and d/d/p/p bind
 * distinct centers at total orders 4, 5 and 6. Repeated-center geometries
 * bind different shells to one atom. Check every coordinate in both spins.
 */
void shell_range_four_center_derivatives() {
  constexpr double omega = 0.3, coefficient = -0.37, step = 1e-4;
  for (const auto angular :
       {std::array<unsigned, 4>{2, 1, 0, 0}, std::array<unsigned, 4>{2, 1, 1, 0},
        std::array<unsigned, 4>{2, 2, 1, 0}, std::array<unsigned, 4>{2, 2, 1, 1}}) {
    for (const bool repeated_center : {false, true}) {
      generativeqc::core::System system;
      system.atoms = {{1, {0.1, -0.2, -0.8}},
                      {1, {0.3, 0.1, 0.7}},
                      {1, {-0.5, 0.6, 0.2}},
                      {1, {0.8, -0.4, 0.3}}};
      system.shells = {{0, angular[0], {{0.8, 1.0}}},
                       {1, angular[1], {{0.6, 1.0}}},
                       {2, angular[2], {{0.7, 1.0}}},
                       {repeated_center ? 1 : 3, angular[3], {{0.9, 1.0}}}};
      system.electron_count = 4;
      system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(system, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      const auto n = generativeqc::molecule::ao_count(system), matrix = n * n;
      std::vector<double> a(matrix), b(matrix);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          a[i * n + j] = std::cos(0.3 * (i + j)) / n;
          b[i * n + j] = std::sin(0.4 * (i + j)) / (2 * n);
        }
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {system}, 1, 0.0, 64U << 20, &raw, diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      DeviceMatrix device_a(a), device_b(b);
      std::array<std::vector<double>, 2> actual;
      for (unsigned spin = 0; spin < 2; ++spin)
        require(execute_cuda_direct_shell_rsh_energy_derivatives_device(
                    plan.get(), spin ? FockSpin::Unrestricted : FockSpin::Restricted, 0.0, 0.0,
                    coefficient, omega, device_a.pointer, spin ? device_b.pointer : nullptr, matrix,
                    actual[spin], detail) == GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
      const auto coordinates = system.atoms.size() * 3U;
      for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
        auto plus = system, minus = system;
        plus.atoms[coordinate / 3].position[coordinate % 3] += step;
        minus.atoms[coordinate / 3].position[coordinate % 3] -= step;
        auto derivative = generativeqc::integrals::build_range_eri(
            plus, generativeqc::integrals::CoulombRange::Long, omega);
        const auto negative = generativeqc::integrals::build_range_eri(
            minus, generativeqc::integrals::CoulombRange::Long, omega);
        for (std::size_t index = 0; index < derivative.size(); ++index)
          derivative[index] = (derivative[index] - negative[index]) / (2 * step);
        for (unsigned spin = 0; spin < 2; ++spin) {
          auto spec = make_hf_fock_spec(spin ? FockSpin::Unrestricted : FockSpin::Restricted);
          spec.derivative_order = 1;
          spec.coulomb.present = false;
          spec.exchange.coefficient = coefficient;
          const auto cpu = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
          const double expected = contract_exact_direct_energy_derivative(
              cpu, n, derivative, a, spin ? b : std::vector<double>{});
          require(actual[spin].size() == 3U * coordinates, "bounded LR source shape changed");
          require(actual[spin][coordinate] == 0.0 && actual[spin][coordinates + coordinate] == 0.0,
                  "disabled bounded RSH source acquired a contribution");
          const double value = actual[spin][2U * coordinates + coordinate];
          require(std::isfinite(value) && std::abs(value - expected) < 3e-8,
                  "four-center bounded LR derivative differs from displaced CPU ERIs");
        }
      }
    }
  }
}

/** Independent CPU ERIs qualify both public source channels, not their sum.
 * Opposing UKS spins make J exactly zero while K remains live; each coefficient
 * mask must also preserve the disabled channel without contaminating its peer.
 */
void full_range_shell_source_oracles(const generativeqc::core::System& system,
                                     std::span<const double> eri_derivatives,
                                     const std::vector<double>& alpha,
                                     const std::vector<double>& beta) {
  const auto dimension = generativeqc::molecule::ao_count(system);
  const auto matrix_size = dimension * dimension;
  const auto coordinates = system.atoms.size() * 3U;
  CudaDirectJkPlan* raw{};
  CudaDirectJkDiagnostic diagnostic;
  std::string detail;
  require(create_cuda_direct_jk_plan(0, {system}, 1, 0.0, 64U << 20, &raw, diagnostic, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
      raw, &destroy_cuda_direct_jk_plan);
  for (const bool unrestricted : {false, true}) {
    for (const bool opposing : {false, true}) {
      auto selected_alpha = alpha;
      auto selected_beta = beta;
      if (opposing) {
        if (unrestricted) {
          for (std::size_t pair = 0; pair < matrix_size; ++pair)
            selected_beta[pair] = -selected_alpha[pair];
        } else {
          std::fill(selected_alpha.begin(), selected_alpha.end(), 0.0);
        }
      }
      DeviceMatrix device_alpha(selected_alpha), device_beta(selected_beta);
      const auto spin = unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted;
      for (unsigned mask = 0; mask < 4; ++mask) {
        const double coulomb = mask & 1U ? 1.7 : 0.0;
        const double exchange = mask & 2U ? -0.23 : 0.0;
        std::vector<double> actual;
        require(execute_cuda_direct_shell_full_range_derivatives_device(
                    plan.get(), spin, coulomb, exchange, device_alpha.pointer,
                    unrestricted ? device_beta.pointer : nullptr, matrix_size, actual,
                    detail) == GENERATIVEQC_STATUS_SUCCESS,
                detail.c_str());
        require(actual.size() == 2U * coordinates, "full-range J/K source shape changed");
        for (unsigned source = 0; source < 2U; ++source) {
          auto spec = make_hf_fock_spec(spin);
          spec.derivative_order = 1;
          spec.coulomb.present = source == 0U && coulomb != 0.0;
          spec.exchange.present = source == 1U && exchange != 0.0;
          spec.coulomb.coefficient = coulomb;
          spec.exchange.coefficient = exchange;
          const auto cpu = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
          for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
            const auto expected = contract_exact_direct_energy_derivative(
                cpu, dimension,
                eri_derivatives.subspan(coordinate * matrix_size * matrix_size,
                                        matrix_size * matrix_size),
                selected_alpha, unrestricted ? selected_beta : std::vector<double>{});
            const auto value = actual[source * coordinates + coordinate];
            require(std::isfinite(value) && std::abs(value - expected) < 3e-10,
                    "independent full-range shell J/K derivative differs from CPU ERIs");
            if (!(mask & (1U << source)))
              require(value == 0.0, "disabled full-range source acquired a contribution");
          }
        }
      }
    }
  }
}

void direct_providers(bool through_f_response) {
  for (unsigned angular : {0U, 1U, 2U, 3U})
    for (auto representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
      // Noncoincident centers and unequal primitive/basis metadata distinguish
      // items with identical dimensions. Through-f values also test public AO
      // expansion ordering. The optional numerical tier adds d/f response
      // without repeating expensive high-angular CPU derivatives for every mask.
      generativeqc::core::System first;
      first.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
      first.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, angular, {{0.6, 1.0}}}};
      // Mixed p/d quartets exercise generated classes that s+d alone cannot
      // reach. Retain unequal contraction lengths and test both AO conventions.
      if (angular == 2) first.shells.push_back({0, 1, {{1.1, 0.6}, {0.3, 0.4}}});
      first.electron_count = 2;
      first.basis_representation = representation;
      auto second = first;
      second.atoms[1].position[2] += 0.13;
      second.shells[1].primitives[0].exponent = 0.9;
      std::string detail;
      require(generativeqc::molecule::validate_and_normalize(first, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      require(generativeqc::molecule::validate_and_normalize(second, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      const bool derivatives = angular < 2 || through_f_response;
      const auto ints = generativeqc::integrals::build_integrals(first, derivatives);
      const auto other = generativeqc::integrals::build_integrals(second, derivatives);
      const std::size_t n = ints.nbf, matrix = n * n;
      std::vector<double> a(matrix), b(matrix), packed_a, packed_b;
      for (std::size_t ij = 0; ij < matrix; ++ij) {
        a[ij] = std::cos(0.3 * (ij / n) + 0.7 * (ij % n)) / n;
        b[ij] = std::sin(0.8 * (ij / n) - 0.2 * (ij % n)) / n;
      }
      if (derivatives) {
        full_range_shell_source_oracles(first, ints.eri_derivative, a, b);
        full_range_shell_source_oracles(second, other.eri_derivative, a, b);
      }
      packed_a = a;
      packed_a.insert(packed_a.end(), a.begin(), a.end());
      packed_b = b;
      packed_b.insert(packed_b.end(), b.begin(), b.end());
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, derivatives ? 1 : 0, 0.0,
                                         64U * 1024U * 1024U, &raw, diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      direct_eri_tile(plan.get(), 0, ints.eri, n);
      direct_eri_tile(plan.get(), 1, other.eri, n);
      if (derivatives) {
        auto value_spec = make_hf_fock_spec(FockSpin::Restricted);
        value_spec.derivative_order = 0;
        require(direct_jk_generated_exchange_value_available(*plan, value_spec) == (angular <= 2),
                "generated SPD/canonical through-f default selection changed");
        if (angular == 3U)
          require(plan->generated_exchange && plan->generated_exchange->bounded_value_capability &&
                      plan->generated_exchange->shared &&
                      !plan->generated_exchange->shared->value_capability,
                  "f-shell value ownership did not use the bounded Direct-HF fallback");
        plan->bounded_value_opt_in = true;
        require(direct_jk_generated_exchange_value_available(*plan, value_spec),
                "explicit bounded qualification did not complete shell value ownership");
        value_spec.derivative_order = 1;
        require(!direct_jk_generated_exchange_value_available(*plan, value_spec),
                "derivative request incorrectly selected generated value exchange");
        plan->bounded_value_opt_in = false;
      }
      // A value-only owner may use generated pure J; exact generic capacity
      // must still be a usable fallback. Both consume nonsymmetric densities
      // and independently formed full ERIs, including spherical d projection.
      const auto shell_count = first.shells.size() + second.shells.size();
      std::size_t primitive_count = 0;
      for (const auto* system : {&first, &second})
        for (const auto& shell : system->shells) primitive_count += shell.primitives.size();
      const auto pure_j_capacity =
          cuda_direct_coulomb_device_bytes(2, n, 4, shell_count, primitive_count);
      CudaDirectJkPlan* pure_j_raw{};
      CudaDirectJkDiagnostic pure_j_diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, 0, 0.0, pure_j_capacity, &pure_j_raw,
                                         pure_j_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> pure_j_plan(
          pure_j_raw, &destroy_cuda_direct_jk_plan);
      require(pure_j_diagnostic.device_bytes <= pure_j_capacity,
              "generated pure J exceeded shape-only capacity");
      require((std::string(pure_j_diagnostic.schedule).find("generated-shell") !=
               std::string::npos) == (angular <= 2),
              "generated pure J admission/class fallback mismatch");

      CudaDirectJkPlan* generated_value_raw{};
      CudaDirectJkDiagnostic generated_value_diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, 0, 0.0, 64U * 1024U * 1024U,
                                         &generated_value_raw, generated_value_diagnostic,
                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)>
          generated_value_plan(generated_value_raw, &destroy_cuda_direct_jk_plan);
      require((std::string(generated_value_diagnostic.schedule).find("coulomb+exchange") !=
               std::string::npos) == (angular <= 2),
              "automatic generated exchange admission/class fallback mismatch");

      CudaDirectJkPlan* fallback_raw{};
      CudaDirectJkDiagnostic fallback_diagnostic;
      const auto fallback_capacity =
          cuda_direct_jk_device_bytes(2, n, 4, shell_count, primitive_count, 0);
      require(
          create_cuda_direct_jk_plan(0, {first, second}, 0, 0.0, fallback_capacity, &fallback_raw,
                                     fallback_diagnostic, detail) == GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> fallback_plan(
          fallback_raw, &destroy_cuda_direct_jk_plan);
      require(fallback_diagnostic.device_bytes == fallback_capacity,
              "bounded generic fallback allocation inventory changed");
      require(diagnostic.nbf == n && diagnostic.batch_size == 2 && diagnostic.device_bytes > 0 &&
                  diagnostic.coordinates_per_item == 6 &&
                  diagnostic.host_preparation_bytes >= diagnostic.host_bytes,
              "direct provider diagnostics mismatch");
      for (bool uhf : {false, true})
        for (bool j : {false, true})
          for (bool k : {false, true}) {
            const bool response = derivatives && (angular < 2 || (uhf && j && k));
            auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
            spec.derivative_order = derivatives ? 1 : 0;
            spec.coulomb.present = j;
            spec.exchange.present = k;
            spec.coulomb.coefficient = 1.7;
            spec.exchange.coefficient = -0.23;
            const auto cpu = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
            std::vector<double> dj, dka, dkb;
            require(execute_cuda_direct_jk(plan.get(), spec, packed_a,
                                           uhf ? packed_b : std::vector<double>{}, dj, dka, dkb,
                                           detail) == GENERATIVEQC_STATUS_SUCCESS,
                    detail.c_str());
            std::vector<double> ej, eka, ekb, gradient;
            for (const auto* data : {&ints, &other}) {
              const auto expected =
                  build_exact_direct_jk(cpu, n, data->eri, a, uhf ? b : std::vector<double>{});
              ej.insert(ej.end(), expected.coulomb.begin(), expected.coulomb.end());
              eka.insert(eka.end(), expected.exchange_alpha.begin(), expected.exchange_alpha.end());
              ekb.insert(ekb.end(), expected.exchange_beta.begin(), expected.exchange_beta.end());
              if (response)
                for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
                  gradient.push_back(contract_exact_direct_energy_derivative(
                      cpu, n,
                      std::span(data->eri_derivative)
                          .subspan(coordinate * matrix * matrix, matrix * matrix),
                      a, uhf ? b : std::vector<double>{}));
            }
            auto compare = [&](const auto& actual, const auto& expected) {
              require(actual.size() == expected.size(), "direct J/K selected shape mismatch");
              for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-10,
                        ("direct CUDA mismatch l=" + std::to_string(angular) +
                         " representation=" + std::to_string(representation) +
                         " spin=" + std::to_string(uhf) + " J=" + std::to_string(j) +
                         " K=" + std::to_string(k) + " element=" + std::to_string(i))
                            .c_str());
            };
            compare(dj, ej);
            compare(dka, eka);
            compare(dkb, ekb);
            direct_device(plan.get(), spec, packed_a, packed_b, ej, eka, ekb);
            if (j) {
              // Generated-capable value plans must preserve independent J/K
              // numerics when J and K are dispatched through separate sources.
              direct_device(pure_j_plan.get(), spec, packed_a, packed_b, ej, eka, ekb);
              direct_device(fallback_plan.get(), spec, packed_a, packed_b, ej, eka, ekb);
            }
            if (angular <= 2 && (j || k)) {
              direct_device(generated_value_plan.get(), spec, packed_a, packed_b, ej, eka, ekb);
            }
            std::vector<double> actual_gradient;
            if (response || !derivatives) {
              const auto status = execute_cuda_direct_energy_derivative(
                  plan.get(), spec, packed_a, uhf ? packed_b : std::vector<double>{},
                  actual_gradient, detail);
              if (response) {
                require(status == GENERATIVEQC_STATUS_SUCCESS, detail.c_str());
                compare(actual_gradient, gradient);
              } else
                require(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "unrequested direct derivatives executed");
            }
          }
      auto fitted = make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted);
      std::vector<double> j{123.0}, ka, kb;
      require(execute_cuda_direct_jk(plan.get(), fitted, packed_a, {}, j, ka, kb, detail) ==
                      GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                  j == std::vector<double>{123.0},
              "direct provider accepted fitted semantics or published partial output");
      CudaDirectJkPlan* too_small{};
      require(create_cuda_direct_jk_plan(0, {first}, 0, 0.0, 1, &too_small, diagnostic, detail) ==
                      GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                  !too_small,
              "direct source ignored its buffer budget");
      auto invalid = first;
      invalid.atoms[0].position[0] = std::numeric_limits<double>::quiet_NaN();
      require(
          create_cuda_direct_jk_plan(0, {invalid}, 0, 0.0, 64U * 1024U * 1024U, &too_small,
                                     diagnostic, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !too_small,
          "nonfinite direct source geometry accepted");
      if (angular == 0 && representation == GENERATIVEQC_BASIS_CARTESIAN) {
        prepared_interaction_source_device(first, ints.eri);
        direct_device_failures(plan.get(), packed_a);
        // Inputs are finite but deliberately outside the stable numerical
        // range. A failed bound must not silently screen the entire source.
        invalid = first;
        invalid.shells[0].primitives[0].coefficient = 1e200;
        require(create_cuda_direct_jk_plan(0, {invalid}, 0, 0.0, 64U * 1024U * 1024U, &too_small,
                                           diagnostic,
                                           detail) == GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                    !too_small,
                "nonfinite direct bound published a source");
        auto spec = make_hf_fock_spec(FockSpin::Restricted);
        auto huge = packed_a;
        std::fill(huge.begin(), huge.end(), std::numeric_limits<double>::max());
        require(execute_cuda_direct_jk(plan.get(), spec, huge, {}, j, ka, kb, detail) ==
                        GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                    j == std::vector<double>{123.0},
                "nonfinite direct matrix published partial output");
        std::fill(huge.begin(), huge.end(), 1e200);
        std::vector<double> gradient{123.0};
        require(
            execute_cuda_direct_energy_derivative(plan.get(), spec, huge, {}, gradient, detail) ==
                    GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                gradient == std::vector<double>{123.0},
            "nonfinite direct derivative published partial output");
        auto opposite = huge;
        for (auto& value : opposite) value = -value;
        spec.spin = FockSpin::Unrestricted;
        spec.exchange.present = false;
        require(execute_cuda_direct_energy_derivative(plan.get(), spec, huge, opposite, gradient,
                                                      detail) == GENERATIVEQC_STATUS_SUCCESS,
                "absent direct exchange contaminated finite total-density response");
        for (double value : gradient) require(value == 0.0, "zero total-density response changed");
      }
    }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--spd-range-only") {
      spd_optional_allocation_fallback();
      spd_canonical_range_values();
      std::cout << "CUDA SPD generated/full and canonical/SR/LR value gates PASS\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--canonical-work-only") {
      canonical_work_census();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--canonical-values-only") {
      spd_optional_allocation_fallback();
      spd_canonical_range_values();
      canonical_screening_rows();
      canonical_screened_values();
      canonical_one_electron_reuse();
      canonical_value_provider();
      std::cout << "CUDA canonical s/p/d/f full/SR/LR value gates PASS\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--mixed-census-only") {
      mixed_coulomb_work_census();
      mixed_coulomb_work_census(true);
      mixed_coulomb_preserves_strict_exchange();
      std::cout << "CUDA mixed Coulomb work census and strict exchange PASS\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--range-response-only") {
      range_exchange_derivatives();
      shell_range_four_center_derivatives();
      canonical_order_two_derivatives();
      std::cout << "CUDA s/p/d/f SR/LR derivative gates PASS\n";
      return 0;
    }
    require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--through-f-response"),
            "expected optional --mixed-census-only, --range-response-only, or "
            "--through-f-response");
    const bool through_f_response = argc == 2;
    spd_optional_allocation_fallback();
    spd_canonical_range_values();
    direct_value_dispatch_selection();
    mixed_coulomb_work_census();
    mixed_coulomb_work_census(true);
    mixed_coulomb_preserves_strict_exchange();
    device_selection();
    range_exchange_provider();
    range_exchange_derivatives();
    shell_range_four_center_derivatives();
    canonical_order_two_derivatives();
    direct_providers(through_f_response);
    std::cout << "CUDA independent J/K: DF layouts/selection and direct through-f values, "
              << (through_f_response ? "through-f" : "s/p") << " derivatives PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
